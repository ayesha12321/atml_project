"""
GEPA Adapter for Contrastive Verifier Refiner Optimization

This adapter integrates GEPA's reflective prompt evolution with the 
contrastive verifier system to optimize critique and refinement prompts.
"""

import re
import sys
import os
import json
from typing import Any, TypedDict
from collections.abc import Mapping, Sequence

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use system env vars

# Set HF token if available
if os.getenv("HF_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")

# Add GEPA to path
sys.path.insert(0, './gepa/src')

from gepa.core.adapter import EvaluationBatch, GEPAAdapter


# =============================================================================
# Data Types
# =============================================================================

class RefinerDataInst(TypedDict):
    """Input data for a single refinement task"""
    question: str
    solution: str  # Low-confidence solution to refine
    ground_truth: str  # Expected answer (from GSM8K #### format)
    original_score: float  # Verifier score before refinement


class RefinerTrajectory(TypedDict):
    """Execution trace for reflection"""
    data: RefinerDataInst
    critique: str
    refined_solution: str
    original_verifier_score: float
    refined_verifier_score: float
    original_correct: bool
    refined_correct: bool


class RefinerRolloutOutput(TypedDict):
    """Output from refinement"""
    refined_solution: str
    final_score: float
    improved: bool


class RefinerReflectiveRecord(TypedDict):
    """Record for reflection LLM"""
    Question: str
    Original_Solution: str
    Critique_Generated: str
    Refined_Solution: str
    Feedback: str


# =============================================================================
# Answer Extraction & Verification (from existing codebase)
# =============================================================================

def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{} format"""
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    return None


def normalize_answer(ans: str | None) -> str | None:
    """Normalize answer for comparison"""
    if ans is None:
        return None
    ans = str(ans).replace(',', '').replace('$', '').replace('%', '')
    ans = ans.strip().lower()
    match = re.search(r'-?\d+\.?\d*', ans)
    if match:
        return match.group(0)
    return ans


def check_correctness(generated: str, ground_truth: str) -> bool:
    """Check if generated answer matches ground truth"""
    gen_ans = extract_boxed_answer(generated)
    gen_norm = normalize_answer(gen_ans)
    gt_norm = normalize_answer(ground_truth)
    
    if gen_norm is None or gt_norm is None:
        return False
    
    return gen_norm == gt_norm


# =============================================================================
# Configurable Refiner
# =============================================================================

class ConfigurableRefiner:
    """
    Refiner with configurable prompts for GEPA optimization.
    """
    
    DEFAULT_CRITIQUE_PROMPT = """Review this solution step by step. Identify any errors in mathematical reasoning, calculations, or logic. Be specific about what went wrong."""
    
    DEFAULT_REFINEMENT_PROMPT = """Based on the critique above, provide a corrected step-by-step solution. End your final answer with \\boxed{answer}."""
    
    def __init__(
        self, 
        model=None, 
        tokenizer=None,
        critique_prompt: str | None = None,
        refinement_prompt: str | None = None,
        device: str = "cuda"
    ):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.critique_prompt = critique_prompt or self.DEFAULT_CRITIQUE_PROMPT
        self.refinement_prompt = refinement_prompt or self.DEFAULT_REFINEMENT_PROMPT
        
    def set_prompts(self, critique_prompt: str, refinement_prompt: str):
        """Update prompts (used by GEPA optimization)"""
        self.critique_prompt = critique_prompt
        self.refinement_prompt = refinement_prompt
    
    def generate_critique(self, question: str, solution: str, max_length: int = 256) -> str:
        """Generate critique using configurable prompt"""
        import torch
        
        prompt = f"""<start_of_turn>user
Question: {question}

Proposed Solution:
{solution}

{self.critique_prompt}

Critique:<end_of_turn>
<start_of_turn>model
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        critique = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<start_of_turn>model" in critique:
            critique = critique.split("<start_of_turn>model")[-1].strip()
        
        return critique
    
    def generate_refinement(self, question: str, original_solution: str, critique: str, max_length: int = 512) -> str:
        """Generate refined solution using configurable prompt"""
        import torch
        
        prompt = f"""<start_of_turn>user
Question: {question}

Original Solution:
{original_solution}

Critique of the solution:
{critique}

{self.refinement_prompt}

Corrected Solution:<end_of_turn>
<start_of_turn>model
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        refinement = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<start_of_turn>model" in refinement:
            refinement = refinement.split("<start_of_turn>model")[-1].strip()
        
        return refinement
    
    def refine_solution(self, question: str, solution: str) -> dict[str, str]:
        """Complete refinement: critique + refinement"""
        critique = self.generate_critique(question, solution)
        refinement = self.generate_refinement(question, solution, critique)
        
        return {
            'critique': critique,
            'refined_solution': refinement
        }


# =============================================================================
# GEPA Adapter
# =============================================================================

class ContrastiveVerifierAdapter(GEPAAdapter[RefinerDataInst, RefinerTrajectory, RefinerRolloutOutput]):
    """
    GEPA Adapter for optimizing refiner prompts using verifier feedback.
    
    Candidate structure:
        {
            "critique_prompt": "...",
            "refinement_prompt": "..."
        }
    
    Scoring:
        - 1.0 if refined solution is correct
        - 0.0 if incorrect (with partial credit for score improvement)
        - -0.5 if refinement made correct solution incorrect
    """
    
    def __init__(
        self,
        verifier_model,
        verifier_tokenizer,
        refiner_model,
        refiner_tokenizer,
        device: str = "cuda"
    ):
        self.device = device
        
        # Store verifier components
        self.verifier_model = verifier_model
        self.verifier_tokenizer = verifier_tokenizer
        
        # Create configurable refiner
        self.refiner = ConfigurableRefiner(
            model=refiner_model,
            tokenizer=refiner_tokenizer,
            device=device
        )
    
    def _score_solution(self, question: str, solution: str) -> float:
        """Score a solution using the verifier"""
        import torch
        
        text = f"Question: {question}\nSolution: {solution}"
        
        encoding = self.verifier_tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            score = self.verifier_model(input_ids, attention_mask)
        
        return score.item()
    
    def _compute_final_score(
        self,
        original_correct: bool,
        refined_correct: bool,
        original_score: float,
        refined_score: float
    ) -> float:
        """
        Compute final score for GEPA optimization.
        
        Scoring logic:
        - Refined correct: 1.0 (success!)
        - Made worse (was correct, now incorrect): -0.5
        - Still incorrect but improved score: partial credit
        """
        if refined_correct:
            return 1.0
        elif original_correct and not refined_correct:
            return -0.5  # Penalize regression
        else:
            # Still incorrect - give partial credit for score improvement
            improvement = max(0, refined_score - original_score)
            return improvement * 0.5  # Scale to [0, 0.5]
    
    def evaluate(
        self,
        batch: list[RefinerDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False
    ) -> EvaluationBatch[RefinerTrajectory, RefinerRolloutOutput]:
        """
        Evaluate the refiner with given critique/refinement prompts.
        """
        # Update refiner prompts from candidate
        self.refiner.set_prompts(
            critique_prompt=candidate["critique_prompt"],
            refinement_prompt=candidate["refinement_prompt"]
        )
        
        outputs: list[RefinerRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[RefinerTrajectory] | None = [] if capture_traces else None
        
        for data in batch:
            try:
                # Run refinement
                result = self.refiner.refine_solution(data["question"], data["solution"])
                critique = result["critique"]
                refined_solution = result["refined_solution"]
                
                # Score refined solution
                refined_score = self._score_solution(data["question"], refined_solution)
                
                # Check correctness
                original_correct = check_correctness(data["solution"], data["ground_truth"])
                refined_correct = check_correctness(refined_solution, data["ground_truth"])
                
                # Compute final score
                final_score = self._compute_final_score(
                    original_correct, refined_correct,
                    data["original_score"], refined_score
                )
                
                output: RefinerRolloutOutput = {
                    "refined_solution": refined_solution,
                    "final_score": final_score,
                    "improved": refined_correct and not original_correct
                }
                outputs.append(output)
                scores.append(final_score)
                
                if trajectories is not None:
                    trajectory: RefinerTrajectory = {
                        "data": data,
                        "critique": critique,
                        "refined_solution": refined_solution,
                        "original_verifier_score": data["original_score"],
                        "refined_verifier_score": refined_score,
                        "original_correct": original_correct,
                        "refined_correct": refined_correct
                    }
                    trajectories.append(trajectory)
                    
            except Exception as e:
                # Handle failures gracefully
                print(f"Error processing example: {e}")
                outputs.append({
                    "refined_solution": "",
                    "final_score": 0.0,
                    "improved": False
                })
                scores.append(0.0)
                
                if trajectories is not None:
                    trajectories.append({
                        "data": data,
                        "critique": f"ERROR: {str(e)}",
                        "refined_solution": "",
                        "original_verifier_score": data["original_score"],
                        "refined_verifier_score": 0.0,
                        "original_correct": False,
                        "refined_correct": False
                    })
        
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories
        )
    
    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[RefinerTrajectory, RefinerRolloutOutput],
        components_to_update: list[str]
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """
        Build reflective dataset for LLM to propose improved prompts.
        """
        trajectories = eval_batch.trajectories
        assert trajectories is not None, "Trajectories required for reflection"
        
        result: dict[str, list[RefinerReflectiveRecord]] = {}
        
        for component in components_to_update:
            records: list[RefinerReflectiveRecord] = []
            
            for traj, score in zip(trajectories, eval_batch.scores):
                data = traj["data"]
                
                # Build detailed feedback
                if traj["refined_correct"]:
                    feedback = f"SUCCESS: Refinement corrected the answer. The refined solution now contains the correct answer '{data['ground_truth']}'."
                elif traj["original_correct"] and not traj["refined_correct"]:
                    feedback = f"REGRESSION: The original solution was correct but refinement made it incorrect. Original answer was correct, refined answer is wrong. Correct answer: '{data['ground_truth']}'. The critique may have been misleading or the refinement introduced new errors."
                else:
                    score_delta = traj["refined_verifier_score"] - traj["original_verifier_score"]
                    if score_delta > 0:
                        feedback = f"PARTIAL IMPROVEMENT: Still incorrect but verifier score improved by {score_delta:.3f}. The correct answer is '{data['ground_truth']}'. The refinement is heading in the right direction but needs to be more thorough."
                    else:
                        feedback = f"NO IMPROVEMENT: Refinement did not help. Verifier score changed by {score_delta:.3f}. The correct answer is '{data['ground_truth']}'. The critique may not have identified the actual errors, or the refinement didn't address them properly."
                
                record: RefinerReflectiveRecord = {
                    "Question": data["question"],
                    "Original_Solution": data["solution"],
                    "Critique_Generated": traj["critique"],
                    "Refined_Solution": traj["refined_solution"],
                    "Feedback": feedback
                }
                records.append(record)
            
            result[component] = records
        
        return result


# =============================================================================
# Utility Functions
# =============================================================================

def load_verifier(checkpoint_path: str, device: str = "cuda"):
    """Load trained verifier model"""
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    import os
    
    class VerifierModel(nn.Module):
        def __init__(self, model_name='microsoft/deberta-v3-base'):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            
            self.scorer = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            )
        
        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            score = self.scorer(cls_embedding)
            return score.squeeze(-1)
    
    # Check if file exists and has reasonable size
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Verifier checkpoint not found: {checkpoint_path}\n"
                               f"Please train the verifier first: python train_verifier.py")
    
    file_size = os.path.getsize(checkpoint_path)
    if file_size < 1000000:  # Less than 1MB is suspicious
        raise ValueError(f"Verifier checkpoint appears corrupted or incomplete: {checkpoint_path}\n"
                        f"File size: {file_size} bytes (expected ~400MB)\n"
                        f"Please re-upload or retrain: python train_verifier.py")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except RuntimeError as e:
        if "failed finding central directory" in str(e) or "PytorchStreamReader" in str(e):
            raise RuntimeError(f"Verifier checkpoint is corrupted: {checkpoint_path}\n"
                              f"This usually happens when the file was not fully uploaded.\n"
                              f"Solutions:\n"
                              f"  1. Re-upload the file using: rsync -avzP verifier_best.pt root@<pod-ip>:/workspace/\n"
                              f"  2. Or retrain: python train_verifier.py") from e
        raise
    tokenizer_name = checkpoint.get('tokenizer_name', 'microsoft/deberta-v3-base')
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = VerifierModel(tokenizer_name).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded verifier from {checkpoint_path} (val_acc: {checkpoint.get('val_acc', 'N/A')})")
    
    return model, tokenizer


def load_refiner_model(model_name: str = "google/gemma-2-2b-it", device: str = "cuda"):
    """Load Gemma model for refinement"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print(f"Loading refiner model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    model.eval()
    print("Refiner model loaded!")
    
    return model, tokenizer


def prepare_training_data(
    scored_file: str,
    original_file: str,
    max_samples: int = 500,
    score_threshold: float = 0.7
) -> tuple[list[RefinerDataInst], list[RefinerDataInst]]:
    """
    Prepare training/validation data from scored outputs.
    Selects low-confidence solutions that need refinement.
    """
    import random
    
    # Load scored data (with error handling for corrupted lines)
    scored_data = []
    skipped = 0
    with open(scored_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                scored_data.append(json.loads(line))
            except json.JSONDecodeError as e:
                skipped += 1
                if skipped <= 3:
                    print(f"Warning: Skipping corrupted line {line_num}: {str(e)[:50]}")
    
    if skipped > 0:
        print(f"Skipped {skipped} corrupted lines in {scored_file}")
    
    if len(scored_data) == 0:
        raise ValueError(f"No valid data found in {scored_file}. File may be corrupted.")
    
    # Load ground truth (with error handling)
    gt_answers = {}
    with open(original_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line)
                # Extract ground truth answer
                match = re.search(r'####\s*(.+)$', item['answer'].strip())
                if match:
                    gt_answers[item['question']] = match.group(1).strip()
            except json.JSONDecodeError:
                pass  # Skip corrupted lines silently
    
    # Find low-confidence solutions
    instances: list[RefinerDataInst] = []
    
    for item in scored_data:
        question = item['question']
        gt = gt_answers.get(question)
        
        if gt is None:
            continue
        
        # Get best solution and its score
        scores = item['verifier_scores']
        solutions = item['generated_answers']
        best_idx = scores.index(max(scores))
        best_score = scores[best_idx]
        best_solution = solutions[best_idx]
        
        # Only include if below threshold (needs refinement)
        if best_score < score_threshold:
            instances.append({
                "question": question,
                "solution": best_solution,
                "ground_truth": gt,
                "original_score": best_score
            })
    
    # Shuffle and limit (if max_samples > 0)
    random.shuffle(instances)
    if max_samples > 0:
        instances = instances[:max_samples]
    
    # Split 90/10 train/val
    split_idx = int(len(instances) * 0.9)
    trainset = instances[:split_idx]
    valset = instances[split_idx:]
    
    print(f"Prepared {len(trainset)} training, {len(valset)} validation examples")
    
    return trainset, valset
