"""
GEPA-based Refiner Prompt Optimizer

Optimizes the critique and refinement prompts of the self-correction refiner
using GEPA's reflective prompt evolution instead of RLHF.

Usage:
    python gepa_optimize_refiner.py --max_iterations 10 --reflection_temp 0.7
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Add paths
sys.path.insert(0, './gepa/src')

import torch

from gepa_refiner_adapter import (
    ContrastiveVerifierAdapter,
    RefinerDataInst,
    load_verifier,
    load_refiner_model,
    prepare_training_data,
)


# =============================================================================
# Seed Prompts (Current Baseline)
# =============================================================================

SEED_PROMPTS = {
    "critique_prompt": """Review this solution step by step. Identify any errors in mathematical reasoning, calculations, or logic. Be specific about what went wrong.""",
    
    "refinement_prompt": """Based on the critique above, provide a corrected step-by-step solution. End your final answer with \\boxed{answer}."""
}


# =============================================================================
# Reflection Prompt Template for GEPA
# =============================================================================

REFLECTION_PROMPT_TEMPLATE = """I provided an assistant with the following instructions to generate critiques and refinements for math solutions:

```
<curr_instructions>
```

The following are examples of different inputs provided to the assistant, along with its critique/refinement outputs and feedback on how well the refinement worked:

```
<inputs_outputs_feedback>
```

Your task is to write improved instructions for the assistant.

Key observations from the examples:
1. Look at cases where refinement SUCCEEDED - what made the critique effective?
2. Look at cases where refinement FAILED - was the critique missing the actual error?
3. Look at REGRESSIONS where correct solutions became incorrect - what went wrong?

When writing new instructions:
- Include specific strategies that worked in successful cases
- Add warnings about common failure patterns observed
- Include domain-specific math reasoning guidance
- Be explicit about how to identify calculation vs. conceptual errors

Provide the new instructions within ``` blocks.
"""


# =============================================================================
# Simple GEPA Optimization Loop (Self-Contained)
# =============================================================================

class SimpleGEPAOptimizer:
    """
    Simplified GEPA optimizer for the refiner prompts.
    Uses the core GEPA principles without the full framework complexity.
    """
    
    def __init__(
        self,
        adapter: ContrastiveVerifierAdapter,
        trainset: list[RefinerDataInst],
        valset: list[RefinerDataInst],
        reflection_model=None,
        reflection_tokenizer=None,
        device: str = "cuda"
    ):
        self.adapter = adapter
        self.trainset = trainset
        self.valset = valset
        self.reflection_model = reflection_model
        self.reflection_tokenizer = reflection_tokenizer
        self.device = device
        
        # State
        self.candidates: list[dict[str, str]] = []
        self.scores: list[float] = []
        self.history: list[dict] = []
    
    def _evaluate_on_valset(self, candidate: dict[str, str]) -> float:
        """Evaluate candidate on full validation set"""
        eval_result = self.adapter.evaluate(self.valset, candidate, capture_traces=False)
        return sum(eval_result.scores) / len(eval_result.scores) if eval_result.scores else 0.0
    
    def _sample_minibatch(self, size: int = 5) -> list[RefinerDataInst]:
        """Sample a minibatch from training set"""
        import random
        return random.sample(self.trainset, min(size, len(self.trainset)))
    
    def _generate_reflection(
        self,
        current_prompt: str,
        reflective_dataset: list[dict],
        component_name: str
    ) -> str:
        """Use reflection LLM to propose improved prompt"""
        
        # Format the reflective dataset
        formatted_examples = []
        for i, record in enumerate(reflective_dataset):
            example = f"""### Example {i+1}
Question: {record.get('Question', 'N/A')[:200]}...
Original Solution (excerpt): {record.get('Original_Solution', 'N/A')[:200]}...
Critique Generated: {record.get('Critique_Generated', 'N/A')[:300]}...
Refined Solution (excerpt): {record.get('Refined_Solution', 'N/A')[:200]}...
Feedback: {record.get('Feedback', 'N/A')}
"""
            formatted_examples.append(example)
        
        examples_text = "\n\n".join(formatted_examples)
        
        # Build reflection prompt
        reflection_prompt = REFLECTION_PROMPT_TEMPLATE.replace(
            "<curr_instructions>", current_prompt
        ).replace(
            "<inputs_outputs_feedback>", examples_text
        )
        
        # Generate using Gemma
        full_prompt = f"""<start_of_turn>user
{reflection_prompt}<end_of_turn>
<start_of_turn>model
"""
        
        inputs = self.reflection_tokenizer(
            full_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.reflection_model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.reflection_tokenizer.eos_token_id
            )
        
        response = self.reflection_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract instruction from response
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[-1].strip()
        
        # Extract content within ``` blocks
        import re
        match = re.search(r'```(?:\w*\n)?(.+?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback: return response after first ```
        if "```" in response:
            parts = response.split("```")
            if len(parts) > 1:
                return parts[1].strip()
        
        return response.strip()
    
    def optimize(
        self,
        seed_candidate: dict[str, str],
        max_iterations: int = 10,
        minibatch_size: int = 5,
        components_to_optimize: list[str] | None = None
    ) -> dict[str, str]:
        """
        Run GEPA optimization loop.
        """
        if components_to_optimize is None:
            components_to_optimize = ["critique_prompt", "refinement_prompt"]
        
        # Initialize with seed
        current_candidate = seed_candidate.copy()
        self.candidates.append(current_candidate.copy())
        
        # Evaluate seed
        seed_score = self._evaluate_on_valset(current_candidate)
        self.scores.append(seed_score)
        print(f"Seed prompt score: {seed_score:.4f}")
        
        best_candidate = current_candidate.copy()
        best_score = seed_score
        
        for iteration in range(max_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{max_iterations}")
            print(f"{'='*60}")
            
            # Round-robin component selection
            component_idx = iteration % len(components_to_optimize)
            component = components_to_optimize[component_idx]
            print(f"Optimizing component: {component}")
            
            # Sample minibatch
            minibatch = self._sample_minibatch(minibatch_size)
            
            # Evaluate current candidate with traces
            eval_result = self.adapter.evaluate(minibatch, current_candidate, capture_traces=True)
            minibatch_score = sum(eval_result.scores) / len(eval_result.scores)
            print(f"Minibatch score: {minibatch_score:.4f}")
            
            # Build reflective dataset
            reflective_dataset = self.adapter.make_reflective_dataset(
                current_candidate, eval_result, [component]
            )
            
            # Generate improved prompt via reflection
            print("Generating improved prompt via reflection...")
            new_prompt = self._generate_reflection(
                current_candidate[component],
                list(reflective_dataset.get(component, [])),
                component
            )
            
            # Create new candidate
            new_candidate = current_candidate.copy()
            new_candidate[component] = new_prompt
            
            print(f"\nProposed new {component}:\n{new_prompt[:300]}...")
            
            # Evaluate new candidate on same minibatch
            new_eval = self.adapter.evaluate(minibatch, new_candidate, capture_traces=False)
            new_minibatch_score = sum(new_eval.scores) / len(new_eval.scores)
            print(f"New minibatch score: {new_minibatch_score:.4f}")
            
            # Accept if improved on minibatch
            if new_minibatch_score > minibatch_score:
                print("✓ Minibatch improved - evaluating on full valset...")
                
                # Full validation
                new_val_score = self._evaluate_on_valset(new_candidate)
                print(f"Validation score: {new_val_score:.4f} (was {best_score:.4f})")
                
                # Accept into candidate pool
                self.candidates.append(new_candidate.copy())
                self.scores.append(new_val_score)
                current_candidate = new_candidate.copy()
                
                if new_val_score > best_score:
                    best_score = new_val_score
                    best_candidate = new_candidate.copy()
                    print(f"★ New best! Score: {best_score:.4f}")
            else:
                print("✗ No improvement on minibatch - keeping current candidate")
            
            # Log history
            self.history.append({
                "iteration": iteration + 1,
                "component": component,
                "minibatch_score_before": minibatch_score,
                "minibatch_score_after": new_minibatch_score,
                "accepted": new_minibatch_score > minibatch_score,
                "best_val_score": best_score
            })
        
        print(f"\n{'='*60}")
        print(f"Optimization complete!")
        print(f"Best validation score: {best_score:.4f} (started at {seed_score:.4f})")
        print(f"Improvement: {(best_score - seed_score):.4f} ({(best_score - seed_score) / max(seed_score, 0.001) * 100:.1f}%)")
        print(f"{'='*60}")
        
        return best_candidate


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GEPA-based refiner prompt optimizer")
    parser.add_argument("--max_iterations", type=int, default=10, help="Number of optimization iterations")
    parser.add_argument("--minibatch_size", type=int, default=5, help="Minibatch size for reflection")
    parser.add_argument("--max_train_samples", type=int, default=-1, help="Max training samples (-1 for all)")
    parser.add_argument("--score_threshold", type=float, default=0.7, help="Score threshold for selecting low-confidence examples")
    parser.add_argument("--verifier_checkpoint", type=str, default="verifier_best.pt", help="Path to verifier checkpoint")
    parser.add_argument("--scored_file", type=str, default="scored_outputs.jsonl", help="Path to scored outputs")
    parser.add_argument("--original_file", type=str, default="generations_google-gemma-2b-it_0_-1.jsonl", help="Path to original generations")
    parser.add_argument("--output_file", type=str, default="evolved_prompts.json", help="Output file for evolved prompts")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()
    
    print("="*60)
    print("GEPA Refiner Prompt Optimizer")
    print("="*60)
    print(f"Max iterations: {args.max_iterations}")
    print(f"Minibatch size: {args.minibatch_size}")
    print(f"Max train samples: {args.max_train_samples}")
    print("="*60)
    
    # Check for required files
    for required_file in [args.verifier_checkpoint, args.scored_file, args.original_file]:
        if not os.path.exists(required_file):
            print(f"ERROR: Required file not found: {required_file}")
            sys.exit(1)
    
    # Load models
    print("\n1. Loading verifier model...")
    verifier_model, verifier_tokenizer = load_verifier(args.verifier_checkpoint, args.device)
    
    print("\n2. Loading refiner model (also used for reflection)...")
    refiner_model, refiner_tokenizer = load_refiner_model("google/gemma-2-2b-it", args.device)
    
    # Prepare data
    print("\n3. Preparing training data...")
    trainset, valset = prepare_training_data(
        scored_file=args.scored_file,
        original_file=args.original_file,
        max_samples=args.max_train_samples,
        score_threshold=args.score_threshold
    )
    
    if len(trainset) == 0:
        print("ERROR: No training examples found with low confidence scores!")
        print("Try lowering --score_threshold")
        sys.exit(1)
    
    # Create adapter
    print("\n4. Creating GEPA adapter...")
    adapter = ContrastiveVerifierAdapter(
        verifier_model=verifier_model,
        verifier_tokenizer=verifier_tokenizer,
        refiner_model=refiner_model,
        refiner_tokenizer=refiner_tokenizer,
        device=args.device
    )
    
    # Create optimizer
    print("\n5. Setting up optimizer...")
    optimizer = SimpleGEPAOptimizer(
        adapter=adapter,
        trainset=trainset,
        valset=valset,
        reflection_model=refiner_model,
        reflection_tokenizer=refiner_tokenizer,
        device=args.device
    )
    
    # Run optimization with exception handling
    print("\n6. Running GEPA optimization...")
    best_prompts = SEED_PROMPTS.copy()  # Default to seed if exception
    
    try:
        best_prompts = optimizer.optimize(
            seed_candidate=SEED_PROMPTS,
            max_iterations=args.max_iterations,
            minibatch_size=args.minibatch_size
        )
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"EXCEPTION OCCURRED: {type(e).__name__}: {e}")
        print(f"{'='*60}")
        print("Saving partial results...")
        
        # Get best candidate so far
        if optimizer.candidates:
            best_idx = optimizer.scores.index(max(optimizer.scores)) if optimizer.scores else 0
            best_prompts = optimizer.candidates[best_idx]
    
    # Save results (always runs, even after exception)
    print(f"\n7. Saving results to {args.output_file}...")
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "seed_prompts": SEED_PROMPTS,
        "evolved_prompts": best_prompts,
        "history": optimizer.history,
        "all_candidates": optimizer.candidates,
        "all_scores": optimizer.scores,
        "final_scores": {
            "seed": optimizer.scores[0] if optimizer.scores else 0,
            "best": max(optimizer.scores) if optimizer.scores else 0
        }
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print full JSON to console
    print(f"\n{'='*60}")
    print("FULL RESULTS JSON:")
    print(f"{'='*60}")
    print(json.dumps(results, indent=2))
    
    print(f"\nDone! Results saved to {args.output_file}")
    
    # Print evolved prompts
    print("\n" + "="*60)
    print("EVOLVED PROMPTS")
    print("="*60)
    for name, prompt in best_prompts.items():
        print(f"\n### {name}:\n{prompt}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

