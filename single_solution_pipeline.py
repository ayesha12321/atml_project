import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import json
from tqdm import tqdm

# Verifier model
class VerifierModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
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

class SingleSolutionSystem:
    def __init__(self, verifier_checkpoint, gemma_model="google/gemma-2-2b-it"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load verifier
        print("Loading verifier...")
        checkpoint = torch.load(verifier_checkpoint, map_location=self.device)
        self.verifier_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False)
        self.verifier = VerifierModel('bert-base-uncased').to(self.device)
        self.verifier.load_state_dict(checkpoint['model_state_dict'])
        self.verifier.eval()
        print(f"Verifier loaded (val_acc: {checkpoint['val_acc']:.4f})")
        
        # Load Gemma
        print(f"Loading {gemma_model}...")
        self.gemma_tokenizer = AutoTokenizer.from_pretrained(gemma_model)
        self.gemma = AutoModelForCausalLM.from_pretrained(
            gemma_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.gemma.eval()
        print("System ready!")
    
    def score_solution(self, question, solution):
        """Score a solution"""
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
            score = self.verifier(input_ids, attention_mask)
        
        return score.item()
    
    def generate_solution(self, question):
        """Generate single solution"""
        prompt = f"""Solve this math problem step by step. Show your work and end with your final answer in \\boxed{{answer}} format.

Question: {question}

Solution:"""
        
        inputs = self.gemma_tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.gemma.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.gemma_tokenizer.eos_token_id
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        solution = self.gemma_tokenizer.decode(generated_ids, skip_special_tokens=True)
        return solution.strip()
    
    def generate_critique(self, question, solution):
        """Generate critique"""
        prompt = f"""Question: {question}

Proposed Solution:
{solution}

Review this solution step by step. Identify any errors in mathematical reasoning, calculations, or logic. Be specific about what went wrong.

Critique:"""
        
        inputs = self.gemma_tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.gemma.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.gemma_tokenizer.eos_token_id
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        critique = self.gemma_tokenizer.decode(generated_ids, skip_special_tokens=True)
        return critique.strip()
    
    def generate_refinement(self, question, original_solution, critique):
        """Generate refinement"""
        prompt = f"""Question: {question}

Original Solution:
{original_solution}

Critique of the solution:
{critique}

Based on the critique above, provide a corrected step-by-step solution. End your final answer with \\boxed{{answer}}.

Corrected Solution:"""
        
        inputs = self.gemma_tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.gemma.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.gemma_tokenizer.eos_token_id
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        refinement = self.gemma_tokenizer.decode(generated_ids, skip_special_tokens=True)
        return refinement.strip()
    
    def process_question(self, question, threshold=0.7):
        """Complete single-solution pipeline"""
        
        # Step 1: Generate one solution
        print("  Generating solution...")
        solution = self.generate_solution(question)
        
        # Step 2: Score it
        score = self.score_solution(question, solution)
        print(f"  Original score: {score:.3f}")
        
        result = {
            'original_solution': solution,
            'original_score': score,
            'refinement_triggered': False,
            'refinement_improved': False,
            'final_solution': solution,
            'final_score': score
        }
        
        # Step 3: Refine if below threshold
        if score < threshold:
            print(f"  Score < {threshold}, triggering refinement...")
            result['refinement_triggered'] = True
            
            # Generate critique
            critique = self.generate_critique(question, solution)
            print(f"  Critique generated")
            
            # Generate refinement
            refined_solution = self.generate_refinement(question, solution, critique)
            print(f"  Refinement generated")
            
            # Re-score
            refined_score = self.score_solution(question, refined_solution)
            print(f"  Refined score: {refined_score:.3f}")
            
            result['critique'] = critique
            result['refined_solution'] = refined_solution
            result['refined_score'] = refined_score
            
            # Use refined if better
            if refined_score > score:
                result['final_solution'] = refined_solution
                result['final_score'] = refined_score
                result['refinement_improved'] = True
                print(f"  ✓ Refinement improved score!")
            else:
                print(f"  ✗ Refinement didn't improve, keeping original")
        else:
            print(f"  Score >= {threshold}, no refinement needed")
        
        return result

def run_single_solution_experiment(
    test_questions_file="gsm8k_test_questions.jsonl",
    output_file="single_solution_results.jsonl",
    verifier_checkpoint="verifier_best.pt",
    threshold=0.7,
    max_questions=None
):
    """Run single-solution experiment"""
    
    # Load questions
    print(f"Loading questions from {test_questions_file}...")
    questions = []
    with open(test_questions_file, 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    
    if max_questions:
        questions = questions[:max_questions]
        print(f"Running on first {max_questions} questions (test mode)")
    
    print(f"Total questions: {len(questions)}\n")
    
    # Initialize system
    system = SingleSolutionSystem(verifier_checkpoint)
    
    # Process each question
    results = []
    stats = {
        'total': len(questions),
        'refinement_triggered': 0,
        'refinement_improved': 0
    }
    
    for i, item in enumerate(questions):
        question = item['question']
        print(f"\n[{i+1}/{len(questions)}] Processing...")
        print(f"Question: {question[:80]}...")
        
        result = system.process_question(question, threshold=threshold)
        
        # Add metadata
        result['question'] = question
        result['ground_truth'] = item['answer']
        
        # Update stats
        if result['refinement_triggered']:
            stats['refinement_triggered'] += 1
        if result['refinement_improved']:
            stats['refinement_improved'] += 1
        
        results.append(result)
    
    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Print statistics
    print(f"\n{'='*60}")
    print("SINGLE-SOLUTION PIPELINE STATISTICS")
    print(f"{'='*60}")
    print(f"Total questions: {stats['total']}")
    print(f"Refinement triggered: {stats['refinement_triggered']} ({stats['refinement_triggered']/stats['total']:.1%})")
    print(f"Refinement improved score: {stats['refinement_improved']} ({stats['refinement_improved']/stats['total']:.1%})")
    
    if stats['refinement_triggered'] > 0:
        success_rate = stats['refinement_improved'] / stats['refinement_triggered']
        print(f"Refinement success rate: {success_rate:.1%}")
    
    print(f"\nResults saved to {output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_single_solution_experiment(
        test_questions_file="gsm8k_test_questions.jsonl",
        output_file="single_solution_results.jsonl",
        verifier_checkpoint="verifier_best.pt",
        threshold=0.7,
        max_questions=100  # Start with 100, change to None for full test
    )