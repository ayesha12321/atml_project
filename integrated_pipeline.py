import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm
from refiner import SelfCorrectionRefiner

# Verifier model (same as train_verifier.py)
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

class IntegratedPipeline:
    def __init__(self, verifier_checkpoint_path, refiner_model_name="google/gemma-2-2b-it"):
        """
        Initialize complete pipeline with verifier + refiner
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load verifier
        print("Loading verifier...")
        checkpoint = torch.load(verifier_checkpoint_path, map_location=self.device)
        self.verifier_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False)
        self.verifier = VerifierModel('bert-base-uncased').to(self.device)
        self.verifier.load_state_dict(checkpoint['model_state_dict'])
        self.verifier.eval()
        print(f"Verifier loaded (val_acc: {checkpoint['val_acc']:.4f})")
        
        # Load refiner
        print("Loading refiner...")
        self.refiner = SelfCorrectionRefiner(refiner_model_name, self.device)
        print("Pipeline ready!")
    
    def score_solution(self, question, solution, max_length=512):
        """
        Score a single solution using verifier
        """
        text = f"Question: {question}\nSolution: {solution}"
        
        encoding = self.verifier_tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            score = self.verifier(input_ids, attention_mask)
        
        return score.item()
    
    def process_question(self, question, solutions, threshold=0.7, max_iterations=1):
        """
        Complete pipeline for one question
        
        Args:
            question: The question text
            solutions: List of generated solutions
            threshold: Score threshold for triggering refinement
            max_iterations: Max number of refinement iterations
        
        Returns:
            dict with final solution, score, and metadata
        """
        # Step 1: Score all solutions
        scores = [self.score_solution(question, sol) for sol in solutions]
        
        # Step 2: Get best solution
        best_idx = scores.index(max(scores))
        best_solution = solutions[best_idx]
        best_score = scores[best_idx]
        
        result = {
            'original_solutions': solutions,
            'original_scores': scores,
            'best_original_idx': best_idx,
            'best_original_solution': best_solution,
            'best_original_score': best_score,
            'refinement_triggered': False,
            'refinement_improved': False,
            'final_solution': best_solution,
            'final_score': best_score
        }
        
        # Step 3: Refinement if below threshold
        if best_score < threshold:
            result['refinement_triggered'] = True
            
            current_solution = best_solution
            current_score = best_score
            
            for iteration in range(max_iterations):
                print(f"  Refining (iteration {iteration + 1})...")
                
                # Generate critique and refinement
                refinement_result = self.refiner.refine_solution(question, current_solution)
                refined_solution = refinement_result['refined_solution']
                critique = refinement_result['critique']
                
                # Score refined solution
                refined_score = self.score_solution(question, refined_solution)
                
                result[f'iteration_{iteration + 1}'] = {
                    'critique': critique,
                    'refined_solution': refined_solution,
                    'refined_score': refined_score
                }
                
                # Check if improved
                if refined_score > current_score:
                    current_solution = refined_solution
                    current_score = refined_score
                    result['refinement_improved'] = True
                else:
                    # No improvement, stop iterating
                    break
            
            # Update final result
            if current_score > best_score:
                result['final_solution'] = current_solution
                result['final_score'] = current_score
        
        return result

def run_pipeline(input_file, output_file, verifier_checkpoint, threshold=0.7, max_questions=None):
    """
    Run complete pipeline on dataset
    """
    # Initialize pipeline
    pipeline = IntegratedPipeline(verifier_checkpoint)
    
    # Load data
    print(f"Loading data from {input_file}...")
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    if max_questions:
        data = data[:max_questions]
    
    print(f"Processing {len(data)} questions with threshold={threshold}...")
    
    # Process each question
    results = []
    stats = {
        'total': len(data),
        'refinement_triggered': 0,
        'refinement_improved': 0,
        'score_improved': 0
    }
    
    for item in tqdm(data, desc="Processing"):
        question = item['question']
        solutions = item['generated_answers']
        
        result = pipeline.process_question(question, solutions, threshold=threshold)
        
        # Add original data
        result['question'] = question
        result['ground_truth'] = item['answer']
        
        # Update stats
        if result['refinement_triggered']:
            stats['refinement_triggered'] += 1
        if result['refinement_improved']:
            stats['refinement_improved'] += 1
        if result['final_score'] > result['best_original_score']:
            stats['score_improved'] += 1
        
        results.append(result)
    
    # Save results
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Print statistics
    print("\n" + "="*60)
    print("PIPELINE STATISTICS")
    print("="*60)
    print(f"Total questions: {stats['total']}")
    print(f"Refinement triggered: {stats['refinement_triggered']} ({stats['refinement_triggered']/stats['total']:.1%})")
    print(f"Refinement improved score: {stats['refinement_improved']} ({stats['refinement_improved']/stats['total']:.1%})")
    print(f"Score improved overall: {stats['score_improved']} ({stats['score_improved']/stats['total']:.1%})")
    
    if stats['refinement_triggered'] > 0:
        success_rate = stats['refinement_improved'] / stats['refinement_triggered']
        print(f"Refinement success rate: {success_rate:.1%}")
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "scored_outputs.jsonl"
    OUTPUT_FILE = "refined_outputs.jsonl"
    VERIFIER_CHECKPOINT = "verifier_best.pt"
    THRESHOLD = 0.7
    MAX_QUESTIONS = 100  # Start with 100 for testing, set to None for full dataset
    
    run_pipeline(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        verifier_checkpoint=VERIFIER_CHECKPOINT,
        threshold=THRESHOLD,
        max_questions=MAX_QUESTIONS
    )