import json
import torch
from transformers import AutoTokenizer
from train_verifier import VerifierModel
from tqdm import tqdm

def score_solutions(model, tokenizer, question, solutions, device, max_length=512):
    """Score a list of solutions for a given question"""
    model.eval()
    scores = []
    
    with torch.no_grad():
        for solution in solutions:
            text = f"Question: {question}\nSolution: {solution}"
            
            encoding = tokenizer(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            score = model(input_ids, attention_mask)
            scores.append(score.item())
    
    return scores

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print("Loading model...")
    checkpoint = torch.load('verifier_best.pt', map_location=device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_name'])
    
    model = VerifierModel(checkpoint['tokenizer_name']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with val_acc: {checkpoint['val_acc']:.4f}")
    
    # Load test data
    print("Loading test data...")
    input_file = "score_outputs.jsonl"
    output_file = "scored_outputs.jsonl"
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in tqdm(f_in, desc="Scoring"):
            item = json.loads(line)
            question = item['question']
            solutions = item['generated_answers']
            
            # Score all solutions
            scores = score_solutions(model, tokenizer, question, solutions, device)
            
            # Add scores to item
            item['verifier_scores'] = scores
            item['best_solution_idx'] = scores.index(max(scores))
            item['best_solution'] = solutions[scores.index(max(scores))]
            
            f_out.write(json.dumps(item) + '\n')
    
    print(f"Scored outputs saved to {output_file}")
    
    # Print example
    print("\nExample scored question:")
    with open(output_file, 'r') as f:
        example = json.loads(f.readline())
        print(f"Question: {example['question'][:100]}...")
        print(f"\nScores: {example['verifier_scores']}")
        print(f"Best idx: {example['best_solution_idx']}")

if __name__ == "__main__":
    main()