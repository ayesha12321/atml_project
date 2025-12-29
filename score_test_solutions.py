import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Verifier model (same as before)
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

def score_solution(model, tokenizer, question, solution, device, max_length=512):
    """Score a single solution"""
    text = f"Question: {question}\nSolution: {solution}"
    
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        score = model(input_ids, attention_mask)
    
    return score.item()

def score_test_set(
    input_file="gsm8k_test_generations.jsonl",
    output_file="gsm8k_test_scored.jsonl",
    verifier_checkpoint="verifier_best.pt"
):
    """Score all test solutions with verifier"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load verifier
    print("Loading verifier...")
    checkpoint = torch.load(verifier_checkpoint, map_location=device)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False)
    model = VerifierModel('bert-base-uncased').to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Verifier loaded (val_acc: {checkpoint['val_acc']:.4f})")
    
    # Load test generations
    print(f"Loading test data from {input_file}...")
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"Loaded {len(data)} questions")
    
    # Score all solutions
    print("\nScoring solutions...")
    for item in tqdm(data, desc="Scoring"):
        question = item['question']
        solutions = item['generated_answers']
        
        # Score each solution
        scores = [score_solution(model, tokenizer, question, sol, device) 
                  for sol in solutions]
        
        # Add to item
        item['verifier_scores'] = scores
        item['best_solution_idx'] = scores.index(max(scores))
        item['best_solution'] = solutions[scores.index(max(scores))]
        item['best_score'] = max(scores)
    
    # Save scored results
    print(f"\nSaving scored results to {output_file}...")
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✓ Scored {len(data)} questions")
    print(f"✓ Saved to {output_file}")

if __name__ == "__main__":
    score_test_set(
        input_file="gsm8k_test_generations.jsonl",
        output_file="gsm8k_test_scored.jsonl",
        verifier_checkpoint="verifier_best.pt"
    )