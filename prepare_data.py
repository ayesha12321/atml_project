import json
import re
from pathlib import Path
from tqdm import tqdm
import random

def extract_boxed_answer(text):
    """Extract answer from \\boxed{} format"""
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    return None

def extract_final_answer(answer_text):
    """Extract final numerical answer from ground truth"""
    # GSM8K format ends with #### NUMBER
    match = re.search(r'####\s*(.+)$', answer_text.strip())
    if match:
        return match.group(1).strip()
    return None

def normalize_answer(ans):
    """Normalize answer for comparison"""
    if ans is None:
        return None
    # Remove common formatting
    ans = str(ans).replace(',', '').replace('$', '').replace('%', '')
    ans = ans.strip().lower()
    # Try to extract just the number
    match = re.search(r'-?\d+\.?\d*', ans)
    if match:
        return match.group(0)
    return ans

def check_correctness(generated, ground_truth):
    """Check if generated answer matches ground truth"""
    gen_ans = extract_boxed_answer(generated)
    gen_norm = normalize_answer(gen_ans)
    gt_norm = normalize_answer(ground_truth)
    
    if gen_norm is None or gt_norm is None:
        return False
    
    return gen_norm == gt_norm

def create_contrastive_pairs(input_jsonl, output_train, output_val, val_split=0.1):
    """Create contrastive pairs from SCORE outputs"""
    
    print("Loading data...")
    data = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"Loaded {len(data)} questions")
    
    all_pairs = []
    stats = {'total_questions': 0, 'correct_sols': 0, 'incorrect_sols': 0, 'pairs_created': 0}
    
    for item in tqdm(data, desc="Creating pairs"):
        question = item['question']
        generated = item['generated_answers']
        
        # Extract ground truth answer
        gt_answer = extract_final_answer(item['answer'])
        if gt_answer is None:
            continue
        
        # Check correctness of each solution
        correct_sols = []
        incorrect_sols = []
        
        for sol in generated:
            if check_correctness(sol, gt_answer):
                correct_sols.append(sol)
                stats['correct_sols'] += 1
            else:
                incorrect_sols.append(sol)
                stats['incorrect_sols'] += 1
        
        # Create pairs: each correct paired with each incorrect
        if len(correct_sols) > 0 and len(incorrect_sols) > 0:
            stats['total_questions'] += 1
            
            # Sample pairs to avoid explosion
            max_pairs_per_question = 10
            for correct in correct_sols:
                sampled_incorrect = random.sample(
                    incorrect_sols, 
                    min(len(incorrect_sols), max_pairs_per_question // len(correct_sols) + 1)
                )
                for incorrect in sampled_incorrect:
                    all_pairs.append({
                        'question': question,
                        'solution_a': correct,
                        'solution_b': incorrect,
                        'label': 1  # A is better than B
                    })
                    stats['pairs_created'] += 1
    
    print(f"\nStatistics:")
    print(f"Questions with both correct/incorrect: {stats['total_questions']}")
    print(f"Total correct solutions: {stats['correct_sols']}")
    print(f"Total incorrect solutions: {stats['incorrect_sols']}")
    print(f"Pairs created: {stats['pairs_created']}")
    
    # Shuffle and split
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * (1 - val_split))
    
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    
    print(f"\nTrain pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    
    # Save
    with open(output_train, 'w') as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + '\n')
    
    with open(output_val, 'w') as f:
        for pair in val_pairs:
            f.write(json.dumps(pair) + '\n')
    
    print(f"\nSaved to {output_train} and {output_val}")

if __name__ == "__main__":
    create_contrastive_pairs(
        input_jsonl="generations_google-gemma-2b-it_0_-1.jsonl",  # Your file
        output_train="train_pairs.jsonl",
        output_val="val_pairs.jsonl",
        val_split=0.1
    )