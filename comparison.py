import json
import re
from collections import defaultdict

def extract_boxed_answer(text):
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    return None

def extract_final_answer(answer_text):
    match = re.search(r'####\s*(.+)$', answer_text.strip())
    if match:
        return match.group(1).strip()
    return None

def normalize_answer(ans):
    if ans is None:
        return None
    ans = str(ans).replace(',', '').replace('$', '').replace('%', '')
    ans = ans.strip().lower()
    match = re.search(r'-?\d+\.?\d*', ans)
    if match:
        return match.group(0)
    return ans

def check_correctness(generated, ground_truth):
    gen_ans = extract_boxed_answer(generated)
    gen_norm = normalize_answer(gen_ans)
    gt_norm = normalize_answer(ground_truth)
    
    if gen_norm is None or gt_norm is None:
        return False
    
    return gen_norm == gt_norm

def compare_methods(scored_file, original_file):
    """Compare different selection strategies"""
    
    print("Loading data...")
    scored_data = []
    with open(scored_file, 'r') as f:
        for line in f:
            scored_data.append(json.loads(line))
    
    # Get ground truth
    gt_answers = {}
    with open(original_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            gt_answers[item['question']] = extract_final_answer(item['answer'])
    
    print(f"Comparing methods on {len(scored_data)} questions...\n")
    
    methods = {
        'Random Selection': 0,
        'First Solution': 0,
        'Majority Vote': 0,
        'Verifier (Top-Scored)': 0,
        'Oracle (Best Possible)': 0
    }
    
    # For threshold analysis
    threshold_stats = defaultdict(lambda: {'correct': 0, 'triggered': 0})
    
    import random
    
    for item in scored_data:
        question = item['question']
        solutions = item['generated_answers']
        scores = item['verifier_scores']
        gt_answer = gt_answers.get(question)
        
        if gt_answer is None:
            continue
        
        # Check correctness
        correctness = [check_correctness(sol, gt_answer) for sol in solutions]
        
        # Method 1: Random
        random_idx = random.randint(0, len(solutions) - 1)
        if correctness[random_idx]:
            methods['Random Selection'] += 1
        
        # Method 2: First solution
        if correctness[0]:
            methods['First Solution'] += 1
        
        # Method 3: Majority vote
        answers = [extract_boxed_answer(sol) for sol in solutions]
        answer_counts = defaultdict(int)
        for ans in answers:
            if ans:
                answer_counts[normalize_answer(ans)] += 1
        
        if answer_counts:
            majority_ans = max(answer_counts, key=answer_counts.get)
            if majority_ans == normalize_answer(gt_answer):
                methods['Majority Vote'] += 1
        
        # Method 4: Verifier top-scored
        best_idx = scores.index(max(scores))
        if correctness[best_idx]:
            methods['Verifier (Top-Scored)'] += 1
        
        # Method 5: Oracle
        if any(correctness):
            methods['Oracle (Best Possible)'] += 1
        
        # Threshold analysis for self-correction triggering
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            if scores[best_idx] < threshold:
                threshold_stats[threshold]['triggered'] += 1
                # In real scenario, you'd self-correct here
                # For now, assume self-correction gives you oracle
                if any(correctness):
                    threshold_stats[threshold]['correct'] += 1
            else:
                # Use verifier's choice
                if correctness[best_idx]:
                    threshold_stats[threshold]['correct'] += 1
    
    total = len(scored_data)
    
    print("="*70)
    print("METHOD COMPARISON")
    print("="*70)
    
    print("\nMethod                    | Accuracy  | Correct/Total")
    print("-" * 70)
    for method, correct in sorted(methods.items(), key=lambda x: x[1], reverse=True):
        acc = correct / total
        print(f"{method:25} | {acc:.2%}    | {correct}/{total}")
    
    print("\n" + "="*70)
    print("VERIFIER + SELF-CORRECTION SIMULATION")
    print("="*70)
    print("\nAssumption: Self-correction on low-confidence gives oracle performance")
    print("\nThreshold | Accuracy | SC Trigger Rate | Correct/Total")
    print("-" * 70)
    
    for threshold in sorted(threshold_stats.keys()):
        stats = threshold_stats[threshold]
        acc = stats['correct'] / total
        trigger_rate = stats['triggered'] / total
        print(f"{threshold:.1f}       | {acc:.2%}    | {trigger_rate:.2%}           | {stats['correct']}/{total}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    verifier_acc = methods['Verifier (Top-Scored)'] / total
    random_acc = methods['Random Selection'] / total
    oracle_acc = methods['Oracle (Best Possible)'] / total
    
    improvement = (verifier_acc - random_acc) / random_acc * 100
    
    print(f"\n→ Verifier improves over random by {improvement:.1f}%")
    print(f"→ Verifier captures {verifier_acc/oracle_acc:.1%} of oracle performance")
    
    # Find best threshold
    best_threshold = max(threshold_stats.keys(), key=lambda t: threshold_stats[t]['correct'])
    best_acc = threshold_stats[best_threshold]['correct'] / total
    
    print(f"\n→ Best threshold for SC triggering: {best_threshold}")
    print(f"  - Accuracy: {best_acc:.2%}")
    print(f"  - SC trigger rate: {threshold_stats[best_threshold]['triggered']/total:.2%}")
    
    if best_acc > verifier_acc:
        gain = (best_acc - verifier_acc) / verifier_acc * 100
        print(f"  - Gain over verifier-only: +{gain:.1f}%")
        print("\n✓ Self-correction on low-confidence solutions helps!")
    else:
        print("\n✗ Self-correction doesn't help - verifier alone is sufficient")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    compare_methods(
        scored_file="scored_outputs.jsonl",
        original_file="generations_google-gemma-2b-it_0_-1.jsonl"
    )