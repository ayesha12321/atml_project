import json
import re
from collections import defaultdict
import numpy as np

def extract_boxed_answer(text):
    """Extract answer from \\boxed{} format"""
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    return None

def extract_final_answer(answer_text):
    """Extract final numerical answer from ground truth"""
    match = re.search(r'####\s*(.+)$', answer_text.strip())
    if match:
        return match.group(1).strip()
    return None

def normalize_answer(ans):
    """Normalize answer for comparison"""
    if ans is None:
        return None
    ans = str(ans).replace(',', '').replace('$', '').replace('%', '')
    ans = ans.strip().lower()
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

def analyze_verifier_performance(scored_file, original_file):
    """Analyze how well the verifier scores correlate with correctness"""
    
    print("Loading data...")
    scored_data = []
    with open(scored_file, 'r') as f:
        for line in f:
            scored_data.append(json.loads(line))
    
    # Get ground truth answers
    gt_answers = {}
    with open(original_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            gt_answers[item['question']] = extract_final_answer(item['answer'])
    
    print(f"Analyzing {len(scored_data)} questions...\n")
    
    # Metrics
    stats = {
        'total_questions': len(scored_data),
        'verifier_correct': 0,
        'random_correct': 0,
        'best_possible': 0,
        'worst_possible': 0,
        'score_ranges': defaultdict(lambda: {'correct': 0, 'total': 0})
    }
    
    all_scores_correct = []
    all_scores_incorrect = []
    
    for item in scored_data:
        question = item['question']
        solutions = item['generated_answers']
        scores = item['verifier_scores']
        gt_answer = gt_answers.get(question)
        
        if gt_answer is None:
            continue
        
        # Check correctness of each solution
        correctness = [check_correctness(sol, gt_answer) for sol in solutions]
        
        # Verifier's choice (highest score)
        best_idx = scores.index(max(scores))
        if correctness[best_idx]:
            stats['verifier_correct'] += 1
        
        # Random baseline
        import random
        random_idx = random.randint(0, len(solutions) - 1)
        if correctness[random_idx]:
            stats['random_correct'] += 1
        
        # Best possible (oracle)
        if any(correctness):
            stats['best_possible'] += 1
        
        # Worst possible
        if all(correctness):
            stats['worst_possible'] += 1
        
        # Score distribution analysis
        for score, is_correct in zip(scores, correctness):
            if is_correct:
                all_scores_correct.append(score)
            else:
                all_scores_incorrect.append(score)
            
            # Binning
            score_bin = int(score * 10) / 10  # Round to 0.1
            stats['score_ranges'][score_bin]['total'] += 1
            if is_correct:
                stats['score_ranges'][score_bin]['correct'] += 1
    
    # Print results
    print("="*60)
    print("VERIFIER PERFORMANCE ANALYSIS")
    print("="*60)
    
    print(f"\nTotal questions analyzed: {stats['total_questions']}")
    
    print("\n--- Selection Accuracy ---")
    print(f"Verifier (top-scored):     {stats['verifier_correct']}/{stats['total_questions']} = {stats['verifier_correct']/stats['total_questions']:.2%}")
    print(f"Random baseline:           {stats['random_correct']}/{stats['total_questions']} = {stats['random_correct']/stats['total_questions']:.2%}")
    print(f"Oracle (best possible):    {stats['best_possible']}/{stats['total_questions']} = {stats['best_possible']/stats['total_questions']:.2%}")
    
    if all_scores_correct and all_scores_incorrect:
        print("\n--- Score Distribution ---")
        print(f"Avg score for CORRECT solutions:   {np.mean(all_scores_correct):.3f} ± {np.std(all_scores_correct):.3f}")
        print(f"Avg score for INCORRECT solutions: {np.mean(all_scores_incorrect):.3f} ± {np.std(all_scores_incorrect):.3f}")
        print(f"Score separation (higher is better): {np.mean(all_scores_correct) - np.mean(all_scores_incorrect):.3f}")
    
    print("\n--- Score Range Calibration ---")
    print("Score Range | Accuracy | Count")
    print("-" * 40)
    for score_range in sorted(stats['score_ranges'].keys()):
        data = stats['score_ranges'][score_range]
        acc = data['correct'] / data['total'] if data['total'] > 0 else 0
        print(f"{score_range:.1f} - {score_range+0.1:.1f}  | {acc:.2%}     | {data['total']}")
    
    # Confidence threshold analysis
    print("\n--- Threshold Analysis (what if we only trust high scores?) ---")
    print("Threshold | Precision | Coverage | Correct")
    print("-" * 50)
    
    for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        above_threshold = 0
        correct_above = 0
        
        for item in scored_data:
            question = item['question']
            solutions = item['generated_answers']
            scores = item['verifier_scores']
            gt_answer = gt_answers.get(question)
            
            if gt_answer is None:
                continue
            
            best_idx = scores.index(max(scores))
            if scores[best_idx] >= threshold:
                above_threshold += 1
                correctness = [check_correctness(sol, gt_answer) for sol in solutions]
                if correctness[best_idx]:
                    correct_above += 1
        
        precision = correct_above / above_threshold if above_threshold > 0 else 0
        coverage = above_threshold / stats['total_questions']
        
        print(f"{threshold:.1f}       | {precision:.2%}      | {coverage:.2%}     | {correct_above}/{above_threshold}")
    
    print("\n" + "="*60)
    
    # Key insight
    if all_scores_correct and all_scores_incorrect:
        separation = np.mean(all_scores_correct) - np.mean(all_scores_incorrect)
        if separation > 0.1:
            print("✓ Verifier shows good score separation - it can distinguish correct from incorrect!")
        else:
            print("✗ Verifier shows poor separation - may need more training or better model")
    
    print("\nRecommendation:")
    verifier_acc = stats['verifier_correct']/stats['total_questions']
    oracle_acc = stats['best_possible']/stats['total_questions']
    
    if verifier_acc > 0.7 * oracle_acc:
        print("→ Verifier is performing well! Use it for solution selection.")
        print("→ Consider using score threshold (e.g., 0.5) to trigger self-correction on low-confidence solutions.")
    else:
        print("→ Verifier needs improvement. Try:")
        print("  - More training epochs")
        print("  - Larger model (e.g., deberta-v3-large)")
        print("  - More contrastive pairs")

if __name__ == "__main__":
    analyze_verifier_performance(
        scored_file="scored_outputs.jsonl",
        original_file="generations_google-gemma-2b-it_0_-1.jsonl"
    )