import json
import re

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

def evaluate_single_solution(results_file="single_solution_results.jsonl"):
    """Evaluate single-solution self-correction"""
    
    print("Loading results...")
    data = []
    with open(results_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"Evaluating {len(data)} questions...\n")
    
    stats = {
        'total': len(data),
        'original_correct': 0,
        'final_correct': 0,
        'refinement_triggered': 0,
        'refinement_helped': 0,
        'refinement_hurt': 0,
        'refinement_no_change': 0
    }
    
    for item in data:
        gt_answer = extract_final_answer(item['ground_truth'])
        
        if gt_answer is None:
            continue
        
        # Check original
        original_correct = check_correctness(item['original_solution'], gt_answer)
        if original_correct:
            stats['original_correct'] += 1
        
        # Check final
        final_correct = check_correctness(item['final_solution'], gt_answer)
        if final_correct:
            stats['final_correct'] += 1
        
        # Refinement analysis
        if item['refinement_triggered']:
            stats['refinement_triggered'] += 1
            
            if final_correct and not original_correct:
                stats['refinement_helped'] += 1
            elif original_correct and not final_correct:
                stats['refinement_hurt'] += 1
            else:
                stats['refinement_no_change'] += 1
    
    # Print results
    print("="*70)
    print("SINGLE-SOLUTION SELF-CORRECTION EVALUATION")
    print("="*70)
    
    print(f"\nTotal questions: {stats['total']}")
    
    print("\n--- ACCURACY (KEY RESULT) ---")
    orig_acc = stats['original_correct'] / stats['total']
    final_acc = stats['final_correct'] / stats['total']
    improvement = final_acc - orig_acc
    relative_improvement = (improvement / orig_acc * 100) if orig_acc > 0 else 0
    
    print(f"Original (1 solution, no refinement): {stats['original_correct']}/{stats['total']} = {orig_acc:.2%}")
    print(f"Final (with self-correction):          {stats['final_correct']}/{stats['total']} = {final_acc:.2%}")
    print(f"Absolute improvement:                  {improvement:+.2%} ({stats['final_correct'] - stats['original_correct']:+d} questions)")
    print(f"Relative improvement:                  {relative_improvement:+.1f}%")
    
    print("\n--- REFINEMENT STATISTICS ---")
    trigger_rate = stats['refinement_triggered'] / stats['total']
    print(f"Refinement triggered: {stats['refinement_triggered']}/{stats['total']} ({trigger_rate:.1%})")
    
    if stats['refinement_triggered'] > 0:
        print(f"\nWhen refinement was triggered:")
        help_rate = stats['refinement_helped'] / stats['refinement_triggered']
        hurt_rate = stats['refinement_hurt'] / stats['refinement_triggered']
        print(f"  Helped (wrong → correct):  {stats['refinement_helped']} ({help_rate:.1%})")
        print(f"  Hurt (correct → wrong):    {stats['refinement_hurt']} ({hurt_rate:.1%})")
        print(f"  No change:                 {stats['refinement_no_change']} ({stats['refinement_no_change']/stats['refinement_triggered']:.1%})")
        
        if stats['refinement_helped'] > 0:
            net_gain = stats['refinement_helped'] - stats['refinement_hurt']
            print(f"\n  Net questions corrected:   {net_gain} ({stats['refinement_helped']} helped - {stats['refinement_hurt']} hurt)")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if improvement > 0:
        print(f"✓ Self-correction works! Accuracy improved by {improvement:.2%}")
        print(f"  - {stats['refinement_helped']} questions corrected")
        if stats['refinement_hurt'] > 0:
            print(f"  - {stats['refinement_hurt']} questions degraded")
        print(f"  - Net benefit: {stats['refinement_helped'] - stats['refinement_hurt']} questions")
        print(f"\n  This proves verifier + refiner can improve single solutions!")
    elif improvement == 0:
        print(f"= Self-correction had no net effect")
    else:
        print(f"✗ Self-correction hurt performance by {improvement:.2%}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    evaluate_single_solution("single_solution_results.jsonl")