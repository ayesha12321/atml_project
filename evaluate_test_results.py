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

def evaluate_test_set(refined_file="gsm8k_test_refined.jsonl"):
    """Evaluate test set results"""
    
    print("Loading test results...")
    data = []
    with open(refined_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"Evaluating {len(data)} test questions...\n")
    
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
        original_correct = check_correctness(item['best_original_solution'], gt_answer)
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
    print("GSM8K TEST SET EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nTotal test questions: {stats['total']}")
    
    print("\n--- ACCURACY (MAIN RESULT) ---")
    orig_acc = stats['original_correct'] / stats['total']
    final_acc = stats['final_correct'] / stats['total']
    improvement = final_acc - orig_acc
    
    print(f"Verifier Only:          {stats['original_correct']}/{stats['total']} = {orig_acc:.2%}")
    print(f"Verifier + Refinement:  {stats['final_correct']}/{stats['total']} = {final_acc:.2%}")
    print(f"Improvement:            {improvement:+.2%} ({stats['final_correct'] - stats['original_correct']:+d} questions)")
    
    print("\n--- Refinement Statistics ---")
    trigger_rate = stats['refinement_triggered'] / stats['total']
    print(f"Refinement triggered: {stats['refinement_triggered']}/{stats['total']} ({trigger_rate:.1%})")
    
    if stats['refinement_triggered'] > 0:
        print(f"\nWhen refinement was triggered:")
        print(f"  Helped (wrong → correct):  {stats['refinement_helped']} ({stats['refinement_helped']/stats['refinement_triggered']:.1%})")
        print(f"  Hurt (correct → wrong):    {stats['refinement_hurt']} ({stats['refinement_hurt']/stats['refinement_triggered']:.1%})")
        print(f"  No change:                 {stats['refinement_no_change']} ({stats['refinement_no_change']/stats['refinement_triggered']:.1%})")
    
    print("\n" + "="*70)
    print("FINAL CONCLUSION")
    print("="*70)
    
    print(f"\n✓ Test Set Accuracy: {final_acc:.2%}")
    print(f"✓ Improvement over verifier-only: {improvement:+.2%}")
    print(f"✓ Questions corrected: {stats['refinement_helped']}")
    print(f"✓ Questions degraded: {stats['refinement_hurt']}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    evaluate_test_set("gsm8k_test_refined.jsonl")