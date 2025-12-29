import json
import re
from collections import defaultdict

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

def evaluate_refinement(refined_file):
    """
    Evaluate refinement results
    """
    print("Loading refined results...")
    data = []
    with open(refined_file, 'r') as f:
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
        question = item['question']
        gt_answer = extract_final_answer(item['ground_truth'])
        
        if gt_answer is None:
            continue
        
        # Check original best solution
        original_correct = check_correctness(item['best_original_solution'], gt_answer)
        if original_correct:
            stats['original_correct'] += 1
        
        # Check final solution
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
    print("REFINEMENT EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nTotal questions: {stats['total']}")
    
    print("\n--- Accuracy ---")
    orig_acc = stats['original_correct'] / stats['total']
    final_acc = stats['final_correct'] / stats['total']
    improvement = final_acc - orig_acc
    
    print(f"Original (verifier only):  {stats['original_correct']}/{stats['total']} = {orig_acc:.2%}")
    print(f"Final (with refinement):   {stats['final_correct']}/{stats['total']} = {final_acc:.2%}")
    print(f"Improvement: {improvement:+.2%} ({stats['final_correct'] - stats['original_correct']:+d} questions)")
    
    print("\n--- Refinement Statistics ---")
    trigger_rate = stats['refinement_triggered'] / stats['total']
    print(f"Refinement triggered: {stats['refinement_triggered']}/{stats['total']} ({trigger_rate:.1%})")
    
    if stats['refinement_triggered'] > 0:
        print(f"\nWhen refinement was triggered:")
        print(f"  Helped (wrong → correct):  {stats['refinement_helped']} ({stats['refinement_helped']/stats['refinement_triggered']:.1%})")
        print(f"  Hurt (correct → wrong):    {stats['refinement_hurt']} ({stats['refinement_hurt']/stats['refinement_triggered']:.1%})")
        print(f"  No change:                 {stats['refinement_no_change']} ({stats['refinement_no_change']/stats['refinement_triggered']:.1%})")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if improvement > 0:
        print(f"✓ Refinement helped! Accuracy improved by {improvement:.2%}")
        print(f"  - {stats['refinement_helped']} questions were corrected")
        if stats['refinement_hurt'] > 0:
            print(f"  - Warning: {stats['refinement_hurt']} questions were made worse")
    elif improvement < 0:
        print(f"✗ Refinement hurt performance by {improvement:.2%}")
        print("  - Consider adjusting threshold or refinement strategy")
    else:
        print("= Refinement had no net effect on accuracy")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    evaluate_refinement("refined_outputs.jsonl")