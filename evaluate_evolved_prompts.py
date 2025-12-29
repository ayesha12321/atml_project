"""
Evaluate Evolved Prompts vs Baseline

Compare the performance of GEPA-evolved prompts against the baseline prompts.

Usage:
    python evaluate_evolved_prompts.py --evolved_prompts evolved_prompts.json
"""

import argparse
import json
import re
import sys
from tqdm import tqdm

import torch


# =============================================================================
# Utility Functions (from existing codebase)
# =============================================================================

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


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_prompts(
    prompts: dict[str, str],
    test_data: list[dict],
    verifier_model,
    verifier_tokenizer,
    refiner_model,
    refiner_tokenizer,
    device: str = "cuda",
    score_threshold: float = 0.7,
    max_samples: int = 100
) -> dict:
    """Evaluate prompts on test data"""
    
    from gepa_refiner_adapter import ConfigurableRefiner
    
    # Create refiner with prompts
    refiner = ConfigurableRefiner(
        model=refiner_model,
        tokenizer=refiner_tokenizer,
        critique_prompt=prompts["critique_prompt"],
        refinement_prompt=prompts["refinement_prompt"],
        device=device
    )
    
    stats = {
        "total": 0,
        "low_confidence": 0,
        "original_correct": 0,
        "final_correct": 0,
        "refinement_triggered": 0,
        "refinement_helped": 0,
        "refinement_hurt": 0
    }
    
    data_to_process = test_data[:max_samples]
    
    for item in tqdm(data_to_process, desc="Evaluating"):
        question = item['question']
        gt = extract_final_answer(item['answer'])
        
        if gt is None:
            continue
        
        solutions = item['generated_answers']
        scores = item['verifier_scores']
        
        # Get best solution
        best_idx = scores.index(max(scores))
        best_score = scores[best_idx]
        best_solution = solutions[best_idx]
        
        stats["total"] += 1
        
        # Check original correctness
        original_correct = check_correctness(best_solution, gt)
        if original_correct:
            stats["original_correct"] += 1
        
        # Check if needs refinement
        if best_score < score_threshold:
            stats["low_confidence"] += 1
            stats["refinement_triggered"] += 1
            
            # Apply refinement
            try:
                result = refiner.refine_solution(question, best_solution)
                refined_solution = result['refined_solution']
                refined_correct = check_correctness(refined_solution, gt)
                
                if refined_correct:
                    stats["final_correct"] += 1
                    if not original_correct:
                        stats["refinement_helped"] += 1
                elif original_correct:
                    stats["refinement_hurt"] += 1
                    
            except Exception as e:
                print(f"Error during refinement: {e}")
                if original_correct:
                    stats["final_correct"] += 1
        else:
            # High confidence - keep original
            if original_correct:
                stats["final_correct"] += 1
    
    # Calculate metrics
    metrics = {
        "total_questions": stats["total"],
        "original_accuracy": stats["original_correct"] / stats["total"] if stats["total"] > 0 else 0,
        "final_accuracy": stats["final_correct"] / stats["total"] if stats["total"] > 0 else 0,
        "low_confidence_rate": stats["low_confidence"] / stats["total"] if stats["total"] > 0 else 0,
        "refinement_help_rate": stats["refinement_helped"] / stats["refinement_triggered"] if stats["refinement_triggered"] > 0 else 0,
        "refinement_hurt_rate": stats["refinement_hurt"] / stats["refinement_triggered"] if stats["refinement_triggered"] > 0 else 0,
        "raw_stats": stats
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate evolved vs baseline prompts")
    parser.add_argument("--evolved_prompts", type=str, default="evolved_prompts.json", help="Path to evolved prompts JSON")
    parser.add_argument("--test_file", type=str, default="scored_outputs.jsonl", help="Path to test data")
    parser.add_argument("--verifier_checkpoint", type=str, default="verifier_best.pt", help="Path to verifier checkpoint")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples to evaluate")
    parser.add_argument("--score_threshold", type=float, default=0.7, help="Score threshold for refinement")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()
    
    print("="*60)
    print("Evaluating Evolved Prompts vs Baseline")
    print("="*60)
    
    # Load evolved prompts
    print("\n1. Loading evolved prompts...")
    with open(args.evolved_prompts, 'r') as f:
        evolved_data = json.load(f)
    
    evolved_prompts = evolved_data["evolved_prompts"]
    baseline_prompts = evolved_data["seed_prompts"]
    
    print(f"   Seed score: {evolved_data['final_scores']['seed']:.4f}")
    print(f"   Best score: {evolved_data['final_scores']['best']:.4f}")
    
    # Load test data
    print("\n2. Loading test data...")
    test_data = []
    with open(args.test_file, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"   Loaded {len(test_data)} examples")
    
    # Load models
    print("\n3. Loading models...")
    from gepa_refiner_adapter import load_verifier, load_refiner_model
    
    verifier_model, verifier_tokenizer = load_verifier(args.verifier_checkpoint, args.device)
    refiner_model, refiner_tokenizer = load_refiner_model("google/gemma-2-2b-it", args.device)
    
    # Evaluate baseline
    print("\n4. Evaluating BASELINE prompts...")
    baseline_metrics = evaluate_prompts(
        prompts=baseline_prompts,
        test_data=test_data,
        verifier_model=verifier_model,
        verifier_tokenizer=verifier_tokenizer,
        refiner_model=refiner_model,
        refiner_tokenizer=refiner_tokenizer,
        device=args.device,
        score_threshold=args.score_threshold,
        max_samples=args.max_samples
    )
    
    # Evaluate evolved
    print("\n5. Evaluating EVOLVED prompts...")
    evolved_metrics = evaluate_prompts(
        prompts=evolved_prompts,
        test_data=test_data,
        verifier_model=verifier_model,
        verifier_tokenizer=verifier_tokenizer,
        refiner_model=refiner_model,
        refiner_tokenizer=refiner_tokenizer,
        device=args.device,
        score_threshold=args.score_threshold,
        max_samples=args.max_samples
    )
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\n{'Metric':<30} {'Baseline':<15} {'Evolved':<15} {'Delta':<15}")
    print("-"*75)
    
    for metric in ["original_accuracy", "final_accuracy", "refinement_help_rate", "refinement_hurt_rate"]:
        baseline_val = baseline_metrics[metric]
        evolved_val = evolved_metrics[metric]
        delta = evolved_val - baseline_val
        delta_str = f"{delta:+.2%}" if delta != 0 else "0.00%"
        
        print(f"{metric:<30} {baseline_val:<15.2%} {evolved_val:<15.2%} {delta_str:<15}")
    
    print("\n" + "="*60)
    
    improvement = evolved_metrics["final_accuracy"] - baseline_metrics["final_accuracy"]
    if improvement > 0:
        print(f"✓ EVOLVED prompts improved accuracy by {improvement:.2%}")
    elif improvement < 0:
        print(f"✗ EVOLVED prompts decreased accuracy by {-improvement:.2%}")
    else:
        print("= No change in accuracy")
    
    print("="*60)
    
    # Save comparison
    comparison = {
        "baseline_metrics": baseline_metrics,
        "evolved_metrics": evolved_metrics,
        "improvement": improvement
    }
    
    output_file = args.evolved_prompts.replace(".json", "_comparison.json")
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to {output_file}")


if __name__ == "__main__":
    main()
