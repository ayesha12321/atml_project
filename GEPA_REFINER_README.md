# GEPA-based Refiner Optimization

Optimize the self-correction refiner's **critique** and **refinement** prompts using GEPA's reflective prompt evolution instead of RLHF.

## Overview

This implementation applies GEPA (Genetic-Pareto) optimization to evolve better prompts for:
1. **Critique Prompt**: Instructions for identifying errors in math solutions
2. **Refinement Prompt**: Instructions for generating corrected solutions

The verifier's scores serve as the feedback signal - GEPA reflects on which critiques/refinements led to improvements.

## Files

| File | Purpose |
|------|---------|
| `gepa_refiner_adapter.py` | GEPA adapter and configurable refiner |
| `gepa_optimize_refiner.py` | Main optimization script |
| `evaluate_evolved_prompts.py` | Compare baseline vs evolved prompts |
| `requirements_gepa.txt` | Dependencies |

## RunPod Setup

### 1. Rent GPU
- GPU: **A40** (48GB) or **RTX A6000** (48GB)
- Template: **PyTorch 2.1**
- Disk: 50GB minimum

### 2. Upload Files
```bash
scp *.py *.jsonl *.pt requirements*.txt root@<pod-ip>:/workspace/
```

### 3. Install Dependencies
```bash
cd /workspace
pip install -r requirements_gepa.txt    
```

### 4. Run Optimization
```bash
# Full optimization (10 iterations, ~7K training examples)
python gepa_optimize_refiner.py

# Quick test (3 iterations, 100 samples)
python gepa_optimize_refiner.py \
    --max_iterations 3 \
    --max_train_samples 100 \
    --minibatch_size 3
```

### 5. Evaluate Results
```bash
python evaluate_evolved_prompts.py \
    --evolved_prompts evolved_prompts.json \
    --max_samples 200
```

## Expected Timeline

| Stage | Time (A40) |
|-------|------------|
| Load models | ~2 min |
| Per iteration | ~3-5 min |
| 10 iterations | ~30-50 min |
| Evaluation | ~10 min |

## Output

After optimization, `evolved_prompts.json` contains:
```json
{
  "seed_prompts": {...},
  "evolved_prompts": {
    "critique_prompt": "...",
    "refinement_prompt": "..."
  },
  "history": [...],
  "final_scores": {
    "seed": 0.XX,
    "best": 0.XX
  }
}
```

## Expected Results

- **Baseline**: Current hardcoded prompts
- **Evolved**: GEPA-optimized prompts with domain-specific strategies
- **Expected improvement**: +3-10% accuracy on refinement tasks
