import json

# Load results
with open('refined_outputs.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# Find cases where refinement was triggered
triggered = [d for d in data if d['refinement_triggered']]

print(f"Found {len(triggered)} cases where refinement was triggered\n")

# Show 3 examples where it helped
improved = [d for d in triggered if d['refinement_improved']]
print(f"=== CASES WHERE REFINEMENT HELPED ({len(improved)}) ===\n")
for i, item in enumerate(improved[:3]):
    print(f"Example {i+1}:")
    print(f"Question: {item['question'][:100]}...")
    print(f"Original score: {item['best_original_score']:.3f}")
    print(f"Final score: {item['final_score']:.3f}")
    print(f"Original: {item['best_original_solution'][:150]}...")
    print(f"Refined: {item['final_solution'][:150]}...")
    print("\n" + "="*60 + "\n")

# Show 3 examples where it didn't help
not_improved = [d for d in triggered if not d['refinement_improved']]
print(f"=== CASES WHERE REFINEMENT DIDN'T HELP ({len(not_improved)}) ===\n")
for i, item in enumerate(not_improved[:3]):
    print(f"Example {i+1}:")
    print(f"Question: {item['question'][:100]}...")
    print(f"Original score: {item['best_original_score']:.3f}")
    if 'iteration_1' in item:
        print(f"Refined score: {item['iteration_1']['refined_score']:.3f}")
        print(f"Critique: {item['iteration_1']['critique'][:200]}...")
        print(f"Refined: {item['iteration_1']['refined_solution'][:150]}...")
    print("\n" + "="*60 + "\n")