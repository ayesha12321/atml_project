# check_data_source.py
import json
from datasets import load_dataset

# Load your data
with open("generations_google-gemma-2b-it_0_-1.jsonl", "r") as f:
    your_data = [json.loads(line) for line in f]

print(f"Your dataset: {len(your_data)} questions")

# Load official GSM8K
train_set = load_dataset("gsm8k", "main", split="train")
test_set = load_dataset("gsm8k", "main", split="test")

train_questions = set(item["question"] for item in train_set)
test_questions = set(item["question"] for item in test_set)
your_questions = set(item["question"] for item in your_data)

# Check overlaps
train_overlap = len(your_questions & train_questions)
test_overlap = len(your_questions & test_questions)
neither = len(your_questions) - train_overlap - test_overlap

print(f"\nOfficial GSM8K train: {len(train_set)}")
print(f"Official GSM8K test: {len(test_set)}")
print(f"Total official: {len(train_set) + len(test_set)}")

print(f"\nYour data breakdown:")
print(f"  Overlap with train: {train_overlap} ({train_overlap/len(your_data)*100:.1f}%)")
print(f"  Overlap with test:  {test_overlap} ({test_overlap/len(your_data)*100:.1f}%)")
print(f"  Not in official:    {neither} ({neither/len(your_data)*100:.1f}%)")

print(f"\nConclusion:")
if test_overlap > 0:
    print(f"⚠️  WARNING: {test_overlap} test questions in your data!")
    print("   This means verifier was trained on test set = data leakage")
if neither > 0:
    print(f"ℹ️  {neither} questions not in official GSM8K")