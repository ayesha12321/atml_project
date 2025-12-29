import json


scored_file = "generations_google-gemma-2b-it_0_-1.jsonl"

scored_data = []
skipped = 0
with open(scored_file, 'r') as f:
    for line_num, line in enumerate(f, 1):
        try:
            scored_data.append(json.loads(line))
        except json.JSONDecodeError as e:
            skipped += 1
            if skipped <= 3:
                print(f"Warning: Skipping corrupted line {line_num}: {str(e)[:50]}")


print(len(scored_data))