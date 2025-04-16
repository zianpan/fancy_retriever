import json
import os

input_file = "train1000_codexglue.json"
output_file = "test_small.json"

with open(input_file, 'r') as f:
    data = json.load(f)

reduced_data = []
max_ctxs = 50
for item in data[:100]:
    reduced_item = item.copy()
    
    if 'ctxs' in item and isinstance(item['ctxs'], list):
        reduced_item['ctxs'] = item['ctxs'][:max_ctxs]
    
    reduced_data.append(reduced_item)

# Save the reduced dataset
with open(output_file, 'w') as f:
    json.dump(reduced_data, f, indent=2)

print(f"Original data had {len(data)} questions")
print(f"Reduced data has {len(reduced_data)} questions, each with maximum {max_ctxs} contexts")