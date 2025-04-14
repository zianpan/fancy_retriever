import json
import os

input_file = "train1000_codexglue.json"
output_file = "test_small.json"

with open(input_file, 'r') as f:
    data = json.load(f)

reduced_data = []

for item in data[:10]:
    reduced_item = item.copy()
    
    if 'ctxs' in item and isinstance(item['ctxs'], list):
        reduced_item['ctxs'] = item['ctxs'][:10]
    
    reduced_data.append(reduced_item)

# Save the reduced dataset
with open(output_file, 'w') as f:
    json.dump(reduced_data, f, indent=2)

print(f"Successfully created {output_file} with top 5 contexts for each question")
print(f"Original data had {len(data)} questions")
print(f"Reduced data has {len(reduced_data)} questions, each with maximum 5 contexts")