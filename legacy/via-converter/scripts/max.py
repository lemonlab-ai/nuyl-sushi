import json

# Step 1: Read the JSON file
with open('taxonomy.json', 'r') as file:
    data = json.load(file)

# Step 2 & 3: Extract 'parentName' values and handle mixed types
parent_names = [str(item['parentName']) for item in data if item['parentName'] is not None]

# Step 4: Find the maximum 'parentName'
max_parent_name = max(parent_names, key=lambda x: (x.isdigit(), x))

print(f"The maximum parentName is: {max_parent_name}")