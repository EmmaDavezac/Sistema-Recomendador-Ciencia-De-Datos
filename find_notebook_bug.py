import json

notebook_path = 'Desarrollo CRISP-DM.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Searching for cells matching the error signature:")
for i, cell in enumerate(nb['cells']):
    content = "".join(cell['source'])
    if "evaluate_model_2" in content or "get_recommendations_model_2" in content:
        print(f"Index: {i}, Type: {cell['cell_type']}")
        print(content[:500])
        print("="*40)
