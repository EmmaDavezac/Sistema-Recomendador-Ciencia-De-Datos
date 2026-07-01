import json
import traceback

notebook_path = 'Desarrollo CRISP-DM.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

globals_dict = {}
code_cells = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']

print(f"Found {len(code_cells)} code cells to execute.")

success_count = 0
for idx, cell in enumerate(code_cells):
    original_code = "".join(cell['source'])
    
    # Strip magic commands (lines starting with % or !)
    code_lines = []
    for line in original_code.splitlines():
        trimmed = line.strip()
        if trimmed.startswith(('%', '!')):
            # Comment it out so we can still print/log it if needed
            code_lines.append(f"# {line}")
        else:
            code_lines.append(line)
            
    code_content = "\n".join(code_lines)
    # Prevent matplotlib blockings
    code_content = "import matplotlib.pyplot as plt\nplt.show = lambda *args, **kwargs: None\n" + code_content
    
    # Skip completely empty cells (after stripping magics)
    if not code_content.strip():
        continue
        
    try:
        exec(code_content, globals_dict)
        success_count += 1
    except Exception as e:
        print(f"ERROR in Cell {idx+1}:")
        print("Code executed:")
        print(code_content)
        print("-" * 40)
        print(traceback.format_exc())
        break

print(f"\nExecution summary: {success_count}/{len(code_cells)} cells executed successfully.")
