import json
import os
import pandas as pd

BASE = "/gpfs/data/oermannlab/gaifl/users/yl14814/SLM_hybrid_project/03_results/baselines"

models = ["llama31","mistral7b","qwen2_7b"]
methods = ["zeroshot","cot","fewshot"]

rows = []

for m in models:
    print("Checking:", m)
    row = {"Model": m}

    for me in methods:
        path = f"{BASE}/{m}/MedQA/MedQA_{me}_chat_full.json"

        if not os.path.exists(path):
            print("❌ missing:", path)
            row[me] = None
            continue

        with open(path) as f:
            data = json.load(f)

        # 🔥 핵심 수정
        if isinstance(data, dict):
            acc = data.get("accuracy", None)

        elif isinstance(data, list):
            correct = sum(1 for x in data if x.get("correct") == True)
            acc = correct / len(data)

        else:
            acc = None

        row[me] = acc

    rows.append(row)

df = pd.DataFrame(rows)

print("\n=== FINAL TABLE ===")
print(df)

save_path = f"{BASE}/final_medqa_table.csv"
df.to_csv(save_path, index=False)

print("\nSaved:", save_path)
