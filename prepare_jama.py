import json
import pandas as pd

# 원본 JAMA 데이터
data = json.load(open("/gpfs/data/oermannlab/gaifl/data/jama_clinical_cases.json"))

rows = []

for item in data:
    q = item.get("question", "")
    a = item.get("answer", "")

    if q and a:
        rows.append({
            "question": q,
            "answer": a,
            "difficulty": "medium"
        })

df = pd.DataFrame(rows)

# 저장
df.to_csv("00_data/splits/jama_train.csv", index=False)

print("Saved:", len(df))
