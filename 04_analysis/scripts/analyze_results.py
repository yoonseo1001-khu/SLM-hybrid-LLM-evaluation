import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = "/gpfs/data/oermannlab/gaifl/users/yl14814/medqa_hybrid_project"
RESULT_DIR = f"{BASE_DIR}/03_results/baselines"

models = ["llama31", "mistral7b", "qwen2_7b"]
datasets = ["MedQA", "HealthBench", "ReMedE"]
prompts = ["zeroshot", "cot", "fewshot"]

def compute_accuracy(path):

    if not os.path.exists(path):
        return None

    with open(path) as f:
        data = json.load(f)

    correct = sum(1 for x in data if x["correct"])
    return correct / len(data)

rows = []

for model in models:
    for dataset in datasets:

        row = {"model": model, "dataset": dataset}

        for prompt in prompts:

            path = f"{RESULT_DIR}/{model}/{dataset}/{dataset}_{prompt}_chat_full.json"

            acc = compute_accuracy(path)

            row[prompt] = acc

        rows.append(row)

df = pd.DataFrame(rows)

print(df)

# CSV 저장
save_path = f"{BASE_DIR}/03_results/baseline_summary.csv"
df.to_csv(save_path, index=False)

print("Saved summary:", save_path)

# --------------------------------
# Heatmap 생성
# --------------------------------

heatmap_data = df.pivot(index="model", columns="dataset", values="fewshot")

plt.figure(figsize=(6,4))
sns.heatmap(heatmap_data, annot=True, cmap="Blues")

plt.title("Few-shot Accuracy Heatmap")

plt.savefig(f"{BASE_DIR}/04_analysis/accuracy_heatmap.png")

plt.show()

# --------------------------------
# Model Ranking
# --------------------------------

model_rank = df.groupby("model")[["zeroshot","cot","fewshot"]].mean()

print("\nModel Ranking\n")
print(model_rank)

model_rank.to_csv(f"{BASE_DIR}/04_analysis/model_ranking.csv")

# --------------------------------
# Dataset Ranking
# --------------------------------

dataset_rank = df.groupby("dataset")[["zeroshot","cot","fewshot"]].mean()

print("\nDataset Ranking\n")
print(dataset_rank)

dataset_rank.to_csv(f"{BASE_DIR}/04_analysis/dataset_ranking.csv")