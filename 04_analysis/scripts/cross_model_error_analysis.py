import os
import json
import pandas as pd

BASE_DIR = "/gpfs/data/oermannlab/gaifl/users/yl14814/medqa_hybrid_project"
RESULT_DIR = f"{BASE_DIR}/03_results/baselines"

models = ["llama31","mistral7b","qwen2_7b"]
dataset = "MedQA"

data = {}

for model in models:

    path = f"{RESULT_DIR}/{model}/{dataset}/{dataset}_fewshot_chat_full.json"

    if not os.path.exists(path):
        print("Missing:", path)
        continue

    with open(path) as f:
        data[model] = json.load(f)

length = len(next(iter(data.values())))

df = pd.DataFrame({
    "question_id": range(length),
    **{m:[x["correct"] for x in data[m]] for m in data}
})

# ---------------------------
# 분석
# ---------------------------

all_wrong = df[(df==False).all(axis=1)]

llama_only = df[
    (df["llama31"]==True) &
    (df["mistral7b"]==False) &
    (df["qwen2_7b"]==False)
]

mistral_only = df[
    (df["mistral7b"]==True) &
    (df["llama31"]==False) &
    (df["qwen2_7b"]==False)
]

qwen_only = df[
    (df["qwen2_7b"]==True) &
    (df["llama31"]==False) &
    (df["mistral7b"]==False)
]

print("All models wrong:", len(all_wrong))
print("Llama only correct:", len(llama_only))
print("Mistral only correct:", len(mistral_only))
print("Qwen only correct:", len(qwen_only))

save_dir = f"{BASE_DIR}/04_analysis"

os.makedirs(save_dir, exist_ok=True)

all_wrong.to_csv(f"{save_dir}/all_models_wrong.csv", index=False)
llama_only.to_csv(f"{save_dir}/llama_only_correct.csv", index=False)
mistral_only.to_csv(f"{save_dir}/mistral_only_correct.csv", index=False)
qwen_only.to_csv(f"{save_dir}/qwen_only_correct.csv", index=False)