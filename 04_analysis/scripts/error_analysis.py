import os
import json
import pandas as pd

# ---------------------------------
# 기본 경로
# ---------------------------------

BASE_DIR = "/gpfs/data/oermannlab/gaifl/users/yl14814/medqa_hybrid_project"
RESULT_DIR = f"{BASE_DIR}/03_results/baselines"

model = "llama31"
dataset = "MedQA"

# ---------------------------------
# 결과 파일 경로
# ---------------------------------

files = {
    "zeroshot": f"{RESULT_DIR}/{model}/{dataset}/{dataset}_zeroshot_chat_full.json",
    "cot": f"{RESULT_DIR}/{model}/{dataset}/{dataset}_cot_chat_full.json",
    "fewshot": f"{RESULT_DIR}/{model}/{dataset}/{dataset}_fewshot_chat_full.json"
}

# ---------------------------------
# 결과 로드
# ---------------------------------

data = {}

for k, v in files.items():

    if os.path.exists(v):

        print(f"Loading: {v}")

        with open(v, "r") as f:
            data[k] = json.load(f)

    else:

        print(f"Missing file: {v}")

# ---------------------------------
# 데이터 확인
# ---------------------------------

if len(data) == 0:
    raise ValueError("No result files found!")

length = len(next(iter(data.values())))

# ---------------------------------
# dataframe 생성
# ---------------------------------

df = pd.DataFrame({
    "question_id": range(length),
    **{k: [x["correct"] for x in v] for k, v in data.items()}
})

# ---------------------------------
# error pattern 분석
# ---------------------------------

all_wrong = df[
    (df["zeroshot"] == False) &
    (df["cot"] == False) &
    (df["fewshot"] == False)
]

only_cot = df[
    (df["cot"] == True) &
    (df["zeroshot"] == False) &
    (df["fewshot"] == False)
]

only_fewshot = df[
    (df["fewshot"] == True) &
    (df["zeroshot"] == False) &
    (df["cot"] == False)
]

# ---------------------------------
# 결과 출력
# ---------------------------------

print("\n===== Error Analysis =====\n")

print("Total questions:", len(df))
print("All prompts wrong:", len(all_wrong))
print("Only CoT correct:", len(only_cot))
print("Only Few-shot correct:", len(only_fewshot))

# ---------------------------------
# 저장
# ---------------------------------

save_dir = f"{BASE_DIR}/04_analysis"
os.makedirs(save_dir, exist_ok=True)

all_wrong.to_csv(f"{save_dir}/all_models_wrong.csv", index=False)
only_cot.to_csv(f"{save_dir}/only_cot_correct.csv", index=False)
only_fewshot.to_csv(f"{save_dir}/only_fewshot_correct.csv", index=False)

print("\nSaved analysis files to:", save_dir)
