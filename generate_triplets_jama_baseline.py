import json
import pandas as pd
from tqdm import tqdm
import os
import random

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("00_data/splits/jama_train.csv")

# 테스트용 (처음엔 작게)
df = df.sample(200).reset_index(drop=True)

triplets = []

SAVE_PATH = "03_results/jama_triplets_baseline.json"

os.makedirs("03_results", exist_ok=True)

# =========================
# GENERATE LOOP
# =========================
for i in tqdm(range(len(df))):

    q = str(df.iloc[i]["question"])
    p = str(df.iloc[i]["answer"])

    # 🔥 random negative (자기 자신 제외)
    while True:
        rand_idx = random.randint(0, len(df) - 1)
        if rand_idx != i:
            break

    n = str(df.iloc[rand_idx]["answer"])

    triplets.append({
        "question": q,
        "positive": p,
        "negative": n
    })

    # 중간 저장
    if len(triplets) % 20 == 0:
        json.dump(triplets, open(SAVE_PATH, "w"), indent=2)

# 최종 저장
json.dump(triplets, open(SAVE_PATH, "w"), indent=2)

print("\n🔥 JAMA Baseline Triplet DONE")
