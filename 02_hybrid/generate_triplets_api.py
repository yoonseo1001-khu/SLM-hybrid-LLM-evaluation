import json
import requests
import pandas as pd
from tqdm import tqdm
import os

# =========================
# API 설정
# =========================
API_URL = "http://10.189.26.12:30002/v1/chat/completions"

# =========================
# 🔥 NEGATIVE GENERATION (정제 포함)
# =========================
def generate_negative(question, answer):

    prompt = f"""
    Question: {question}
    Correct Answer: {answer}

    Generate a medically plausible but incorrect answer.
    Return ONLY one short sentence.
    """

    data = {
        "model": "llama-3-3-70b-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    response = requests.post(API_URL, json=data)
    result = response.json()

    full_text = result["choices"][0]["message"]["content"]

    # =========================
    # 🔥 CLEANING (핵심)
    # =========================
    neg = full_text.split("\n")[0]  # 첫 줄만
    neg = neg.replace("Note:", "")
    neg = neg.strip()

    return neg

# =========================
# LOAD DATA
# =========================
BASE_PATH = os.path.abspath(os.path.join(os.getcwd(), ".."))
DATA_PATH = os.path.join(BASE_PATH, "00_data", "splits", "medqa", "train.csv")
SAVE_PATH = os.path.join(BASE_PATH, "03_results", "medqa_final", "triplets_api.json")

df = pd.read_csv(DATA_PATH)

# 테스트용 (처음엔 작게)
df = df.sample(200)

triplets = []

# =========================
# GENERATE LOOP
# =========================
for i, row in tqdm(df.iterrows(), total=len(df)):

    q = str(row["question"])
    a = str(row["answer"])

    try:
        neg = generate_negative(q, a)

        triplets.append({
            "question": q,
            "positive": a,
            "negative": neg
        })

    except Exception as e:
        print("Error:", e)
        continue

    # =========================
    # 🔥 중간 저장 (중요)
    # =========================
    if len(triplets) % 20 == 0:
        json.dump(triplets, open(SAVE_PATH, "w"), indent=2)

# =========================
# FINAL SAVE
# =========================
json.dump(triplets, open(SAVE_PATH, "w"), indent=2)

print("\n🔥 Triplet generation DONE")
