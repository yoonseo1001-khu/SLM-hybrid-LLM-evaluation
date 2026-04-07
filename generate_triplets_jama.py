import json
import requests
import pandas as pd
from tqdm import tqdm
import os

# =========================
# API 설정
# =========================
API_URL = "http://10.189.26.12:30002/v1/chat/completions"

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

    # 🔥 CLEAN
    neg = full_text.split("\n")[0].strip()

    return neg


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("00_data/splits/jama_train.csv")

# 테스트용 (나중에 전체로 바꿔도 됨)
df = df.sample(200)

triplets = []

SAVE_PATH = "03_results/jama_triplets.json"

os.makedirs("03_results", exist_ok=True)

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

    # 중간 저장
    if len(triplets) % 20 == 0:
        json.dump(triplets, open(SAVE_PATH, "w"), indent=2)

# 최종 저장
json.dump(triplets, open(SAVE_PATH, "w"), indent=2)

print("\n🔥 JAMA Triplet DONE")
