import json
import requests
import pandas as pd
from tqdm import tqdm
import os

API_URL_1 = "http://10.189.26.12:30002/v1/chat/completions"
API_URL_2 = "http://10.189.26.12:30002/v1/chat/completions"  # 다른 모델로 바꿀 수 있음

def generate_negative(api_url, question, answer):

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

    response = requests.post(api_url, json=data)
    result = response.json()

    text = result["choices"][0]["message"]["content"]
    return text.split("\n")[0].strip()


df = pd.read_csv("00_data/splits/medqa/train.csv").sample(200)

triplets = []
SAVE_PATH = "03_results/multi_triplets.json"

os.makedirs("03_results", exist_ok=True)

for _, row in tqdm(df.iterrows(), total=len(df)):

    q = row["question"]
    a = row["answer"]

    try:
        neg1 = generate_negative(API_URL_1, q, a)
        neg2 = generate_negative(API_URL_2, q, a)

        triplets.append({
            "question": q,
            "positive": a,
            "negative_1": neg1,
            "negative_2": neg2
        })

    except:
        continue

json.dump(triplets, open(SAVE_PATH, "w"), indent=2)

print("DONE")
