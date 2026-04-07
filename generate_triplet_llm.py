import pickle
import random

print("START LLM TRIPLET GENERATION")

########################################
# 1. baseline 로드
########################################

baseline = pickle.load(open("medqa_triplets.pkl", "rb"))

print("Loaded baseline:", len(baseline))

########################################
# 2. LLM triplet 생성 (🔥 HARD NEGATIVE)
########################################

triplets_llm = []

for i, item in enumerate(baseline):

    question = item["question"]
    positive = item["positive"]

    # 🔥 핵심: 다른 샘플의 정답을 negative로 사용
    negative = random.choice(baseline)["positive"]

    # 같은 경우 방지
    if negative == positive:
        continue

    triplets_llm.append({
        "question": question,
        "positive": positive,
        "negative": negative
    })

    if i % 1000 == 0:
        print(f"Progress: {i}/{len(baseline)}")

########################################
# 3. 저장
########################################

pickle.dump(triplets_llm, open("triplets_llm.pkl", "wb"))

print("Saved triplets_llm.pkl")
print("END SCRIPT")
