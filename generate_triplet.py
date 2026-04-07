import random
import pickle
from datasets import load_dataset

print("START SCRIPT")

########################################
# 1. 데이터 로드
########################################

dataset = load_dataset(
    "bigbio/med_qa",
    "med_qa_en_4options_bigbio_qa",
    trust_remote_code=True
)

train_data = dataset["train"]

print("Dataset size:", len(train_data))

########################################
# 2. triplet 생성
########################################

triplets = []

for i in range(len(train_data)):

    item = train_data[i]

    question = item["question"]

    # choices는 list 형태
    choices = item["choices"]

    # answer는 정답 텍스트
    correct_answer = item["answer"]

    # negative 생성 (정답 제외)
    negatives = [c for c in choices if c != correct_answer]

    if len(negatives) == 0:
        continue

    negative = random.choice(negatives)

    triplets.append({
        "question": question,
        "positive": correct_answer,
        "negative": negative
    })

    if i % 100 == 0:
        print(f"Progress: {i}/{len(train_data)}")

########################################
# 3. 저장
########################################

pickle.dump(triplets, open("medqa_triplets.pkl", "wb"))

print("Saved medqa_triplets.pkl")
print("END SCRIPT")
