import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# -----------------------------------
# 기본 설정
# -----------------------------------

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

BASE_PATH = "/gpfs/data/oermannlab/gaifl/users/yl14814/medqa_hybrid_project"

RESULT_PATH = f"{BASE_PATH}/03_results/baselines/llama31/MedQA"
DEBUG_PATH = f"{BASE_PATH}/03_results/debug/llama31/MedQA"

os.makedirs(RESULT_PATH, exist_ok=True)

# -----------------------------------
# 모델 로드
# -----------------------------------

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded")

# -----------------------------------
# 데이터 로드
# -----------------------------------

print("Loading dataset...")

dataset = load_dataset("bigbio/med_qa", "med_qa_en")

test_data = dataset["test"]

print("Dataset loaded:", len(test_data))

# -----------------------------------
# prompt 생성
# -----------------------------------

def build_fewshot_prompt(example):

    fewshot = """
Question:
A patient with crushing chest pain radiating to the left arm most likely has:

Options:
A. Pneumonia
B. Myocardial infarction
C. GERD
D. Panic attack

Answer: B
"""

    labels = ["A", "B", "C", "D"]

    options_text = ""

    for i, c in enumerate(example["choices"]):
        options_text += f"{labels[i]}. {c}\n"

    prompt = f"""
You are a medical expert solving USMLE questions.

Example:

{fewshot}

Now answer the question.

Question:
{example["question"]}

Options:
{options_text}

Answer with only one letter.

Answer:
"""

    return prompt

# -----------------------------------
# gold label 변환
# -----------------------------------

def get_gold_letter(example):

    correct_text = example["answer"][0]

    labels = ["A", "B", "C", "D"]

    for i, c in enumerate(example["choices"]):

        if c == correct_text:

            return labels[i]

    return None

# -----------------------------------
# answer 추출
# -----------------------------------

import re

def extract_answer(text):

    match = re.search(r"(A|B|C|D)", text)

    if match:

        return match.group(1)

    return "INVALID"

# -----------------------------------
# inference
# -----------------------------------

results = []

correct = 0

for example in tqdm(test_data):

    prompt = build_fewshot_prompt(example)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    pred = extract_answer(decoded)

    gold = get_gold_letter(example)

    is_correct = pred == gold

    if is_correct:

        correct += 1

    results.append({
        "gold": gold,
        "prediction": pred,
        "correct": is_correct
    })

accuracy = correct / len(test_data)

print("Final accuracy:", accuracy)

# -----------------------------------
# 저장
# -----------------------------------

save_file = f"{RESULT_PATH}/MedQA_fewshot_chat_full.json"

with open(save_file, "w") as f:

    json.dump(results, f, indent=2)

print("Saved:", save_file)
