import argparse
import os
import torch
import json
import re
from tqdm import tqdm
from collections import Counter
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# =========================
# Argument
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--method", type=str, required=True)
args = parser.parse_args()

MODEL_NAME = args.model
METHOD = args.method

# =========================
# Path
# =========================
BASE_PATH = "/gpfs/data/oermannlab/gaifl/users/yl14814/SLM_hybrid_project"

BASELINE_RESULT_DIR = f"{BASE_PATH}/03_results/baselines/{MODEL_NAME}/MedQA"
DEBUG_RESULT_DIR = f"{BASE_PATH}/03_results/debug/{MODEL_NAME}/MedQA"

os.makedirs(BASELINE_RESULT_DIR, exist_ok=True)
os.makedirs(DEBUG_RESULT_DIR, exist_ok=True)

# =========================
# Model mapping
# =========================
MODEL_MAP = {
    "llama31": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "qwen2_7b": "Qwen/Qwen2-7B-Instruct"
}

MODEL_ID = MODEL_MAP[MODEL_NAME]

print("Loading model:", MODEL_ID)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# =========================
# Dataset
# =========================
dataset = load_dataset(
    "bigbio/med_qa",
    "med_qa_en_4options_bigbio_qa",
    trust_remote_code=True
)

test_data = dataset["test"]

print("Test size:", len(test_data))

# =========================
# Helper functions
# =========================
def get_choices(example):

    if isinstance(example["choices"], dict):
        return example["choices"]["text"]

    if isinstance(example["choices"], list):
        return example["choices"]

    return []

def get_gold_letter(example):

    choices = get_choices(example)
    labels = ["A","B","C","D"]

    if isinstance(example["answer"], list):

        gold_text = example["answer"][0].strip().lower()

        for i, choice in enumerate(choices):
            if choice.strip().lower() == gold_text:
                return labels[i]

    if isinstance(example["answer"], str):
        ans = example["answer"].strip().upper()
        if ans in ["A","B","C","D"]:
            return ans

    return "INVALID"

# =========================
# Extractors
# =========================
def extract_choice(text):

    text = text.strip()

    match = re.search(r"answer\s*[:\-]?\s*(A|B|C|D)", text, re.IGNORECASE)
    if match:
        return match.group(1)

    match = re.match(r"^\s*(A|B|C|D)[\.\)]?", text)
    if match:
        return match.group(1)

    return "INVALID"

def extract_cot_answer(text):

    text = text.strip()

    patterns = [
        r"final answer\s*[:\-]?\s*(A|B|C|D)",
        r"answer\s*[:\-]?\s*(A|B|C|D)",
        r"correct option\s*(A|B|C|D)",
        r"option\s*(A|B|C|D)",
        r"\b(A|B|C|D)\b"
    ]

    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return "INVALID"

# =========================
# Prompts (노트북 그대로)
# =========================
def build_zero_shot_prompt(example):

    question = example["question"]
    choices = get_choices(example)

    labels = ["A", "B", "C", "D"]

    options_text = "\n".join(
        [f"{labels[i]}. {choice}" for i, choice in enumerate(choices)]
    )

    return f"""
You are a medical expert solving a USMLE clinical question.

Question:
{question}

Options:
{options_text}

Choose the correct answer.

Answer:
"""

def build_cot_prompt(example):

    question = example["question"]
    choices = get_choices(example)

    options_text = "\n".join(
        [f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]
    )

    return f"""
You are a medical expert answering a USMLE clinical question.

Question:
{question}

Options:
{options_text}

Think through the problem step by step.

At the end of your reasoning, output ONLY the letter of the correct answer.

Format strictly as:

Final Answer: X

Where X is one of A, B, C, or D.

Reasoning:
"""

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

    question = example["question"]
    choices = get_choices(example)

    options_text = "\n".join(
        [f"{chr(65+i)}. {choice}" for i,choice in enumerate(choices)]
    )

    return f"""
You are a medical expert solving USMLE questions.

Example:
{fewshot}

Now answer the following question.

Question:
{question}

Options:
{options_text}

Answer with only the letter (A, B, C, or D).

Answer:
"""

# =========================
# Inference (노트북 그대로)
# =========================
def run_inference_batch(data, prompt_builder, extractor, batch_size=4):

    correct = 0
    results = []

    prompts = [prompt_builder(data[i]) for i in range(len(data))]

    for i in tqdm(range(0, len(prompts), batch_size)):

        batch_prompts = prompts[i:i+batch_size]

        batch_examples = [
            data[k] for k in range(i, min(i+batch_size, len(data)))
        ]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )

        for j in range(len(batch_prompts)):

            input_len = inputs["input_ids"].shape[1]

            generated_tokens = outputs[j][input_len:]

            decoded = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )

            prediction = extractor(decoded)
            gold = get_gold_letter(batch_examples[j])

            is_correct = prediction == gold

            if is_correct:
                correct += 1

            results.append({
                "question": batch_examples[j]["question"],
                "gold": gold,
                "prediction": prediction,
                "correct": is_correct,
                "model_output": decoded
            })

    accuracy = correct / len(data)
    return accuracy, results

# =========================
# RUN
# =========================
if METHOD == "zeroshot":
    prompt_fn = build_zero_shot_prompt
    extractor = extract_choice

elif METHOD == "cot":
    prompt_fn = build_cot_prompt
    extractor = extract_cot_answer

elif METHOD == "fewshot":
    prompt_fn = build_fewshot_prompt
    extractor = extract_choice

save_path = f"{BASELINE_RESULT_DIR}/MedQA_{METHOD}_chat_full.json"

acc, res = run_inference_batch(
    test_data,
    prompt_fn,
    extractor
)

print(f"{METHOD} Accuracy:", acc)

with open(save_path, "w") as f:
    json.dump({
        "accuracy": acc,
        "results": res
    }, f, indent=2)

print("Saved:", save_path)
