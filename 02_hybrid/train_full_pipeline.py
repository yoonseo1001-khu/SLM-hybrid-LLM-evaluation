
import os
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import ast

# =========================
# PATH
# =========================
BASE_PATH = os.path.abspath(os.path.join(os.getcwd(), ".."))
DATA_PATH = os.path.join(BASE_PATH, "00_data", "splits", "medqa", "train.csv")
RESULT_PATH = os.path.join(BASE_PATH, "03_results")

os.makedirs(RESULT_PATH, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)
df = df.sample(min(2000, len(df)))

print("Loaded:", len(df))

# =========================
# TEXT → TENSOR
# =========================
def text_to_tensor(text_list):
    max_len = 12
    tensor = []

    for text in text_list:
        ids = [ord(c) % 10000 for c in text[:max_len]]
        ids += [0] * (max_len - len(ids))
        tensor.append(ids)

    return torch.tensor(tensor).to(DEVICE)

# =========================
# DATASET
# =========================
class TripletDataset(Dataset):
    def __init__(self, df, mode="baseline"):
        self.df = df.reset_index(drop=True)
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        q = str(row["question"])
        p = str(row["answer"])

        # =========================
        # 🔥 IMPROVED HARD NEGATIVE
        # =========================
        if self.mode == "hybrid":
            choices = row.get("choices", None)

            if isinstance(choices, str):
                choices = ast.literal_eval(choices)

            if isinstance(choices, list) and len(choices) > 1:
                # 정답 제외
                candidates = [c for c in choices if c != p]

                if len(candidates) > 0:
                    # 🔥 길이 기준으로 가장 비슷한 답 선택
                    candidates_sorted = sorted(
                        candidates,
                        key=lambda x: abs(len(str(x)) - len(str(p)))
                    )
                    neg = str(candidates_sorted[0])
                else:
                    neg = str(self.df.sample(1).iloc[0]["answer"])
            else:
                neg = str(self.df.sample(1).iloc[0]["answer"])

        else:
            # baseline
            neg = str(self.df.sample(1).iloc[0]["answer"])

        difficulty = row["difficulty"] if "difficulty" in self.df.columns else "medium"

        return q, p, neg, difficulty

# =========================
# MODEL
# =========================
class SimpleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10000, 64)
        self.linear = nn.Linear(64, 64)

    def forward(self, x):
        input_ids = text_to_tensor(x)
        emb = self.embedding(input_ids)
        emb = emb.mean(dim=1)
        return self.linear(emb)

model = SimpleEncoder().to(DEVICE)

# =========================
# LOSS
# =========================
criterion = nn.TripletMarginLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# =========================
# TRAIN
# =========================
def train_loop(df, mode="baseline", epochs=3):

    dataset = TripletDataset(df, mode)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    losses = []

    for epoch in range(epochs):
        total_loss = 0

        for q, p, n, _ in tqdm(dataloader):

            q_emb = model(q)
            p_emb = model(p)
            n_emb = model(n)

            loss = criterion(q_emb, p_emb, n_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(dataloader)
        losses.append(avg)

        print(f"[{mode}] Epoch {epoch} Loss: {avg}")

    return losses

# =========================
# CURRICULUM
# =========================
def curriculum_train(df):

    if "difficulty" not in df.columns:
        return train_loop(df, "hybrid")

    easy = df[df["difficulty"] == "easy"]
    medium = df[df["difficulty"] == "medium"]
    hard = df[df["difficulty"] == "hard"]

    all_losses = []

    for stage, stage_df in enumerate([easy, medium, hard]):

        if len(stage_df) == 0:
            continue

        print(f"\nStage {stage}")
        losses = train_loop(stage_df, "hybrid", epochs=1)
        all_losses.extend(losses)

    return all_losses

# =========================
# EVAL
# =========================
def eval_accuracy(df):

    sample = df.sample(200)
    correct = 0

    for _, row in sample.iterrows():

        q = str(row["question"])
        p = str(row["answer"])
        n = str(df.sample(1).iloc[0]["answer"])

        q_emb = model([q])
        p_emb = model([p])
        n_emb = model([n])

        if torch.cosine_similarity(q_emb, p_emb) > torch.cosine_similarity(q_emb, n_emb):
            correct += 1

    return correct / len(sample)

# =========================
# RUN
# =========================
print("\nBASELINE")
baseline = train_loop(df, "baseline")

print("\nHYBRID")
hybrid = train_loop(df, "hybrid")

print("\nCURRICULUM")
curriculum = curriculum_train(df)

acc = eval_accuracy(df)
print("Accuracy:", acc)

# =========================
# SAVE
# =========================
results = {
    "baseline": baseline,
    "hybrid": hybrid,
    "curriculum": curriculum,
    "accuracy": acc
}

json.dump(results, open(os.path.join(RESULT_PATH, "final.json"), "w"), indent=4)

# =========================
# PLOT
# =========================
plt.figure()
plt.plot(baseline, label="baseline")
plt.plot(hybrid, label="hybrid")
plt.plot(curriculum, label="curriculum")

plt.legend()
plt.title("Final Comparison")
plt.savefig(os.path.join(RESULT_PATH, "final_plot.png"))

print("\n🔥 DONE (hard negative improved)")
