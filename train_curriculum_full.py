import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("00_data/splits/jama_final/train.csv")

# difficulty별 분리
easy = df[df["difficulty"] == "easy"]
medium = df[df["difficulty"] == "medium"]
hard = df[df["difficulty"] == "hard"]

stages = [easy, medium, hard]

def text_to_tensor(text_list):
    max_len = 12
    tensor = []

    for text in text_list:
        ids = [ord(c) % 10000 for c in text[:max_len]]
        ids += [0] * (max_len - len(ids))
        tensor.append(ids)

    return torch.tensor(tensor).to(DEVICE)

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
criterion = nn.TripletMarginLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for i, stage in enumerate(stages):

    print(f"\n===== Stage {i} =====")

    if len(stage) < 2:
        continue

    dataset = list(zip(stage["question"], stage["answer"], stage["answer"].sample(len(stage))))

    loader = DataLoader(dataset, batch_size=4)

    for epoch in range(1):

        total = 0

        for q, p, n in loader:

            q_emb = model(q)
            p_emb = model(p)
            n_emb = model(n)

            loss = criterion(q_emb, p_emb, n_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()

        print(f"Stage {i} Loss:", total / len(loader))
