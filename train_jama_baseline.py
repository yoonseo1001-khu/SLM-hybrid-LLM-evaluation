import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD TRIPLETS
# =========================
triplets = json.load(open("03_results/jama_triplets_baseline.json"))

print("Loaded:", len(triplets))

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
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["question"], item["positive"], item["negative"]

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
dataset = TripletDataset(triplets)
loader = DataLoader(dataset, batch_size=4, shuffle=True)  # 안정성 위해 4

losses = []

for epoch in range(3):

    total = 0

    for q, p, n in tqdm(loader):

        q_emb = model(q)
        p_emb = model(p)
        n_emb = model(n)

        loss = criterion(q_emb, p_emb, n_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    avg = total / len(loader)
    losses.append(avg)

    print(f"[JAMA BASELINE] Epoch {epoch} Loss:", avg)

# =========================
# SAVE
# =========================
os.makedirs("03_results", exist_ok=True)

json.dump(losses, open("03_results/jama_baseline_loss.json", "w"))

plt.figure()
plt.plot(losses, label="JAMA Baseline")
plt.legend()
plt.title("JAMA Baseline Training")
plt.savefig("03_results/jama_baseline_plot.png")

print("\n🔥 JAMA BASELINE TRAIN DONE")
