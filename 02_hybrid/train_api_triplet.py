import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD API TRIPLETS
# =========================
triplets = json.load(open("../03_results/medqa_final/triplets_api.json"))

print("Loaded triplets:", len(triplets))

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
    def __init__(self, triplets):
        self.data = triplets

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
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

losses = []

for epoch in range(3):

    total_loss = 0

    for q, p, n in tqdm(dataloader):

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

    print(f"[API] Epoch {epoch} Loss: {avg}")

# =========================
# SAVE
# =========================
os.makedirs("../03_results/medqa_final", exist_ok=True)

json.dump(losses, open("../03_results/medqa_final/api_loss.json", "w"))

# =========================
# PLOT
# =========================
plt.figure()
plt.plot(losses, label="API Hybrid")

plt.legend()
plt.title("API Triplet Training")

plt.savefig("../03_results/medqa_final/api_plot.png")

print("\n🔥 API TRAINING DONE")
