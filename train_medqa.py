#LLM triplet 버전 최종본#
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

print("Start script...")

########################################
# 1. GPU 강제 (🔥 핵심)
########################################

assert torch.cuda.is_available(), "❌ GPU not available"
device = torch.device("cuda")
print("Using device:", device)

########################################
# 2. Dataset
########################################

class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        print(f"Loading dataset from {path}...")
        self.data = pickle.load(open(path, "rb"))
        print(f"Loaded dataset size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["question"], item["positive"], item["negative"]

########################################
# 3. Collate (🔥 중요)
########################################

def collate_fn(batch):
    qs, ps, ns = zip(*batch)
    return list(qs), list(ps), list(ns)

########################################
# 4. Dataset 로드
########################################

print("Before dataset load")
dataset = TripletDataset("hybrid_7_3.pkl")
print("After dataset load")

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,        # 🔥 메모리 안정
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0        # 🔥 필수 (OOM 방지)
)

########################################
# 5. 모델
########################################

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10000, 128)

    def forward(self, x):
        return self.embedding(x)

model = Encoder().to(device)

########################################
# 6. Loss / Optimizer
########################################

criterion = nn.TripletMarginLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

########################################
# 7. Tokenizer (🔥 길이 축소)
########################################

def dummy_tokenize(text):
    return torch.randint(0, 10000, (5,))   # 🔥 5로 줄임

########################################
# 8. Training
########################################

print("Start training...")

for epoch in range(3):
    print(f"\nEpoch {epoch}")

    for i, (q, p, n) in enumerate(dataloader):

        q = torch.stack([dummy_tokenize(x) for x in q]).to(device)
        p = torch.stack([dummy_tokenize(x) for x in p]).to(device)
        n = torch.stack([dummy_tokenize(x) for x in n]).to(device)

        q_emb = model(q)
        p_emb = model(p)
        n_emb = model(n)

        loss = criterion(q_emb, p_emb, n_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Step {i} | Loss: {loss.item():.4f}")

print("Training finished!")
