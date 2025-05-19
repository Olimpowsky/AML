import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

features = pd.read_csv("elliptic_txs_features.csv", header=None)
classes = pd.read_csv("elliptic_txs_classes.csv")
edges = pd.read_csv("elliptic_txs_edgelist.csv")

features.columns = ["txId","time_step"] + [f"feature_{i}" for i in range(1,166)]
classes.columns = ["txId","class"]
edges.columns= ["source","target"]

df = pd.merge(features, classes, on="txId")
df = df[df["class"] != "unknown"].reset_index(drop=True)
df["class"] = df["class"].map({"1":0,"2":1})
df = df.sort_values("time_step").reset_index(drop=True)

X_np = df.drop(["txId","time_step","class"], axis=1).values.astype(np.float32)
y_np = df["class"].values
tx_to_idx = {tx: i for i, tx in enumerate(df["txId"])}

N = len(df)
src = edges["source"].map(tx_to_idx)
tgt = edges["target"].map(tx_to_idx)
valid = src.notna() & tgt.notna()
src = src[valid].astype(int).to_numpy()
tgt = tgt[valid].astype(int).to_numpy()

x = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.long)
edge_index = torch.tensor(np.vstack([src, tgt]), dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits = TimeSeriesSplit(n_splits=5)
f1_scores = []

for fold, (train_idx, test_idx) in enumerate(splits.split(df), 1):
    model = GCN(x.shape[1], hidden_feats=64, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    data = data.to(device)
    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=1).cpu().numpy()
    f1 = f1_score(y_np[test_idx], preds[test_idx])
    f1_scores.append(f1)
    print(f"Fold {fold} — F1-score: {f1:.4f}")

print(f"Średni F1-score: {np.mean(f1_scores):.4f}")
