# ============================================================
# QRNG — Análise dos bits gerados
# Inatel — Mariana Bissaro Weiss
# ============================================================

import numpy as np
import os
import matplotlib.pyplot as plt

# --- Carrega os bits mais recentes ---
pasta = sorted(os.listdir('resultados'))[-1]  # pega a pasta mais recente
caminho = f'resultados/{pasta}/bits_qrng.npy'
bits = np.load(caminho)

print(f"✓ Bits carregados de: {caminho}")
print(f"  Total: {len(bits):,} bits")

# --- Min-Entropia ---
p1 = bits.mean()
p0 = 1 - p1
p_max = max(p0, p1)
h_min = -np.log2(p_max)

print(f"\n--- Análise estatística ---")
print(f"P(0) = {p0:.6f}")
print(f"P(1) = {p1:.6f}")
print(f"Bias = {abs(p1 - 0.5):.6f}")
print(f"Min-Entropia = {h_min:.6f} bits/bit")

# --- Transformer para detectar padrões ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEQ_LEN = 64
BATCH_SIZE = 512
EPOCHS = 10

# Cria sequências
X, y = [], []
for i in range(len(bits) - SEQ_LEN):
    X.append(bits[i:i+SEQ_LEN])
    y.append(bits[i+SEQ_LEN])

X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(np.array(y), dtype=torch.float32)

# Split treino/teste
split = int(0.8 * len(X))
train_ds = TensorDataset(X[:split], y[:split])
test_ds  = TensorDataset(X[split:], y[split:])
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# Modelo
class TransformerQRNG(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=1, nhead=1, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(SEQ_LEN, 1)

    def forward(self, x):
        x = self.transformer(x)
        x = x.squeeze(-1)
        return self.fc(x).squeeze(-1)

model = TransformerQRNG()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

print("\n--- Treinamento do Transformer ---")
accs = []
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_dl:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_dl:
            preds = (torch.sigmoid(model(xb)) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += len(yb)
    acc = correct / total * 100
    accs.append(acc)
    print(f"Época {epoch+1}/{EPOCHS} — Acurácia teste: {acc:.2f}%")

print(f"\n✓ Acurácia final: {acc:.2f}%")
print(f"  (esperado ~50% para QRNG ideal)")

# --- Gráficos ---
# 1. Distribuição dos bits
plt.figure(figsize=(6,4))
plt.bar(['0', '1'], [p0*100, p1*100], color=['steelblue', 'tomato'])
plt.axhline(50, color='black', linestyle='--', label='Ideal (50%)')
plt.title('Distribuição dos bits QRNG')
plt.ylabel('Porcentagem (%)')
plt.ylim(49, 51)
plt.legend()
plt.tight_layout()
plt.savefig(f'resultados/{pasta}/distribuicao_bits.png')
plt.show()

# 2. Acurácia por época
epocas = list(range(1, EPOCHS+1))
plt.figure(figsize=(6,4))
plt.plot(epocas, accs, marker='o', color='steelblue')
plt.axhline(50, color='red', linestyle='--', label='Aleatório ideal (50%)')
plt.title('Acurácia do Transformer por Época')
plt.xlabel('Época')
plt.ylabel('Acurácia (%)')
plt.ylim(45, 65)
plt.legend()
plt.tight_layout()
plt.savefig(f'resultados/{pasta}/acuracia_transformer.png')
plt.show()

print(f"✓ Gráficos salvos em resultados/{pasta}/")