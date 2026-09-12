"""
Small VAE (PyTorch) that learns the shape of real rainfall event sequences
from the De Bilt KNMI record and generates synthetic ones.


"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

torch.manual_seed(0)
np.random.seed(0)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR.parent / 'data' / 'result.txt'

# ---------------------------------------------------------------
# 1. Load data and build training windows
# ---------------------------------------------------------------
df = pd.read_csv(
    DATA_PATH,
    comment='#', names=['STN', 'YYYYMMDD', 'RD'], skipinitialspace=True
).dropna()
df['RD_mm'] = df['RD'].astype(float).clip(lower=0) / 10.0
series = df['RD_mm'].values

WINDOW = 30  # 30-day rainfall sequences
STRIDE = 5   # overlap windows to get more training examples

windows = []
for i in range(0, len(series) - WINDOW, STRIDE):
    w = series[i:i + WINDOW]
    # Keep only windows that contain some real rain (skip long dry stretches
    # so the model spends its capacity on actual rain events, not zeros)
    if w.sum() > 20:  # at least 2mm total over the month, arbitrary filter
        windows.append(w)

windows = np.array(windows, dtype=np.float32)
print(f"Training windows: {windows.shape[0]} sequences of length {WINDOW}")

# Normalize with log1p (rainfall is skewed) then scale to roughly [0,1]
log_windows = np.log1p(windows)
scale = log_windows.max()
data = log_windows / scale

X = torch.from_numpy(data)

# ---------------------------------------------------------------
# 2. Define a small VAE
# ---------------------------------------------------------------
LATENT_DIM = 12

class VAE(nn.Module):
    def __init__(self, seq_len=WINDOW, latent_dim=LATENT_DIM, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(seq_len, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, seq_len), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta):
    recon_loss = F.mse_loss(recon, x, reduction='sum') / x.shape[0]
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    return recon_loss + beta * kld, recon_loss, kld


# ---------------------------------------------------------------
# 3. Train (with KL annealing to avoid posterior collapse)
# ---------------------------------------------------------------
model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

EPOCHS = 400
BATCH_SIZE = 64
n = X.shape[0]
MAX_BETA = 0.02  # keep the KL term weak so the decoder can't ignore z

for epoch in range(EPOCHS):
    beta = MAX_BETA * min(1.0, epoch / 100)  # ramp up over first 100 epochs
    perm = torch.randperm(n)
    total_loss, total_recon, total_kld = 0.0, 0.0, 0.0
    n_batches = 0
    for i in range(0, n, BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        batch = X[idx]
        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        loss, recon_loss, kld = vae_loss(recon, batch, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kld += kld.item()
        n_batches += 1
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS}  loss: {total_loss/n_batches:.4f}  "
              f"recon: {total_recon/n_batches:.4f}  kld: {total_kld/n_batches:.4f}  beta: {beta:.4f}")

# ---------------------------------------------------------------
# 4. Generate synthetic sequences and invert the normalization
# ---------------------------------------------------------------
model.eval()
with torch.no_grad():
    z_samples = torch.randn(6, LATENT_DIM)
    synthetic = model.dec(z_samples).numpy()

def invert(norm_seq):
    return np.expm1(norm_seq * scale)

synthetic_mm = np.array([invert(s) for s in synthetic])

# Pick a few real windows for comparison
real_idx = np.random.choice(len(windows), 6, replace=False)
real_examples = windows[real_idx]

# ---------------------------------------------------------------
# 5. Plot real vs synthetic
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 6, figsize=(18, 5), sharey=True)
for i in range(6):
    axes[0, i].bar(range(WINDOW), real_examples[i], color='steelblue')
    axes[0, i].set_title(f"Real #{i+1}", fontsize=9)
    axes[1, i].bar(range(WINDOW), synthetic_mm[i], color='indianred')
    axes[1, i].set_title(f"Synthetic #{i+1}", fontsize=9)
axes[0, 0].set_ylabel("Real windows\n(mm/day)")
axes[1, 0].set_ylabel("VAE-generated\n(mm/day)")
fig.suptitle("30-day rainfall sequences: real (De Bilt) vs VAE-generated")
plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'vae_scenarios.png', dpi=150)
print("\nPlot saved.")

# ---------------------------------------------------------------
# 6. A rough sanity check: compare summary statistics
# ---------------------------------------------------------------
print("\n--- Sanity check: 30-day window totals (mm) ---")
print(f"Real windows      - mean: {windows.sum(axis=1).mean():.1f}, "
      f"std: {windows.sum(axis=1).std():.1f}, "
      f"max: {windows.sum(axis=1).max():.1f}")
print(f"Synthetic windows - mean: {synthetic_mm.sum(axis=1).mean():.1f}, "
      f"std: {synthetic_mm.sum(axis=1).std():.1f}, "
      f"max: {synthetic_mm.sum(axis=1).max():.1f}")

torch.save(model.state_dict(), SCRIPT_DIR / 'vae_rainfall_model.pt')
