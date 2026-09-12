# Flood Risk & AI Scenario Analysis - Sample Work

Two small, self-contained analyses done — one probabilistic/statistical, one generative AI.
Both use the same public rainfall dataset (KNMI, De Bilt, 1901–2025) from
two different angles.

## Contents

- **[`gev-rainfall/`](./gev-rainfall)** — Fits a Generalized
  Extreme Value (GEV) distribution to annual maximum daily rainfall and
  derives return periods (e.g. "what's a 1-in-100-year rainfall event").
  Classic actuarial/hydrological extreme value analysis.

- **[`vae-scenarios/`](./vae-scenarios)** — Trains a small
  Variational Autoencoder (PyTorch) on 30-day rainfall sequences and uses
  it to generate new synthetic ones — a minimal example of using generative
  AI to extend a set of rainfall/flood scenarios.

Each folder has its own README with full method, results, and known
limitations.

## Data

`data/result.txt` — KNMI daily precipitation record, station 550 (De Bilt),
1901–2025. Public data, free to use with attribution to KNMI (see the
file's own header). Source: https://daggegevens.knmi.nl

## Running it

```bash
pip install -r requirements.txt
python gev-rainfall/gev_analysis.py
python vae-scenarios/vae_scenarios.py
```
