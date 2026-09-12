# VAE Rainfall Scenario Generator (PyTorch)

Demonstrates using a generative model
(a Variational Autoencoder) to synthesize new rainfall event sequences
from a real historical record.

## Data
Same KNMI De Bilt daily rainfall record (`RD`, station 550, 1901–2025)
used GEV analysis. Rather than annual maxima, this uses
overlapping 30-day windows (stride 5 days) that contain some rainfall,
giving ~8,500 training sequences.

## Method
- **Preprocessing**: log1p-transform (rainfall is heavily right-skewed),
  then scale to [0,1].
- **Model**: a small fully-connected VAE — encoder (30 → 128 → 128 → 12
  latent dims), decoder mirrors it back out to a 30-day sequence.
- **Training**: 400 epochs, Adam optimizer, KL-term annealed in slowly
  (β ramped from 0 to 0.02 over the first 100 epochs) — a plain β=0.5 run
  collapsed to the decoder ignoring the latent code entirely and just
  outputting a near-constant sequence (documented in the script history:
  first attempt gave synthetic std of 0.7mm vs a real std of 35.6mm).
  Annealing the KL weight fixed this.
- **Generation**: sample z ~ N(0, I), decode, invert the log1p/scaling.

## Result
Synthetic 30-day totals: mean 37.7mm, std 14.4mm, max 56.5mm, vs real
windows: mean 71.5mm, std 35.6mm, max 293.4mm. The model captures the
general shape (rain in short bursts, dry stretches between) but
undershoots the extreme tail — expected from a plain VAE with MSE
reconstruction loss, which naturally smooths toward the mean rather than
reproducing rare extreme events. See `vae_scenarios.png` for a side-by-side
of six real vs six generated sequences.

## Honest limitations
- This is a proof of concept, not a validated scenario generator.
- Single station, single variable (rainfall only — no flood extent,
  routing, or dependency structure between locations).

## Files
- `vae_scenarios.py` — full training + generation script
- `vae_scenarios.png` — real vs synthetic sequence comparison
- `vae_rainfall_model.pt` — trained model weights
