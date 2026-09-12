# De Bilt Extreme Rainfall — GEV Analysis

Demonstrates fitting a Generalized Extreme Value (GEV)
distribution to annual maximum daily rainfall and deriving return periods.
## Data
- Source: KNMI daily precipitation, station 550 (De Bilt), `RD` variable
  (24-hour sum, 08:00–08:00 UTC), tenths of mm.
- Record: 1901–2025 (125 complete years; the partial 2026 year is excluded
  from the fit).

## Method
1. Extract the annual maximum 1-day rainfall for each complete year (block
   maxima approach).
2. Fit a GEV distribution (`scipy.stats.genextreme`) to the 125 annual
   maxima.
3. Derive return levels (rainfall amount associated with a given return
   period) by inverting the fitted CDF.
4. Check fit quality with a KS test and a return-level plot against the
   empirical (Weibull plotting-position) values.

## Results

Fitted GEV parameters: shape (ξ) ≈ 0.031, location ≈ 29.9 mm, scale ≈ 7.5 mm.
A positive but small shape parameter indicates a slightly heavy-tailed
distribution — consistent with the occasional very wet day (e.g. 1952,
2024) sitting above what a purely exponential tail would predict.

| Return period (yr) | Return level (mm/day) |
|---|---|
| 2   | 32.7 |
| 5   | 41.5 |
| 10  | 47.5 |
| 25  | 55.2 |
| 50  | 61.2 |
| 100 | 67.2 |
| 200 | 73.3 |

KS test: statistic = 0.035, p-value ≈ 1.0 — no evidence against the GEV fit.

## Limitations
- Single-station, single-day-duration analysis; 
- Stationary fit: the parameters are estimated over the full 1901–2025
  record and do not account for a possible trend from climate change. With
  125 years of data the tail is reasonably well constrained, but the extreme
  return periods (100+ years) still carry real estimation uncertainty from
  having a finite sample. 
## Files
- `gev_analysis.py` — full analysis script
- `gev_analysis.png` — histogram/fit and return-level plots
- `annual_maxima_1day.csv` — the 125 annual maxima used in the fit
