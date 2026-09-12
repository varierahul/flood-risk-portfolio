import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ---- 1. Load data ----
DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'result.txt'
cols = ['STN', 'YYYYMMDD', 'RD']
df = pd.read_csv(
    DATA_PATH,
    comment='#',
    names=cols,
    skipinitialspace=True
)

# Drop any stray header/blank rows that survived
df = df.dropna()
df['YYYYMMDD'] = df['YYYYMMDD'].astype(int)
df['RD'] = df['RD'].astype(float)

# Check for any negative/placeholder codes
print("Unique negative values (if any):", df.loc[df['RD'] < 0, 'RD'].unique())
print("Min RD:", df['RD'].min(), "Max RD:", df['RD'].max())

# Convert tenths of mm -> mm
df['RD_mm'] = df['RD'].clip(lower=0) / 10.0

# Parse date, extract year
df['date'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
df['year'] = df['date'].dt.year

print("\nRecord spans:", df['date'].min().date(), "to", df['date'].max().date())
print("Number of years:", df['year'].nunique())
print("Total days:", len(df))

# ---- 2. Annual maxima (1-day) ----
annual_max = df.groupby('year')['RD_mm'].max()

# Drop incomplete first/last years if they have very few days
day_counts = df.groupby('year').size()
print("\nYears with <300 days of data (partial years):")
print(day_counts[day_counts < 300])

# Use only reasonably complete years for the fit (>=300 days)
complete_years = day_counts[day_counts >= 300].index
am = annual_max.loc[annual_max.index.isin(complete_years)]

print(f"\nUsing {len(am)} annual maxima (years {am.index.min()}-{am.index.max()})")
print("\nTop 10 annual maxima (mm):")
print(am.sort_values(ascending=False).head(10))

# ---- 3. Also compute 3-day cumulative annual maxima ----
df_sorted = df.sort_values('date').set_index('date')
df_sorted['RD_3day'] = df_sorted['RD_mm'].rolling(3, min_periods=3).sum()
annual_max_3day = df_sorted.groupby(df_sorted.index.year)['RD_3day'].max()
am3 = annual_max_3day.loc[annual_max_3day.index.isin(complete_years)]

# ---- 4. Fit GEV distribution (1-day) ----
# scipy's genextreme uses a sign convention where c = -shape (xi)
c, loc, scale = stats.genextreme.fit(am.values)
xi = -c  # convert to standard GEV shape parameter (xi>0 heavy tail)
print(f"\n--- GEV fit (1-day annual max) ---")
print(f"shape (xi) = {xi:.4f}, location = {loc:.3f}, scale = {scale:.3f}")

# ---- 5. Return levels ----
return_periods = [2, 5, 10, 25, 50, 100, 200]
return_levels = {}
for T in return_periods:
    p = 1 - 1/T
    level = stats.genextreme.ppf(p, c, loc=loc, scale=scale)
    return_levels[T] = level

print("\nReturn period (yr) -> Return level (mm/day)")
for T, lvl in return_levels.items():
    print(f"{T:>4} -> {lvl:6.1f} mm")

# ---- 6. Goodness of fit check (KS test) ----
ks_stat, ks_p = stats.kstest(am.values, 'genextreme', args=(c, loc, scale))
print(f"\nKS test statistic={ks_stat:.4f}, p-value={ks_p:.4f}")

# ---- 7. Plots ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Histogram + fitted density
ax = axes[0]
x = np.linspace(am.min()*0.8, am.max()*1.3, 300)
ax.hist(am.values, bins=15, density=True, alpha=0.6, color='steelblue', edgecolor='white', label='Annual maxima')
ax.plot(x, stats.genextreme.pdf(x, c, loc=loc, scale=scale), 'r-', lw=2, label='Fitted GEV')
ax.set_xlabel('Annual max 1-day rainfall (mm)')
ax.set_ylabel('Density')
ax.set_title('De Bilt: Annual Maximum Daily Rainfall\nGEV Fit')
ax.legend()

# Return level plot
ax2 = axes[1]
T_plot = np.logspace(np.log10(1.1), np.log10(500), 200)
p_plot = 1 - 1/T_plot
levels_plot = stats.genextreme.ppf(p_plot, c, loc=loc, scale=scale)
ax2.plot(T_plot, levels_plot, 'b-', lw=2, label='GEV return level')

# Empirical points (Weibull plotting position)
am_sorted = np.sort(am.values)
n = len(am_sorted)
ranks = np.arange(1, n+1)
T_emp = (n + 1) / (n + 1 - ranks)
ax2.scatter(T_emp, am_sorted, color='black', s=20, zorder=5, label='Empirical')

ax2.set_xscale('log')
ax2.set_xlabel('Return period (years)')
ax2.set_ylabel('Rainfall (mm/day)')
ax2.set_title('Return Level Plot')
ax2.legend()
ax2.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / 'gev_analysis.png', dpi=150)
print("\nPlot saved.")

# ---- 8. Save annual maxima + summary to CSV for reference ----
am.to_csv(Path(__file__).resolve().parent / 'annual_maxima_1day.csv', header=['annual_max_mm'])
