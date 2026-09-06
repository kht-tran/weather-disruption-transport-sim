import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

sig = pd.read_csv('data/ciaran_disruption_signature.csv')
val = pd.read_csv('data/layer_e_validation_ciaran.csv')

sig = sig[sig['in_abm_network'] == True]
df = sig.merge(val[['city', 'abm_stranded']], on='city', how='inner')

# Remove Palma: ratio method fails (off-season near-zero reference week)
df2 = df[df['city'] != 'Palma'].copy()
rho, p = stats.spearmanr(df2['ciaran_score'], df2['abm_stranded'])
print(f"Excl. Palma (n={len(df2)}): Spearman rho = {rho:.3f}, p = {p:.4f}")

# Single panel
fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")

ax.scatter(df2['ciaran_score'], df2['abm_stranded'],
           color="#2c7bb6", alpha=0.75, s=45, zorder=3)

# OLS trend line for visual clarity
m, b = np.polyfit(df2['ciaran_score'], df2['abm_stranded'], 1)
x_line = np.linspace(df2['ciaran_score'].min(), df2['ciaran_score'].max(), 100)
ax.plot(x_line, m * x_line + b, color="#d7191c", lw=1.5, zorder=2,
        label=f"Spearman correlation\n$\\rho={rho:.3f}$, $p={p:.3f}$, $n={len(df2)}$")

# Label cities with high stranding or notable OPDI score
for _, row in df2.iterrows():
    if row['abm_stranded'] > 5 or abs(row['ciaran_score']) > 0.15:
        ax.annotate(row['city'],
                    (row['ciaran_score'], row['abm_stranded']),
                    textcoords='offset points', xytext=(4, 3), fontsize=7.5,
                    color="#333333")

ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("OPDI disruption score — Storm Ciaran (Nov 2023)", fontsize=9.5)
ax.set_ylabel("ABM stranded passengers", fontsize=9.5)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")

plt.tight_layout()
plt.savefig('data/ciaran_validation_scatter.png', dpi=200, bbox_inches='tight')
plt.savefig('data/ciaran_validation_scatter.pdf', dpi=300, bbox_inches='tight')
print("Plot saved: data/ciaran_validation_scatter.png / .pdf")
