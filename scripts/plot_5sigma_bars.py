"""
5σ flux-density upper limits: horizontal grouped bar chart, all targets, all channels.

Output: plots/5sigma_bars.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "results/all_sigma_ensemble.coldat"

CHANNEL_COLORS = {
    "ch1 (3.6 µm)": "#4e9af1",
    "ch2 (4.5 µm)": "#f4a261",
    "ch3 (5.8 µm)": "#2dc56e",
    "ch4 (8.0 µm)": "#e45c5c",
}
CH_KEYS = list(CHANNEL_COLORS.keys())

df = pd.read_csv(DATA, sep=r"\s+", comment="#",
                 names=["name", "ch1", "ch2", "ch3", "ch4"])

# Sort by ch2 limit, NaN last
df_sorted = df.sort_values("ch2", na_position="last").reset_index(drop=True)

n = len(df_sorted)
bar_h = 0.18
group_gap = 0.08
group_h = 4 * bar_h + group_gap

fig_h = max(7, n * group_h + 1.2)
fig, ax = plt.subplots(figsize=(10, fig_h))

y_centers = np.arange(n) * group_h
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_h

for j, (label, color) in enumerate(CHANNEL_COLORS.items()):
    col = ["ch1", "ch2", "ch3", "ch4"][j]
    vals = df_sorted[col].values
    ys = y_centers + offsets[j]
    for i, (v, y) in enumerate(zip(vals, ys)):
        if np.isfinite(v):
            ax.barh(y, v, height=bar_h * 0.88, color=color,
                    alpha=0.88, label=label if i == 0 else "")
        else:
            # No data marker
            ax.text(50, y, "—", va="center", ha="left",
                    color="gray", fontsize=7, alpha=0.6)

ax.set_yticks(y_centers)
ax.set_yticklabels(df_sorted["name"], fontsize=8)
ax.set_xscale("log")
ax.set_xlabel(r"$F_{5\sigma}\ [\mu\mathrm{Jy}]$", fontsize=12)
ax.set_title("Spitzer IRAC  5σ Flux-Density Upper Limits — All Magnetars",
             fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{x:g}" if x < 1000 else f"{x/1000:.0f}k"
))
ax.grid(axis="x", which="both", ls="--", alpha=0.35, lw=0.7)
ax.set_axisbelow(True)

handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.88)
           for c in CHANNEL_COLORS.values()]
ax.legend(handles, CH_KEYS, title="IRAC channel", loc="lower right",
          fontsize=9, title_fontsize=9, framealpha=0.85)

plt.tight_layout()
out = ROOT / "plots" / "5sigma_bars.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)
