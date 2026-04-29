"""
5σ ch2 upper limit vs. Galactic latitude |b| for all magnetars.

Crowded, low-latitude fields produce higher confusion noise → weaker limits.
Output: plots/limit_vs_glat.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
SIGMA = ROOT / "results/all_sigma_ensemble.coldat"
COORDS = ROOT / "mag_info/mag_dec.coldat"

df_s = pd.read_csv(SIGMA, sep=r"\s+", comment="#",
                   names=["name", "ch1", "ch2", "ch3", "ch4"])
df_c = pd.read_csv(COORDS, sep=r"\s+", comment="#",
                   names=["name", "ra", "dec"])

df = pd.merge(df_s, df_c, on="name")

# Galactic latitude from RA/Dec
coords = SkyCoord(ra=df["ra"].values * u.deg,
                  dec=df["dec"].values * u.deg, frame="icrs")
df["glat"] = coords.galactic.b.deg
df["absglat"] = np.abs(df["glat"])

# Keep only rows with finite ch2 limit
df2 = df[np.isfinite(df["ch2"])].copy()

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))

sc = ax.scatter(
    df2["absglat"], df2["ch2"],
    c=np.log10(df2["ch2"]),
    cmap="plasma_r", s=60, edgecolors="white", lw=0.6,
    zorder=4,
)

for _, row in df2.iterrows():
    label = row["name"].replace("sg", "SGR ").replace("4u", "4U ") \
                       .replace("sw", "SW ").replace("1e", "1E ") \
                       .replace("ps", "PSR ") .replace("cx", "CX ") \
                       .replace("ax", "AX ").replace("xt", "XTE ")
    ax.text(row["absglat"] + 0.4, row["ch2"], label,
            fontsize=6.5, va="center", alpha=0.85,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

ax.set_yscale("log")
ax.set_xlabel(r"Galactic latitude $|b|$ [deg]", fontsize=12)
ax.set_ylabel(r"$F_{5\sigma}^{\rm ch2}\ [\mu\mathrm{Jy}]$", fontsize=12)
ax.set_title("Confusion noise vs. Galactic Plane Distance\n"
             "Spitzer IRAC ch2 (4.5 µm), all magnetars",
             fontsize=12, fontweight="bold")

cb = fig.colorbar(sc, ax=ax, pad=0.01)
cb.set_label(r"$\log_{10}(F_{5\sigma}\ [\mu\mathrm{Jy}])$", fontsize=10)

ax.grid(which="both", ls="--", alpha=0.3, lw=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
out = ROOT / "plots" / "limit_vs_glat.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)
