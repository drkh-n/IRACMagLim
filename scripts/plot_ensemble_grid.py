"""
Ensemble PRF injection grid: 3×3 subplots, one per injection position.

Each panel shows a small cutout centred on that injection position with the
aperture and sky annulus overlaid. The panel whose position is closest to the
catalogued X-ray position is highlighted with a red border.

Output: plots/ensemble_grid_overlay.png
"""
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.constants import MOSAIC_PIXEL_SCALE_ARCSEC, PSIZE_ASEC

CHANNEL = 1
TARGET  = "4u0142+61"
RA, DEC = 26.593363, 61.750885
MOSAIC  = ROOT / "simtar_partial/4u0142+61/mosaici1/Combine/mosaic.fits"
CSV     = ROOT / "results/photometry/ensemble_data.csv"

RAP_CAM  = 2.0
RINN_CAM = 2.2
ROUT_CAM = 3.0
# Cutout half-size around each injection position (in mosaic pixels)
HALF_PIX = 14


def cam_to_mosaic(r, ch):
    return r * PSIZE_ASEC[ch - 1] / MOSAIC_PIXEL_SCALE_ARCSEC


# Load full mosaic once
with fits.open(MOSAIC) as hdul:
    image = hdul[0].data.astype(float)
    hdr   = hdul[0].header

wcs = WCS(hdr)
xc, yc = wcs.all_world2pix([[RA, DEC]], 0)[0]
xc, yc = float(xc), float(yc)

# Unique injection positions, sorted by (x, y) so the grid is spatially ordered
df = pd.read_csv(CSV)
pos = (df[(df["name"] == TARGET) & (df["channel"] == CHANNEL)]
       [["x_pos", "y_pos"]].drop_duplicates()
       .sort_values(["x_pos", "y_pos"]).values)   # 9 rows

# Recovered fluxes per position (averaged over the two scale levels)
flux_per_pos = (
    df[(df["name"] == TARGET) & (df["channel"] == CHANNEL)]
    .groupby(["x_pos", "y_pos"])["flux"].mean()
    .reset_index()
    .sort_values(["x_pos", "y_pos"])["flux"].values
)

ap_r   = cam_to_mosaic(RAP_CAM,  CHANNEL)
rin_r  = cam_to_mosaic(RINN_CAM, CHANNEL)
rout_r = cam_to_mosaic(ROUT_CAM, CHANNEL)

h, w = image.shape

# Global Z-scale from the region around the target
bx = int(BOX := 60)
gx0 = max(int(xc) - bx, 0);  gx1 = min(int(xc) + bx, w)
gy0 = max(int(yc) - bx, 0);  gy1 = min(int(yc) + bx, h)
gsub = image[gy0:gy1, gx0:gx1]
interval = ZScaleInterval(contrast=0.25)
vmin, vmax = interval.get_limits(gsub[np.isfinite(gsub)])

# Which position is closest to the true X-ray position?
dists = np.hypot(pos[:, 0] - xc, pos[:, 1] - yc)
target_idx = int(np.argmin(dists))

# ── 3×3 figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(9, 9),
                         gridspec_kw={"hspace": 0.35, "wspace": 0.15})

for k, (ax, (px, py)) in enumerate(zip(axes.flat, pos)):
    cx, cy = int(px), int(py)
    x0 = max(cx - HALF_PIX, 0);  x1 = min(cx + HALF_PIX, w)
    y0 = max(cy - HALF_PIX, 0);  y1 = min(cy + HALF_PIX, h)
    sub = image[y0:y1, x0:x1].copy()
    if np.any(np.isnan(sub)):
        sub[np.isnan(sub)] = np.nanmedian(sub[np.isfinite(sub)])

    # Local centre of the injection position
    lx = px - x0
    ly = py - y0

    ax.imshow(sub, origin="lower", cmap="inferno",
              interpolation="nearest", vmin=vmin, vmax=vmax)

    # Aperture + annulus
    for r, color, ls, lw in [
        (ap_r,   "cyan",   "-",  1.6),
        (rin_r,  "yellow", "--", 1.1),
        (rout_r, "yellow", "--", 1.1),
    ]:
        ax.add_patch(mpatches.Circle(
            (lx, ly), r,
            edgecolor=color, facecolor="none", lw=lw, linestyle=ls,
        ))
    ax.plot(lx, ly, "+", color="cyan", ms=8, markeredgewidth=1.5)

    # Highlight the panel nearest the X-ray position with a red spine
    if k == target_idx:
        for spine in ax.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(2.5)
        ax.plot(lx, ly, "x", color="red", ms=10, markeredgewidth=2.2, zorder=5)

    # Mean recovered flux label
    ax.set_title(f"pos {k+1}  ⟨F⟩={flux_per_pos[k]:.1f} µJy",
                 fontsize=7.5, pad=3)
    ax.set_xticks([]);  ax.set_yticks([])

# Shared legend below the grid
legend_elements = [
    mpatches.Patch(edgecolor="cyan",   facecolor="none", lw=1.6,
                   label=f"Aperture  $r_{{\\rm ap}}={RAP_CAM}$ native px"),
    mpatches.Patch(edgecolor="yellow", facecolor="none", lw=1.1, linestyle="--",
                   label=f"Sky annulus  {RINN_CAM}–{ROUT_CAM} native px"),
    plt.Line2D([0], [0], marker="x", color="red", lw=0,
               ms=9, markeredgewidth=2.2, label="X-ray position (nearest panel)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           fontsize=8.5, framealpha=0.88, bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    f"4U 0142+61 — IRAC ch{CHANNEL} (3.6 µm)  ·  Ensemble 3×3 injection grid\n"
    f"Each panel: {2*HALF_PIX}×{2*HALF_PIX} mosaic-pixel cutout centred on one injection position",
    fontsize=11, fontweight="bold", y=0.995,
)

fig.subplots_adjust(bottom=0.10, top=0.93)
out = ROOT / "plots" / "ensemble_grid_overlay.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)
