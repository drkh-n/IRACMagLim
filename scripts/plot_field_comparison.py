"""
Side-by-side Spitzer IRAC 4.5 µm field comparison: easy vs hard detection.

Easy: 4U 0142+61 (detected, SNR=36, F=23.3 µJy)
Hard: SGR 1806-20 (undetected, 5σ limit ~34,300 µJy)

Motivates ensemble photometry: single aperture measurement is unreliable
in a structured, confused background.

Output: plots/field_comparison_ch2.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.constants import MOSAIC_PIXEL_SCALE_ARCSEC, PSIZE_ASEC

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHANNEL = 2  # 4.5 µm

TARGETS = [
    dict(
        name="4U 0142+61",
        tag="4u0142+61",
        ra=26.593363,
        dec=61.750885,
        mosaic=ROOT / "simtar_partial/4u0142+61/mosaici2/Combine/mosaic.fits",
        flux_label=r"Detected: $F_{4.5} = 23.3\,\mu$Jy, SNR = 36",
        limit_label=r"$F_{5\sigma} = 33.7\,\mu$Jy",
        panel_label="(a) Easy field",
    ),
    dict(
        name="SGR 1806−20",
        tag="sg1806-20",
        ra=272.16391,
        dec=-20.411070,
        mosaic=ROOT / "simtar_partial/sg1806-20/mosaici2/Combine/mosaic.fits",
        flux_label=r"Undetected",
        limit_label=r"$F_{5\sigma} = 34{,}346\,\mu$Jy",
        panel_label="(b) Hard field",
    ),
]

# Aperture and annulus radii in native camera pixels
RAP_CAM = 2.0
RINN_CAM = 2.2
ROUT_CAM = 3.0

BOX_ARCSEC = 60.0   # cutout half-size in arcsec

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def cam_to_mosaic(r_cam, channel):
    """Convert radius in native IRAC pixels to mosaic pixels."""
    return r_cam * PSIZE_ASEC[channel - 1] / MOSAIC_PIXEL_SCALE_ARCSEC


def radec_to_pixel(hdr, ra, dec):
    wcs = WCS(hdr)
    px, py = wcs.all_world2pix([[ra, dec]], 0)[0]
    return float(px), float(py)


def load_cutout(mosaic_path, xc, yc, half_pix):
    with fits.open(mosaic_path) as hdul:
        image = hdul[0].data.astype(float)
        hdr = hdul[0].header

    h, w = image.shape
    x0 = max(int(xc) - half_pix, 0)
    x1 = min(int(xc) + half_pix, w)
    y0 = max(int(yc) - half_pix, 0)
    y1 = min(int(yc) + half_pix, h)

    sub = image[y0:y1, x0:x1].copy()
    # Replace NaNs with local median for display
    if np.any(np.isnan(sub)):
        sub[np.isnan(sub)] = np.nanmedian(sub)

    sub_xc = xc - x0
    sub_yc = yc - y0
    return sub, sub_xc, sub_yc, hdr


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
fig.subplots_adjust(left=0.04, right=0.96, bottom=0.18, top=0.88, wspace=0.12)

half_pix = int(BOX_ARCSEC / MOSAIC_PIXEL_SCALE_ARCSEC) // 2

ap_mpix   = cam_to_mosaic(RAP_CAM,  CHANNEL)
rinn_mpix = cam_to_mosaic(RINN_CAM, CHANNEL)
rout_mpix = cam_to_mosaic(ROUT_CAM, CHANNEL)

for ax, tgt in zip(axes, TARGETS):
    with fits.open(tgt["mosaic"]) as hdul:
        hdr = hdul[0].header

    xc, yc = radec_to_pixel(hdr, tgt["ra"], tgt["dec"])
    sub, sub_xc, sub_yc, _ = load_cutout(tgt["mosaic"], xc, yc, half_pix)

    # Z-scale stretch
    interval = ZScaleInterval(contrast=0.25)
    vmin, vmax = interval.get_limits(sub)

    im = ax.imshow(sub, origin="lower", cmap="gray",
                   interpolation="nearest", vmin=vmin, vmax=vmax)

    # Aperture
    ap_circle = mpatches.Circle(
        (sub_xc, sub_yc), ap_mpix,
        edgecolor="red", facecolor="none", lw=1.8, linestyle="-",
    )
    # Inner annulus
    ann_in = mpatches.Circle(
        (sub_xc, sub_yc), rinn_mpix,
        edgecolor="yellow", facecolor="none", lw=1.5, linestyle="--",
    )
    # Outer annulus
    ann_out = mpatches.Circle(
        (sub_xc, sub_yc), rout_mpix,
        edgecolor="yellow", facecolor="none", lw=1.5, linestyle="--",
    )
    for patch in (ap_circle, ann_in, ann_out):
        ax.add_patch(patch)

    ax.plot(sub_xc, sub_yc, "+", color="cyan",
            markersize=12, markeredgewidth=1.8)

    # Pixel scale bar: 10 arcsec
    bar_pix = 10.0 / MOSAIC_PIXEL_SCALE_ARCSEC
    bar_x0 = 0.07 * sub.shape[1]
    bar_y  = 0.08 * sub.shape[0]
    ax.plot([bar_x0, bar_x0 + bar_pix], [bar_y, bar_y],
            color="white", lw=2.5)
    ax.text(bar_x0 + bar_pix / 2, bar_y + 1.8, r'$10^{\prime\prime}$',
            color="white", ha="center", va="bottom", fontsize=10)

    # Colorbar
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("MJy sr$^{-1}$", fontsize=9)

    # Panel label (top-left)
    ax.text(0.03, 0.97, tgt["panel_label"],
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            color="white", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45, ec="none"))

    # Source info (bottom)
    ax.set_title(
        f"{tgt['name']}  ·  IRAC ch{CHANNEL} (4.5 µm)\n"
        f"{tgt['flux_label']}   {tgt['limit_label']}",
        fontsize=10, pad=6,
    )
    ax.set_xlabel("Mosaic pixel offset", fontsize=9)
    ax.set_ylabel("Mosaic pixel offset" if ax is axes[0] else "", fontsize=9)

# Aperture legend (shared)
legend_elements = [
    mpatches.Patch(edgecolor="red",    facecolor="none", lw=1.8, label=f"Aperture ($r={RAP_CAM}$ native px)"),
    mpatches.Patch(edgecolor="yellow", facecolor="none", lw=1.5, linestyle="--",
                   label=f"Sky annulus ({RINN_CAM}–{ROUT_CAM} native px)"),
    plt.Line2D([0], [0], marker="+", color="cyan", lw=0,
               markersize=10, markeredgewidth=1.8, label="X-ray position"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           fontsize=9, framealpha=0.85, bbox_to_anchor=(0.5, 0.02))

fig.suptitle(
    "Spitzer IRAC 4.5 µm: easy vs. hard detection environments",
    fontsize=13, fontweight="bold", y=0.98,
)

out = ROOT / "plots" / "field_comparison_ch2.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)
