"""
Spectral Energy Distribution of 4U 0142+61 in the infrared.

Our Spitzer IRAC measurements (ch1, ch2, ch4) + literature points from
Wang et al. (2006) + Hare et al. (2024) JWST, overlaid with power-law
model slopes (α = −0.96 magnetospheric, α = +4/3 passive disk).

Output: plots/sed_4u0142.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------------
# Data points
# ---------------------------------------------------------------------------
# IRAC central wavelengths in µm
IRAC_LAMBDA = {1: 3.6, 2: 4.5, 3: 5.8, 4: 8.0}

# Our 5σ upper limits (µJy), from all_sigma_ensemble.coldat for 4u0142+61
our_limits = {
    1: 25.712,
    2: 33.687,
    4: 43.705,
}

# Wang et al. (2006) Ks + IRAC detections (µJy)
# Ks ~2.17 µm, F=11.0±1.9 µJy (Kaplan et al. 2009 compilation)
# IRAC ch1 23.3 µJy, ch2 23.7 µJy (Wang+2006)
wang_lam  = np.array([2.17, 3.6, 4.5])
wang_flux = np.array([11.0, 23.3, 23.7])
wang_err  = np.array([1.9,  1.2,  1.4])

# Hare et al. (2024) JWST NIRCam F360M + F460M (µJy), α_IR = −0.96
hare_lam  = np.array([3.6, 4.6])
hare_flux = np.array([22.5, 18.1])
hare_err  = np.array([1.5,  1.3])

# ---------------------------------------------------------------------------
# Model power-laws  F_ν ∝ ν^α  (plotted relative to Wang ch1 anchor)
# ---------------------------------------------------------------------------
lam_model = np.logspace(np.log10(1.5), np.log10(10.0), 200)
nu_model  = 3e14 / lam_model          # Hz (λ in µm)

nu_anchor = 3e14 / 3.6
F_anchor  = 23.3                      # µJy at 3.6 µm

def powerlaw(alpha):
    return F_anchor * (nu_model / nu_anchor) ** alpha

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

# Model slopes
ax.plot(lam_model, powerlaw(-0.96), color="#9b59b6", lw=1.8, ls="--",
        label=r"Magnetospheric: $\alpha=-0.96$ (Hare+2024)")
ax.plot(lam_model, powerlaw(4/3), color="#e67e22", lw=1.8, ls="-.",
        label=r"Passive disk: $\alpha=+4/3$")

# Our upper limits (downward triangles)
for ch, flim in our_limits.items():
    lam = IRAC_LAMBDA[ch]
    ax.annotate("", xy=(lam, flim * 0.65), xytext=(lam, flim),
                arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=1.8))
    ax.plot(lam, flim, "v", color="#e74c3c", ms=10, zorder=4,
            label=r"This work: $5\sigma$ limit" if ch == 1 else "")

# Wang+2006
ax.errorbar(wang_lam, wang_flux, yerr=wang_err,
            fmt="s", color="#2980b9", ms=7, capsize=4, elinewidth=1.4,
            label="Wang et al. (2006)")

# Hare+2024 JWST
ax.errorbar(hare_lam, hare_flux, yerr=hare_err,
            fmt="D", color="#27ae60", ms=7, capsize=4, elinewidth=1.4,
            label="Hare et al. (2024) JWST")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Wavelength $\lambda$ [$\mu$m]", fontsize=12)
ax.set_ylabel(r"Flux density $F_\nu$ [$\mu$Jy]", fontsize=12)
ax.set_title("Infrared SED of 4U 0142+61", fontsize=13, fontweight="bold")
ax.set_xlim(1.5, 10.0)
ax.set_ylim(4.0, 80.0)

ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{x:g}"
))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{x:g}"
))
ax.set_xticks([2, 3, 3.6, 4.5, 5.8, 8.0])

ax.legend(fontsize=9, framealpha=0.88, loc="upper right")
ax.grid(which="both", ls="--", alpha=0.3, lw=0.7)

plt.tight_layout()
out = ROOT / "plots" / "sed_4u0142.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)
