"""
Figure 2.2 — PRF channel panels.

For each IRAC channel, shows the raw 100× oversampled PRF side-by-side with
the rebinned PRF on the native mosaic pixel grid.

Usage:
    cd /mnt/c/Users/d.nurzhakyp/Desktop/IRACMagLim
    python scripts/plot_prf_channels.py --prf_dir ./prf --outfile ./results/figures/fig_prf_channels.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import LogNorm

# Make sure src/ is on the path when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.psf_utils import make_PSF

CHANNEL_WAVELENGTHS = {1: '3.6 µm', 2: '4.5 µm', 3: '5.8 µm', 4: '8.0 µm'}


def load_prf(prf_dir: str, channel: int) -> np.ndarray:
    """Load the raw APEX PRF FITS file for a given channel."""
    # APEX PRF files are typically named like ch1_prf.fits or similar
    candidates = [
        f'ch{channel}_prf.fits',
        f'IRAC{channel}_col129_row129.fits',
        f'irac_ch{channel}_prf.fits',
        f'PRF_ch{channel}.fits',
    ]
    for name in candidates:
        path = os.path.join(prf_dir, name)
        if os.path.exists(path):
            with fits.open(path) as hdul:
                return hdul[0].data.astype(float)
    # Fallback: first .fits file in prf_dir that contains the channel number
    for fname in sorted(os.listdir(prf_dir)):
        if fname.endswith('.fits') and str(channel) in fname:
            path = os.path.join(prf_dir, fname)
            with fits.open(path) as hdul:
                return hdul[0].data.astype(float)
    raise FileNotFoundError(
        f"Could not find PRF file for channel {channel} in {prf_dir}. "
        f"Tried: {candidates}"
    )


def main():
    parser = argparse.ArgumentParser(description='Plot raw vs rebinned PRF for all IRAC channels.')
    parser.add_argument('--prf_dir', default='./prf', help='Directory containing APEX PRF FITS files')
    parser.add_argument('--outfile', default='./results/figures/fig_prf_channels.png',
                        help='Output PNG path')
    parser.add_argument('--channels', nargs='+', type=int, default=[1, 2, 3, 4],
                        help='IRAC channels to plot (default: 1 2 3 4)')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    n_ch = len(args.channels)
    fig, axes = plt.subplots(n_ch, 2, figsize=(8, 3 * n_ch))
    if n_ch == 1:
        axes = axes[np.newaxis, :]

    for row, ch in enumerate(args.channels):
        try:
            raw_prf = load_prf(args.prf_dir, ch)
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            continue

        # Process PRF: trim → shift (no sub-pixel) → congrid
        # Use center of image as nominal position (no sub-pixel shift)
        rebinned = make_PSF(raw_prf, x=0.0, y=0.0, channel=ch, scale=1.0,
                            trim_pixels=50, verbose=False)

        wl = CHANNEL_WAVELENGTHS.get(ch, f'ch{ch}')

        # ---- Left panel: raw oversampled PRF ----
        ax_raw = axes[row, 0]
        cy_raw, cx_raw = np.array(raw_prf.shape) // 2
        half_raw = 200  # show central 400×400 of the oversampled PRF
        sl = np.s_[max(0, cy_raw - half_raw):cy_raw + half_raw,
                    max(0, cx_raw - half_raw):cx_raw + half_raw]
        crop_raw = raw_prf[sl]
        vmin = np.percentile(crop_raw[crop_raw > 0], 1) if np.any(crop_raw > 0) else 1e-6
        im0 = ax_raw.imshow(crop_raw, origin='lower', cmap='inferno',
                            norm=LogNorm(vmin=vmin, vmax=crop_raw.max()))
        ax_raw.set_title(f'Ch {ch} ({wl}) — raw PRF (100× oversamp.)', fontsize=9)
        ax_raw.set_xlabel('Oversampled pixel')
        ax_raw.set_ylabel('Oversampled pixel')
        plt.colorbar(im0, ax=ax_raw, label='Normalized flux')

        # ---- Right panel: rebinned PRF on mosaic grid ----
        ax_reb = axes[row, 1]
        if rebinned is not None:
            vmin2 = np.percentile(rebinned[rebinned > 0], 1) if np.any(rebinned > 0) else 1e-6
            im1 = ax_reb.imshow(rebinned, origin='lower', cmap='inferno',
                                norm=LogNorm(vmin=vmin2, vmax=rebinned.max()))
            ax_reb.set_title(f'Ch {ch} ({wl}) — rebinned (0.6″/pix grid)', fontsize=9)
            ax_reb.set_xlabel('Mosaic pixel')
            ax_reb.set_ylabel('Mosaic pixel')
            plt.colorbar(im1, ax=ax_reb, label='Normalized flux')
        else:
            ax_reb.text(0.5, 0.5, 'Processing failed', ha='center', va='center',
                        transform=ax_reb.transAxes)

    fig.suptitle('IRAC PRF: 100× oversampled (left) vs. rebinned to mosaic pixel grid (right)',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(args.outfile, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.outfile}")


if __name__ == '__main__':
    main()
