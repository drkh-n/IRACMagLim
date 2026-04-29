"""
Figure 2.5 — Ensemble photometry linearity check.

For each IRAC channel, plots recovered flux vs. injected flux for all ensemble
grid positions, with the ensemble standard deviation (σ_ens) shown as error bars
and a 1:1 reference line. A flat σ_ens across injection levels confirms the
background-dominated regime required for the 5σ limit derivation.

Usage:
    cd /mnt/c/Users/d.nurzhakyp/Desktop/IRACMagLim
    python scripts/plot_ensemble_linearity.py \
        --infile ./results/photometry/all_phot.coldat \
        --outfile ./results/figures/fig_ensemble_linearity.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.constants import APCOR, FACTORS, MAGIC_NUMBER

CHANNEL_WAVELENGTHS = {1: '3.6 µm', 2: '4.5 µm', 3: '5.8 µm', 4: '8.0 µm'}


def sigma_by_window(values: np.ndarray, window_size: int) -> np.ndarray:
    """Return sample std for each non-overlapping window of window_size."""
    stds = []
    for i in range(0, len(values), window_size):
        stds.append(np.std(values[i:i + window_size], ddof=1))
    return np.array(stds)


def mean_by_window(values: np.ndarray, window_size: int) -> np.ndarray:
    means = []
    for i in range(0, len(values), window_size):
        means.append(np.mean(values[i:i + window_size]))
    return np.array(means)


def main():
    parser = argparse.ArgumentParser(
        description='Plot ensemble photometry linearity (recovered vs injected flux).')
    parser.add_argument('--infile', default='./results/photometry/all_phot.coldat',
                        help='Ensemble photometry CSV/coldat produced by main.py')
    parser.add_argument('--outfile', default='./results/figures/fig_ensemble_linearity.png',
                        help='Output PNG path')
    parser.add_argument('--grid', type=int, default=3,
                        help='Grid size used during ensemble run (default 3 for 3×3)')
    parser.add_argument('--magnetar', default=None,
                        help='Plot only this magnetar name (default: first found)')
    args = parser.parse_args()

    if not os.path.exists(args.infile):
        print(f"ERROR: input file not found: {args.infile}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    df = pd.read_csv(args.infile, comment='#')

    # Select magnetar
    names = df['name'].unique()
    if args.magnetar:
        if args.magnetar not in names:
            print(f"ERROR: magnetar '{args.magnetar}' not found. Available: {list(names)}")
            sys.exit(1)
        target_name = args.magnetar
    else:
        target_name = names[0]
        print(f"Plotting: {target_name} (use --magnetar to select another)")

    sub = df[df['name'] == target_name]

    n_scales = len(FACTORS)
    window_size = args.grid ** 2

    channels = sorted(sub['channel'].unique())
    fig, axes = plt.subplots(1, len(channels), figsize=(4.5 * len(channels), 4.5), sharey=False)
    if len(channels) == 1:
        axes = [axes]

    for ax, ch in zip(axes, channels):
        ch_data = sub[sub['channel'] == ch]

        # Determine flux column and unit conversion
        if 'flux' in ch_data.columns:
            flux_col = 'flux'
            is_uJy = 'unit' in ch_data.columns and 'uJy' in str(ch_data['unit'].iloc[0])
            conv = 1.0 if is_uJy else MAGIC_NUMBER * APCOR[ch - 1]
        else:
            flux_col = 'phot_flux_(µJy)' if 'phot_flux_(µJy)' in ch_data.columns else ch_data.columns[-1]
            conv = 1.0

        phot_vals = np.array(ch_data[flux_col]) * conv  # µJy

        if len(phot_vals) != n_scales * window_size:
            ax.text(0.5, 0.5,
                    f'Expected {n_scales * window_size} rows,\ngot {len(phot_vals)}',
                    ha='center', va='center', transform=ax.transAxes, color='red')
            ax.set_title(f'Ch {ch} ({CHANNEL_WAVELENGTHS.get(ch, "")})')
            continue

        # Injected flux levels
        if 'expected_flux' in ch_data.columns:
            exp_vals = np.array(ch_data['expected_flux']) * conv
            injected = mean_by_window(exp_vals, window_size)
        else:
            # Fall back to FACTORS × median σ estimate (approximate)
            rough_sigma = np.std(phot_vals[:window_size], ddof=1)
            injected = FACTORS * rough_sigma * 5.0 / 5.0  # rough scale

        recovered_mean = mean_by_window(phot_vals, window_size)
        recovered_std  = sigma_by_window(phot_vals, window_size)

        # Scatter plot: individual points
        colors = plt.cm.tab10(np.linspace(0, 0.4, n_scales))
        for i in range(n_scales):
            w0 = i * window_size
            w1 = w0 + window_size
            pts = phot_vals[w0:w1]
            inj_i = injected[i] if i < len(injected) else np.nan
            ax.scatter([inj_i] * len(pts), pts, color=colors[i], alpha=0.6, s=30,
                       label=f'FACTOR={FACTORS[i]}×' if i == 0 else f'FACTOR={FACTORS[i]}×')
            ax.errorbar(inj_i, recovered_mean[i], yerr=recovered_std[i],
                        fmt='D', color=colors[i], markersize=7, capsize=4, zorder=5)

        # 1:1 reference line
        xlim_min = min(injected) * 0.7
        xlim_max = max(injected) * 1.3
        ref = np.linspace(xlim_min, xlim_max, 100)
        ax.plot(ref, ref, 'k--', lw=1, label='1:1 (perfect recovery)')

        wl = CHANNEL_WAVELENGTHS.get(ch, f'ch{ch}')
        ax.set_title(f'Ch {ch} ({wl})\n{target_name}', fontsize=10)
        ax.set_xlabel('Injected flux (µJy)')
        ax.set_ylabel('Recovered flux (µJy)')
        ax.legend(fontsize=8)

        # Annotate σ_ens
        for i, (inj_i, std_i) in enumerate(zip(injected, recovered_std)):
            ax.annotate(f'σ={std_i:.2f} µJy',
                        xy=(inj_i, recovered_mean[i]),
                        xytext=(5, 10), textcoords='offset points', fontsize=8, color=colors[i])

    fig.suptitle(f'Ensemble photometry: recovered vs. injected flux — {target_name}',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(args.outfile, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.outfile}")


if __name__ == '__main__':
    main()
