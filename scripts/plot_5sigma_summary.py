"""
Figure 2.6 — 5σ detection limit summary.

Bar chart of F_5σ per magnetar per IRAC channel, with published detections
(4U 0142+61 and 1E 2259+586 at 4.5 µm) overlaid as horizontal lines or
annotated points for visual validation.

Usage:
    cd /mnt/c/Users/d.nurzhakyp/Desktop/IRACMagLim
    python scripts/plot_5sigma_summary.py \
        --infile ./results/all_sigma_ensemble.coldat \
        --outfile ./results/figures/fig_5sigma_summary.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Published Spitzer IRAC flux measurements for validation sources (µJy)
# Source: Wang et al. 2006 (4U 0142+61); Kaplan & Chakrabarty 2009 (1E 2259+586)
LITERATURE_FLUXES = {
    '4U0142': {2: 32.0},    # ch2 = 4.5 µm
    '1E2259': {2: 6.3},     # ch2 = 4.5 µm
}

CHANNEL_WAVELENGTHS = ['3.6 µm', '4.5 µm', '5.8 µm', '8.0 µm']
CHANNEL_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def main():
    parser = argparse.ArgumentParser(description='Plot 5σ limit summary bar chart.')
    parser.add_argument('--infile', default='./results/all_sigma_ensemble.coldat',
                        help='Tab-separated coldat file produced by analysis.py')
    parser.add_argument('--outfile', default='./results/figures/fig_5sigma_summary.png',
                        help='Output PNG path')
    parser.add_argument('--log', action='store_true', default=True,
                        help='Use log scale on y-axis (default: True)')
    args = parser.parse_args()

    if not os.path.exists(args.infile):
        print(f"ERROR: input file not found: {args.infile}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    # Read coldat: comment lines start with #
    df = pd.read_csv(args.infile, sep='\t', comment='#',
                     names=['name', 'ch1', 'ch2', 'ch3', 'ch4'])
    df = df.dropna(how='all')

    targets = df['name'].tolist()
    n_targets = len(targets)
    n_channels = 4

    x = np.arange(n_targets)
    bar_width = 0.18
    offsets = np.linspace(-(n_channels - 1) / 2, (n_channels - 1) / 2, n_channels) * bar_width

    fig, ax = plt.subplots(figsize=(max(7, 2 * n_targets), 5))

    for ch_idx, (col, wl, color) in enumerate(
            zip(['ch1', 'ch2', 'ch3', 'ch4'], CHANNEL_WAVELENGTHS, CHANNEL_COLORS)):
        vals = pd.to_numeric(df[col], errors='coerce').values
        bars = ax.bar(x + offsets[ch_idx], vals, bar_width,
                      label=f'Ch {ch_idx + 1} ({wl})', color=color, alpha=0.8, edgecolor='k',
                      linewidth=0.5)

        # Annotate bar values
        for xi, v in zip(x + offsets[ch_idx], vals):
            if np.isfinite(v):
                ax.text(xi, v * 1.05, f'{v:.1f}', ha='center', va='bottom',
                        fontsize=7, rotation=60)

    # Overlay literature detections
    lit_label_added = set()
    for t_idx, target_name in enumerate(targets):
        # Try to match target name to literature key
        for lit_key, ch_fluxes in LITERATURE_FLUXES.items():
            if lit_key.replace(' ', '').lower() in target_name.replace(' ', '').lower():
                for ch_num, flux in ch_fluxes.items():
                    ch_idx = ch_num - 1
                    label = f'Published detection ({CHANNEL_WAVELENGTHS[ch_idx]})' \
                        if (lit_key, ch_num) not in lit_label_added else None
                    ax.scatter(t_idx + offsets[ch_idx], flux,
                               marker='*', s=120, color='black', zorder=10,
                               label=label)
                    if label:
                        lit_label_added.add((lit_key, ch_num))

    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Flux limit / detected flux (µJy)', fontsize=10)
    ax.set_title('5σ detection limits per magnetar per IRAC channel\n'
                 '(★ = published Spitzer detections used for validation)', fontsize=11)
    if args.log:
        ax.set_yscale('log')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.outfile, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.outfile}")


if __name__ == '__main__':
    main()
