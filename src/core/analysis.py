# ; Copyright (c) 2000-2050, Banzai Astrophysics.  All rights reserved.
# ;	Unauthorized reproduction prohibited without touting Darkhan's
# ;	name. Please help me live forever by continuing the tradition
# ;	of honoring science nerds of the past by putting their name in
# ;	your code that uses theirs.
# ;
# ;+
# ; NAME:
# ; FLUX_SNR5
# ;
# ; PURPOSE:
# ; Calculates the 5-sigma sensitivity limits for IRAC data from ensemble
# ; aperture photometry results.
# ;
# ; METHOD:
# ; For each target and IRAC channel, the ensemble photometry data contains
# ; N_scales × window_size measurements — one group per PRF injection flux level
# ; (see FACTORS in constants.py). Within each group, all positions have the same
# ; injected flux, so std(recovered_flux, ddof=1) across those positions measures
# ; the real photometric noise (background-dominated). The 5σ upper limit is:
# ;
# ;   5σ_limit = 5 × mean(σ_ensemble per group)    [µJy]
# ;
# ; This is valid for undetected magnetars (background-dominated regime).
# ;
# ; INPUTS:
# ; infile: CSV with ensemble photometry results (output of EnsemblePhotometry.run()).
# ; outfile: Tab-separated output file with 5σ limits per channel.
# ;
# ; MODIFICATION HISTORY:
# ;   Written by: Darkhan Nurzhakyp, 2025 September 30
# ;-

import argparse
import logging

import numpy as np
import pandas as pd

from utils.constants import APCOR, FACTORS, MAGIC_NUMBER


# -----------------------------------------------------------------
#               Mini-Routines
# -----------------------------------------------------------------

def _std_by_window(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute sample std (ddof=1) for each non-overlapping window of `window_size`.

    Parameters
    ----------
    data : array of length n_windows * window_size
    window_size : number of elements per window

    Returns
    -------
    stds : array of length n_windows
    """
    stds = []
    for i in range(0, len(data), window_size):
        window = data[i: i + window_size]
        stds.append(np.std(window, ddof=1))
    return np.array(stds)


# -----------------------------------------------------------------
#                             Main Routine
# -----------------------------------------------------------------

def process_all_magnetars(
    infile: str,
    outfile: str,
    grid: int = 3,
    plot: bool = False,
) -> None:
    """
    Compute 5σ upper flux limits for each magnetar and IRAC channel.

    Reads ensemble photometry CSV produced by EnsemblePhotometry.run(),
    groups measurements by (name, channel, flux level), computes σ_ensemble
    as std of recovered fluxes, and writes 5 × mean(σ_ensemble) per channel.

    Parameters
    ----------
    infile : str
        Path to ensemble photometry CSV.
    outfile : str
        Path for output tab-separated file.
    grid : int
        Grid size used during the ensemble run (default 3 for 3×3).
        window_size is computed as grid² so it stays in sync with the config.
    plot : bool
        Reserved for future diagnostic plots (not yet implemented).
    """
    logger = logging.getLogger(__name__)

    df = pd.read_csv(infile, comment='#')
    result_lines = ['# name\tch1_sens5(µJy)\tch2_sens5(µJy)\tch3_sens5(µJy)\tch4_sens5(µJy)']

    n_scales = len(FACTORS)
    window_size = grid ** 2          # positions per flux level = grid × grid
    expected_len = n_scales * window_size

    for name in np.unique(df['name']):
        sub = df[df['name'] == name]
        row_result = [name]

        for ch in range(1, 5):
            ch_data = sub[sub['channel'] == ch]

            if len(ch_data) == 0:
                logger.warning(f"No ensemble data for {name} ch{ch}")
                row_result.append(np.nan)
                continue

            # Select flux column (uJy if available, else raw counts)
            if 'flux' in ch_data.columns:
                flux_col = 'flux'
            else:
                flux_col = 'phot_flux_(µJy)'

            phot_values = np.array(ch_data[flux_col])

            # Guard: data length must be exactly n_scales × window_size.
            # If any grid position failed during the run, windowing would be
            # silently wrong without this check.
            if len(phot_values) != expected_len:
                logger.warning(
                    f"Expected {expected_len} rows for {name} ch{ch} "
                    f"({n_scales} scales × {window_size} positions), "
                    f"got {len(phot_values)}. Skipping — check for failed grid positions."
                )
                row_result.append(np.nan)
                continue

            # Unit conversion factor (only needed if data is in raw counts)
            is_uJy = 'unit' in ch_data.columns and 'uJy' in str(ch_data['unit'].iloc[0])
            const = 1.0 if is_uJy else MAGIC_NUMBER * APCOR[ch - 1]

            # Compute σ_ensemble for each injection flux level.
            # Each window of window_size measurements shares the same injected flux;
            # their scatter reflects real photometric noise at that position region.
            stds = _std_by_window(phot_values, window_size)
            sigma_ensemble = stds * const  # µJy

            logger.info(f"{name} ch{ch}: σ_ensemble per flux level = {sigma_ensemble} µJy")

            # Linearity check: mean(recovered_flux) should ≈ expected_flux (within aperture).
            # Expected aperture flux ≈ expected_flux / APCOR (APCOR corrects for wing losses).
            if 'expected_flux' in ch_data.columns:
                expected_vals = np.array(ch_data['expected_flux'])
                for i in range(n_scales):
                    w_start = i * window_size
                    w_end = w_start + window_size
                    mean_recovered = np.mean(phot_values[w_start:w_end]) * const
                    expected = expected_vals[w_start] * const
                    if expected != 0:
                        recovery = mean_recovered / expected
                        # Expect recovery ≈ 1/APCOR because circ_apphot only captures
                        # the in-aperture fraction of the injected flux.
                        expected_recovery = 1.0 / APCOR[ch - 1]
                        logger.info(
                            f"  {name} ch{ch} scale {i+1}: "
                            f"recovery = {recovery:.3f} "
                            f"(expect ~{expected_recovery:.3f} = 1/APCOR)"
                        )
                        if abs(recovery - expected_recovery) / expected_recovery > 0.15:
                            logger.warning(
                                f"  Recovery fraction deviates >15% from 1/APCOR for "
                                f"{name} ch{ch} scale {i+1}. Check PRF normalization."
                            )

            # 5σ upper limit: average σ_ensemble across flux levels, then multiply by 5.
            # Valid for background-dominated (faint) sources where σ_ensemble ≈ constant.
            limit_5sigma = 5.0 * float(np.mean(sigma_ensemble))
            logger.info(f"{name} ch{ch}: 5σ limit = {limit_5sigma:.4f} µJy")
            row_result.append(limit_5sigma)

        result_lines.append(
            '\t'.join(v if isinstance(v, str) else f'{v:.6f}' for v in row_result)
        )

    with open(outfile, 'w') as f:
        f.write('\n'.join(result_lines) + '\n')

    print(f"Results saved to {outfile}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute 5-sigma flux limits from ensemble photometry results."
    )
    parser.add_argument('-i', '--input', type=str, default="result.coldat",
                        help="Ensemble photometry CSV/coldat file")
    parser.add_argument('-o', '--output', type=str, default="./../results/snr5_result.coldat",
                        help="Output coldat file for 5-sigma limits")
    parser.add_argument('--plot', action='store_true',
                        help="Generate diagnostic plots (not yet implemented)")
    args = parser.parse_args()

    process_all_magnetars(args.input, args.output, plot=args.plot)


if __name__ == "__main__":
    main()
