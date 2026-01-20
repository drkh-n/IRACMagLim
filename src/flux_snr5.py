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
# ; This here Python script calculates the 5-sigma sensitivity limits for IRAC data by performing linear regression on signal-to-noise ratio versus flux measurements across multiple channels.
# ;
# ; CATEGORY:
# ; Astronomy, Sensitivity Analysis, IRAC Data Processing
# ;
# ; CALLING SEQUENCE:
# ; python flux_snr5.py -i input_file -o output_file [--plot]
# ;
# ; INPUTS:
# ; input_file: The input data file containing photometry results with columns for source name, channel, flux factors, photometry measurements, and uncertainties.
# ;
# ; OUTPUTS:
# ; output_file: The output file containing 5-sigma sensitivity limits for each source across all IRAC channels.
# ;
# ; KEYWORD PARAMETERS:
# ; --plot: Set this flag to generate diagnostic plots showing the linear fit and 5-sigma flux determination for each channel.
# ;
# ; PROCEDURE:
# ; The script reads photometry data, calculates signal-to-noise ratios, performs linear regression of SNR versus flux, and determines the flux value corresponding to SNR=5 for each source and channel.
# ;
# ; EXAMPLE:
# ; Calculate 5-sigma sensitivity limits and generate plots:
# ;
# ; python flux_snr5.py -i result.coldat -o snr5_result.coldat --plot
# ;
# ; MODIFICATION HISTORY:
# ; 	Written by:	Darkhan Nurzhakyp 2025 September 30
# ;	September,2025	Any additional mods get described here.  Remember to
# ;			change the stuff above if you add a new keyword or
# ;			something!
# ;-

import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from scipy import stats
import logging

# Per-channel no_bgnd values (µJy)
# NO_BGND_PER_CH = [2.975601012159931, 8.47*1.120, 8.47*1.135, 8.47*1.221]
WINDOW_SIZE = 9
AP_COR = [1.125, 1.120, 1.135, 1.221]

# ;-----------------------------------------------------------------
# ;               Mini-Routines (main routine comes last)
# ;-----------------------------------------------------------------

def std(data, window_size):
    stds = []
    for i in range(0, len(data), window_size):
        window = data[i : i + window_size]
        stds.append(np.std(window))
    return np.array(stds)

# ;-----------------------------------------------------------------
# ;-----------------------------------------------------------------
# ;                             Main Routine
# ;-----------------------------------------------------------------
# ;-----------------------------------------------------------------

def process_all_magnetars(infile, outfile, plot=True):
    logger = logging.getLogger(__name__)
    
    df = pd.read_csv(infile, comment='#')
    result_lines = ['# name\tch1_sens5(µJy)\tch2_sens5(µJy)\tch3_sens5(µJy)\tch4_sens5(µJy)']

    one_shot = pd.read_csv('results/all_one_shot.csv', comment='#')
    for name in np.unique(df['name']):
        sub = df[df['name'] == name]
        row_result = [name]

        one_shot_name = one_shot[one_shot['name'] == name]

        for ch in range(1, 5):
            ch_data = sub[sub['channel'] == ch]
            if len(ch_data) == 0:
                row_result.append(np.nan)
                continue

            expected = np.array(ch_data['expected_flux'])
            phot_values = np.array(ch_data['phot_flux_(µJy)'])

            if len(phot_values) % WINDOW_SIZE != 0:
                print(f"⚠️ Warning: samples not divisible by {WINDOW_SIZE} for {name}, channel {ch}")
                row_result.append(np.nan)
                continue

            const = 8.47 * AP_COR[ch-1]
            stds = std(phot_values, WINDOW_SIZE)
            y_data = stds * const
            x_data = expected[::WINDOW_SIZE] * const

            logger.info(f"Magnetar: {name}, Channel: {ch}")
            logger.info(f"y_data (sigma ensemble): {y_data}")
            logger.info(f"x_data (expected flux): {x_data}")

            one_shot_line = one_shot_name[one_shot_name['channel'] == ch]
            print(one_shot_line)

            # guard: ensure we actually have a matching one-shot row
            if len(one_shot_line) == 0 or one_shot_line['sigma'].isnull().all():
                logger.warning(f"Missing one-shot sigma for {name}, channel {ch}")
                print(f"⚠️ Warning: missing one-shot sigma for {name}, channel {ch}")
                row_result.append(np.nan)
                continue

            # use positional indexing (.iloc) because the filtered DataFrame
            # preserves original row labels which may not include 0,1,2...
            sigma_5 = 5.0 * float(one_shot_line['sigma'].iloc[0])
            logger.debug(f"sigma_5 (5 * one_shot_sigma): {sigma_5:.6f}")

            # --- Perform linear regression ---
            # if len(x_data) >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
            logger.info(f"Linear regression results for {name} ch{ch}: Slope={slope:.6f}, Intercept={intercept:.6f}, R-value={r_value:.6f}, P-value={p_value:.6e}, Standard error={std_err:.6f}")
            # else:
            #     slope, intercept, r_value, p_value, std_err = np.nan, np.nan, 0, np.nan, np.nan
            #     logger.warning(f"Insufficient data points for linear regression (only {len(x_data)} points) - setting results to NaN for {name} ch{ch}")

            print(f"Slope: {slope}")
            print(f"Intercept: {intercept}")
            print(f"R-value: {r_value}")
            print(f"P-value: {p_value}")
            print(f"Standard error: {std_err}")

            sigma_en_at_5 = slope * sigma_5 + intercept
            logger.info(f"{name} ch{ch}: Calculated sigma_ensemble@Flux=5*sigma = {sigma_en_at_5:.6f} µJy (slope={slope:.6f} * {sigma_5:.6f} + intercept={intercept:.6f})")
            print(sigma_en_at_5)
            print(f"→ {name} ch{ch}: Sigma_ensemble@Flux=5*sigma = {sigma_en_at_5:.2f} µJy")

            # --- Optional plotting ---
            if plot:
                plt.figure(figsize=(7,5))
                plt.plot(x_data, y_data, 'o')

                # plot the linear fit line
                y_fit = None
                try:
                    x_fit = np.linspace(np.nanmin(x_data), np.nanmax(x_data), 200)
                    y_fit = slope * x_fit + intercept
                    plt.plot(x_fit, y_fit, '-', color='orange', linewidth=1.8, label=f'Linear fit')
                except Exception:
                    y_fit = None
                    pass

                plt.title(f"Simulated Upper Limit Flux \nMagnetar: {name}, IRAC: {ch}")
                plt.xlabel("Simulated Source Flux (µJy)")
                plt.ylabel("Sigma Ensemble (uJy)")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"plots/{name}_ch{ch}_sens.png", dpi=150)
                plt.close()

            row_result.append(sigma_en_at_5)
        # format output row
        result_lines.append('\t'.join(f'{v}' if isinstance(v, str) else f'{v:.6f}' for v in row_result))

    # write file
    with open(outfile, 'w') as f:
        f.write('\n'.join(result_lines) + '\n')

    print(f"✅ Results saved to {outfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default="result.coldat", help="coldat file with circapphot data")
    parser.add_argument('-o', '--output', type=str, default="./../results/snr5_result.coldat", help="Output coldat result")
    parser.add_argument('--plot', action='store_true', help="Plot SNR vs Flux curve for each channel")
    args = parser.parse_args()
    
    process_all_magnetars(args.input, args.output, plot=args.plot)

if __name__ == "__main__":
    main()
