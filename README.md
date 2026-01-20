## PhotometryPy

Comprehensive README describing how PhotometryPy works, how to run it, configuration details, data layout, and troubleshooting.

## Overview

PhotometryPy is a small pipeline to perform ensemble aperture photometry on simulated/real mosaicked infrared images (IRAC-like). The code: reads a list of targets, converts sky coordinates to pixel coordinates using the image WCS, performs a one-shot circular aperture photometry measurement, then carries out an ensemble of simulated PRF placements around the source to measure the noise/flux behaviour and estimate sensitivity (5-sigma) limits.

Key features:
- One-shot circular aperture photometry for each target and channel.
- Ensemble photometry by placing and scaling PRFs on a grid to measure photometric scatter.
- Automatic aggregation of intermediate measurements and calculation of 5-sigma sensitivities.

## Quick start

1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Edit `configs/default.yml` if you want to change file paths or photometry parameters.

3. Run the pipeline (uses `configs/default.yml` by default):

```bash
python main.py -i configs/default.yml
```

This will:
- Read the magnetar list from `magnetars_list_path_file`.
- For each target, run `ensemble_photometry` (per-channel) which will write to an intermediate CSV indicated by `intermediate_path_file`.
- After all targets, the script runs `process_all_magnetars` (flux_snr5) to compute 5-sigma sensitivities and write `result_path_file`.

## Project layout and important files

- `main.py` — entry point that loads the YAML config, iterates targets, runs ensemble photometry, and computes SNR=5 results.
- `requirements.txt` — Python dependencies (astropy, scipy, numpy, pandas, matplotlib, pyyaml).
- `configs/default.yml` — default configuration file (paths and photometry parameters).
- `mag_info/` — contains `CORRECTED_mag_dec.coldat` (target name, RA, DEC).
- `prf/` — contains PRF/PSF FITS files used when placing simulated PRFs (e.g. `apex_sh_IRAC1_...fits`).
- `simtar_partial/` — top-level directory with per-target mosaics (expected subdirs like `mosaici1/`, `mosaici2/`, ...).
- `results/` — output files (intermediate and final results).
- `plots/` — generated figures (per-target photometry plots and diagnostics).
- `src/` — core modules and library code:
  - `config.py` — YAML loader and validator (produces an `AppConfig` dataclass).
  - `ensemble_photometry.py` — main per-target photometry workflow.
  - `idl_circapphot.py` — aperture photometry routines (ported from IDL: `circ_apphot`, `get_annulus`).
  - `psf.py` — PSF object and helpers (trimming, resampling, normalization).
  - `utils.py` — image/PSF I/O helpers, plotting, saving CSV results.
  - `flux_snr5.py` — aggregation and linear-regression based computation of 5-sigma sensitivities.
  - `combine_results.py` — helper to merge one-shot and sensitivity outputs into a combined table.
  - `logging_setup.py` — logging configuration used by `main.py`.

## Configuration (keys explained)

The YAML file expects a flat mapping (see `configs/default.yml`). `src/config.py` maps these keys into the `AppConfig` dataclass and validates values.

Important keys (and how they map):
- `magnetars_list_path_file` — path to the file listing targets (name RA DEC). Mapped to `AppConfig.magnetars_list_path_file`.
- `data_path_folder` — directory containing per-target mosaics (`simtar_partial` in this repo). Mapped to `data_path_folder`.
- `prf_path` — directory containing PRF/PSF FITS files. Mapped to `prf_path`.
- `intermediate_path_file` — path to CSV where every ensemble photometry measurement is appended (created by `save_results`). Mapped to `intermediate_path_file`.
- `result_path_file` — final aggregated sensitivity file (written by `flux_snr5.process_all_magnetars`). Mapped to `result_path_file`.
- `channels` — list of channels to process (allowed values: 1,2,3,4).
- `grid` — integer grid size for ensemble PRF placements (NxN grid).
- `spacing` — pixel spacing between simulated PRF centers.
- Aperture radii keys (note: the YAML uses `rap_(cam_pix)`, `rbackin_(cam_pix)`, `rbackout_(cam_pix)`): these are converted to `rap_cam_pix`, `rbackin_cam_pix`, `rbackout_cam_pix` in `AppConfig` and are used to compute radii for each channel with a pixel scale correction.

Validation and behavior notes:
- `config.load_yaml_config()` validates types and ranges (e.g., radii ordering rap < rbackin < rbackout, positive spacing, grid >= 1, channels subset of [1,2,3,4]).
- It also checks that `magnetars_list_path_file` is a file and `data_path_folder` and `prf_path` are directories; missing paths will raise a `ValueError`.

## How the pipeline works (detailed flow)

For each target listed in `magnetars_list_path_file`:
1. Build `mag_config` with paths and photometry parameters derived from the top-level `AppConfig`.
2. For each requested channel (1–4):
   - Resolve PSF and mosaic FITS paths. The code expects PRFs named like `apex_sh_IRAC{ch}_col129_row129_x100.fits` in `prf/` and mosaics at `<data_path_folder>/<target>/mosaici{ch}/Combine/mosaic.fits`.
   - Convert RA/DEC to pixel coordinates using `astropy.wcs.WCS` applied to the mosaic FITS header.
   - Run `circ_apphot()` (one-shot) on the real mosaic at the target pixel position to get `total_counts` and `sigma`. One-shot results are appended to `results/all_one_shot.csv` by `save_one_shot()`.
   - Compute aperture and annulus sizes in pixels for the current channel using a per-channel pixel scale (`PSIZE_ASEC` in `ensemble_photometry.py`) relative to mosaic pixel scale.
   - Create a set of normalization scales (FACTORS * measured sigma) and for each scale place a resampled+shifted+normalized PSF into a copy of the image at positions defined by a grid (centered on the target) with spacing `spacing`.
   - For each simulated image perform `circ_apphot()` and append a measurement to `intermediate_path_file` via `save_results()`.
   - Save plots of the aperture/annulus overlays and other diagnostics under `plots/`.

After all targets are processed, `main.py` calls `process_all_magnetars(intermediate_path_file, result_path_file)` from `flux_snr5.py` which:
- Reads the intermediate ensemble results and `results/all_one_shot.csv`.
- Bins ensemble photometry into windows (default WINDOW_SIZE=9), computes per-window standard deviations, and performs a linear regression of sigma_ensemble vs expected flux.
- Uses the one-shot sigma (multiplied by 5) to calculate the ensemble sigma corresponding to a 5-sigma detection (the code computes `sigma_en_at_5 = slope * sigma_5 + intercept`).
- Writes final sensitivity results to `result_path_file` (tab-separated values, header begins with `#`). It also optionally saves diagnostic plots per source and channel under `plots/`.

## File formats (important fields)

- Intermediate CSV (`intermediate_path_file`): appended rows via `save_results()` with header (if file not present):

  name, ra, dec, channel, x_pos, y_pos, expected_flux, phot_flux_(µJy), phot_sigma_(µJy)

- One-shot CSV (`results/all_one_shot.csv`) columns:

  name, phot, sigma, channel

- Final sensitivity output (`result_path_file`) is a tab-separated file where each row is `name` followed by ch1..ch4 sens5 values (µJy). See `flux_snr5.py` for exact formatting.

## Module responsibilities (short)

- `main.py` — orchestrates the run (config load → per-target ensemble photometry → SNR calculations).
- `src/config.py` — loads and normalizes YAML config into `AppConfig`, validates inputs.
- `src/utils.py` — FITS reading, PSF placement, plotting, CSV helpers.
- `src/psf.py` — PSF trimming, resampling (`congrid`/`PSF.congrid`), shifting and normalization.
- `src/idl_circapphot.py` — core aperture photometry: `get_annulus()` and `circ_apphot()` (returns total counts and sigma among other values).
- `src/ensemble_photometry.py` — high-level per-target workflow that calls the utilities above.
- `src/flux_snr5.py` — aggregates ensemble results and computes 5-sigma sensitivity via regression.
- `src/combine_results.py` — utilities for merging one-shot and sensitivity outputs into a single table (useful post-processing).

## Contract (inputs / outputs / errors)

- Inputs: YAML config, target list file (name RA DEC), per-target mosaics and PRF FITS.
- Outputs: `results/all_one_shot.csv`, `intermediate_path_file` (ensemble rows), `result_path_file` (5-sigma sensitivities), plots under `plots/`.
- Error modes:
  - Missing files or directories → `config.load_yaml_config()` will raise ValueError earlier, ensemble_photometry will log errors and continue to next channel/target.
  - WCS conversion failures → logged and that channel is skipped for that target.
  - PSF trimming/resizing issues → PSF methods may raise ValueError if trim is too large.

Success: program completes and writes the final `result_path_file`. Partial failures will be logged in `logs/photometrypy.log` and processing will continue for other targets.

## Common troubleshooting and tips

- If the script complains about missing PRF files, confirm `prf_path` in `configs/default.yml` points to the directory containing `apex_sh_IRAC{n}...fits` files.
- If WCS conversion fails, inspect the mosaic FITS headers: `astropy.wcs.WCS` requires valid WCS keywords. You can open the file with `astropy.io.fits` to inspect headers.
- If the YAML loader raises `Invalid configuration`, fix the indicated keys or create directories/files expected by the config.
- If you want to run only the SNR step on an existing intermediate file:

```bash
python -c "from src.flux_snr5 import process_all_magnetars; process_all_magnetars('results/photometry/all_phot.coldat', 'results/snr5_output.coldat', plot=True)"
```

## Notes, assumptions, and future suggestions

- The pipeline assumes mosaics are at `<data_path_folder>/<target>/mosaici{ch}/Combine/mosaic.fits` and PRFs follow the naming convention in the repo. If your data are arranged differently, update `main.py` or restructure data folders.
- The code currently writes intermediate outputs by appending rows to CSVs. For very large runs you might prefer batching or using a database/file format that supports parallel writes.
- Suggested small improvements: add CLI flags in `main.py` to restrict to a subset of targets, parallelize per-target processing, parameterize PSF trim size and output directories.

## Contact / authorship

Repository owner: `drkh-n` (local repo metadata). For code comments and history, see file headers (e.g., `flux_snr5.py` includes author metadata).

