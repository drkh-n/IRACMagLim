## PhotometryPy

Ensemble aperture photometry pipeline for computing 5σ flux upper limits on Spitzer/IRAC magnetar observations.

## Overview

PhotometryPy processes mosaicked infrared images (Spitzer IRAC-like) to estimate per-channel photometric sensitivity limits. For each target the pipeline:

1. Runs one-shot circular aperture photometry at the target sky position.
2. Injects PRFs at a grid of positions around the target at multiple flux levels (ensemble simulation).
3. Computes the 5σ upper limit as `5 × mean(σ_ensemble)` across all injection levels, where `σ_ensemble` is the standard deviation of recovered fluxes within each injection level group.

Key features:
- Two photometry backends: custom `circ_apphot` (IDL-ported) or `photutils`.
- Ensemble PRF injection controlled by `grid`, `spacing`, and `psf_trim_pixels`.
- Results written in configurable formats (`.csv` and/or `.coldat`).
- Partial results preserved per-target: if a later target fails, earlier results are already on disk.

## Quick start

1. Create a Python environment and install the package in editable mode:

```bash
python -m venv .venv
# Activate the virtual environment:
# On Windows: .venv\Scripts\activate
# On macOS/Linux: source .venv/bin/activate

pip install -e .[dev]
```

2. Run the tests to verify the installation:

```bash
pytest
```

3. Edit `configs/default.yml` to set your data paths and photometry parameters.

4. Run the pipeline:

```bash
python main.py -i configs/default.yml
```

Add `-v` to enable DEBUG-level logging:

```bash
python main.py -i configs/default.yml -v
```

## Project layout

```
PhotometryPy/
├── main.py                        # Entry point
├── pyproject.toml                 # Build system and dependencies
├── configs/
│   └── default.yml                # Default configuration
├── mag_info/                      # Target list files (name RA DEC)
├── prf/                           # APEX PRF FITS files
├── simtar_partial/                # Per-target mosaics
├── results/                       # Output files
│   └── photometry/                # Intermediate per-target results
├── plots/                         # Diagnostic plots (if save_plots: true)
└── src/
    ├── config/
    │   └── loader.py              # YAML loader, AppConfig dataclass, validation
    ├── core/
    │   ├── photometry.py          # EnsemblePhotometry class and perform_photometry()
    │   ├── analysis.py            # process_all_magnetars(): 5σ limit computation
    │   ├── idl_circapphot.py      # Aperture photometry routines (circ_apphot, get_annulus)
    │   ├── psf.py                 # PSF class: trimming, resampling, normalization
    │   └── combine_results.py     # Merge one-shot and ensemble outputs into one table
    └── utils/
        ├── constants.py           # MOSAIC_PIXEL_SCALE_ARCSEC, PSIZE_ASEC, APCOR, FACTORS, counts_to_ujy()
        ├── io.py                  # load_fits_image(), get_PSF(), save_rows()
        ├── psf_utils.py           # make_PSF(), place_PSF()
        ├── plotting.py            # make_plot(), save_x_profile()
        ├── common.py              # Shared utility functions
        └── custom_logger.py       # configure_logging()
```

## Configuration

The pipeline is driven by a YAML config file. All keys are flat (no nesting).

### Required keys

| Key | Description |
|-----|-------------|
| `magnetars_list_path_file` | Path to target list file (whitespace-separated: `name ra dec`) |
| `data_path_folder` | Directory containing per-target mosaic subdirectories |
| `prf_path` | Directory containing APEX PRF FITS files |
| `channels` | List of IRAC channels to process, e.g. `[1, 2, 3, 4]` |
| `grid` | Grid size for PRF injection (N×N positions) |
| `spacing` | Pixel spacing between PRF injection positions |
| `rap_(cam_pix)` | Aperture radius in native IRAC camera pixels |
| `rbackin_(cam_pix)` | Inner sky annulus radius in native IRAC camera pixels |
| `rbackout_(cam_pix)` | Outer sky annulus radius in native IRAC camera pixels |

Radii must satisfy `rap < rbackin < rbackout`. The loader converts these to mosaic pixels per channel using the native IRAC pixel scales from `utils/constants.py`.

### Optional keys (with defaults)

| Key | Default | Description |
|-----|---------|-------------|
| `ensemble_phot_data` | `results/photometry/ensemble_data` | Base path (without extension) for ensemble photometry output |
| `one_shot_phot_data` | `results/photometry/one_shot_data` | Base path for one-shot photometry output |
| `output_file` | `results/output` | Base path for the final 5σ limit file |
| `output_formats` | `['csv']` | List of output formats: `csv`, `coldat`, or both |
| `uJy_units` | `true` | Write flux values in µJy; if `false`, raw MJy/sr counts |
| `photometry_method` | `circ_apphot` | Aperture photometry backend: `circ_apphot` or `photutils` |
| `save_plots` | `false` | Save aperture overlay and profile diagnostic plots |
| `start_sim` | `true` | Run the ensemble PRF injection; if `false`, only one-shot photometry |
| `psf_trim_pixels` | `50` | Pixels to trim from each edge of the raw APEX PRF before downsampling |

The legacy keys `intermediate_path_file` and `result_path_file` are also accepted as fallbacks for `ensemble_phot_data` and `output_file` respectively, for backwards compatibility.

## How the pipeline works

### Per-target (main.py → EnsemblePhotometry)

For each target in the target list:

1. Build an `EnsemblePhotometryConfig` from the top-level `AppConfig`.
2. For each requested IRAC channel (1–4):
   - Open `<data_path_folder>/<name>/mosaici{ch}/Combine/mosaic.fits` and extract image data and WCS.
   - Convert RA/Dec to mosaic pixel coordinates via `astropy.wcs.WCS`.
   - Scale aperture/annulus radii from native IRAC pixels to mosaic pixels using `PSIZE_ASEC[ch] / MOSAIC_PIXEL_SCALE_ARCSEC`.
   - Run `perform_photometry()` at the target position (**one-shot**) and record flux/sigma.
   - Load the APEX PRF: `<prf_path>/apex_sh_IRAC{ch}_col129_row129_x100.fits`.
   - For each injection scale in `FACTORS × σ_one_shot`, place a resampled+shifted PRF at each of the `grid × grid` positions centered on the target (**ensemble**).
   - Run `perform_photometry()` on each simulated image and collect rows.
3. Write this target's one-shot and ensemble rows immediately (append mode) via `save_rows()`.

### 5σ limit computation (analysis.py → process_all_magnetars)

After all targets are processed:

1. Read the ensemble photometry CSV.
2. For each (target, channel), group the recovered fluxes into windows of size `grid²` (one window per injection flux level).
3. Compute `σ_ensemble = std(recovered_flux)` within each window.
4. Compute the 5σ upper limit: `5 × mean(σ_ensemble)` (valid in the background-dominated regime).
5. Write a tab-separated output file with one row per target and columns for each channel's 5σ limit in µJy.

## Output files

| File | Description |
|------|-------------|
| `<one_shot_phot_data>.<fmt>` | One-shot photometry: `name, phot, sigma, channel, unit` |
| `<ensemble_phot_data>.<fmt>` | Ensemble measurements: `name, ra, dec, channel, x_pos, y_pos, scale, expected_flux, flux, sigma, unit` |
| `<output_file>` | Final 5σ limits: tab-separated, columns `name, ch1_sens5(µJy), ch2_sens5(µJy), ch3_sens5(µJy), ch4_sens5(µJy)` |

## Running the analysis step standalone

To recompute 5σ limits from an existing ensemble file:

```bash
python -c "
from src.core.analysis import process_all_magnetars
process_all_magnetars(
    'results/photometry/ensemble_data.csv',
    'results/output.coldat',
    grid=3,
)
"
```

Or using the module's own CLI:

```bash
python src/core/analysis.py -i results/photometry/ensemble_data.csv -o results/output.coldat
```

## Troubleshooting

- **Missing PRF files**: confirm `prf_path` points to the directory containing `apex_sh_IRAC{n}_col129_row129_x100.fits`.
- **WCS conversion failures**: inspect the mosaic FITS header with `astropy.io.fits` — valid WCS keywords are required.
- **`Invalid configuration` on startup**: the loader validates all required keys, path existence, and radius ordering; the error message will identify the failing check.
- **`Expected N rows … got M`** in analysis: some grid positions failed during the ensemble run. Check the log for warnings about `make_PSF` or photometry failures on those positions.
- **Partial results**: if a target fails mid-run, previously written rows are preserved. Re-run after fixing the issue; stale output files are removed at the start of each run.

## Contact / authorship

Author: Darkhan Nurzhakyp (`darkhan.nurzhakyp@nu.edu.kz`). See `pyproject.toml` for package metadata.
