import argparse
import logging
import sys

import pandas as pd

from src.config.loader import load_yaml_config
from src.core.analysis import process_all_magnetars
from src.core.photometry import (
    ENSEMBLE_HEADERS,
    ONE_SHOT_HEADERS,
    EnsemblePhotometry,
    EnsemblePhotometryConfig,
)
from src.utils.custom_logger import configure_logging
from src.utils.io import save_rows


def setup_logger(level: str = 'INFO') -> logging.Logger:
    try:
        configure_logging(level=level)
        return logging.getLogger(__name__)
    except Exception as e:
        print(f"Error setting up logging: {e}", file=sys.stderr)
        sys.exit(1)


def main(config_file: str = 'configs/default.yml', verbose: bool = False) -> None:
    log_level = 'DEBUG' if verbose else 'INFO'
    logger = setup_logger(level=log_level)
    logger.info("Starting IRACMagLim run")

    # ── Load configuration ─────────────────────────────────────────────
    try:
        cfg = load_yaml_config(filename=config_file)
        logger.debug(f"Loaded configuration: {cfg.as_dict()}")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration error: {e}")
        return

    # ── Read target list ───────────────────────────────────────────────
    try:
        mag_list = pd.read_csv(
            cfg.magnetars_list_path_file,
            comment='#', sep=r'\s+',
            names=['name', 'ra', 'dec'],
        )
        logger.info(f"Found {len(mag_list)} targets")
    except OSError as e:
        logger.error(f"Cannot read target list: {e}")
        return

    # ── Determine output paths ─────────────────────────────────────────
    ensemble_base = cfg.ensemble_phot_data
    one_shot_base = cfg.one_shot_phot_data
    limit_outfile = (
        cfg.output_file
        if cfg.output_file.endswith(('.coldat', '.txt', '.csv'))
        else cfg.output_file + '.coldat'
    )

    # Remove stale output files so save_rows starts fresh.
    import os
    for base in (ensemble_base, one_shot_base):
        for fmt in cfg.output_formats:
            path = f"{base}.{fmt}"
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Removed existing file: {path}")

    # ── Per-target photometry ──────────────────────────────────────────
    total = len(mag_list)
    for i, row in mag_list.iterrows():
        name, ra, dec = row['name'], row['ra'], row['dec']
        logger.info(f"Processing {i+1}/{total}: {name}  RA={ra}  Dec={dec}")

        ensemble_cfg = EnsemblePhotometryConfig(
            name=name,
            x_coord=ra,
            y_coord=dec,
            channels=cfg.channels,
            psf_file_path=cfg.prf_path,
            image_file_path=f"{cfg.data_path_folder}/{name}",
            ap_radius=cfg.rap_cam_pix,
            inner_ann_radius=cfg.rbackin_cam_pix,
            outer_ann_radius=cfg.rbackout_cam_pix,
            grid=cfg.grid,
            spacing=cfg.spacing,
            output_formats=cfg.output_formats,
            uJy_units=cfg.uJy_units,
            photometry_method=cfg.photometry_method,
            save_plots=cfg.save_plots,
            start_sim=cfg.start_sim,
            psf_trim_pixels=cfg.psf_trim_pixels,
        )

        try:
            result = EnsemblePhotometry(ensemble_cfg, verbose=verbose).run()
        except Exception as e:
            logger.error(f"Unexpected error for {name}: {e}", exc_info=True)
            continue

        # Write this target's rows immediately (append mode) so partial
        # results are preserved even if a later target fails.
        if result.one_shot_rows:
            save_rows(one_shot_base, result.one_shot_rows,
                      cfg.output_formats, ONE_SHOT_HEADERS, mode='a')
        if result.ensemble_rows:
            save_rows(ensemble_base, result.ensemble_rows,
                      cfg.output_formats, ENSEMBLE_HEADERS, mode='a')

    # ── Compute 5σ limits ─────────────────────────────────────────────
    input_limit_file = (
        ensemble_base + '.csv' if 'csv' in cfg.output_formats
        else ensemble_base + '.coldat'
    )
    try:
        logger.info("Computing 5σ limits from ensemble photometry results")
        process_all_magnetars(input_limit_file, limit_outfile, grid=cfg.grid)
    except Exception as e:
        logger.error(f"Error computing 5σ limits: {e}", exc_info=True)
        return

    logger.info("IRACMagLim run completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble aperture photometry for Spitzer IRAC magnetar upper limits."
    )
    parser.add_argument('-i', '--config', default='configs/default.yml',
                        help="YAML configuration file")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Enable DEBUG logging")
    args = parser.parse_args()
    main(args.config, verbose=args.verbose)
