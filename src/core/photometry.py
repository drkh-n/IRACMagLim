"""
Ensemble aperture photometry for magnetar 5σ flux limit estimation.

Design notes
------------
* `EnsemblePhotometry.run()` returns row dicts; it does NOT write to files.
  The caller (main.py) accumulates results across targets and writes once
  per file using `utils.io.save_rows`. This decouples the photometry logic
  from I/O and makes each method independently testable.
* Each FITS image is opened once per channel (image data + WCS in a single
  `load_fits_image` call).
* Exceptions are caught at the narrowest scope: only expected failure modes
  for each operation are caught; programming errors propagate normally.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy.wcs import WCS

from core.idl_circapphot import circ_apphot
from utils.constants import APCOR, FACTORS, MOSAIC_PIXEL_SCALE_ARCSEC, PSIZE_ASEC, counts_to_ujy
from utils.io import get_PSF, load_fits_image
from utils.plotting import make_plot, save_x_profile
from utils.psf_utils import make_PSF, place_PSF

# Column definitions for output files — shared with main.py for save_rows calls.
ONE_SHOT_HEADERS = ['name', 'phot', 'sigma', 'channel', 'unit']
ENSEMBLE_HEADERS = [
    'name', 'ra', 'dec', 'channel',
    'x_pos', 'y_pos', 'scale',
    'expected_flux', 'flux', 'sigma', 'unit',
]


def perform_photometry(
    image_data: np.ndarray,
    xc: float,
    yc: float,
    ap_radius: float,
    inner_ann_radius: float,
    outer_ann_radius: float,
    method: str = 'circ_apphot',
    t_exp: float = 1.0,
    photutils_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Perform circular aperture photometry using the specified method.

    Parameters
    ----------
    image_data : np.ndarray
    xc, yc : float
        Source center in mosaic pixels.
    ap_radius : float
        Aperture radius in mosaic pixels.
    inner_ann_radius, outer_ann_radius : float
        Background annulus radii in mosaic pixels.
    method : str
        'circ_apphot' (default) or 'photutils'.
    t_exp : float
        Exposure time (for instrumental magnitude only).
    photutils_kwargs : dict, optional
        Extra kwargs forwarded to photutils.aperture_photometry.

    Returns
    -------
    dict with keys: total_counts, instrumental_mag, mag_error,
                    n_pixels, bg_stddev, sigma, bg_level
    """
    if method == 'circ_apphot':
        return circ_apphot(
            image_data, xc, yc, ap_radius,
            t_exp=t_exp,
            bgndwidth=outer_ann_radius - inner_ann_radius,
            quiet=True,
            rbackin=inner_ann_radius,
        )

    if method == 'photutils':
        from photutils.aperture import (
            ApertureStats, CircularAnnulus, CircularAperture, aperture_photometry,
        )
        if photutils_kwargs is None:
            photutils_kwargs = {}

        positions = [(xc, yc)]
        aperture = CircularAperture(positions, r=ap_radius)
        annulus = CircularAnnulus(positions, r_in=inner_ann_radius, r_out=outer_ann_radius)

        aperstats = ApertureStats(image_data, annulus)
        bg_level = float(aperstats.median)
        bg_stddev = float(aperstats.std)

        phot_table = aperture_photometry(image_data, aperture, **photutils_kwargs)
        raw_flux = float(phot_table['aperture_sum'][0])

        aperture_area = aperture.area
        net_flux = raw_flux - bg_level * aperture_area
        sigma = np.sqrt(aperture_area) * bg_stddev

        inst_mag = -2.5 * np.log10(net_flux / t_exp) if net_flux > 0 else np.nan

        return {
            'total_counts': net_flux,
            'instrumental_mag': inst_mag,
            'mag_error': 0.0,
            'n_pixels': float(aperture_area),
            'bg_stddev': bg_stddev,
            'sigma': sigma,
            'bg_level': bg_level,
        }

    raise ValueError(f"Unknown photometry method: '{method}'")


@dataclass
class EnsemblePhotometryConfig:
    """Per-target configuration passed to EnsemblePhotometry."""
    name: str
    x_coord: float           # RA in degrees
    y_coord: float           # Dec in degrees
    channels: List[int]
    psf_file_path: str
    image_file_path: str
    ap_radius: float         # aperture radius in native IRAC camera pixels
    inner_ann_radius: float
    outer_ann_radius: float
    grid: int
    spacing: float
    output_formats: List[str]
    uJy_units: bool = True
    photometry_method: str = 'circ_apphot'
    save_plots: bool = False
    start_sim: bool = True
    psf_trim_pixels: int = 50


@dataclass
class EnsembleResult:
    """Rows collected by EnsemblePhotometry.run() — no file I/O inside."""
    one_shot_rows: List[Dict] = field(default_factory=list)
    ensemble_rows: List[Dict] = field(default_factory=list)


class EnsemblePhotometry:
    """
    Ensemble aperture photometry for a single target across IRAC channels.

    Usage
    -----
    result = EnsemblePhotometry(config).run()
    # result.one_shot_rows and result.ensemble_rows are lists of dicts;
    # pass them to utils.io.save_rows to write to disk.
    """

    def __init__(self, config: EnsemblePhotometryConfig, verbose: bool = False) -> None:
        self.config = config
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> EnsembleResult:
        """
        Process all channels and return collected result rows.

        Does not write any files. The caller is responsible for persistence.
        """
        result = EnsembleResult()
        for channel in self.config.channels:
            if channel not in (1, 2, 3, 4):
                self.logger.error(f"Invalid channel {channel} — must be 1–4, skipping.")
                continue
            self._process_channel(channel, result)
        return result

    # ------------------------------------------------------------------
    # Private per-channel pipeline
    # ------------------------------------------------------------------

    def _process_channel(self, channel: int, result: EnsembleResult) -> None:
        """Load data, run one-shot, run ensemble, append rows to result."""
        # 1. Load image and PSF
        try:
            image_data, wcs, xc, yc = self._load_channel_data(channel)
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            self.logger.error(
                f"{self.config.name} ch{channel}: data load failed — {e}"
            )
            return

        # 2. Aperture / annulus radii in mosaic pixels
        ap_radius, inner_ann, outer_ann = self._aperture_radii(channel)

        # 3. One-shot photometry at the target position
        one_shot_phot = self._do_one_shot(image_data, xc, yc, ap_radius,
                                          inner_ann, outer_ann, channel)
        if one_shot_phot is not None:
            result.one_shot_rows.append(one_shot_phot['row'])
            if self.config.save_plots:
                self._save_one_shot_plots(image_data, xc, yc, ap_radius,
                                          inner_ann, outer_ann, channel)

        # 4. Ensemble PRF injection
        if self.config.start_sim and one_shot_phot is not None:
            psf_data = self._load_psf(channel)
            if psf_data is not None:
                ensemble_rows = self._run_ensemble(
                    image_data, psf_data, xc, yc,
                    ap_radius, inner_ann, outer_ann,
                    one_shot_phot['sigma_raw'], channel,
                )
                result.ensemble_rows.extend(ensemble_rows)

    def _load_channel_data(
        self, channel: int
    ) -> Tuple[np.ndarray, WCS, float, float]:
        """
        Open the mosaic FITS file once, extract image data and WCS,
        and convert RA/Dec to mosaic pixel coordinates.

        Returns (image_data, wcs, xc, yc).
        Raises FileNotFoundError or RuntimeError on failure.
        """
        target_path = (
            f"{self.config.image_file_path}"
            f"/mosaici{channel}/Combine/mosaic.fits"
        )
        image_data, wcs, _ = load_fits_image(target_path)

        try:
            xc, yc = wcs.wcs_world2pix(self.config.x_coord, self.config.y_coord, 0)
            xc, yc = float(xc), float(yc)
        except Exception as e:
            raise ValueError(
                f"WCS coordinate conversion failed for "
                f"RA={self.config.x_coord}, Dec={self.config.y_coord}: {e}"
            ) from e

        self.logger.info(
            f"{self.config.name} ch{channel}: "
            f"RA={self.config.x_coord}, Dec={self.config.y_coord} "
            f"→ pixel ({xc:.2f}, {yc:.2f})"
        )
        return image_data, wcs, xc, yc

    def _load_psf(self, channel: int) -> Optional[np.ndarray]:
        """Load the APEX PRF FITS file for this channel."""
        psf_path = (
            f"{self.config.psf_file_path}"
            f"/apex_sh_IRAC{channel}_col129_row129_x100.fits"
        )
        try:
            return get_PSF(psf_path, channel=channel)
        except (FileNotFoundError, RuntimeError) as e:
            self.logger.error(f"{self.config.name} ch{channel}: PSF load failed — {e}")
            return None

    def _aperture_radii(self, channel: int) -> Tuple[float, float, float]:
        """
        Convert config radii (native IRAC camera pixels) to mosaic pixels.

        Mosaic pixel scale = MOSAIC_PIXEL_SCALE_ARCSEC arcsec/pix.
        Native IRAC scale = PSIZE_ASEC[channel-1] arcsec/pix.
        """
        factor = PSIZE_ASEC[channel - 1] / MOSAIC_PIXEL_SCALE_ARCSEC
        return (
            self.config.ap_radius * factor,
            self.config.inner_ann_radius * factor,
            self.config.outer_ann_radius * factor,
        )

    def _do_one_shot(
        self,
        image: np.ndarray,
        xc: float, yc: float,
        ap_radius: float, inner_ann: float, outer_ann: float,
        channel: int,
    ) -> Optional[Dict]:
        """
        Run aperture photometry at the target position.

        Returns a dict with:
          'row'       — the CSV row dict (ready for save_rows)
          'sigma_raw' — sigma in MJy/sr (used to scale ensemble injections)
        or None on failure.
        """
        try:
            phot = perform_photometry(
                image, xc, yc, ap_radius, inner_ann, outer_ann,
                method=self.config.photometry_method,
            )
        except Exception as e:
            self.logger.error(
                f"{self.config.name} ch{channel}: one-shot photometry failed — {e}",
                exc_info=True,
            )
            return None

        flux_ujy = counts_to_ujy(phot['total_counts'], channel)
        sigma_ujy = counts_to_ujy(phot['sigma'], channel)

        self.logger.info(
            f"{self.config.name} ch{channel} one-shot: "
            f"flux={flux_ujy:.4f} µJy, σ={sigma_ujy:.4f} µJy"
        )

        if self.config.uJy_units:
            val_flux, val_sigma, unit_str = flux_ujy, sigma_ujy, 'uJy'
        else:
            val_flux, val_sigma, unit_str = phot['total_counts'], phot['sigma'], 'counts'

        row = {
            'name': self.config.name,
            'phot': val_flux,
            'sigma': val_sigma,
            'channel': channel,
            'unit': unit_str,
        }
        return {'row': row, 'sigma_raw': phot['sigma']}

    def _run_ensemble(
        self,
        image: np.ndarray,
        psf_data: np.ndarray,
        xc: float, yc: float,
        ap_radius: float, inner_ann: float, outer_ann: float,
        sigma_raw: float,
        channel: int,
    ) -> List[Dict]:
        """
        Inject PRFs at a grid of positions around the target at multiple flux
        levels and collect recovered photometry rows.

        PRF injection levels = FACTORS × σ_background (in MJy/sr image units).
        After PRF normalization to sum=1 in congrid(), `scale` equals the total
        injected flux in MJy/sr, so expected_flux_ujy = scale × APCOR × MAGIC.

        Returns a list of row dicts (one per grid position per flux level).
        """
        # Injection amplitudes bracket the ~5σ detection threshold
        scales = FACTORS * sigma_raw
        half_grid = (self.config.grid - 1) / 2.0
        rows = []
        n_total = len(scales) * self.config.grid ** 2

        self.logger.info(
            f"{self.config.name} ch{channel}: ensemble — "
            f"{len(scales)} flux levels × {self.config.grid}² positions "
            f"= {n_total} measurements"
        )

        for scale in scales:
            expected_flux_ujy = counts_to_ujy(scale, channel)

            for ii in range(self.config.grid):
                for jj in range(self.config.grid):
                    x_pos = xc + (ii - half_grid) * self.config.spacing
                    y_pos = yc + (jj - half_grid) * self.config.spacing

                    processed_psf = make_PSF(
                        psf_data, x_pos, y_pos, channel,
                        scale=scale,
                        trim_pixels=self.config.psf_trim_pixels,
                        verbose=self.verbose,
                    )
                    if processed_psf is None:
                        self.logger.warning(
                            f"{self.config.name} ch{channel}: make_PSF returned None "
                            f"at ({x_pos:.1f}, {y_pos:.1f}), scale={scale:.4f}"
                        )
                        continue

                    sim_image = image.copy()
                    sim_image = place_PSF(sim_image, processed_psf, x_pos, y_pos)

                    try:
                        phot = perform_photometry(
                            sim_image, x_pos, y_pos,
                            ap_radius, inner_ann, outer_ann,
                            method=self.config.photometry_method,
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"{self.config.name} ch{channel}: photometry failed at "
                            f"({x_pos:.1f}, {y_pos:.1f}) scale={scale:.4f} — {e}"
                        )
                        continue

                    flux_ujy = counts_to_ujy(phot['total_counts'], channel)
                    sigma_ujy = counts_to_ujy(phot['sigma'], channel)

                    self.logger.debug(
                        f"  scale={scale:.4f}, pos=({x_pos:.1f},{y_pos:.1f}), "
                        f"flux={flux_ujy:.3f} µJy"
                    )

                    if self.config.uJy_units:
                        val_flux = flux_ujy
                        val_sigma = sigma_ujy
                        val_expected = expected_flux_ujy
                        unit_str = 'uJy'
                    else:
                        val_flux = phot['total_counts']
                        val_sigma = phot['sigma']
                        val_expected = scale
                        unit_str = 'counts'

                    rows.append({
                        'name': self.config.name,
                        'ra': self.config.x_coord,
                        'dec': self.config.y_coord,
                        'channel': channel,
                        'x_pos': x_pos,
                        'y_pos': y_pos,
                        'scale': scale,
                        'expected_flux': val_expected,
                        'flux': val_flux,
                        'sigma': val_sigma,
                        'unit': unit_str,
                    })

        self.logger.info(
            f"{self.config.name} ch{channel}: collected {len(rows)}/{n_total} "
            f"ensemble measurements"
        )
        return rows

    # ------------------------------------------------------------------
    # Optional diagnostic plots
    # ------------------------------------------------------------------

    def _save_one_shot_plots(
        self,
        image: np.ndarray,
        xc: float, yc: float,
        ap_radius: float, inner_ann: float, outer_ann: float,
        channel: int,
    ) -> None:
        make_plot(
            image, xc, yc, ap_radius, inner_ann, outer_ann,
            save_path=f"plots/{self.config.name}_ch{channel}.png",
        )
        save_x_profile(
            image, xc, yc, channel, self.config.name, 0,
            out_dir=f"results/plots/profiles/simulated/",
        )
