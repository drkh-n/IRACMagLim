"""
PSF processing and image injection utilities for PhotometryPy.
"""
import logging
from typing import Optional

import numpy as np
from astropy.io import fits

from core.psf import PSF

logger = logging.getLogger(__name__)


def make_PSF(
    psf_data: np.ndarray,
    x: float,
    y: float,
    channel: int,
    scale: float = 1.0,
    trim_pixels: int = 50,
    verbose: bool = False,
    save_psf: bool = False,
    outdir: str = "./simulated_psf",
    filename: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Process a raw APEX PRF (trim → shift → congrid → scale) and return the
    result ready for injection into a mosaic image.

    After `congrid`, the PSF is renormalized to sum=1 (inside `PSF.congrid`),
    so `scale` directly represents the total injected flux in image units.

    Parameters
    ----------
    psf_data : np.ndarray
        Raw oversampled PRF image (e.g. APEX x100 IRAC PRF).
    x, y : float
        Target position in mosaic pixels (used for sub-pixel shift).
    channel : int
        IRAC channel (1–4), sets the pixel scale ratio.
    scale : float
        Total flux amplitude in image units (MJy/sr).
    trim_pixels : int
        Pixels to remove from each edge of the raw PRF before downsampling.
    verbose : bool
        Log extra debug info.
    save_psf : bool
        If True, write the processed PSF to a FITS file.
    outdir : str
        Directory for saved PSF files.
    filename : str, optional
        Filename for saved PSF (auto-generated if None).

    Returns
    -------
    np.ndarray or None
        Processed PSF image, or None if processing fails.
    """
    import os

    psf_obj = PSF(psf_data, channel=channel)
    logger.debug(f"Raw PRF shape: {psf_obj.get_psf_image().shape}")

    try:
        psf_obj.trim_psf(trim_pixels)
    except ValueError as e:
        logger.error(f"PRF trim failed (trim_pixels={trim_pixels}): {e}")
        return None

    logger.debug(f"PRF shape after trim: {psf_obj.get_psf_image().shape}")

    new_dim = psf_obj.compute_dimensions()
    logger.debug(f"Target mosaic dimension: {new_dim}")

    psf_obj.shift_psf(x - int(x), y - int(y))
    logger.debug(f"Sub-pixel shift: ({x - int(x):.3f}, {y - int(y):.3f})")

    # congrid resamples to mosaic pixel scale and normalizes sum to 1.
    psf_obj.congrid(new_dim, method='linear')
    logger.debug(f"PRF shape after congrid: {psf_obj.get_psf_image().shape}")

    psf_obj.psf_image *= scale
    psf_img = psf_obj.get_psf_image()

    if save_psf:
        if filename is None:
            filename = f"PSF_channel{channel}_x{x:.2f}_y{y:.2f}.fits"
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, filename)
        hdu = fits.PrimaryHDU(psf_img.astype(np.float32))
        hdu.header['CHANNEL'] = channel
        hdu.header['XSHIFT'] = (x - int(x), 'Fractional X shift [pix]')
        hdu.header['YSHIFT'] = (y - int(y), 'Fractional Y shift [pix]')
        hdu.header['SCALE'] = (float(scale) if np.isfinite(scale) else -1.0, 'PRF scale')
        hdu.writeto(outpath, overwrite=True)
        logger.debug(f"Saved processed PRF to {outpath}")

    return psf_img


def place_PSF(
    image_data: np.ndarray,
    processed_psf_image: np.ndarray,
    x: float,
    y: float,
) -> np.ndarray:
    """
    Add a processed PRF into a mosaic image at position (x, y).

    Parameters
    ----------
    image_data : np.ndarray
        Target mosaic image (modified in-place).
    processed_psf_image : np.ndarray
        PRF array to inject (already trimmed, shifted, and scaled).
    x, y : float
        Center position in mosaic pixels.

    Returns
    -------
    np.ndarray
        Modified image.
    """
    h, w = image_data.shape
    cx, cy = int(x), int(y)
    half = processed_psf_image.shape[0] // 2

    x0 = max(cx - half, 0)
    x1 = min(cx + half + 1, w)
    y0 = max(cy - half, 0)
    y1 = min(cy + half + 1, h)

    px0 = max(half - cx, 0)
    px1 = px0 + (x1 - x0)
    py0 = max(half - cy, 0)
    py1 = py0 + (y1 - y0)

    image_data[y0:y1, x0:x1] += processed_psf_image[py0:py1, px0:px1]
    return image_data
