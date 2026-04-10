"""
FITS file I/O and tabular data persistence for PhotometryPy.
"""
import csv
import logging
import os
from typing import Dict, List

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)


def load_fits_image(file_path: str):
    """
    Open a FITS file once and return (image_data, wcs, header).

    Loading in one call avoids opening the file twice (once for WCS,
    once for data), which was the previous pattern.

    Parameters
    ----------
    file_path : str

    Returns
    -------
    tuple: (image_data: np.ndarray, wcs: WCS, header: fits.Header)

    Raises
    ------
    FileNotFoundError
    RuntimeError
    """
    try:
        with fits.open(file_path) as hdul:
            data = np.array(hdul[0].data, dtype=np.float64)
            header = hdul[0].header
            wcs = WCS(header)
        return data, wcs, header
    except FileNotFoundError:
        raise FileNotFoundError(f"FITS file not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading FITS file {file_path}: {e}") from e


def get_PSF(psf_file_path: str, channel: int = 1) -> np.ndarray:
    """
    Read a PSF FITS file and return the image array.

    Parameters
    ----------
    psf_file_path : str
    channel : int

    Returns
    -------
    np.ndarray

    Raises
    ------
    FileNotFoundError
    RuntimeError
    """
    try:
        with fits.open(psf_file_path) as hdul:
            return np.array(hdul[0].data)
    except FileNotFoundError:
        raise FileNotFoundError(f"PSF file not found: {psf_file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading PSF FITS file: {e}") from e


def save_rows(
    base_filename: str,
    rows: List[Dict],
    formats: List[str],
    headers: List[str],
    mode: str = 'a',
) -> None:
    """
    Write a list of row dicts to CSV and/or coldat files.

    Parameters
    ----------
    base_filename : str
        Path without extension (e.g. 'results/photometry/ensemble_data').
    rows : list of dict
        Each dict maps header name → value.
    formats : list of str
        Subset of ['csv', 'coldat'].
    headers : list of str
        Column order. Rows missing a key are written as empty string.
    mode : str
        'a' to append (default), 'w' to overwrite.
    """
    if not rows:
        return

    dir_name = os.path.dirname(base_filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    for fmt in formats:
        if fmt == 'csv':
            fname = f"{base_filename}.csv"
            delimiter = ','
        elif fmt == 'coldat':
            fname = f"{base_filename}.coldat"
            delimiter = '\t'
        else:
            logger.warning(f"Unknown format '{fmt}', skipping.")
            continue

        file_exists = os.path.isfile(fname) and mode == 'a'
        try:
            with open(fname, mode, newline='') as f:
                writer = csv.writer(f, delimiter=delimiter)
                if not file_exists:
                    if fmt == 'coldat':
                        f.write('# ' + '\t'.join(headers) + '\n')
                    else:
                        writer.writerow(headers)
                for row in rows:
                    writer.writerow([row.get(h, '') for h in headers])
        except OSError as e:
            logger.error(f"Error writing to {fname}: {e}")
