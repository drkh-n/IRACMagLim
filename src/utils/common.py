# This module is kept for backward compatibility only.
# Import directly from the specific modules instead:
#   utils.io       — FITS I/O, save_rows
#   utils.psf_utils — make_PSF, place_PSF
#   utils.plotting  — make_plot, save_grid_plot, save_x_profile

from utils.io import get_PSF, load_fits_image, save_rows          # noqa: F401
from utils.psf_utils import make_PSF, place_PSF                   # noqa: F401
from utils.plotting import make_plot, save_grid_plot, save_x_profile  # noqa: F401

# Legacy alias kept for any code that still calls save_data_multi_format with a single dict.
def save_data_multi_format(base_filename, data, formats, headers):
    save_rows(base_filename, [data], formats, headers)
