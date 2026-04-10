import math
from typing import Tuple

import numpy as np
from skimage.transform import resize

from utils.constants import MOSAIC_PIXEL_SCALE_ARCSEC


class PSF:
    def __init__(self, psf_image: np.ndarray, channel: int = 1) -> None:
        """
        Initializes the PSF object with a PSF image.

        Args:
            psf_image: The 2D image of the Point Spread Function.
            channel: The channel number (1, 2, 3, or 4).
        """
        self.psf_image = psf_image
        self.channel = channel
        self.original_shape = psf_image.shape
        self.trimmed_shape = psf_image.shape
        self.current_shape = self.psf_image.shape
        self.scale = self.get_scale(channel)

    def get_scale(self, channel: int) -> float:
        """
        Returns the scale factor based on the channel.

        Args:
            channel: The channel number (1, 2, 3, or 4).

        Returns:
            The scale factor.
        """
        if channel == 1:
            p_prf = 1.221
        elif channel == 2:
            p_prf = 1.213
        elif channel == 3:
            p_prf = 1.222
        elif channel == 4:
            p_prf = 1.220
        else:
            # Default fallback if channel is invalid/unknown, though typically checked upstream
            p_prf = 1.221

        # The APEX PRF is oversampled x100 relative to the native IRAC pixel scale.
        # Scale factor = (native_pix_scale / 100) / mosaic_pix_scale
        # This maps each oversampled PRF pixel to the corresponding fraction of a mosaic pixel.
        return (p_prf / 100.0) / MOSAIC_PIXEL_SCALE_ARCSEC

    def trim_psf(self, trim_pixels: int) -> None:
        """
        Trims the PSF image equally from all sides.

        Args:
            trim_pixels: The number of pixels to trim from each side (top, bottom, left, right).

        Raises:
            ValueError: If trim_pixels is invalid or too large.
        """
        if not isinstance(trim_pixels, int) or trim_pixels < 0:
            raise ValueError("trim_pixels must be a non-negative integer.")

        if (
            2 * trim_pixels >= self.psf_image.shape[0]
            or 2 * trim_pixels >= self.psf_image.shape[1]
        ):
            raise ValueError(
                "Trim amount is too large for the current PSF image dimensions."
            )

        self.psf_image = self.psf_image[
            trim_pixels:-trim_pixels, trim_pixels:-trim_pixels
        ]
        self.trimmed_shape = self.psf_image.shape

    def compute_dimensions(self) -> int:
        """
        Computes the target dimension for the PSF image after scaling.

        Returns:
            The new dimension size (assumes square scaling logic roughly).
        """
        # The logic here seems to differentiate rounding based on channel
        if self.channel == 1 or self.channel == 3:
            return math.ceil(self.psf_image.shape[0] * self.scale)
        elif self.channel == 2 or self.channel == 4:
            return math.floor(self.psf_image.shape[0] * self.scale)
        else:
            return int(self.psf_image.shape[0] * self.scale)

    def shift_psf(self, frac_x: float, frac_y: float) -> None:
        """
        Shifts the PSF image by fractional pixel amounts using rolling (integer shift approximation).

        Args:
            frac_x: Fractional shift in the x-direction.
            frac_y: Fractional shift in the y-direction.
        """
        # Note: This implementation does an integer roll based on the scaled fraction.
        # It might be an approximation.
        shift_x = int(frac_x / self.scale)
        shift_y = int(frac_y / self.scale)
        self.psf_image = np.roll(self.psf_image, shift_x, axis=1)
        self.psf_image = np.roll(self.psf_image, shift_y, axis=0)

    def normalize_psf(self, norm: float = 1.0) -> None:
        """
        Normalizes the PSF image so that its sum equals `norm`.

        Args:
            norm: Normalization factor (target sum).
        """
        total = np.sum(self.psf_image)
        if total != 0:
            self.psf_image /= total
            self.psf_image *= norm

    def congrid(self, new_shape: int, method: str = 'linear') -> None:
        """
        Resample the PSF image to a new shape using interpolation.

        Args:
            new_shape: Target dimension (creates a square new_shape x new_shape).
            method: Interpolation method: 'nearest', 'linear', or 'cubic'.
        """
        # The original code passed 'new_dimensions' which is an int, but used it as (new_rows, new_cols)
        # So we assume square resizing if a single int is passed, or tuple if supported.
        if isinstance(new_shape, int):
            new_rows = new_shape
            new_cols = new_shape
        elif isinstance(new_shape, tuple) and len(new_shape) == 2:
            new_rows, new_cols = new_shape
        else:
            # Fallback
            new_rows = new_shape
            new_cols = new_shape

        order_map = {'nearest': 0, 'linear': 1, 'cubic': 3}
        order = order_map.get(method, 1)

        # resize expects (rows, cols).
        # skimage.transform.resize preserves average intensity per pixel, NOT total flux.
        # After downsampling the oversampled APEX PRF, the pixel count drops dramatically
        # (e.g., 229² → 5²), so the sum of the array changes by N_new/N_old.
        # We renormalize immediately after to ensure sum=1, so that `scale` in make_PSF
        # directly represents the total injected flux in image units.
        self.psf_image = resize(
            self.psf_image,
            (new_rows, new_cols),
            order=order,
            mode='reflect',
            anti_aliasing=False,
        )
        self.normalize_psf(norm=1.0)

    def get_psf_image(self) -> np.ndarray:
        """
        Returns the current PSF image.

        Returns:
            The 2D image of the Point Spread Function.
        """
        return self.psf_image

    def plot_psf_image(self) -> None:
        """
        Plots the PSF image using matplotlib.
        """
        import matplotlib.pyplot as plt

        plt.imshow(self.psf_image, cmap='viridis', origin='lower')
        plt.colorbar(label='Intensity')
        plt.title(f'PSF Image (Channel {self.channel})')
        plt.xlabel('X Pixels')
        plt.ylabel('Y Pixels')
        plt.show()
