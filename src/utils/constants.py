import numpy as np

# Pixel scale of the super-resolved IRAC mosaics (arcsec/pix).
# This is NOT the native IRAC detector pixel scale; the mosaics are
# drizzled to 0.6 arcsec/pix. Used in PSF.get_scale() and aperture scaling.
MOSAIC_PIXEL_SCALE_ARCSEC = 0.6

# Native IRAC detector pixel scales per channel (arcsec/pix).
# Source: IRAC Instrument Handbook Table 2.2.
# Channels 1–4 correspond to 3.6, 4.5, 5.8, 8.0 µm.
PSIZE_ASEC = [1.221, 1.213, 1.222, 1.220]

# Aperture correction factors per channel for a ~2 native-pixel aperture radius.
# Converts aperture flux to total flux: F_total = F_aperture * APCOR.
# Source: IRAC Instrument Handbook Table 4.7 (Warm Mission corrections).
APCOR = [1.125, 1.120, 1.135, 1.221]

# Unit conversion: MJy/sr per mosaic pixel → µJy.
# Derivation: pixel_solid_angle = (MOSAIC_PIXEL_SCALE_ARCSEC * π/648000)² sr
#             MAGIC_NUMBER = pixel_solid_angle * 1e12  [MJy → Jy → µJy]
#           = (0.6 * π/648000)² * 1e12 ≈ 8.464 µJy / (MJy/sr · pixel)
# The same value applies to all channels because all mosaics share the 0.6″/pix scale.
MAGIC_NUMBER = 8.47

# PRF injection flux levels as multiples of the one-shot photometric noise σ.
# Chosen to bracket the ~5σ detection threshold (3.8σ below, 7.0σ above).
# The scatter of recovered fluxes at each level gives σ_ensemble, and the
# 5σ upper limit is 5 × mean(σ_ensemble), valid for background-dominated sources.
FACTORS = np.array([3.8, 7.0])

def counts_to_ujy(counts: float, channel: int) -> float:
    """
    Convert circ_apphot total_counts (sum of MJy/sr pixel values) to µJy.

    Applies aperture correction and pixel solid-angle unit conversion.
    Use this everywhere instead of inline `counts * APCOR[ch-1] * MAGIC_NUMBER`.

    Parameters
    ----------
    counts : float
        Background-subtracted aperture sum in MJy/sr units (from circ_apphot).
    channel : int
        IRAC channel number (1–4).

    Returns
    -------
    float
        Flux in µJy.
    """
    return counts * APCOR[channel - 1] * MAGIC_NUMBER