import numpy as np
import numbers


def get_annulus(arr, xc, yc, rin, rout, justdex=False):
    """
    Extracts values or indices from an array within a specified annulus.

    This function is a Python/NumPy translation of the IDL function get_annulus.
    It returns the pixel values or indices where the pixel centers are
    between rin and rout from the center of the given xc, yc pixel.

    Parameters
    ----------
    arr : np.ndarray
        The input 2D array.
    xc : number
        The x-coordinate (column index) of the annulus center.
    yc : number
        The y-coordinate (row index) of the annulus center.
    rin : number
        The inner radius of the annulus.
    rout : number
        The outer radius of the annulus.
    justdex : bool, optional
        If True, returns the flat 1D indices of the pixels within the annulus,
        replicating the behavior of IDL's `where()`.
        If False (default), returns a 1D array of the values of the pixels
        within the annulus.

    Returns
    -------
    np.ndarray
        If justdex is False, a 1D array of the values in the annulus.
        If justdex is True, a 1D array of the flat indices in the annulus.
        
    Raises
    ------
    TypeError
        If 'arr' is not a NumPy array or if xc, yc, rin, or rout are not numbers.
    ValueError
        If 'arr' is not a 2D array.
    """
    # --- Input Validation (Pythonic approach) ---
    # In Python, it's better to raise errors than return error codes like -1.
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input 'arr' must be a NumPy array.")
    if arr.ndim != 2:
        raise ValueError("Input 'arr' must be a 2D array.")
    if not all(isinstance(val, numbers.Number) for val in [xc, yc, rin, rout]):
        raise TypeError("xc, yc, rin, and rout must be numbers.")

    # --- Coordinate Grid Creation ---
    # This is the NumPy equivalent of the nested for-loops in the IDL code.
    # np.indices creates two arrays; one holding the y-coordinate (row) of
    # each pixel, and the other holding the x-coordinate (column).
    ny, nx = arr.shape
    y_indices, x_indices = np.indices((ny, nx))

    # --- Find Pixels within the Annulus ---
    # Calculate the radial distance of each pixel from the center (xc, yc).
    # This is a vectorized operation, which is much more efficient than a loop.
    dist = np.sqrt((x_indices - xc)**2 + (y_indices - yc)**2)

    # Create a boolean mask where the condition for the annulus is met.
    # This is equivalent to the 'where' clause in the IDL function.
    mask = (dist >= rin) & (dist <= rout)

    # --- Return Value ---
    if justdex:
        # If justdex is True, return the flat 1D indices of the pixels.
        # np.flatnonzero is the direct equivalent of IDL's where() on a mask.
        return np.flatnonzero(mask)
    else:
        # If justdex is False (default), return the values from the array
        # using the boolean mask.
        return arr[mask]


def circ_apphot(im, xc, yc, raper, t_exp, bgndwidth, quiet, rbackin):
    """
    Performs circular aperture photometry.

    This function is a Python translation of the IDL procedure `circapphot`.
    It calculates the total counts, instrumental magnitude, and associated errors
    for a source in an image, given an aperture radius. The background is
    estimated from an annulus near the edge of the image.

    Parameters
    ----------
    im : np.ndarray
        The input 2D image array.
    xc : number
        The x-coordinate (column index) of the aperture center.
    yc : number
        The y-coordinate (row index) of the aperture center.
    raper : number
        The radius of the circular aperture for photometry.
    t_exp : number
        The exposure time.
    bgndwidth : number, optional
        The width of the background annulus in pixels. Default is 5.
    quiet : bool, optional
        If True, suppresses printed output. Default is False.
    rbackin : number, optional
        The inner radius of the background annulus. If None (default), it is
        calculated automatically.

    Returns
    -------
    dict
        A dictionary containing the photometry results:
        - 'total_counts': Background-subtracted counts (tot).
        - 'instrumental_mag': Instrumental magnitude (imag).
        - 'mag_error': The magnitude error (magerr).
        - 'n_pixels': Number of pixels in the aperture (npix).
        - 'bg_stddev': Standard deviation of the background pixels (pixsig).
        - 'sigma': The photometric error in counts (sigma).
        - 'bg_level': The median background level (bgndlvl).
    """
    # --- Input Validation ---
    if not isinstance(im, np.ndarray):
        raise TypeError("Input 'im' must be a NumPy array.")
    if im.ndim != 2:
        raise ValueError("Input 'im' must be a 2D array.")
    if not all(isinstance(val, numbers.Number) for val in [xc, yc, raper, t_exp]):
         raise TypeError("xc, yc, raper, and t_exp must be numbers.")

    if not quiet:
        print('Photometry taken with approx. circular apertures and edge backgrounds.')

    # --- Find counts in aperture ---
    # Equivalent to: ann=get_annulus(im,xc,yc,0,raper)
    ann = get_annulus(im, xc, yc, 0, raper)
    # Equivalent to: npix=n_elements(ann)
    npix = ann.size

    # --- Find background ---
    # The original IDL code determines a background radius based on image size.
    # Note: IDL's size() returns dimensions as [cols, rows] for a 2D array,
    # while NumPy's .shape attribute is (rows, cols).
    ny, nx = im.shape

    if rbackin is None:
        # This logic mirrors: rguess=x1/2-3-5
        rguess = nx / 2.0 - 8.0
        rback = rguess
    else:
        # This mirrors: if keyword_set(rbackin) then rback=rbackin
        rback = rbackin

    bgnd = get_annulus(im, xc, yc, rback, rback + bgndwidth)

    # Handle case where background annulus is empty (e.g., on a small image)
    if bgnd.size == 0:
        if not quiet:
            print("Warning: Background annulus is empty. Setting background to 0.")
        bgndlvl = 0.0
        pixsig = 0.0
    else:
        # Equivalent to: bgndlvl=median(bgnd) and pixsig=stdev(bgnd)
        bgndlvl = np.median(bgnd)
        pixsig = np.std(bgnd)

    if not quiet:
        print(f'background radii in,out,level= {rback:.2f}, {rback + bgndwidth:.2f}, {bgndlvl:.4f}')

    # --- Final calculations ---
    # sigma=sqrt(npix)*pixsig
    sigma = np.sqrt(npix) * pixsig
    # tot=total(ann)-npix*bgndlvl
    tot = np.sum(ann) - npix * bgndlvl

    # Calculate magnitude and error, with checks for non-positive counts
    if tot > 0 and t_exp > 0:
        # imag=-2.5*alog10(tot/t_exp)
        imag = -2.5 * np.log10(tot / t_exp)

        # Calculate error, ensuring arguments to log10 are positive
        term_plus_sigma = tot + sigma
        term_minus_sigma = tot - sigma

        if term_plus_sigma > 0 and term_minus_sigma > 0:
            # m2=-2.5*alog10((tot+sigma)/t_exp)
            m2 = -2.5 * np.log10(term_plus_sigma / t_exp)
            # m1=-2.5*alog10((tot-sigma)/t_exp)
            m1 = -2.5 * np.log10(term_minus_sigma / t_exp)
            # magerr=abs(m2-m1)/2.
            magerr = abs(m2 - m1) / 2.0
        else:
            # Cannot calculate error if flux +/- sigma is not positive
            magerr = np.nan
    else:
        # Cannot calculate magnitude for non-positive total counts
        imag = np.nan
        magerr = np.nan

    results = {
        'total_counts': tot,
        'instrumental_mag': imag,
        'mag_error': magerr,
        'n_pixels': npix,
        'bg_stddev': pixsig,
        'sigma': sigma,
        'bg_level': bgndlvl
    }

    return results