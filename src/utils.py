import os
import csv
from astropy.io import fits

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from src.psf import PSF


def get_PSF(psf_file_path, channel=1):
    """
    Reads a PSF FITS file and returns the PSF image as a numpy array.

    Parameters
    ----------
    psf_file_path : str
        The file path to the PSF FITS file.
    channel : int
        The channel number (1, 2, 3, or 4).

    Returns
    -------
    np.ndarray
        The PSF image data.
    """
    try:
        with fits.open(psf_file_path) as hdul:
            psf_data = hdul[0].data
            return psf_data
    except FileNotFoundError:
        raise FileNotFoundError(f"PSF file not found at {psf_file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading PSF FITS file: {e}")

def get_Image(image_file_path):
    """
    Reads an astronomical image FITS file and returns the image data as a numpy array.

    Parameters
    ----------
    image_file_path : str
        The file path to the image FITS file.

    Returns
    -------
    np.ndarray
        The image data.
    """
    try:
        with fits.open(image_file_path) as hdul:
            image_data = hdul[0].data
            return image_data
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found at {image_file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading image FITS file: {e}")

def place_PSF(image_data, processed_psf_image, x, y):
    image_shape = image_data.shape
    source_x, source_y = int(x), int(y)

    psf_half_size = processed_psf_image.shape[0] // 2

    x_start = max(source_x - psf_half_size, 0)
    x_end = min(source_x + psf_half_size + 1, image_shape[1])

    y_start = max(source_y - psf_half_size, 0)
    y_end = min(source_y + psf_half_size + 1, image_shape[0])

    psf_x_start = max(psf_half_size - source_x, 0)
    psf_x_end = psf_x_start + (x_end - x_start)

    psf_y_start = max(psf_half_size - source_y, 0)
    psf_y_end = psf_y_start + (y_end - y_start)
    
    image_data[y_start:y_end, x_start:x_end] += processed_psf_image[psf_y_start:psf_y_end, psf_x_start:psf_x_end]
    return image_data

def make_PSF(psf_data, x, y, channel, norm=1.0, verbose=False,
             save_psf=False, outdir="./simulated_psf", filename=None):
    # Initialize PSF object
    psf_obj = PSF(psf_data, channel=channel)
    
    # PSF Pipeline
    if verbose:
        print(f"Original PSF shape: {psf_obj.get_psf_image().shape}")

    # Trim PSF
    trim_pixels = 50
    try:
        psf_obj.trim_psf(trim_pixels)
        if verbose:
            print(f"PSF shape after trimming by {trim_pixels} pixels: {psf_obj.get_psf_image().shape}")
    except ValueError as e:
        print(f"Error trimming PSF: {e}")
        return
    
    new_dimensions = psf_obj.compute_dimensions()
    if verbose:
        print(f"Computed new dimensions for PSF: {new_dimensions}")

    # Shift PSF
    psf_obj.shift_psf(x-int(x), y-int(y))
    if verbose:
        print(f"PSF shifted by ({x-int(x)}, {y-int(y)}) fractional pixels.")

    # Scale PSF
    psf_obj.congrid(new_dimensions, method='linear')
    if verbose:
        print(f"PSF resampled to shape: {psf_obj.get_psf_image().shape}")

    # Normalize PSF (use passed normalization)
    # psf_obj.normalize_psf(norm=norm)
    # if verbose:
    #     print(f"PSF normalized. Sum: {np.sum(psf_obj.get_psf_image()):.4f}")

    # PSF image after processing
    psf_img = psf_obj.get_psf_image()

    # Return PSF image
    if save_psf:
        if filename is None:
            filename = f"PSF_channel{channel}_x{x:.2f}_y{y:.2f}.fits"

        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, filename)

        hdu = fits.PrimaryHDU(psf_img.astype(np.float32))
        hdr = hdu.header

        hdr["CHANNEL"] = channel
        hdr["XSHIFT"] = (x - int(x), "Fractional X shift [pix]")
        hdr["YSHIFT"] = (y - int(y), "Fractional Y shift [pix]")

        if np.isfinite(norm):
            hdr["NORM"] = (float(norm), "PSF normalization")
        else:
            hdr["NORM"] = (-1.0, "PSF normalization (invalid)")
            hdr["COMMENT"] = "Original normalization was NaN or Inf"
            hdr.add_history("Normalization value was NaN or Inf")

        hdu.writeto(outpath, overwrite=True)

        if verbose:
            print(f"PSF saved to {outpath}")

    return psf_img

def save_grid_plot(sim_images_with_pos, name, channel, norm, xc, yc, ap_radius, inner_ann_radius, outer_ann_radius, filename, box_size=50):
    """
    Create and save a grid plot (n x n) of simulated images with aperture/annulus overlays.

    Parameters
    ----------
    sim_images_with_pos : list of tuples
        Each element is (simulated_image, x_pos, y_pos). The order should match the grid scanning order.
    xc, yc : float
        Center pixel coordinates of the original target (used for title only).
    ap_radius, inner_ann_radius, outer_ann_radius : float
        Radii in pixels for the aperture and annulus.
    filename : str
        Path to save the output PNG file.
    box_size : int
        Size of the square box (in pixels) to crop around each position for display.
    """
    n = int(np.ceil(np.sqrt(len(sim_images_with_pos))))
    # increase vertical spacing between rows and horizontal spacing between cols
    fig, axes = plt.subplots(n, n, figsize=(4*n, 4*n), gridspec_kw={'hspace': 0.45, 'wspace': 0.25})
    axes = np.array(axes).reshape(n, n)

    # Prepare all sub-images but keep the crop centered on the central target (xc, yc)
    # so the simulated sources will appear to move within a fixed frame.
    half_box = box_size // 2
    sub_images = []
    center_cx = int(xc)
    center_cy = int(yc)
    for sim_image, x_pos, y_pos in sim_images_with_pos:
        # use the central coordinates to determine the crop window for every panel
        y0 = max(center_cy - half_box, 0)
        y1 = min(center_cy + half_box, sim_image.shape[0])
        x0 = max(center_cx - half_box, 0)
        x1 = min(center_cx + half_box, sim_image.shape[1])
        sub = sim_image[y0:y1, x0:x1]

        # compute position of the simulated source relative to the fixed crop
        sub_cx = x_pos - x0
        sub_cy = y_pos - y0
        sub_images.append((sub, sub_cx, sub_cy))

    if channel == 1 or channel == 2:
        vmax = 0.4
    elif channel == 3 or channel == 4:
        vmax = 10.0

    # Plot each subplot
    idx = 0
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if idx < len(sub_images):
                sub, sub_cx, sub_cy = sub_images[idx]
                im = ax.imshow(sub, origin='lower', cmap='gray', interpolation='nearest', vmax=vmax)
                # Draw aperture and annuli
                aperture = Circle((sub_cx, sub_cy), ap_radius, edgecolor='red', facecolor='none', lw=1.2)
                ann_inner = Circle((sub_cx, sub_cy), inner_ann_radius, edgecolor='yellow', facecolor='none', lw=1.0, linestyle='--')
                ann_outer = Circle((sub_cx, sub_cy), outer_ann_radius, edgecolor='yellow', facecolor='none', lw=1.0, linestyle='--')
                ax.add_patch(aperture)
                ax.add_patch(ann_inner)
                ax.add_patch(ann_outer)
                ax.plot(sub_cx, sub_cy, marker='+', color='cyan', markersize=8)
                ax.set_title(f"pos {idx+1}: ({sim_images_with_pos[idx][1]:.1f}, {sim_images_with_pos[idx][2]:.1f})")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')
            idx += 1

    fig.suptitle(f"Simulated PRFs on {name} at center ({xc:.3f}, {yc:.3f}) \n IRAC: {channel}, Normalization: {norm}", fontsize=16)

    # Reserve space on the right for a vertical colorbar that won't overlap the subplots
    # Adjust the subplot area to leave room (right edge smaller to make space)
    fig.subplots_adjust(right=0.86)

    # Create an axis for the colorbar to the right of the panels and attach the colorbar to it
    # cax position: [left, bottom, width, height] in figure coordinates
    cbar_ax = fig.add_axes([0.89, 0.12, 0.02, 0.76])
    fig.colorbar(im, cax=cbar_ax)

    # Use tight_layout but respect the reserved right-side area; shrink top to leave more room
    plt.tight_layout(rect=[0, 0.03, 0.86, 0.92])
    # Ensure directory exists
    out_dir = os.path.dirname(filename)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(filename, dpi=150)
    plt.close(fig)

def make_plot(simulated_image, xc, yc, ap_radius, inner_ann_radius, outer_ann_radius, safe=False, safe_path=None):
    box_size = 50
    half_box = box_size // 2
    cx, cy = int(xc), int(yc)

    y0 = max(cy - half_box, 0)
    y1 = min(cy + half_box, simulated_image.shape[0])
    x0 = max(cx - half_box, 0)
    x1 = min(cx + half_box, simulated_image.shape[1])

    sub_image = simulated_image[y0:y1, x0:x1]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(sub_image, origin='lower', cmap='gray', interpolation='nearest', vmax=0.4)
    ax.set_title(f"PRF placement ({xc:.2f}, {yc:.2f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Coordinates in the sub-image frame
    sub_cx = cx - x0
    sub_cy = cy - y0

    # Aperture (solid) and annuli (dashed)
    aperture = Circle((sub_cx, sub_cy), ap_radius, edgecolor='red', facecolor='none', lw=1.5, linestyle='-')
    ann_inner = Circle((sub_cx, sub_cy), inner_ann_radius, edgecolor='yellow', facecolor='none', lw=1.2, linestyle='--')
    ann_outer = Circle((sub_cx, sub_cy), outer_ann_radius, edgecolor='yellow', facecolor='none', lw=1.2, linestyle='--')

    ax.add_patch(aperture)
    ax.add_patch(ann_inner)
    ax.add_patch(ann_outer)

    # Mark the exact center
    ax.plot(sub_cx, sub_cy, marker='+', color='cyan', markersize=10, markeredgewidth=1.5)
    plt.tight_layout()
    if safe:
        plt.savefig(f"{safe_path}")
    plt.close(fig)
    # plt.show()

def save_x_profile(image, x_pos, y_pos, channel, name, norm, out_dir=None, half_width=10):
    """
    Save an X vs Intensity plot for a horizontal cut through the image at a fixed Y.

    The function samples the image along X while holding Y fixed at the nearest
    integer pixel to `y_pos`. It collects intensities for X in [x_pos-half_width, x_pos+half_width]
    (clipped to image bounds) and saves a PNG with metadata in the filename.

    Parameters
    ----------
    image : 2D numpy.ndarray
        The image array to sample.
    x_pos, y_pos : float
        Pixel coordinates (floating allowed). y_pos determines the row to sample;
        x_pos is the center of the sampled X-range.
    channel : int
        IRAC channel number (used for labeling/filename).
    name : str
        Target name (used for labeling/filename).
    norm : float
        Normalization value (used for labeling/filename).
    out_dir : str or None
        Directory to save the PNG. If None, the function uses a `plots` directory
        in the current working directory.
    half_width : int
        Half-width in pixels for the sampled range around x_pos (default 10).

    Returns
    -------
    str
        The absolute path to the saved PNG file.
    """
    # Validate image
    if image is None or image.ndim != 2:
        raise ValueError("`image` must be a 2D numpy array")

    h, w = image.shape
    # clamp the y coordinate to valid row indices
    cy = int(round(y_pos))
    cy = max(0, min(cy, h - 1))

    cx = int(round(x_pos))

    x0 = max(cx - half_width, 0)
    x1 = min(cx + half_width, w - 1)

    xs = np.arange(x0, x1 + 1)
    intensities = image[cy, xs]

    # Prepare plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, intensities, marker='o', linestyle='-')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Intensity')
    ax.set_title(f"X profile at Y={y_pos:.2f} for {name}  (ch{channel}, norm={norm})")
    ax.grid(True, linestyle='--', alpha=0.6)

    # Prepare output path
    if out_dir is None:
        out_dir = 'plots/profiles/'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Sanitize numeric formatting for filename
    fname = os.path.join(out_dir, f"{name}_ch{channel}_norm{norm}_xprof_x{int(round(x_pos))}_y{int(round(y_pos))}.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)

    return os.path.abspath(fname)

def save_results(filename, data):
    header_list = [
    'name', 'ra', 'dec', 'channel', 'x_pos', 'y_pos', 
    'expected_flux', 'phot_flux_(µJy)', 'phot_sigma_(µJy)'
    ]

    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(header_list)

        writer.writerow(data)

def save_one_shot(filename, name, phot, sigma, channel):
    header_list = [
        'name', 'phot', 'sigma', 'channel'
    ]

    file_exists = os.path.isfile(filename)
    data = [name, phot, sigma, channel]
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(header_list)

        writer.writerow(data)
