"""
Diagnostic plotting utilities for IRACMagLim.
"""
import logging
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

logger = logging.getLogger(__name__)


def make_plot(
    image: np.ndarray,
    xc: float,
    yc: float,
    ap_radius: float,
    inner_ann_radius: float,
    outer_ann_radius: float,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a cutout of the image with aperture and annulus overlaid.

    Parameters
    ----------
    image : np.ndarray
    xc, yc : float
        Source center in mosaic pixels.
    ap_radius, inner_ann_radius, outer_ann_radius : float
        Aperture and annulus radii in mosaic pixels.
    save_path : str, optional
        If given, save to this path instead of showing interactively.
    """
    box_size = 50
    half = box_size // 2
    cx, cy = int(xc), int(yc)

    y0 = max(cy - half, 0)
    y1 = min(cy + half, image.shape[0])
    x0 = max(cx - half, 0)
    x1 = min(cx + half, image.shape[1])
    sub = image[y0:y1, x0:x1]
    sub_cx, sub_cy = xc - x0, yc - y0

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(sub, origin='lower', cmap='gray', interpolation='nearest', vmax=0.4)
    ax.set_title(f"PRF placement ({xc:.2f}, {yc:.2f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for r, color, style in [
        (ap_radius, 'red', '-'),
        (inner_ann_radius, 'yellow', '--'),
        (outer_ann_radius, 'yellow', '--'),
    ]:
        ax.add_patch(Circle((sub_cx, sub_cy), r, edgecolor=color,
                             facecolor='none', lw=1.5, linestyle=style))
    ax.plot(sub_cx, sub_cy, '+', color='cyan', markersize=10, markeredgewidth=1.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)


def save_grid_plot(
    sim_images_with_pos: List[Tuple[np.ndarray, float, float]],
    name: str,
    channel: int,
    norm: float,
    xc: float,
    yc: float,
    ap_radius: float,
    inner_ann_radius: float,
    outer_ann_radius: float,
    filename: str,
    box_size: int = 50,
) -> None:
    """
    Save a grid of cutout images showing all PRF injection positions.
    """
    n = int(np.ceil(np.sqrt(len(sim_images_with_pos))))
    fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n),
                             gridspec_kw={'hspace': 0.45, 'wspace': 0.25})
    axes = np.array(axes).reshape(n, n)

    half = box_size // 2
    cx0, cy0 = int(xc), int(yc)
    vmax = 0.4 if channel in (1, 2) else 10.0

    idx = 0
    im_ref = None
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if idx < len(sim_images_with_pos):
                sim_img, x_pos, y_pos = sim_images_with_pos[idx]
                y0 = max(cy0 - half, 0)
                y1 = min(cy0 + half, sim_img.shape[0])
                x0 = max(cx0 - half, 0)
                x1 = min(cx0 + half, sim_img.shape[1])
                sub = sim_img[y0:y1, x0:x1]
                sub_cx = x_pos - x0
                sub_cy = y_pos - y0

                im_ref = ax.imshow(sub, origin='lower', cmap='gray',
                                   interpolation='nearest', vmax=vmax)
                for r, color, style in [
                    (ap_radius, 'red', '-'),
                    (inner_ann_radius, 'yellow', '--'),
                    (outer_ann_radius, 'yellow', '--'),
                ]:
                    ax.add_patch(Circle((sub_cx, sub_cy), r, edgecolor=color,
                                        facecolor='none', lw=1.2, linestyle=style))
                ax.plot(sub_cx, sub_cy, '+', color='cyan', markersize=8)
                ax.set_title(f"pos {idx+1}: ({x_pos:.1f}, {y_pos:.1f})")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')
            idx += 1

    fig.suptitle(
        f"Simulated PRFs on {name} at ({xc:.3f}, {yc:.3f})\nIRAC: {channel}, Norm: {norm}",
        fontsize=16,
    )
    if im_ref is not None:
        fig.subplots_adjust(right=0.86)
        cbar_ax = fig.add_axes([0.89, 0.12, 0.02, 0.76])
        fig.colorbar(im_ref, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0.03, 0.86, 0.92])
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def save_x_profile(
    image: np.ndarray,
    x_pos: float,
    y_pos: float,
    channel: int,
    name: str,
    norm: float,
    out_dir: Optional[str] = None,
    half_width: int = 10,
) -> str:
    """
    Save a horizontal intensity profile through the image at fixed y_pos.

    Returns
    -------
    str
        Absolute path to the saved plot.
    """
    if image is None or image.ndim != 2:
        raise ValueError("`image` must be a 2D numpy array")

    h, w = image.shape
    cy = max(0, min(int(round(y_pos)), h - 1))
    cx = int(round(x_pos))
    x0 = max(cx - half_width, 0)
    x1 = min(cx + half_width, w - 1)

    xs = np.arange(x0, x1 + 1)
    intensities = image[cy, xs]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, intensities, marker='o', linestyle='-')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Intensity')
    ax.set_title(f"X profile at Y={y_pos:.2f} | {name} ch{channel} norm={norm}")
    ax.grid(True, linestyle='--', alpha=0.6)

    if out_dir is None:
        out_dir = 'plots/profiles/'
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(
        out_dir,
        f"{name}_ch{channel}_norm{norm}_xprof_x{cx}_y{cy}.png",
    )
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return os.path.abspath(fname)
