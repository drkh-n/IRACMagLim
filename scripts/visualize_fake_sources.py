from astropy.wcs import WCS
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from src.idl_circapphot import circ_apphot
from src.utils import get_PSF, get_Image, place_PSF, make_PSF

FACTORS = np.array([3.8, 7.0])
APCOR = [1.125, 1.120, 1.135, 1.221]
PSIZE_ASEC = [1.221, 1.213, 1.222, 1.220]

def visualize_ensemble_context(image_data, psf_data, xc, yc, channel, name, sigma_level, 
                               configs, aperture_props, save_path=None):
    """
    Visualizes the target with ALL ensemble sources injected simultaneously.
    
    Parameters
    ----------
    image_data : np.ndarray
        The original science image.
    psf_data : np.ndarray
        The raw PSF data.
    xc, yc : float
        Pixel coordinates of the central target.
    channel : int
        IRAC channel.
    sigma_level : float
        The sigma scaling factor (e.g., 3.8 or 7.0).
    configs : dict
        Configuration dictionary containing 'grid' and 'spacing'.
    aperture_props : dict
        Dictionary containing 'ap_radius', 'inner_ann_radius', 'outer_ann_radius'.
    """
    
    # 1. Create a copy of the image to inject ALL sources
    viz_image = image_data.copy()
    
    grid_size = configs['grid']
    spacing = configs['spacing']
    
    # We need to calculate the injection flux scale based on the sigma level
    # Assuming we pass the calculated scale or derive it. For visualization, 
    # we usually need the 'scale' variable from the ensemble loop.
    # If you don't have the exact flux scale, you can pass an approximate one 
    # or the actual calculated 'scale' from ensemble_photometry.py
    
    # List to store grid positions for plotting crosses
    # grid_positions = []

    # # 2. Inject all sources
    # print(f"Generating context image for Sigma={sigma_level}...")
    # for ii in range(grid_size):
    #     for jj in range(grid_size):
    #         x_offset = (ii - 1) * spacing
    #         y_offset = (jj - 1) * spacing
    #         x_pos = xc + x_offset
    #         y_pos = yc + y_offset
            
    #         grid_positions.append((x_pos, y_pos))
            
    #         # Skip the center if you don't want to double-inject on the target
    #         # (Optional: usually ensemble grid includes 0,0 offset)
    #         if x_offset == 0 and y_offset == 0:
    #             continue

            # Make and Place PSF
            # Note: We use sigma_level as a proxy for 'scale' here. 
            # In your real code, ensure 'scale' is the flux count, not just the sigma factor.
            # processed_psf = make_PSF(psf_data, x_pos, y_pos, channel, norm=sigma_level, verbose=False)
            # viz_image = place_PSF(viz_image, processed_psf, x_pos, y_pos)

    # 3. Plotting
    box_size = 60 # Slightly larger box to see the grid
    half_box = box_size // 2
    cx, cy = int(xc), int(yc)
    
    y0 = max(cy - half_box, 0)
    y1 = min(cy + half_box, viz_image.shape[0])
    x0 = max(cx - half_box, 0)
    x1 = min(cx + half_box, viz_image.shape[1])
    
    sub_image = viz_image[y0:y1, x0:x1]
    v_max = np.percentile(sub_image, 85)
    v_min = np.percentile(sub_image, 10)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(sub_image, origin='lower', cmap='gray', interpolation='nearest', vmax=v_max, vmin=v_min)
    # ax.set_title(f"Ensemble Context: {sigma_level:.2f} Sigma\nCenter ({xc:.2f}, {yc:.2f})")
    ax.set_title(f"Magnetar {name}\nIRAC {channel}\nRA={configs['x_coord']}, DEC={configs['y_coord']}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 4. Draw Aperture/Annulus ONLY on Central Source
    # Transform center coords to subplot coords
    sub_cx_target = xc - x0
    sub_cy_target = yc - y0
    
    ap_radius = aperture_props['ap_radius']
    in_ann = aperture_props['inner_ann_radius']
    out_ann = aperture_props['outer_ann_radius']
    
    # Solid Red Aperture
    aperture = Circle((sub_cx_target, sub_cy_target), ap_radius, 
                      edgecolor='red', facecolor='none', lw=2, linestyle='-')
    # Dashed Yellow Annuli
    ann_inner = Circle((sub_cx_target, sub_cy_target), in_ann, 
                       edgecolor='yellow', facecolor='none', lw=1.5, linestyle='--')
    ann_outer = Circle((sub_cx_target, sub_cy_target), out_ann, 
                       edgecolor='yellow', facecolor='none', lw=1.5, linestyle='--')
    
    ax.add_patch(aperture)
    ax.add_patch(ann_inner)
    ax.add_patch(ann_outer)
    
    # 5. Draw Crosses at ALL Grid Positions
    # for (gx, gy) in grid_positions:
    #     sub_gx = gx - x0
    #     sub_gy = gy - y0
        
    #     # Check if point is within the crop window
    #     if 0 <= sub_gx < (x1-x0) and 0 <= sub_gy < (y1-y0):
    #         ax.plot(sub_gx, sub_gy, marker='+', color='cyan', markersize=8, markeredgewidth=1)
    ax.plot(sub_cx_target, sub_cy_target, marker='+', color='cyan', markersize=8, markeredgewidth=1)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved context plot to {save_path}")
    
    plt.close(fig)

def visualize_ensemble_grid(image_data, psf_data, xc, yc, channel, sigma_level, 
                            configs, aperture_props, save_path=None):
    """
    Creates a multipanel plot where each panel shows ONE injected source.
    The view (crop) is centered on the INJECTED source, so the aperture 
    is always in the middle of the panel.
    """
    
    grid_size = configs['grid']
    spacing = configs['spacing']
    
    # Setup Figure Grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(4*grid_size, 4*grid_size))
    # Ensure axes is 2D array even if grid=1
    if grid_size == 1: axes = np.array([[axes]])
    
    # Box size for the cutout
    box_size = 50 
    half_box = box_size // 2
    
    print(f"Generating grid view for Sigma={sigma_level}...")

    for ii in range(grid_size):
        for jj in range(grid_size):
            # Handle axes indexing (row=jj, col=ii)
            ax = axes[jj, ii] 
            
            # 1. Calculate Injection Position (same as in photometry loop)
            x_offset = (ii - 1) * spacing
            y_offset = (jj - 1) * spacing
            x_pos = xc + x_offset
            y_pos = yc + y_offset
            
            # 2. Generate Image with ONE source injected at (x_pos, y_pos)
            sim_image = image_data.copy()
            processed_psf = make_PSF(psf_data, x_pos, y_pos, channel, norm=sigma_level, verbose=False)
            sim_image = place_PSF(sim_image, processed_psf, x_pos, y_pos)
            
            # 3. Define Crop FIXED on the CENTRAL TARGET (xc, yc)
            crop_cx, crop_cy = int(xc), int(yc)
            # Use a fixed center for the view, regardless of where the source was injected
            y0 = max(crop_cy - half_box, 0)
            y1 = min(crop_cy + half_box, sim_image.shape[0])
            x0 = max(crop_cx - half_box, 0)
            x1 = min(crop_cx + half_box, sim_image.shape[1])

            sub_image = sim_image[y0:y1, x0:x1]
            v_max = np.percentile(sub_image, 85)
            v_min = np.percentile(sub_image, 10)
            im = ax.imshow(sub_image, origin='lower', cmap='gray', interpolation='nearest', vmax=v_max, vmin=v_min)

            # 4. Calculate Aperture Position relative to the FIXED crop origin (x0, y0)
            # The aperture must follow the injected source, but be plotted in the fixed frame.
            sub_inj_x = x_pos - x0
            sub_inj_y = y_pos - y0

            ap_radius = aperture_props['ap_radius']
            in_ann = aperture_props['inner_ann_radius']
            out_ann = aperture_props['outer_ann_radius']

            # Draw circles at the *relative* injected position (sub_inj_x, sub_inj_y)
            aperture = Circle((sub_inj_x, sub_inj_y), ap_radius, edgecolor='red', facecolor='none', lw=1.5)
            ann_inner = Circle((sub_inj_x, sub_inj_y), in_ann, edgecolor='yellow', facecolor='none', lw=1, linestyle='--')
            ann_outer = Circle((sub_inj_x, sub_inj_y), out_ann, edgecolor='yellow', facecolor='none', lw=1, linestyle='--')

            ax.add_patch(aperture)
            ax.add_patch(ann_inner)
            ax.add_patch(ann_outer)
            ax.plot(sub_inj_x, sub_inj_y, marker='+', color='cyan', markersize=6)
            
            # Clean up axes
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Optional: Add label to show offset
            ax.text(0.05, 0.95, f"Off: ({x_offset:.1f}, {y_offset:.1f})", 
                    transform=ax.transAxes, color='white', fontsize=8, verticalalignment='top')
            
            # Add title only to the middle top plot to save space
            # if ii == 1 and jj == 0:
            #     ax.set_title(f"Injected Sources ({sigma_level:.2f} scale)", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved grid plot to {save_path}")
    plt.close(fig)

def ensemble_photometry(configs, verbose=False):
    logger = logging.getLogger(__name__)

    x_deg = configs['x_coord']
    y_deg = configs['y_coord']
    channels = configs['channels']

    # Ensure plots directory exists
    if not os.path.exists("plots/targets"):
        os.makedirs("plots/targets", exist_ok=True)

    for channel in channels:
        if channel not in [1, 2, 3, 4]:
            print(f"Invalid channel: {channel}. Must be 1, 2, 3, or 4.")
            return
        
        # =======================================================================
        # 1. Read PSF and Image data for the channel
        psf_file_path = configs['psf_file_path'] + f"/apex_sh_IRAC{channel}_col129_row129_x100.fits"
        target_file_path = configs['image_file_path'] + f"/mosaici{channel}/Combine/mosaic.fits"
        
        # 1.1 convert RA/DEC to pixel coordinates
        try:
            wcs = WCS(target_file_path) 
            xc, yc = wcs.wcs_world2pix(x_deg, y_deg, 0)
            logger.info(f"Channel {channel}: Converted RA={x_deg}, DEC={y_deg} to pixel coordinates x={xc:.2f}, y={yc:.2f}")
        except Exception as e:
            logger.error(f"Failed to create WCS or convert coordinates for {configs['name']} ch{channel}: {e} (file: {target_file_path})")
            print(f"Error creating WCS: {e}")
            continue

        if verbose:
            print(f"Target coordinates in pixels: x={xc:.2f}, y={yc:.2f}")
    
        try:
            psf_data = get_PSF(psf_file_path, channel=channel)
        except (FileNotFoundError, RuntimeError) as e:
            logger.error(f"Failed to load PSF for {configs['name']} ch{channel}: {e} (file: {psf_file_path})")
            print(f"Error: {e}")
            continue
        
        try:
            image_data = get_Image(target_file_path)
        except (FileNotFoundError, RuntimeError) as e:
            logger.error(f"Failed to load image for {configs['name']} ch{channel}: {e} (file: {target_file_path})")
            print(f"Error: {e}")
            continue
        
        # Aperture and annulus sizes in pixels
        ap_radius = configs['ap_radius'] * (PSIZE_ASEC[channel-1] / 0.6)
        inner_ann_radius = configs['inner_ann_radius'] * (PSIZE_ASEC[channel-1] / 0.6)
        outer_ann_radius = configs['outer_ann_radius'] * (PSIZE_ASEC[channel-1] / 0.6)

        try:
            result = circ_apphot(
                image_data, xc+0., yc+0., ap_radius, 1.0,
                bgndwidth=outer_ann_radius-inner_ann_radius,
                quiet=True, rbackin=inner_ann_radius
            )
            # Log one-shot photometry results
            phot_flux_ujy = result['total_counts'] * APCOR[channel-1] * 8.47
            phot_sigma_ujy = result['sigma'] * APCOR[channel-1] * 8.47
            logger.info(f"One-shot photometry for {configs['name']} ch{channel}: pixel_coords=({xc:.2f}, {yc:.2f}), total_counts={result['total_counts']:.6f}, sigma={result['sigma']:.6f}, flux={phot_flux_ujy:.6f} µJy, sigma={phot_sigma_ujy:.6f} µJy")
            if verbose:
                print(f"Photometry result: {phot_flux_ujy}")
                print(f"Photometric error (sigma): {phot_sigma_ujy}")
        except Exception as e:
            logger.error(f"Error in one-shot aperture photometry for {configs['name']} ch{channel}: {e}")
            print(f"Error in aperture photometry: {e}")
            continue

        one_shot_file = 'results/all_one_shot.csv'
        # save_one_shot(one_shot_file, configs['name'], result['total_counts']*APCOR[channel-1]*8.47, result['sigma']*APCOR[channel-1]*8.47, channel)

        # make_plot(image_data, xc, yc, ap_radius, inner_ann_radius, outer_ann_radius, safe=True, safe_path=f"plots/{configs['name']}_ch{channel}.png")
        
        measured_sigma = result['sigma']
        print(measured_sigma)
        
        scales = FACTORS * result['sigma']
        
        # ========================================================================
        # NEW: Visualization of Ensemble Setup (3.8 sigma and 7.0 sigma)
        # ========================================================================
        logger.info(f"Generating ensemble visualizations for {configs['name']} ch{channel}...")
        
        # specific properties for visualization
        ap_props = {
            'ap_radius': ap_radius,
            'inner_ann_radius': inner_ann_radius,
            'outer_ann_radius': outer_ann_radius
        }

        # Iterate over both factors (3.8 and 7.0) to generate plots for each
        for i, scale in enumerate(scales):
            factor_name = FACTORS[i] # e.g., 3.8 or 7.0
            
            # 1. Context Plot (One image, all crosses, aperture on center)
            # ctx_filename = f"plots/targets/{configs['name']}_ch{channel}_{factor_name}sigma_context.png"
            ctx_filename = f"plots/targets/{configs['name']}_ch{channel}_.png"
            visualize_ensemble_context(
                image_data, psf_data, xc, yc, channel, configs['name'],
                sigma_level=scale, # Pass the actual flux count (scale)
                configs=configs, 
                aperture_props=ap_props, 
                save_path=ctx_filename
            )

            # 2. Grid Plot (Multiple panels, aperture tracks the source)
            grid_filename = f"plots/targets/{configs['name']}_ch{channel}_{factor_name}sigma_grid.png"
            visualize_ensemble_grid(
                image_data, psf_data, xc, yc, channel, 
                sigma_level=scale, # Pass the actual flux count (scale)
                configs=configs, 
                aperture_props=ap_props, 
                save_path=grid_filename
            )
        # ========================================================================

        logger.info(f"Starting ensemble photometry for {configs['name']} ch{channel}: scales={scales}, grid={configs['grid']}x{configs['grid']}, spacing={configs['spacing']}")
        sim_images_with_pos = []
        ensemble_count = 0
        for scale in scales:
            for ii in range(configs['grid']):
                for jj in range(configs['grid']):
                    x_offset = (ii - 1) * configs['spacing']
                    y_offset = (jj - 1) * configs['spacing']
                    x_pos = xc + x_offset
                    y_pos = yc + y_offset

                    # ========================================================================
                    # 2. Place PRF at specified coordinates in a copy of the image
                    processed_psf_image = make_PSF(psf_data, x_pos, y_pos, channel, scale, verbose=verbose)
                    simulated_image = image_data.copy()
                    simulated_image = place_PSF(simulated_image, processed_psf_image, x_pos, y_pos)

                    # ========================================================================
                    # 3. Perform Circular Aperture Photometry
                    try:
                        result = circ_apphot(
                            simulated_image, x_pos, y_pos, ap_radius, 1.0,
                            bgndwidth=outer_ann_radius-inner_ann_radius,
                            quiet=True, rbackin=inner_ann_radius
                        )
                        ensemble_count += 1
                        phot_flux_ujy = result['total_counts'] * APCOR[channel-1] * 8.47
                        phot_sigma_ujy = result['sigma'] * APCOR[channel-1] * 8.47
                        logger.info(f"Ensemble photometry {ensemble_count} for {configs['name']} ch{channel}: scale={scale:.6f}, pixel_coords=({x_pos:.2f}, {y_pos:.2f}), total_counts={result['total_counts']:.6f}, sigma={result['sigma']:.6f}, flux={phot_flux_ujy:.6f} µJy, sigma={phot_sigma_ujy:.6f} µJy")
                        sim_images_with_pos.append((simulated_image, x_pos, y_pos))
                        if verbose:
                            print(f"Photometry result: {phot_flux_ujy}")
                            print(f"Photometric error (sigma): {phot_sigma_ujy}")
                    except Exception as e:
                        logger.error(f"Error in ensemble aperture photometry for {configs['name']} ch{channel} at scale={scale:.6f}, pos=({x_pos:.2f}, {y_pos:.2f}): {e}")
                        print(f"Error in aperture photometry: {e}")
                        continue

                    # =======================================================================
                    # 4. Save results to CSV
                    # save_results(configs['intermediate_path_file'],
                    #     [
                    #         configs['name'],
                    #         x_deg,
                    #         y_deg,
                    #         channel,
                    #         x_pos,
                    #         y_pos,
                    #         scale,
                    #         result['total_counts'],
                    #         result['sigma']
                    #     ])
                    
        logger.info(f"Completed ensemble photometry for {configs['name']} ch{channel}: {ensemble_count} measurements total")