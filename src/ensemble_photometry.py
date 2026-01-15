from astropy.wcs import WCS

import numpy as np
import logging


from src.idl_circapphot import circ_apphot
from src.utils import get_PSF, get_Image, place_PSF, make_PSF, make_plot, save_one_shot, save_results

FACTORS = np.array([3.8, 7.0])
APCOR = [1.125, 1.120, 1.135, 1.221]
PSIZE_ASEC = [1.221, 1.213, 1.222, 1.220]


def ensemble_photometry(configs, verbose=False):
    logger = logging.getLogger(__name__)

    x_deg = configs['x_coord']
    y_deg = configs['y_coord']
    channels = configs['channels']

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
        save_one_shot(one_shot_file, configs['name'], result['total_counts']*APCOR[channel-1]*8.47, result['sigma']*APCOR[channel-1]*8.47, channel)

        make_plot(image_data, xc, yc, ap_radius, inner_ann_radius, outer_ann_radius, safe=True, safe_path=f"plots/{configs['name']}_ch{channel}.png")

        # Prepare directory for plots: put plots next to results file in a "plots" directory
        # results_dir = os.path.dirname(configs['result_path_file']) if os.path.dirname(configs['result_path_file']) else '.'
        # plots_dir = os.path.join('results', 'plots')
        
        measured_sigma = result['sigma']
        print(measured_sigma)
        
        scales = FACTORS * result['sigma']
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
                    # 2. Place PRF at specified coordinates in a copy of the image so
                    #    successive placements don't accumulate on the same image
                    processed_psf_image = make_PSF(psf_data, x_pos, y_pos, channel, scale, verbose=verbose, save_psf=True, filename=f"{configs['name']}_ch{channel}.fits")
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
                    save_results(configs['intermediate_path_file'],
                        [
                            configs['name'],
                            x_deg,
                            y_deg,
                            channel,
                            x_pos,
                            y_pos,
                            scale,
                            result['total_counts'],
                            result['sigma']
                        ])
                    
                    # =======================================================================
                    # 5. Save X profile plot
                    # try:
                    #     out_fname = save_x_profile(simulated_image, x_pos, y_pos, channel, configs['name'], norm, out_dir=f"{plots_dir}/profiles/simulated/")
                    #     max_idx = np.unravel_index(np.argmax(processed_psf_image), processed_psf_image.shape)
                    #     out_fname = save_x_profile(processed_psf_image, max_idx[0], max_idx[1], channel, "processed_psf", norm, out_dir=f"{plots_dir}/profiles/processed_psf/")
                    #     if verbose:
                    #         print(f"Saved X profile plot to {out_fname}")
                    # except Exception as e:
                    #     print(f"Error saving X profile plot: {e}")

                # =======================================================================
                # 6. Save grid plot of all simulated images with apertures/annuli
                # out_fname = os.path.join("plots/grids", f"{configs['name']}_ch{channel}scale{scale}.png")
                # save_grid_plot(sim_images_with_pos, configs['name'], channel, scale, xc, yc, ap_radius, inner_ann_radius, outer_ann_radius, out_fname)
        
        logger.info(f"Completed ensemble photometry for {configs['name']} ch{channel}: {ensemble_count} measurements total")
       