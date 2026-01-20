import argparse
import pandas as pd

from src.config import load_yaml_config
from src.logging_setup import configure_logging
import logging
from src.utils import get_Image
from astropy.wcs import WCS
from src.ensemble_photometry import PSIZE_ASEC, APCOR
from src.idl_circapphot import circ_apphot
            
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry

import numpy as np
# ========================================================================
# Main Execution
# =======================================================================
channel = 4  # Test for channel 1
def test(config_file='configs/default.yml', verbose=False):

    cfg = load_yaml_config(filename=config_file)
    
    mag = '4u0142+61'
    ra_deg = 26.593363
    dec_deg = 61.750885

    print(f"Processing target: {mag}")
    path_target = cfg.data_path_folder + '/' + mag
    target_file_path = path_target + f"/mosaici{channel}/Combine/mosaic.fits"
    wcs = WCS(target_file_path) 
    xc, yc = wcs.wcs_world2pix(ra_deg, dec_deg, 0)
    
    image_data = get_Image(target_file_path)

    ap_radius = cfg.rap_cam_pix * (PSIZE_ASEC[channel-1] / 0.6)
    inner_ann_radius = cfg.rbackin_cam_pix * (PSIZE_ASEC[channel-1] / 0.6)
    outer_ann_radius = cfg.rbackout_cam_pix * (PSIZE_ASEC[channel-1] / 0.6)

    result = circ_apphot(
            image_data, xc+0., yc+0., ap_radius, 1.0,
            bgndwidth=outer_ann_radius-inner_ann_radius,
            quiet=True, rbackin=inner_ann_radius
        )
    
    print(f"IDL Photometry result: {result['total_counts']:.4f}")
    print(f"IDL Photometric error (sigma): {result['sigma']:.4f}")





    positions = [(xc, yc)]   # photutils uses (x, y)

    aperture = CircularAperture(positions, r=ap_radius)
    annulus = CircularAnnulus(positions, r_in=inner_ann_radius, r_out=outer_ann_radius)

    phot_table = aperture_photometry(image_data, aperture, method='center')
    raw_flux = phot_table['aperture_sum'][0]
    
    aperture_area = aperture.area_overlap(image_data)
    
    from photutils.aperture import ApertureStats
    aperstats = ApertureStats(image_data, annulus)
    bkg_mean = aperstats.median
    
    total_bkg = bkg_mean * aperture_area
    total_flux = raw_flux - total_bkg

    pixsig = aperstats.stddev
    npix = 

    print(f"Flux = {total_flux[0]:.4f}")
    print(f"Error: ")

    exit()

    # =======================================================================

    ny, nx = 200, 200

    # create empty image with background noise
    np.random.seed(42)
    # data = np.random.normal(loc=1000, scale=20, size=(ny, nx))
    data = np.zeros((ny, nx))

    # inject a fake star
    xc, yc = 100, 100
    true_flux = 1000   # total counts of the star

    for y in range(ny):
        for x in range(nx):
            r = np.sqrt((x-xc)**2 + (y-yc)**2)
            data[y, x] += true_flux * np.exp(-r**2 / (2*2.5**2)) / (2*np.pi*2.5**2)

    # visualize
    import matplotlib.pyplot as plt
    plt.imshow(data, origin='lower')
    plt.colorbar()
    plt.title("Synthetic image with one star")
    plt.show()

    ap_radius = 8
    inner_ann_radius = 10
    outer_ann_radius = 12



    result = circ_apphot(
            data, xc+0., yc+0., ap_radius, 1.0,
            bgndwidth=outer_ann_radius-inner_ann_radius,
            quiet=True, rbackin=inner_ann_radius
        )
    
    print("IDL Photometry result:", result['total_counts'])
    print("IDL Photometric error (sigma):", result['sigma'])







    pos = [(xc, yc)]   

    aperture = CircularAperture(pos, r=ap_radius)
    annulus = CircularAnnulus(pos, r_in=inner_ann_radius, r_out=outer_ann_radius)

    phot_table = aperture_photometry(data, aperture, method='subpixel')
    raw_flux = phot_table['aperture_sum'][0]
    
    aperture_area = aperture.area_overlap(data)
    
    from photutils.aperture import ApertureStats
    aperstats = ApertureStats(data, annulus)
    bkg_mean = aperstats.median
    
    total_bkg = bkg_mean * aperture_area
    print("Photutils Flux =", (raw_flux - total_bkg))

    
if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Simulate PSF placement and perform aperture photometry.")
    argparse.add_argument('-i', '--config', type=str, default="configs/default.yml", help="YAML config file with circapphot parameters")
    args = argparse.parse_args()

    test(args.config)