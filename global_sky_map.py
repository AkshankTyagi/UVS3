# Global UV Sky Map : Reads the Diffused, Zodiacal UV, stars for the whole sky

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc

from astropy.coordinates import SkyCoord, get_body
from astropy.time import Time, TimeDelta
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

from configparser import ConfigParser
config = ConfigParser()

from Params_configparser import get_folder_loc
from star_spectrum import * 
from diffused_data import *
from zodiacal_light import *
from Coordinates import *

# include the parameter file and sattelite TLE file
folder_loc, params_file = get_folder_loc()

def read_parameter_file(filename= params_file):
    file_loc_set = 'Params_0'
    param_set = 'Params_1'
    
    config = ConfigParser()
    config.read(filename)
    diffused_file = config.get(file_loc_set, 'diffused_BG_file')
    Sol_spec_file = config.get(file_loc_set, 'Sol_spectra_file')
    Zod_dist_file = config.get(file_loc_set, 'Zod_dist_table')

    diffused_wavelength = config.get(param_set, 'BG_wavelength')
    diffused_wavelength = [int(val) for val in diffused_wavelength[1:-1].split(',')]

    return diffused_file, diffused_wavelength, Sol_spec_file, Zod_dist_file


# def get_diffused_global(wavelength):
#     diffused_file, diffused_wavelength, _, _ = read_parameter_file()


#     if wavelength in diffused_wavelength:
#         filename = diffused_file+f'RA_sorted_flux_{wavelength}.feather'
#         fits_filename = diffused_file+f"scattered_1e10_{wavelength}_a40_g6{os.sep}scattered.fits"
#         with fits.open(fits_filename) as hdul:
#             data_gal = hdul[0].data
#             header_gal = hdul[0].header
#         wcs_gal = WCS(header_gal)
#         hdr_eq = header_gal.copy()
#         hdr_eq['CTYPE1'] = 'RA---TAN'
#         hdr_eq['CTYPE2'] = 'DEC--TAN'
#         wcs_eq = WCS(hdr_eq)

#         try:
#             df = pd.read_feather(filename).iloc[:, [0, 1, 2]]
#             df.columns = ['ra', 'dec', 'flux']
#             print(f'Read diffused_UV_data{os.sep}RA_sorted_flux_{wavelength}.feather for global diffused data @ {wavelength}.')

#         except FileNotFoundError:
#             print("df is empty. File not found.")
#             return None
#         x_ra, y_dec = wcs_eq.wcs_world2pix(df['ra'].values, df['dec'].values, 1)
#         data_eq = 
#         # data_eq[footprint == 0] = np.nan

#     return data_gal, wcs_eq #,diffused_data_global


def get_zodiacal_global(wavelength, time_arr):
    _, _, Sol_spec_file, Zod_dist_file = read_parameter_file()
    
    # Read the Solar Spectra File
    sol_wavelengths, sol_spectra = read_zodiacal_spectrum(Sol_spec_file)

    w_index = index_greater_than(sol_wavelengths, wavelength)
    ref_index = index_greater_than(sol_wavelengths, 5000)
    sol_flux = sol_spectra[w_index]
    ref_flux = sol_spectra[ref_index]

    # Read the Zodiacal light Distribution table
    zod_array, table_lon, table_lat = read_zodiacal_distribution(Zod_dist_file)

    # Create interpolator with linear interpolation and extrapolation
    interpolator = RegularGridInterpolator(
        (table_lon, table_lat),  # grid: latitude, longitude
        zod_array,
        method='linear',
        bounds_error=False,
        fill_value=None  # None allows extrapolation
    )

    wcs = make_aitoff_wcs_deg(bin_size=2) # 2 degree bins for Zodiacal map

    nx, ny = int(wcs.array_shape[1]), int(wcs.array_shape[0])
    x, y = np.arange(1, nx + 1), np.arange(1, ny + 1)
    X, Y = np.meshgrid(x, y)
    ra, dec = wcs.all_pix2world(X, Y, 1)
    # valid = np.isfinite(ra) & np.isfinite(dec)
    Equatorial_mesh = np.column_stack((ra, dec)) # [valid]
    # print(f"Created RA-Dec mesh for Zodiacal light with shape {ra[valid].shape}.\n{list(zip(ra[valid].ravel(), dec[valid].ravel()))[:5]} ...")
    del ra, dec

    df = pd.DataFrame(Equatorial_mesh, columns=['ra', 'dec'])

    for f, t in enumerate(time_arr):    # f represents frame number
        # Calculate the elongation and beta angle for each mesh point
        elong_arr, beta_arr = pointing_geometry(Equatorial_mesh, t)

        # Calculate the Zodiacal spectra for each point
        points = np.column_stack((np.abs(elong_arr), np.abs(beta_arr)))
        zod_intensity = interpolator(points)

        # Scale factor
        zod_scale = zod_intensity / ref_flux

        df[f'{f}'] = zod_scale * sol_flux # Zodiacal data for frame f, in units of flux at the given wavelength, scaled by the zodiacal intensity at 5000A.

    return df, wcs


# dif_data, wcs = get_diffused_global(2300)
# # df2 = get_zodiacal_global(2300, [Time.now()])

# fig = plt.figure(figsize=(10, 5), facecolor="black")
# ax = plt.subplot(111)
# ax.set_facecolor("black")
# diff_cmap = mc.LinearSegmentedColormap.from_list("diffused", [(0,0,0), (0.2,0.4,1)]) # Black to blue gradient

# im_dif = ax.imshow(
#     dif_data,
#     origin="lower",
#     cmap=diff_cmap,
#     norm=mc.LogNorm(vmin=100, vmax=10000),
#     alpha=1,
#     label = f'Diffused ISRF ({2300} $\\AA$)'
# )

# cb2 = plt.colorbar(im_dif, ax=ax, fraction=0.04, pad=0.08)
# cb2.set_label("Diffuse background", color="white")
# cb2.ax.tick_params(colors="white")
# cb2.outline.set_edgecolor("white")

# # plt.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.5)
# # ax.tick_params(colors="white")
# ax.set_title('Full Sky Map in Equatorial Coordinates', color="white", pad=20)
# # ax.set_xlabel(r'Right Ascension $^\circ$', color="white")
# # ax.set_ylabel(r'Declination $^\circ$', color="white")
# plt.show()







# plt.figure(figsize=(10, 5))
# ax = plt.subplot(111, projection='aitoff')

# ax.set_facecolor("black")
# ax.set_title('Full Sky Map in Equatorial Coordinates', color="white", pad=20)
# # set labels
# ax.set_xlabel(r'Right Ascension $^\circ$', color="white")
# ax.set_ylabel(r'Declination $^\circ$')

# diffused_wave = 2300
# zodiacal_wave = 2300

# # if diffused_data != [0]:
# diffused_sky_df = get_diffused_global(diffused_wave)
# aitoff_coords_diff = ra_dec_to_aitoff(diffused_sky_df["ra"].values, diffused_sky_df["dec"].values)
# diffused_cmap = mc.LinearSegmentedColormap.from_list("diffused", [(0,0,0), (0.2,0.4,1)]) # Black to blue gradient
# sc_diff = ax.scatter(
#     aitoff_coords_diff[0],
#     aitoff_coords_diff[1],
#     c=diffused_sky_df["flux"].values,
#     cmap=diffused_cmap,
#     norm=mc.LogNorm(vmin=diffused_sky_df["flux"].values[diffused_sky_df["flux"].values > 0].min(),
#                 vmax=diffused_sky_df["flux"].values.max()),
#     s=10,
#     alpha=0.9,
#     label = f'Diffused ISRF ({diffused_wave} $\\AA$)'
# )

# # if zodiacal_data != [0]:
# zod_sky_df = get_zodiacal_global(zodiacal_wave, [Time.now()])
# aitoff_coords_zod = ra_dec_to_aitoff(zod_sky_df["ra"].values, zod_sky_df["dec"].values)

# zod_cmap = mc.LinearSegmentedColormap.from_list("zod", [(0,0,0), (1,0.4,1)]) # Black to pink gradient
# sc_zod = ax.scatter(
#     aitoff_coords_zod[0],
#     aitoff_coords_zod[1],
#     c=zod_sky_df["0"].values,
#     cmap=zod_cmap,
#     norm=mc.LogNorm(vmin=zod_sky_df["0"].values[zod_sky_df["0"].values > 0].min(),
#                 vmax=zod_sky_df["0"].values.max()),
#     s=20,
#     alpha=0.9,
#     label = f'Zodiacal light ({zodiacal_wave} $\\AA$)'
# )
# ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
# ax.tick_params(colors="white")

# # plt.colorbar(sc_diff, ax=ax, orientation="horizontal", pad=0.12, label="Diffuse background")

# # plt.colorbar(sc_zod, ax=ax, orientation="horizontal", pad=0.05, label="Zodiacal light")
# plt.savefig("Full_sky_map.png", dpi=300, bbox_inches='tight')
# plt.show()





# print(df.shape, "\n", df2.shape)
# print(f"Size of Zodiacal data of shape {df2.shape} : {df2.memory_usage(deep=True).sum() / (1024**2):.2f} MB")


# fits_file = f'{fits_dir}\scattered_1e10_1100_a40_g6\scattered.fits'

# with fits.open(fits_file) as hdul:
#     data = hdul[0].data
#     wcs = WCS(hdul[0].header)

# ny, nx = data.shape

# print(data.shape, wcs)

