import numpy as np
import csv
import pandas as pd
import time

from configparser import ConfigParser

from star_spectrum import * 
# from star_spectrum import GET_STAR_TEMP
from Params_configparser import get_folder_loc
from star_data import is_point_in_polygon

folder_loc, params_file = get_folder_loc()
# print(" diffused data")

def read_parameter_file(filename= params_file):
    config = ConfigParser()
    config.read(filename)
    file_loc_set = 'Params_0'
    param_set = 'Params_1'
    diffused_file = config.get(file_loc_set, 'diffused_BG_file')
    diffused_wavelength = config.get(param_set, 'BG_wavelength')
    diffused_wavelength = [int(val) for val in diffused_wavelength[1:-1].split(',')]
    return diffused_wavelength, diffused_file


def get_diffused_in_FOV( data ):
    print('Reading Diffused UV ISRF in the FOV.')
    wavelength_array, diffused_BG_file = read_parameter_file()

    diffused_data = {}
    
    for wavelength in wavelength_array:
        t1 = time.perf_counter()
        filename = diffused_BG_file+f'RA_sorted_flux_{wavelength}.feather'
        diffused_data[f'{wavelength}'] = []
        try:
            # df = pd.read_csv(filename, header=None, sep = ',', skipinitialspace=False).iloc[:, [0,1,2]]
            df = pd.read_feather(filename).iloc[:, [0, 1, 2]]
            df.columns = ['ra', 'dec', 'flux']
            # print(f'Read diffused_UV_data{os.sep}RA_sorted_flux_{wavelength}.feather for diffused data @ {wavelength}.')

        except FileNotFoundError:
            print("df is empty. File not found.")

        # Convert to numpy arrays once (fast)
        ra_arr = df['ra'].to_numpy()
        dec_arr = df['dec'].to_numpy()
        flux_arr = df['flux'].to_numpy()
        del df  # free memory
        
        for f in range(len(data)): # f represents frame number
            # print('frame :',f)
            _, _, frame_corner = zip(data[f])
            frame_corner= frame_corner[0]
            xmin, ymin = frame_corner.min(axis=0)
            xmax, ymax = frame_corner.max(axis=0)
            # print(f"diffused)Frame {f+1} has {len(data[f])} stars, and frame corners = {frame_corner}")
            # mdf = df[['ra', 'dec', 'flux']]
            # Apply a buffer to include points close to the polygon boundary
            buffer = 0.01
            mask_bbox = (
                (ra_arr >= (xmin - buffer)) &
                (ra_arr <= (xmax + buffer)) &
                (dec_arr >= (ymin - buffer)) &
                (dec_arr <= (ymax + buffer))
            )
            idx_candidates = np.nonzero(mask_bbox)[0]   # integer indices

            # Now, use is_point_in_polygon to keep only the points inside the rotated FOV
            polygon = frame_corner  # The polygon is defined by the rotated FOV corners
            filtered_points = []
            for ra, dec, flux  in zip(ra_arr[idx_candidates], dec_arr[idx_candidates], flux_arr[idx_candidates]):
                # ra, dec, flux = point
                if is_point_in_polygon(ra, dec, polygon):
                    filtered_points.append([ra, dec, flux])

            diffused_data[f'{wavelength}'].append(filtered_points)

        print(f"Total Time for {wavelength}A wavelength: {time.perf_counter() - t1:.3f}s, per frame: {(time.perf_counter() - t1)/len(data):.3f}s")

    return diffused_data, wavelength_array

def calc_total_diffused_flux(diffused_data):
    data = diffused_data
    c = list(zip(*data))
    tot_flux  = sum(c[2])
    return tot_flux

def random_scatter_data(diffused_data):
    data = diffused_data
    c = list(zip(*data))
    ra, dec, fluxes = c[0], c[1], c[2]
    pixel_size = np.radians(0.1)*np.radians(0.1)

    # print(fluxes, loc)
    ra_norm = []
    dec_norm = []
    for i, flux in enumerate(fluxes):
        ra_N = np.random.normal(ra[i], 0.07, size= int(1e4*flux*pixel_size))
        dec_N = np.random.normal(dec[i], 0.07, size= int(1e4*flux*pixel_size))
        for j in range(len(ra_N)):
            ra_norm.append(ra_N[j])
            dec_norm.append(dec_N[j])

    return ra_norm, dec_norm
