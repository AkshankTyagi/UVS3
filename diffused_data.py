import numpy as np
import csv
import pandas as pd
import os

from configparser import ConfigParser

from star_spectrum import * 
# from star_spectrum import GET_STAR_TEMP
from Params_configparser import get_folder_loc
from star_data import is_point_in_polygon

folder_loc, params_file = get_folder_loc()
# print(" diffused data")

def read_parameter_file(filename= params_file, param_set = 'Params_1'):
    config = ConfigParser()
    config.read(filename)
    global sat_name
    sat_name = config.get(param_set, 'sat_name')
    diffused_wavelength = config.get(param_set, 'BG_wavelength')
    diffused_wavelength = [int(val) for val in diffused_wavelength[1:-1].split(',')]
    return diffused_wavelength


def get_diffused_in_FOV( data ):
    # print('read_csv')
    wavelength_array = read_parameter_file()
    diffused_data = {}
    for wavelength in wavelength_array:

        filename = fr"{folder_loc}diffused_UV_data{os.sep}RA_sorted_flux_{wavelength}.feather"
        diffused_data[f'{wavelength}'] = []
        try:
            # df = pd.read_csv(filename, header=None,
            #                 sep = ',', skipinitialspace=False).iloc[:, [0,1,2]]
            df = pd.read_feather(filename).iloc[:, [0, 1, 2]]
            df.columns = ['ra', 'dec', 'flux']
            print(f'read {filename}, finished for diffused data.')
            # print (df[:10])
            # print (type(df[0]['ra']))
        except FileNotFoundError:
            print("df is empty. File not found.")



        for f in range(len(data)): # f represents frame number
            # print('frame :',f)
            _, _, frame_corner = zip(data[f])
            frame_corner= frame_corner[0]
            xmin, ymin = frame_corner.min(axis=0)
            xmax, ymax = frame_corner.max(axis=0)
            # print(f"diffused)Frame {f+1} has {len(data[f])} stars, and frame corners = {frame_corner}")

            mdf = df[['ra', 'dec', 'flux']]
            # Apply a buffer to include points close to the polygon boundary
            buffer = 0.01
            q = f'ra >= @xmin - {buffer} & ra <= @xmax + {buffer} & dec >= @ymin - {buffer} & dec <= @ymax + {buffer}' 
            mdf = mdf.query(q)
            mdf = mdf.values.tolist()

            # Now, use is_point_in_polygon to keep only the points inside the rotated FOV
            polygon = frame_corner  # The polygon is defined by the rotated FOV corners
            filtered_points = []
            for point in mdf:
                ra, dec, flux = point
                if is_point_in_polygon(ra, dec, polygon):
                    filtered_points.append([ra, dec, flux])

            diffused_data[f'{wavelength}'].append([f, filtered_points])

    # print (diffused_data[0])

    return diffused_data

def calc_total_diffused_flux(diffused_data):
    _, data = zip(diffused_data)
    c = list(zip(*data[0]))
    tot_flux  = sum(c[2])
    return tot_flux

def random_scatter_data(diffused_data):
    frame, data = zip(diffused_data)
    # print(data)
    c = list(zip(*data[0]))
    # print(c)
    # print(c[1])
    ra, dec, fluxes = c[0], c[1], c[2]

    # print(fluxes, loc)
    ra_norm = []
    dec_norm = []
    for i, flux in enumerate(fluxes):
        # print(flux, loc[0][i]
        # print(ra,dec)
        ra_N = np.random.normal(ra[i], 0.07, size= int(flux))
        dec_N = np.random.normal(dec[i], 0.07, size= int(flux))
        for j in range(len(ra_N)):
            # if 
            ra_norm.append(ra_N[j])
            dec_norm.append(dec_N[j])
    
  
    # output = [ra_norm, dec_norm]
    # print(ra_norm, dec_norm)
    return ra_norm, dec_norm
