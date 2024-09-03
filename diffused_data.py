import numpy as np
import csv
import pandas as pd
import os

from configparser import ConfigParser

from star_spectrum import * 
# from star_spectrum import GET_STAR_TEMP
from Params_configparser import get_folder_loc

folder_loc, params_file = get_folder_loc()
# print(" diffused data")

def read_parameter_file(filename= params_file, param_set = 'Params_1'):
    config = ConfigParser()
    config.read(filename)
    global sat_name
    sat_name = config.get(param_set, 'sat_name')
    diffused_wavelength = config.get(param_set, 'BG_wavelength')
    return diffused_wavelength


def get_diffused_in_FOV( data ):
    # print('read_csv')
    wavelength = read_parameter_file()
    filename = fr"{folder_loc}diffused_UV_data{os.sep}RA_sorted_flux_{wavelength}.csv"
    diffused_data = []
    try:
        df = pd.read_csv(filename, header=None,
                         sep = ',', skipinitialspace=False).iloc[:, [0,1,2]]
        df.columns = ['ra', 'dec', 'flux']
        # print (df[:10])
        # print (type(df[0]['ra']))
    except FileNotFoundError:
        print("df is empty. File not found.")

    # print('read_csv finished')

    for f in range(len(data)): # f represents frame number
        # print('frame :',f)
        _, _, frame_boundary = zip(data[f])
        frame_boundary = frame_boundary[0]
        [xmin, ymin, xmax, ymax] = frame_boundary

        mdf = df[['ra', 'dec', 'flux']]
        q = 'ra >= @xmin-0.05 & ra <= @xmax+0.05 & dec >= @ymin-0.05 & dec <= @ymax+0.05' 
        mdf = mdf.query(q)
        mdf = mdf.values.tolist()
        diffused_data.append([f, mdf])

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
            ra_norm.append(ra_N[j])
            dec_norm.append(dec_N[j])
    
  
    # output = [ra_norm, dec_norm]
    # print(ra_norm, dec_norm)
    return ra_norm, dec_norm
