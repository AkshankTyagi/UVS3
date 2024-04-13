import numpy as np
import csv

from configparser import ConfigParser

from star_spectrum import * 
# from star_spectrum import GET_STAR_TEMP
from view_orbit import get_folder_loc

folder_loc, params_file = get_folder_loc()

def read_parameter_file(filename= params_file, param_set = 'Params_1'):
    config = ConfigParser()
    config.read(filename)
    global sat_name
    sat_name = config.get(param_set, 'sat_name')
    diffused_wavelength = config.get(param_set, 'BG_wavelength')
    return diffused_wavelength


def get_diffused_in_FOV( data ):
    wavelength = read_parameter_file()
    diffused_data = []
    for f in range(len(data)): # f represents frame number
        print('frame :',f)
        _, _, frame_boundary = zip(data[f])
        frame_boundary = frame_boundary[0]
        loc =[]
        fluxes = []
        with open(fr"{folder_loc}diffused_UV_data\flux_{wavelength}.csv", 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for i, row in enumerate(csv_reader):
                for j, flux_location in enumerate(row):
                    flux, ra, dec = flux_location.split(', ')
                    flux = float(flux[1:])
                    ra = float(ra)
                    dec = float(dec[:-1])
                    if ra == 'nan':
                        continue
                    elif (ra >= frame_boundary[0]and  ra <= frame_boundary[2]):
                        if (dec >= frame_boundary[1] and  dec <= frame_boundary[3]):
                            loc.append((ra,dec))
                            fluxes.append(flux)
                            # print(i, j, ra, dec, flux)   
                  
        print(len(fluxes))
        diffused_data.append([f, fluxes, loc])

    return diffused_data

def calc_total_diffused_flux(flux):
    tot_flux = 0
    for num_photon in flux:
        tot_flux = tot_flux +num_photon
    return tot_flux

def random_scatter_data(diffused_data):
    frame, fluxes, loc = zip(diffused_data)
    print(fluxes, loc)
    ra_norm = []
    dec_norm = []
    for i, flux in enumerate(fluxes):
        ra, dec = zip(loc[i])
        print(ra,dec)
        ra_norm.append(np.random.normal(ra, 0.05, size= int(flux)))
        dec_norm.append(np.random.normal(dec, 0.05, size= int(flux)))
    
    a = 0.5
    out = [ra_norm, dec_norm]

    return a, out
