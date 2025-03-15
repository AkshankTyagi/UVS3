import math
import numpy as np
from configparser import ConfigParser
from astropy.io import fits
from astropy.wcs import WCS
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mc

from Params_configparser import get_folder_loc
from Coordinates import *

# def DEGRAD(deg):
#     return math.radians(deg)

# def RADDEG(rad):
#     return math.degrees(rad)

folder_loc, params_file = get_folder_loc()

def read_parameter_file(filename= params_file, param_set = 'Params_1'):
    config = ConfigParser()
    config.read(filename)
    global sat_name, Interval, spectra_width
    sat_name = config.get(param_set, 'sat_name')
    diffused_wavelength = config.get(param_set, 'BG_wavelength')
    diffused_wavelength = [int(val) for val in diffused_wavelength[1:-1].split(',')]
    return diffused_wavelength

def plot_diffused_bg(data, wavelength):

    data= data/3000

    colors = [(0, 0, 0), (0, 0, 1)]  # Black to blue
    cmap_name = 'black_to_blue'
    BtoB_cmap = mc.LinearSegmentedColormap.from_list(cmap_name, colors)
    # print(wavelength)
    plt.imshow(data, cmap= BtoB_cmap, vmin=0, vmax= 1)
    plt.colorbar()
    plt.title(f'diffused_UV_background{wavelength}')
    plt.savefig(fr'{folder_loc}diffused_UV_data{os.sep}scattered_{wavelength}.png')
    # plt.savefig(fr'C:\Users\Akshank Tyagi\Documents\GitHub\UV-Sky-Simulations\diffused_data\diffused_bg_1100.png')
    
    plt.show()

# gl= [0] #longitude
# gb= [0] #latitude
# ra, dec = conv_gal_to_eq(gl,gb)
# print (gl,gb,'--->',ra,dec)

# Example usage
wavelength = 1300
file_path = fr"C:\Users\Akshank Tyagi\Documents\GitHub\UV-Sky-Simulations\diffused_data\RA_sorted_flux_{wavelength}.csv"

print ('working')
# for x in [wavelength]:
#     fits_filename = fr"{folder_loc}diffused_UV_data{os.sep}scattered_1e10_{x}_a40_g6{os.sep}scattered.fits"
#     # fits_filename = fr'C:\Users\Akshank Tyagi\Documents\GitHub\UV-Sky-Simulations\diffused_data/scattered_100000[(1100, 1130)]_mag4.fits'
#     ra, dec = get_world_coordinates( 1800, 900, fits_filename)
#     print(f"ra,dec:",ra,dec)
#     with fits.open(fits_filename) as hdul:
#         data = hdul[0].data
        # for row in data:
        #     print (row)
        # plot_diffused_bg(data, wavelength)

wavelength_array = read_parameter_file()
print(wavelength_array, wavelength_array[0]) 
for wavelength in [wavelength]:
    file_path = fr"{folder_loc}diffused_UV_data{os.sep}RA_sorted_flux_{wavelength}.csv"

    if os.path.exists(file_path):
        print (f'{file_path} file exists.')
    else:
        print('Running diffused_Background.py to create sorted files of diffused UV BG from Jayant Murthy (2016) data')
        # wavelength = read_parameter_file()
        for wavelength in [1300]:
            fits_filename = f"{folder_loc}diffused_UV_data{os.sep}scattered_1e10_{wavelength}_a40_g6{os.sep}scattered.fits"
            print(fits_filename)

            # plot_diffused_bg(fits_filename, wavelength)

            gl= [0] #longitude
            gb= [0] #latitude
            ra, dec = conv_gal_to_eq(gl,gb)
            print (gl,gb,'--->',ra,dec)

            x_pixel = [1800]
            y_pixel = [900]

            glon, glat = get_world_coordinates(x_pixel, y_pixel, fits_filename)
            ra, dec = conv_gal_to_eq(glon,glat)
            print("RA:", ra)
            print("Dec:", dec)

            x_range= 3600
            x_array = np.arange(1, 3601)
            y_range = 1800

            # Obtain the Flux values
            with fits.open(fits_filename) as hdul:
                global values
                # Print the header of the PrimaryHDU (HDU 0)
                fits.info(fits_filename)
                values = hdul[0].data

            print("values_obtained")

            # from the Pixels converting to galctic to Equatorial coordinates + flux grid
            grid = []
            for i in range( y_range+1):
                # if i%20== 0:
                    # print("line-",i)
                glon, glat = get_world_coordinates(x_array, [i]*3600, fits_filename)
                # print(glon, glat)
                ra_line, dec_line = conv_gal_to_eq(glon,glat)
                
                line = zip(values[i-1], ra_line, dec_line)
                grid.append(line)

            print("cordinate_transformation_done")

            # write the csv file with flux and respective coordinate
            with open(fr"{folder_loc}diffused_UV_data{os.sep}flux_{wavelength}.csv", 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                # Write the data array to the CSV file row by row
                for row in grid:
                    csv_writer.writerow(row)

            print(f"flux_{wavelength}.csv saved")
            del values
            del grid
            del x_array

            list = []
            with open(fr"{folder_loc}diffused_UV_data{os.sep}flux_{wavelength}.csv", 'r', newline='') as csvfile:
                csv_reader = csv.reader(csvfile)
                for r,row in enumerate(csv_reader):
                    print(r)
                    for flux_location in row:
                        if flux_location == '(o.o, nan, nan)':
                            continue

                        flux, ra, dec = flux_location.split(', ')

                        if ra == 'nan':
                            continue

                        flux = float(flux[1:])
                        ra = float(ra)
                        dec = float(dec[:-1])
                        list.append([ra, dec, flux])

            sorted_entries = sorted(list, key= lambda x: x[0])

            with open(fr"{folder_loc}diffused_UV_data{os.sep}RA_sorted_flux_{wavelength}.csv", 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                # Write the data array to the CSV file row by row
                for row in sorted_entries:
                    csv_writer.writerow(row)

            print(f"RA_sorted_flux_{wavelength}.csv saved")


# sorted_list=[]
# with open(fr"{folder_loc}diffused_UV_data/flux_{wavelength}.csv", 'r', newline='') as csvfile:
#     csv_reader = csv.reader(csvfile)
#     already_sorted = 0
#     for r,row in enumerate(csv_reader):
#         print(r, len(sorted_list),'---',len(sorted_list)-already_sorted)
#         already_sorted = len(sorted_list)

#         for flux_location in row:

#             if flux_location == '(o.o, nan, nan)':
#                 continue

#             flux, ra, dec = flux_location.split(', ')

#             if ra == 'nan':
#                 continue

#             flux = float(flux[1:])
#             ra = float(ra)
#             dec = float(dec[:-1])
#             if len(sorted_list) == 0:
#                 sorted_list.append([ra, dec, flux])
#             elif len(sorted_list) == 1:
#                 if ra <= sorted_list[0][0]:
#                     sorted_list.insert(0,[ra, dec, flux])
#                 else:
#                     sorted_list.append([ra,dec,flux])
#             else:
#                 for i in range(len(sorted_list)-1):
#                     if ra<= sorted_list[i][0]:
#                         sorted_list.insert(i, [ra,dec,flux])
#                         break
#                     elif (ra> sorted_list[i][0] and ra<= sorted_list[-1][0]): 
#                         continue
#                     else:
#                         sorted_list.append([ra,dec,flux])
#                         break


# with open(fr"{folder_loc}diffused_UV_data\RA_sorted_flux_{wavelength}.csv", 'w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     # Write the data array to the CSV file row by row
#     for row in sorted_list:
#         csv_writer.writerow(row)

# print(f"RA_sorted_flux_{wavelength}.csv saved")


# with open(fr"{folder_loc}diffused_UV_data\ra_{wavelength}.csv", 'w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     # Write the data array to the CSV file row by row
#     for row in ra_grid:
#         csv_writer.writerow(row)
# print(f"ra_{wavelength}.csv saved")
# with open(fr"{folder_loc}diffused_UV_data\dec_{wavelength}.csv", 'w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     # Write the data array to the CSV file row by row
#     for row in dec_grid:
#         csv_writer.writerow(row)
# print(f"dec_{wavelength}.csv saved")

# center is at BQF900

