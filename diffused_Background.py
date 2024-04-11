import math
import numpy as np
from configparser import ConfigParser
from astropy.io import fits
from astropy.wcs import WCS
import csv


from view_orbit import get_folder_loc

folder_loc = get_folder_loc()
params_file = f'{folder_loc}init_parameter.txt'


# def DEGRAD(deg):
#     return math.radians(deg)

# def RADDEG(rad):
#     return math.degrees(rad)

def read_parameter_file(filename= params_file, param_set = 'Params_1'):
    config = ConfigParser()
    config.read(filename)
    global sat_name, Interval, spectra_width
    sat_name = config.get(param_set, 'sat_name')
    return sat_name

def invert_gal_to_eq(gl, gb):

    # Convert Galactic coordinates from degrees to radians
    l = np.radians(gl)
    b = np.radians(gb)

    # Transformation matrix from Galactic to Equatorial coordinates, // J.C.Liu et al(A&A 536, A102 (2011))//wrong
    A = np.array([
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [0.4941094279, -0.4448296300, 0.7469822445],
        [-0.8676661490, -0.1980763734, 0.4559837762]
    ])

    # Calculate Cartesian coordinates in Galactic system
    xyz_galactic = np.array([np.cos(l) * np.cos(b), np.sin(l) * np.cos(b), np.sin(b)])

    # Apply the inverse of the transformation matrix to get Equatorial coordinates
    A_inv = np.linalg.inv(A)
    xyz_equatorial = np.dot(A_inv, xyz_galactic)

    # Convert Equatorial coordinates from Cartesian to spherical (RA, Dec)
    ra_rad = np.arctan2(xyz_equatorial[1], xyz_equatorial[0])
    dec_rad = np.arcsin(xyz_equatorial[2])

    # Convert Equatorial coordinates from radians to degrees
    ra = np.degrees(ra_rad)
    dec = np.degrees(dec_rad)
    for i, ra_i in enumerate(ra):
        if ra_i<0:
            ra[i] =360 + ra_i
    
    return ra, dec


def get_world_coordinates(x, y, fits_file):
    with fits.open(fits_file) as hdul:
        wcs = WCS(hdul[0].header)  # Create WCS object from FITS header
        # print(wcs)
        # Convert pixel coordinates to world coordinates (RA, Dec)
        ra, dec = wcs.all_pix2world(x, y, 1)  # Assumes pixel indices start from 1
        # print(ra,dec)
        return ra, dec

gl= [0] #longitude
gb= [0] #latitude
ra, dec = invert_gal_to_eq(gl,gb)
print (gl,gb,'--->',ra,dec)

# Example usage
x_pixel = [1800]
y_pixel = [900]

wavelength = 2300
fits_filename = f"{folder_loc}diffused_data\scattered_1e10_{wavelength}_a40_g6\scattered.fits"
print(fits_filename)


glon, glat = get_world_coordinates(x_pixel, y_pixel, fits_filename)
ra, dec = invert_gal_to_eq(glon,glat)
print("RA:", ra)
print("Dec:", dec)

x_range= 3600
x_array = np.arange(1, 3601)
print(x_array)
y_range = 1800


ra_grid = []
dec_grid = []

with fits.open(fits_filename) as hdul:
    global values
    # Print the header of the PrimaryHDU (HDU 0)
    fits.info(fits_filename)
    values = hdul[0].data

print("values_recieved")
# csv_file = fr"{folder_loc}diffused_data\flux_{wavelength}.csv"
with open(fr"{folder_loc}diffused_data\flux_{wavelength}.csv", 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the data array to the CSV file row by row
    for row in values:
        csv_writer.writerow(row)
print(f"flux_{wavelength}.csv saved")

# from the Pixels converting to galctic to Equatorial coordinates
for i in range(1, y_range+1):
    if i%20== 0:
        print("line-",i)
    glon, glat = get_world_coordinates(x_array, [i]*3600, fits_filename)
    # print(glon, glat)
    ra_line, dec_line = invert_gal_to_eq(glon,glat)
    # ra_line.append(ra)
    # dec_line.append(dec)
    ra_grid.append(ra_line)
    dec_grid.append(dec_line)

print("cordinate_transformation_done")

with open(fr"{folder_loc}diffused_data\ra_{wavelength}.csv", 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the data array to the CSV file row by row
    for row in ra_grid:
        csv_writer.writerow(row)
print(f"ra_{wavelength}.csv saved")
with open(fr"{folder_loc}diffused_data\dec_{wavelength}.csv", 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the data array to the CSV file row by row
    for row in dec_grid:
        csv_writer.writerow(row)
print(f"dec_{wavelength}.csv saved")

# center is at BQF900
