# Functions to read and filter sky catalogue data
# Author: Ravi Ram

import datetime
import numpy as np
import pandas as pd
from configparser import ConfigParser
from Params_configparser import get_folder_loc


# Hipparcos Catalogue [hip_main.dat]
# http://cdsarc.u-strasbg.fr/ftp/cats/I/239/ 
# FILENAME = r'C:\Users\Akshank Tyagi\Documents\GitHub\spg-iiap-UV-Sky-Simulation\hip_main.dat'

folder_loc, params_file = get_folder_loc()

def read_parameter_file(filename= params_file, param_set = 'Params_1'):
    config = ConfigParser()
    config.read(filename)
    global hipp_file 
    hipp_file = config.get(param_set, 'hipparcos_catalogue')
    # Field of View size:
    width = float(config.get(param_set, 'width'))
    height = float(config.get(param_set, 'height'))
    star_mag_min_threshold = float(config.get(param_set, 'starmag_min_threshold'))
    star_mag_max_threshold = float(config.get(param_set, 'starmag_max_threshold'))
    return  width, height ,star_mag_min_threshold, star_mag_max_threshold

read_parameter_file()

# read hipparcos catalogue 'hip_main.dat'
def read_hipparcos_data(FILENAME = hipp_file):
    # Field H1: Hipparcos Catalogue (HIP) identifier
    # Field H5: V magnitude
    # Fields H8–9:  The right ascension, α , and declination, δ (in degrees)    
    _, _, star_mag_min_threshold, star_mag_max_threshold = read_parameter_file()
    print (f'Stars apparent magnitude Threshold= {[star_mag_min_threshold, star_mag_max_threshold]}')

    try:
        df = pd.read_csv(FILENAME, header=None,
                         sep = '|', skipinitialspace=True).iloc[:, [1, 5, 8, 9, 11, 37, 76]]
        df.columns = ['hip', 'mag', 'ra_deg', 'de_deg', 'trig_parallax', 'B-V', 'Spectral_type']

        df['mar_size'] = 2*(star_mag_max_threshold - df['mag'])
        
        # filter data above
        max_threshold =  star_mag_max_threshold
        min_threshold =  star_mag_min_threshold
        q = 'mag <= @max_threshold & mag >= @min_threshold' 
        df = df.query(q) 

        return df  
    
    except FileNotFoundError:
        print("df is empty. File not found.")

#------------------------------------------------------------

# def filter_by_fov(mdf, ra, de, chi): 
#     # frame field of view
#     # get valid boundaries  
#     width, height, _, _ = read_parameter_file()
#     xmin, ymin, xmax, ymax = get_frame_boundaries( width, height, ra, de)
#     frame_boundaries = [xmin, ymin, xmax, ymax]
#     # print(frame_boundaries)
#     # if mdf[0]:
#     # extract useful columns
#     mdf = mdf[['ra_deg', 'de_deg', 'mar_size','hip','mag', 'trig_parallax', 'B-V', 'Spectral_type']]
#     # filter data within the boundaries    
#     q = 'ra_deg >= @xmin & ra_deg <= @xmax & de_deg >= @ymin & de_deg <= @ymax' 
#     mdf = mdf.query(q)
#     # print(mdf)
#     # return filtered data
#     return mdf, frame_boundaries

#""" Check if a point (ra, dec) is inside an FOV defined by `poly` vertices. """
def is_point_in_polygon(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y == min(p1y, p2y):
            if x <= max(p1x, p2x) and x >= min(p1x, p2x):
                return True
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if x  == xints:
                            return True
                    if p1x == p2x or x <= xints:
                        # print(f"{i}: {x}, {y} : {p1x}, {p2x}, {p1y}, {p2y}, xints {xints}, {not inside}")
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside #or any(np.isclose([x, y], poly).all(axis=1))


# get valid frame boundary corners then rotate them by angle chi
def get_frame_boundaries(w, h, x, y, chi=0):
    # Convert chi to radians for rotation
    chi = np.deg2rad(chi)

    # Calculate the four corners of the unrotated FOV (xmin, ymin, etc.)
    xmin = float(x) - float(w) / 2.0
    xmax = float(x) + float(w) / 2.0
    ymin = float(y) - float(h) / 2.0
    ymax = float(y) + float(h) / 2.0
    
    # Define the unrotated corners as (RA, Dec) pairs
    corners = np.array([[xmin, ymin],  # bottom-left
                        [xmin, ymax],  # top-left
                        [xmax, ymax], # top-right
                        [xmax, ymin]]) # bottom-right

    # Translate the FOV center to the origin for rotation
    translated_corners = corners - np.array([x, y])

    # Apply 2D rotation matrix to each corner (rotation around z-axis)
    rotation_matrix = np.array([[np.cos(chi), -np.sin(chi)],
                                [np.sin(chi),  np.cos(chi)]])
    
    rotated_corners = np.dot(translated_corners, rotation_matrix.T)

    # Translate the corners back to their original position
    rotated_corners += np.array([x, y])

    # radius = np.sqrt((w/2)**2 + (h/2)**2)
    # return corners, rotated_corners, radius

    return rotated_corners 

# camera fov : height◦ × width◦
def filter_by_fov(mdf, ra, de, chi): 
    # Frame field of view
    # Get valid boundaries  
    w, h, _, _ = read_parameter_file()
    
    # Get the rotated corners of the FOV
    rotated_corners = get_frame_boundaries(w, h, ra, de, chi)
    
    # Calculate min and max RA and Dec from rotated corners
    min_ra, min_dec = rotated_corners.min(axis=0)
    max_ra, max_dec = rotated_corners.max(axis=0)

    # Extract useful columns
    mdf = mdf[['ra_deg', 'de_deg', 'mar_size', 'hip', 'mag', 'trig_parallax', 'B-V', 'Spectral_type']]
    
    # Filter data within the boundaries
    q = 'ra_deg >= @min_ra & ra_deg <= @max_ra & de_deg >= @min_dec & de_deg <= @max_dec' 
    mdf_filtered = mdf.query(q)

    # Convert rotated corners to a list of tuples for polygon testing
    polygon = [tuple(corner) for corner in rotated_corners]
    # Apply polygonal filtering
    mdf_filtered = mdf_filtered[mdf_filtered.apply(lambda row: is_point_in_polygon(row['ra_deg'], row['de_deg'], polygon), axis=1)  ]
    # frame_boundaries = [min_ra, min_dec, max_ra, max_dec]
    
    return mdf_filtered, rotated_corners #, frame_boundaries #

# def get_frame_boundaries(w, h, x, y):
#     # set x boundaries
#     xmin = float(x) - float(w) / 2.0
#     xmax = float(x) + float(w) / 2.0
#     xmin = 0 if xmin<0 else xmin   
#     xmax = 360 if xmax>360 else xmax

    
#     # set y boundaries
#     ymin = float(y) - float(h) / 2.0
#     ymax = float(y) + float(h) / 2.0
#     ymin = -90 if ymin<-90 else ymin
#     ymax = 90 if ymax>90 else ymax
    
#     # return limits 
#     return  xmin, ymin, xmax, ymax
