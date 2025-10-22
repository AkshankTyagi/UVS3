# Main propogator file to run the animation, contains few General functions like Get Folder location used by all other files.
# Functions to calculate state vectors of satellite, RA, Dec of FOV and plot the animation.
# Author: Ravi Ram(starfield view from satellite in orbit), Akshank Tyagi(Spectroscopic analysis)
# Jayant Murthy code ascl:1512.012

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs72
from astropy.coordinates import SkyCoord, get_body
from astropy.time import Time, TimeDelta


from configparser import ConfigParser
config = ConfigParser()
# import pickle

# Run the parameter File updators
from Params_configparser import *
from Satellite_configparser import *

# Import Functions from other code files
from star_data import *#, filter_by_fov, read_hipparcos_data
from plot import *#,animate
from star_spectrum import *#,GET_SPECTRA
from diffused_data import *
from zodiacal_light import *
# from diffused_Background import *
from Coordinates import *


# include the parameter file and sattelite TLE file
folder_loc, params_file = get_folder_loc()
sat_file = f'{folder_loc}Satellite_TLE.txt'

# get Simulation parameters from the init_params file
def read_parameter_file(filename=params_file ):
    config.read(filename)
    file_loc_set = 'Params_0'
    param_set = 'Params_1'

    global roll_rate_hrs, sat_name
    hipp_file = config.get(file_loc_set, 'hipparcos_catalogue')
    castelli_file = config.get(file_loc_set, 'Castelli_data')
    sat_name = config.get(param_set, 'sat_name')
    roll =config.get(param_set, 'roll')
    if roll == "True":
        print(roll)
        roll_rate_hrs = float(config.get(param_set, 'roll_rate_hrs'))
    else:
        roll_rate_hrs = False
    N_revolutions = config.get(param_set, 'number of Revolutions')
    N_frames = config.get(param_set, 'N_frames')
    T_slice = config.get(param_set, 't_slice')

    allignment = config.get(param_set, 'allignment_with_orbit')
    if allignment != 'False':
        allignment = float(allignment)
    elif allignment == 'False':
        print('Allignment_with_orbit = False, The FOV will be alligned to the RA and DEC axis')
    theta = float(config.get(param_set, 'inclination_from_V'))
    
    print(f'Satellite name: {sat_name},  N_revolutions: {N_revolutions}, Num frames: {N_frames}, T_slice: {T_slice}')
    print(f'FOV Allignment with orbit = {allignment}, FOV Inclination from V = {theta}, \nSatellite roll: {roll},  roll_rate_hrs: {roll_rate_hrs}' )
    return hipp_file, castelli_file, sat_name, float(T_slice), N_frames, float(N_revolutions), roll, theta, allignment #, Threshold

# get trigger fo the Simulation accesories from the init_params file
def read_components(filename= params_file, param_set = 'Params_2'):
    config = ConfigParser()
    config.read(filename)
    global diffused_bg
    save_data = config.get(param_set, 'Save_data')
    diffused_bg = config.get(param_set, 'diffused_bg')  
    zodiacal_bg = config.get(param_set, 'zodiacal_bg')
    return diffused_bg, zodiacal_bg, save_data

# get satellite TLE data from the TLE file
def read_satellite_TLE(filename= sat_file, sat_name = 'ISS'):
    config.read(filename)
    if sat_name in config:
        line1 = config.get(sat_name, 'line1')
        line2 = config.get(sat_name, 'line2')
        return line1, line2
    else:
        print(f"WARNING: Satellite TLE for {sat_name} not found in \n{filename}\n--------------------")
        raise ValueError(f"Satellite TLE for {sat_name} not found in {filename}")

    return 0

# get satellite object from TLE (2 lines data)
def get_satellite(line1, line2):
    global mu, r, a
    
    # create satellite object from TLE
    satellite = twoline2rv(line1, line2, wgs72)

    # constants
    mu = satellite.mu           # Earth’s gravitational parameter (km³/s²)
    r = satellite.radiusearthkm # Radius of the earth (km).

    # orbital parameters
    a = satellite.a * r
    apo, peri = satellite.alta * r, satellite.altp * r

    print(f'mu ={mu} km^3/s^2, Earth Radius = {r} km ')
    print(f'Orbital Perigee Height : {peri:.2f} km, Apogee Height:{apo:.2f} km \nDistances from Center of Earth: Perigee = {r+peri} km, Semi major = {a} km, Apogee = {r+apo} km' )

    return satellite

# get celestial coordinates of Sun, Moon
def get_celestial_positions(time_arr):

    solar = []
    lunar = []
    for time in time_arr:
        sun_icrs = get_body("sun",time).icrs
        moon_icrs = get_body("moon",time).icrs
        solar.append((sun_icrs.ra.hour, sun_icrs.dec.deg))
        lunar.append((moon_icrs.ra.hour, moon_icrs.dec.deg))

    # print('\nCelestial positions of Sun and Moon calculated')
    # print(solar[:2],lunar[:2])
    return {
        "sun": solar,
        "moon": lunar
    }

# get ra and dec from state vectors
def get_ra_dec_from_sv(r, v, theta):
    # normalize
    v_n = np.linalg.norm(v)
    r_n = np.linalg.norm(r)
    V_unit =  v / v_n
    R_unit = r / r_n
    # print('V --',v, v_n, V_unit)
    # print('R --',r, r_n, R_unit)

    if theta == 0 :
        U = V_unit
    elif (theta <= 180 and theta > 0):
        U = V_unit*np.cos( np.deg2rad(theta) ) + R_unit* np.sin( np.deg2rad(theta) )
    
    # direction cosines
    l = U[0]; m = U[1]; n = U[2]; 
    # print(l,m,n)

    # declination
    delta = np.arcsin(n)*180/np.pi   

    # right ascension
    np.cosd = lambda x : np.cos( np.deg2rad(x) )
    if m >0:  alfa = np.rad2deg(np.arccos(l/np.cosd(delta)))
    else: alfa = 360 - np.rad2deg(np.arccos(l/np.cosd(delta)))
    # return

    #calculate Normal vector to the orbital plane
    N = np.cross(R_unit, V_unit)

    # RA: A vector in the xy-plane (tangential along constant declination)
    RA = np.array([-U[1], U[0], 0])  # Tangent in xy-plane
    angle_rad = angle_between_vectors(N, RA)

    angle_deg = np.degrees(angle_rad)  # angle between Normal and RA - angle_deg

    return alfa, delta, angle_deg

# returns a list of state vectors, ra, dec for a given sgp4 satellite object
def propagate(sat, time_start, time_end, dt, theta):

    #  list of datetime
    time_arr = time_start + TimeDelta(np.arange(0, time_end, dt), format='sec')
    
    # state vectors, ra, dec for each time step
    position = []; velocity = []
    right_ascension = []; declination = []
    angle_from_normal = []
    for t in time_arr:
        dt_frac = t.datetime.second + t.datetime.microsecond*1e-6
        p, v = sat.propagate(t.datetime.year, t.datetime.month, t.datetime.day, t.datetime.hour, t.datetime.minute, dt_frac)
        ra, dec, angle = get_ra_dec_from_sv(p, v, theta)
        # list packing
        position.append(p); velocity.append(v)
        right_ascension.append(ra); declination.append(dec), angle_from_normal.append(angle)
    # print(f"Satellite FOV RA: {right_ascension[:20]}, \nDec: {declination[:20]}, Angle: \n{angle_from_normal[:20]}")

    # slice into columnsṇ
    pos, vel   = list(zip(*position)), list(zip(*velocity))   # print (position)
    X, Y, Z    = np.array(pos[0]), np.array(pos[1]), np.array(pos[2])
    VX, VY, VZ = np.array(vel[0]), np.array(vel[1]), np.array(vel[2])
    state_vectors = [X, Y, Z, VX, VY, VZ]
    celestial_coordinates = [np.array(right_ascension), np.array(declination), np.array(angle_from_normal)] # print(celestial_coordinates)

    return time_arr, state_vectors, celestial_coordinates

# get list of star data in view along with satellite state_vectors
def get_simulation_data(sat, df, start_time, sim_secs, time_step, theta, allignment, roll=False):
    
    # state_vectors, celestial_coordinates
    tr, sc, cc = propagate(sat, start_time, sim_secs, time_step, theta)
    # parse celestial_coordinates
    ra, dec, angle_to_normal = cc

    # [TESTING] Roll about velocity direction 
    if roll:
        if roll_rate_hrs:         # deg per hr
            roll_rate_sec = roll_rate_hrs/3600.0 # deg per sec
        # modify ra with roll rate
            ra = [ri + roll_rate_sec * i for i, ri in enumerate (ra)]

    # find the required angle that the FOV heightis inclined to the Dec axis
    if allignment == 'False':
        chi = [0]*len(angle_to_normal)
    else:
        chi = angle_to_normal + allignment - 180

    frame_row_list = [] # all frames boundary + star data

    for frame, (r, d, chi_angle) in enumerate(zip(ra, dec, chi)):
        # print (frame, tdf_values) # print(frame, frame_boundary) # print(f"Frame {frame+1} has {len(tdf_values)} stars, and frame corners = {frame_boundary}")
        tdf_values, frame_boundary = filter_by_fov(df, r, d, chi_angle) 
        tdf_values = tdf_values.values.tolist()
        frame_row_list.append([frame, tdf_values, frame_boundary ]) # print (frame_row_list)

    # get celestial positions of Sun, Moon
    sol_positions = get_celestial_positions(tr) # print( sol_positions)

    return tr, sc, frame_row_list, sol_positions

# Save a csv file with all required star information 
def write_to_csv(data, diffused_ISRF_data, zod_data, sol_positions, sat_name, start_time):
    print('writing Simulation output to csv') 
    os.makedirs(f'{folder_loc}Output', exist_ok=True)
    csv_file = f'{folder_loc}Output{os.sep}{sat_name}-{start_time.datetime.strftime("%d_%m_%Y")}_data.csv'
    header =['Frame Number', 'Hip #', 'RA', 'Dec', 'V mag', 'Parallax', 'B-V', 'Spectral Type', 'Sim size']
    zodiacal_data, zod_wavelengths = zod_data
    diffused_data, diffused_wavelengths = diffused_ISRF_data

    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([f"Simulation output for {sat_name} at Start time: {start_time.datetime}"])
        csv_writer.writerow(header)
        for i in range(len(data)):
            frame, d, frame_boundary = data[i]
            frame = int(frame)

            csv_writer.writerow([]) # empty row between frames
            csv_writer.writerow([frame+1,f'FOV:', f'[{frame_boundary[0]}, {frame_boundary[1]}, {frame_boundary[2]}, {frame_boundary[3]}]', f'Sun:', sol_positions['sun'][i], f'moon: ', sol_positions['moon'][i]]) 
            # csv_writer.writerow([frame+1, "Celestial_data for frame:", "      ", ]) #  "Diffused UV ISRF in frame (Total_photons):", ]) 

            if diffused_data != [0]: # diffused UV ISRF present in the frame
                diffused_summary = [f"{wl}: {calc_total_diffused_flux(diffused_data[str(wl)][i]):.4f}" for wl in diffused_wavelengths]
                csv_writer.writerow([frame+1, "Diffused UV ISRF in frame (Total_photons):", "      ", *diffused_summary])

            if zod_data != [0]: # zodiacal UV present in the frame
                tot_flux = calc_total_zodiacal_flux(zodiacal_data[i])
                zod_summary = [f"{wl}: {tot_flux[j]:.4f}" for j, wl in enumerate(zod_wavelengths)]
                csv_writer.writerow([frame+1, "Zodiacal UV in frame (Total_photons):", "      ", *zod_summary])

            if d: # stars present in the frame
                for j in range(len(d)):
                    ra, dec, size, hip, mag, parallax, B_V, Spectral_type = zip(d[j])
                    csv_writer.writerow([frame+1, hip[0], ra[0], dec[0], mag[0], parallax[0], B_V[0], Spectral_type[0], f"{size[0]:.2f}"])
            else:
                csv_writer.writerow([frame+1, "Empty frame", None, None, None, None, None, None, None])

    print(f'Star Data saved in: Demo_file{os.sep}{sat_name}-{start_time.datetime.strftime("%d_%m_%Y")}_data.csv\n----------------')
    return 1


def main():
    # global data
    hipp_file, castelli_dir, sat_name, t_slice, n_frames, N_revolutions, roll, theta, allignment  = read_parameter_file(params_file)
    line1, line2 = read_satellite_TLE(sat_file, sat_name)
    

    # create satellite object
    satellite = get_satellite(line1, line2)

    # read star data
    df = read_hipparcos_data(hipp_file)
    
    # time period for one revolution
    t_rev = 2 * np.pi * (a**3/mu)**0.5
    t_period = N_revolutions* t_rev
    print(f"Orbital Time period of Satellite = {t_rev} sec; Run Time of Simulation: {t_period} " )

    # each time slice
    if t_slice:
        t_step = int(t_period / t_slice) + 1
        print(f"Num Frames: {t_step}; time interval = {t_slice} sec")
    else:
        # set Number of frames
        if n_frames:
            t_slice = t_period/int(n_frames)
            print(f"Num Frames: {t_step}; time interval = {t_slice} sec")
        else:
            print('T_slice not found')

    # simulation starts from current time to one full orbit
    start = Time.now()          
    print(f"Start time of Simulation (UTC): {start}\n------------------")

    # times, state_vectors, celestial_coordinates
    time_arr, state_vectors, celestial_data, sol_position = get_simulation_data(satellite, df, start, t_period, t_slice, theta, allignment, roll)
    Spectra = GET_SPECTRA(castelli_dir, celestial_data)

    diffused_bg, zodiacal_bg, save_data = read_components()

    if diffused_bg == 'True':
        diffused_data, diffused_wavelengths = get_diffused_in_FOV(celestial_data)
        # print( diffused_data["1100"][0], diffused_data["1100"][0][0], diffused_data["1100"][0][0][0],diffused_data["1100"][0][0][1] , diffused_data["1100"][0][0][2] )
    else: 
        diffused_data = [0]
        print('Diffused UV Background not included in the simulation')

    if zodiacal_bg == 'True':
        # print('Zodiacal Background not included in the simulation yet')
        zodiacal_data, zod_wavelengths = get_zodiacal_in_FOV(celestial_data, time_arr)
    else: 
        zodiacal_data = [0]
        zod_wavelengths = [0]
        print('Zodiacal Background not included in the simulation')


    if save_data == 'True':
        write_to_csv(celestial_data, (diffused_data, diffused_wavelengths), (zodiacal_data, zod_wavelengths), sol_position, sat_name, start)
    else:
        print('Star Data not Saved.\n')

    
    #  animate
    animate(time_arr, state_vectors, celestial_data, sol_position, Spectra, diffused_data, (zodiacal_data, zod_wavelengths), r)
    return

# main
if __name__ == '__main__':
    main()
    
    