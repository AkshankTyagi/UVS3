# Main propogator file to run the animation, contains few General functions like Get Folder location used by all other files.
# Functions to calculate state vectors of satellite, RA, Dec of FOV and plot the animation.
# Author: Ravi Ram(starfield view from satellite in orbit), Akshank Tyagi(Spectroscopic analysis)
# Jayant Murthy code ascl:1512.012

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs72
from skyfield.api import load, Topos

# from skyfield.almanac import find_discrete, risings_and_settings

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
from diffused_Background import *
from Coordinates import *


# include the parameter file and sattelite TLE file
folder_loc, params_file = get_folder_loc()
sat_file = f'{folder_loc}Satellite_TLE.txt'


def read_parameter_file(filename=params_file, param_set = 'Params_1'):
    config.read(filename)

    global roll_rate_hrs, sat_name

    hipp_file = config.get(param_set, 'hipparcos_catalogue')
    castelli_file = config.get(param_set, 'Castelli_data')
    sat_name = config.get(param_set, 'sat_name')
    roll = config.get(param_set, 'roll')
    if (roll == True):
        roll_rate_hrs = float(config.get(param_set, 'roll_rate_hrs'))
    else:
        roll_rate_hrs = False
    N_revolutions = config.get(param_set, 'number of Revolutions')
    N_frames = config.get(param_set, 'N_frames')
    T_slice = config.get(param_set, 't_slice')

    allignment = config.get(param_set, 'allignment_with_orbit')
    if allignment != 'False':
        allignment = float(allignment)
    theta = float(config.get(param_set, 'inclination_from_V'))
    
    print('sat_name:', sat_name, ', roll:',roll,',  roll_rate_hrs:',roll_rate_hrs, ',  N_revolutions:',N_revolutions, ',  N_frames:', N_frames, ',  T_slice:', T_slice)
    print('Allignment_with_orbit=', allignment, ', Inclination_from_V =', theta )
    return hipp_file, castelli_file, sat_name, float(T_slice), N_frames, float(N_revolutions), roll, theta, allignment #, Threshold

def read_satellite_TLE(filename= sat_file, sat_name = 'ISS'):
    config.read(filename)
    line1 = config.get(sat_name, 'line1')
    line2 = config.get(sat_name, 'line2')
    return line1, line2

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
    #e = satellite.ecco
    apo, peri = satellite.alta * r, satellite.altp * r
    print('Perigee : %5.2f km, Apogee : %5.2f km' % (peri, apo))
    print(f'mu ={mu} km^3/s^2, Earth Radius = {r} km \nDistances from Center of Earth: Perigee ={r+peri}km, Semi major ={a}km, Apogee ={r+apo}km' )
    # perigee and apogee
    # return
    return satellite


# get celestial coordinates of Sun, Moon
def get_celestial_positions(time_arr):
    # Load ephemeris data
    
    eph = load('de421.bsp')
    earth, sun, moon = eph['earth'], eph['sun'], eph['moon']

    solar = []
    lunar = []
    for time in time_arr:
        # Calculate the position of the Sun relative to Earth at the given time
        astrometric_sun = earth.at(time).observe(sun)
        apparent_sun = astrometric_sun.apparent()
        ra_sun, dec_sun, _ = apparent_sun.radec()

        # Calculate the position of the Moon relative to Earth at the given time
        astrometric_moon = earth.at(time).observe(moon)
        apparent_moon = astrometric_moon.apparent()
        ra_moon, dec_moon, _ = apparent_moon.radec()

        solar.append((ra_sun.hours- 1/60, dec_sun.degrees +4/60))
        lunar.append((ra_moon.hours -8/60, dec_moon.degrees))

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
    # N = N / np.linalg.norm(N)  # Normalize the normal vector

    # Tangential RA and Dec vectors at the point where V intersects the sphere
    # RA: A vector in the xy-plane (tangential along constant declination)
    RA = np.array([-U[1], U[0], 0])  # Tangent in xy-plane
    # RA = RA / np.linalg.norm(RA)
    # Dec: A vector in the vertical plane perpendicular to RA (tangential along constant RA)
    # Dec = np.cross(U, RA)  # Perpendicular to both V_intersect and RA
    # Dec = Dec / np.linalg.norm(Dec)  # Normalize Dec to unit length
    angle_rad = angle_between_vectors(N, RA)
    angle_deg = np.degrees(angle_rad)

    # print("angle between Normal and RA",angle_deg)

    return alfa, delta, angle_deg

# returns a list of state vectors, ra, dec for a
# given sgp4 satellite object
def propagate(sat, time_start, time_end, dt, theta):
    # time
    # end = np.arange(0.0,  time_end, dt)
    # time_arr = time_start + end.astype('timedelta64[s]') 
    # print(time_arr)
    end = np.arange(0.0,  time_end*1000, dt*1000)
    # print(end)
    # list of datetime
    time_arr = time_start + end.astype('timedelta64[ms]') 
    # print(time_arr)   
    # state vectors, ra, dec for each time step
    position = []; velocity = []
    right_ascension = []; declination = []
    angle_from_normal = []
    for j in time_arr.tolist():
        second = float(str(j.second) + "." + str(j.microsecond))
        p, v = sat.propagate(j.year, j.month, j.day, j.hour, j.minute, second)
        ra, dec, angle = get_ra_dec_from_sv(p, v, theta)
        # list packing
        position.append(p); velocity.append(v)
        right_ascension.append(ra); declination.append(dec), angle_from_normal.append(angle)

    # slice into columnsṇ
    pos, vel   = list(zip(*position)), list(zip(*velocity))
    # print (position)
    X, Y, Z    = np.array(pos[0]), np.array(pos[1]), np.array(pos[2])
    VX, VY, VZ = np.array(vel[0]), np.array(vel[1]), np.array(vel[2])
    state_vectors = [X, Y, Z, VX, VY, VZ]
    celestial_coordinates = [np.array(right_ascension), np.array(declination), np.array(angle_from_normal)]
    # print(celestial_coordinates)

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
    chi = angle_to_normal + allignment - 180

    # for (alpha, chi_angle) in zip(angle_to_normal, chi):
        # print("angle between Normal and RA",alpha)
        # print("angle of rotation of FOV",chi_angle)
    print(chi)
    # Create an empty all frames data
    frame_row_list = []

    for frame, (r, d, chi_angle) in enumerate(zip(ra, dec, chi)):
        # print(frame, (r, d))

        tdf_values, frame_boundary = filter_by_fov(df, r, d, chi_angle)
        # print (frame, tdf_values)
        # print(frame, frame_boundary)
        tdf_values = tdf_values.values.tolist()
        # 
        frame_row_list.append([frame, tdf_values, frame_boundary ])
    # print (frame_row_list)

    time_arr2 = []
    ts = load.timescale()
    for time in tr:
        start_time =  time.astype('M8[ms]').astype(datetime.datetime)
        time_arr2.append(ts.utc(start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second))
    sol_positions = get_celestial_positions(time_arr2)
    # print( sol_positions)

    return tr, sc, frame_row_list, sol_positions

# Save a csv file with all required star information 
def write_to_csv(data, sol_positions, sat_name):
    # print('writing to csv')
    # print(data[0:2])
    csv_file = f'{folder_loc}Demo_file{os.sep}{sat_name}_data.csv'
    header =['Frame Number', 'hip', 'ra', 'dec', 'mag', 'parallax', 'B_V', 'Spectral_type', 'size', 'Frame Boundaries']

    # dz.to_csv(csv_file, index=False)
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        for i in range(len(data)):
            frame, d, frame_boundary = zip(data[i])
            
            frame = int(frame[0])
            d = d[0]
            # print(d)
            if d:
                for j in range(len(d)):
                    # print(j, len(d))
                    # print(zip(*d[j]))
                    ra, dec, size, hip, mag, parallax, B_V, Spectral_type = zip(d[j])
                    csv_writer.writerow([frame+1, hip[0], ra[0], dec[0], mag[0], parallax[0], B_V[0], Spectral_type[0], size[0], frame_boundary[0]])
            else:
                csv_writer.writerow([frame+1, "Empty frame", None, None, None, None, None, None, None, frame_boundary[0]])
            
            csv_writer.writerow([frame+1, "Celestial_data for frame", 'Sun', sol_positions['sun'][i], 'moon', sol_positions['moon'][i]])
            
        print('Star Data saved in:', csv_file)


def main():
    global data
    hipp_file, castelli_dir, sat_name, t_slice, n_frames, N_revolutions, roll, theta, allignment  = read_parameter_file(params_file,'Params_1')
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
        print(f"N_Frames: {t_step}; t_slice = {t_slice} sec")
    else:
        # set Number of frames
        if n_frames:
            t_slice = t_period/int(n_frames)
            print(f"N_Frames: {t_step}; t_slice = {t_slice} sec")
        else:
            print('T_slice not found')

    # simulation starts from current time to one full orbit
    start = np.datetime64(datetime.datetime.now()) #+ np.timedelta64(10, 'D')
    print(f"Start time of Simulation: {start}")
    # times, state_vectors, celestial_coordinates  
    time_arr, state_vectors, celestial_data, sol_position  = get_simulation_data(satellite, df, start, t_period, t_slice, theta, allignment,  roll)
    Spectra = GET_SPECTRA(castelli_dir, celestial_data)
    # print(celestial_data)
    # print(sol_position)
    diffused_data = get_diffused_in_FOV(celestial_data)

    # print(Spectra.frame)
    # print(Spectra.wavelength)
    # print(Spectra.spectra_per_star)
    # with open('star_data.pkl',"wb") as f:
    #     data = celestial_coordinates
    #     pickle.dump(data, f)
    # write_to_csv(celestial_data, sol_position, sat_name)
    #  animate
    animate(time_arr, state_vectors, celestial_data, sol_position, Spectra, diffused_data, r)
    return

# main
if __name__ == '__main__':
    main()
    
    