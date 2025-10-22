from configparser import ConfigParser
import os
# from Params_configparser import get_folder_loc


def get_folder_loc():
    # folder_loc = fr'C:\Users\Akshank Tyagi\Documents\GitHub\spg-iiap-UV-Sky-Simulation'  
    folder_loc = os.getcwd()
    if not folder_loc.endswith(os.sep):
        folder_loc += os.sep
    params_file = fr'{folder_loc}init_parameter.txt'
    # print(os.sep,"\n",folder_loc, params_file )
    return folder_loc, params_file 

folder_loc, _ = get_folder_loc()

config = ConfigParser()

config['Params_0'] = {
    'hipparcos_catalogue' : fr'{folder_loc}Star_catalogue{os.sep}hip_main.dat', #path to the Hipparcos file Star_catalogue\hip_main.dat
    'Castelli_data' : fr'{folder_loc}Castelli_data{os.sep}ckp00', #path to the ckp00 file of the Castelli Kurucz Atlas
    'dust_C_section' : fr'{folder_loc}Castelli_data{os.sep}crossec1.dat', #path to the dust_C file of the Castelli Kurucz Atlas

    # Diffused UV data files
    'diffused_BG_file': fr"{folder_loc}diffused_UV_data{os.sep}",
    # Zodiacal light files
    'Sol_spectra_file':  fr'{folder_loc}Zodiacal_light_data{os.sep}zodiacal_spec.txt', 
    'Zod_dist_table': fr'{folder_loc}Zodiacal_light_data{os.sep}leinert_dist.txt',
}

config['Params_1'] = {
    'sat_name' : 'Astrosat',
    'roll' : False,
    'roll_rate_hrs' : False,

    # Specify either Number of frames or period in sec after which the next Frame is given
    'number of Revolutions' : 1,
    'N_frames' : False,
    't_slice' : 20, # Seconds,

    # Camera Field of View in Deg default 9.3 X 7
    'width': 1, # 0.5 (shorter) width
    'height': 2, # 1 (longer) height
    'starmag_min_threshold' : 0, #threshold for what bright stars we want to avoid
    'starmag_max_threshold' : 7, #threshold for what maximum apaarent magnitude stars we want/can to look at

    # Direction of Detector from the velocity of Satellite 
    'allignment_with_orbit' : 90, # gives angle of longer side of slit (height) with the ORBITAL PLANE from 0 to 90 degrees or False, (default 90)
    'inclination_from_V': 0, #gives the angle of inclination 0 to 180 deg, of the the camera from V vector in ORBITAL PLANE (default 0)

    # Staring Mode parameters
    'staring RA' : 0, # in degrees (default 0)
    'staring Dec' : 90, # in degrees (default 90)
    'staring_time' : 10, # in minutes (default 0)

    # Spectrum Parameters (UV Band Wavelengths in Angstroms)
    'limit_min': 1000,
    'limit_max': 3800,
    'BG_wavelength':  [1100, 1500, 2300], #2300 only

    #Animation parameters
    # set view
    'azm': 104,
    'ele': 60,
    'longitudinal_spectral_width' : 0.1, #Declination width of spectral spread to fall on detector in degrees
    'interval_bw_Frames' : 1000 # milliSec
}

config['Params_2'] = {
    'sun': True,
    'moon':  True,
    'galactic_plane': True,
    'diffused_bg': True,
    'zodiacal_bg': True, #True
    'Spectra': True,
    'fix_start': False,
    'Staring_mode': False, #True
    'save_animation': False, #True,
    'Save_data': False #True 
}


with open(f'{folder_loc}init_parameter.txt',"w") as f:
    # print("Writing to parameter file")
    config.write(f)