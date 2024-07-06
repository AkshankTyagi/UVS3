from configparser import ConfigParser
# from Params_configparser import get_folder_loc

def get_folder_loc():
    folder_loc = fr'C:\Users\Akshank Tyagi\Documents\GitHub\spg-iiap-UV-Sky-Simulation\\'
    params_file = fr'{folder_loc}init_parameter.txt'
    return folder_loc, params_file

folder_loc, _ = get_folder_loc()

config = ConfigParser()

config['Params_1'] = {
    'hipparcos_catalogue' : fr'{folder_loc}hip_main.dat', #path to the Hipparcos file
    'Castelli_data' : fr'{folder_loc}Castelli\ckp00', #path to the ckp00 file of the Castelli Kurucz Atlas
    'sat_name' : 'ISS',
    'roll' : False,
    'roll_rate_hrs' : False,
    # TBA Directional Cosines of Detector from the velocity of Satellite 
    'number of Revolutions' : 1,
    # Specify either Number of frames or period in sec after which the next Frame is given
    'N_frames' : False,
    't_slice' : 100, # Seconds,

    # Camera Field of View in Deg default 9.3 X 7
    'width': 0.5, #RA width
    'height': 2, #Dec height
    'starmag_min_threshold' : 0, #threshold for what bright stars we want to avoid
    'starmag_max_threshold' : 9, #threshold for what apaarent magnitude stars we want to look at

    # Spectrum Parameters (UV Band Wavelengths in Angstroms)
    'limit_min': 100,
    'limit_max': 3800,
    'BG_wavelength': 2300, #only [1100, 1500, 2300]

    #Animation parameters
    # set view
    'azm': -59,
    'ele': 41,
    'longitudinal_spectral_width' : 0.1, #Declination width of spectral spread to fall on detector in degrees
    'interval_bw_Frames' : 1000 # milliSec
}


with open(f'{folder_loc}init_parameter.txt',"w") as f:
    config.write(f)