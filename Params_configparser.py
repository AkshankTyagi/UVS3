from configparser import ConfigParser

config = ConfigParser()

config['Params_1'] = {
    'hipparcos_catalogue' : r'C:\Users\Akshank Tyagi\Documents\GitHub\spg-iiap-UV-Sky-Simulation\hip_main.dat', #path to the Hipparcos file
    'Castelli_data' : r'C:\Users\Akshank Tyagi\Documents\GitHub\spg-iiap-UV-Sky-Simulation\Castelli\ckp00', #path to the ckp00 file of the Castelli Kurucz Atlas
    'sat_name' : 'ISS',
    'roll' : False,
    'roll_rate_hrs' : False,
    # TBA Directional Cosines of Detector from the velocity of Satellite 
    'number of Revolutions' : 0.1,
    # Specify either Number of frames or period in sec after which the next Frame is given
    'N_frames' : False,
    't_slice' : 2, # Seconds,

    # Camera Field of View in Deg default 9.3 X 7
    'width': 0.5, #RA width
    'height': 7, #Dec height
    'star_mag_threshold' : 8.5, #threshold for what apaarent magnitude stars we want to look at
    
    # Spectrum Parameters (UV Band Wavelengths in Angstroms)
    'limit_min': 100,
    'limit_max': 3800,

    #Animation parameters
    # set view
    'azm': 40,
    'ele': 25,
    'interval_bw_Frames' : 1000 # milliSec
}


with open('init_parameter.txt',"w") as f:
    config.write(f)