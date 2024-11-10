from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from configparser import ConfigParser
import math
import os
import pandas as pd
from scipy.interpolate import interp1d
from Params_configparser import get_folder_loc

class StellarSpectrum:
    def __init__(self, v1 =0, v2= []):
        self.temperature = v1
        self.spectrum = v2
        # self.wavelength = v3
        # self.photons = v4
        # self.scale = v5


class Spectral_FOV:
    def __init__(self):
        self.frame = []
        self.wavelength = []
        self.spectra_per_star = []
        self.ra = []
        self.dec = []
        self.scale = []
        self.photons = []
        self.frame_size = []

ERG_TO_PHOT = 50341166.81  # number of photons per erg for 1 Angstrom wavelength
N_CASTELLI_MODELS = 76
gas_to_dust = 5.8e21

# Castelli_data--
# spec_dir= r"C:\Users\Akshank Tyagi\Documents\GitHub\spg-iiap-UV-Sky-Simulation\Castelli\ckp00"
# params_file = r'C:\Users\Akshank Tyagi\Documents\GitHub\spg-iiap-UV-Sky-Simulation\init_parameter.txt'

folder_loc, params_file = get_folder_loc()

def read_parameter_file(filename= params_file, param_set = 'Params_1'):
    config = ConfigParser()
    config.read(filename)
    global Castelli_data , dust_c_section
    Castelli_data = config.get(param_set, 'Castelli_data')
    dust_c_section = config.get(param_set, 'dust_C_section')
    # Wavelength range:
    min_lim = float(config.get(param_set, 'limit_min'))
    max_lim = float(config.get(param_set, 'limit_max'))
    return min_lim, max_lim

_, _ = read_parameter_file()

def index_greater_than(lst, value):
    for i, elem in enumerate(lst):
        if elem >= value:
            return i
    return None  

def GET_STAR_TEMP(sptype):
    sptype = str(sptype)
    temperature = 0 

    if sptype.startswith("sd"):
        sptype = sptype[2:]
    

    if sptype.startswith("O3"):
        temperature = 5
    elif sptype.startswith("O4"):
        temperature = 7
    elif sptype.startswith("O5"):
        temperature = 10
    elif sptype.startswith("O6"):
        temperature = 11
    elif sptype.startswith("O7"):
        temperature = 13
    elif sptype.startswith("O8"):
        temperature = 15
    elif sptype.startswith("O9"):
        temperature = 18
    elif sptype.startswith("B0"):
        temperature = 20
    elif sptype.startswith("B1"):
        temperature = 25
    elif sptype.startswith("B2"):
        temperature = 28
    elif sptype.startswith("B3"):
        temperature = 31
    elif sptype.startswith("B4"):
        temperature = 33
    elif sptype.startswith("B5"):
        temperature = 35
    elif sptype.startswith("B6"):
        temperature = 36
    elif sptype.startswith("B7"):
        temperature = 37
    elif sptype.startswith("B8"):
        temperature = 41
    elif sptype.startswith("B9"):
        temperature = 49
    elif sptype.startswith("A0"):
        temperature = 51
    elif sptype.startswith("A1"):
        temperature = 52
    elif sptype.startswith("A2"):
        temperature = 53
    elif sptype.startswith("A3"):
        temperature = 56
    elif sptype.startswith("A4"):
        temperature = 56
    elif sptype.startswith("A5"):
        temperature = 56
    elif sptype.startswith("A6"):
        temperature = 57
    elif sptype.startswith("A7"):
        temperature = 58
    elif sptype.startswith("A8"):
        temperature = 59
    elif sptype.startswith("A9"):
        temperature = 60
    elif sptype.startswith("F0"):
        temperature = 60
    elif sptype.startswith("F1"):
        temperature = 63
    elif sptype.startswith("F2"):
        temperature = 61
    elif sptype.startswith("F3"):
        temperature = 62
    elif sptype.startswith("F4"):
        temperature = 62
    elif sptype.startswith("F5"):
        temperature = 63
    elif sptype.startswith("F6"):
        temperature = 63
    elif sptype.startswith("F7"):
        temperature = 63
    elif sptype.startswith("F8"):
        temperature = 64
    elif sptype.startswith("F9"):
        temperature = 65
    elif sptype.startswith("G0"):
        temperature = 65
    elif sptype.startswith("G1"):
        temperature = 66
    elif sptype.startswith("G2"):
        temperature = 66
    elif sptype.startswith("G3"):
        temperature = 66
    elif sptype.startswith("G4"):
        temperature = 66
    elif sptype.startswith("G5"):
        temperature = 66
    elif sptype.startswith("G6"):
        temperature = 66
    elif sptype.startswith("G7"):
        temperature = 67
    elif sptype.startswith("G8"):
        temperature = 67
    elif sptype.startswith("G9"):
        temperature = 67
    elif sptype.startswith("K0"):
        temperature = 68
    elif sptype.startswith("K1"):
        temperature = 69
    elif sptype.startswith("K2"):
        temperature = 70
    elif sptype.startswith("K3"):
        temperature = 70
    elif sptype.startswith("K4"):
        temperature = 71
    elif sptype.startswith("K5"):
        temperature = 72
    elif sptype.startswith("K6"):
        temperature = 72
    elif sptype.startswith("K7"):
        temperature = 73
    elif sptype.startswith("K8"):
        temperature = 73
    elif sptype.startswith("K9"):
        temperature = 73
    elif sptype.startswith("M0"):
        temperature = 74
    elif sptype.startswith("M1"):
        temperature = 74
    elif sptype.startswith("M2"):
        temperature = 75
    elif sptype.startswith("M3"):
        temperature = 75
    elif sptype.startswith("M4"):
        temperature = 75
    elif sptype.startswith("M5"):
        temperature = 75
    elif sptype.startswith("M6"):
        temperature = 75
    elif sptype.startswith("M7"):
        temperature = 75
    elif sptype.startswith("M8"):
        temperature = 75
    elif sptype.startswith("M9"):
        temperature = 75
    elif sptype.startswith("O"):
        temperature = 13
    elif sptype.startswith("B"):
        temperature = 35
    elif sptype.startswith("M"):
        temperature = 75
    elif sptype.startswith("C"):
        temperature = 75
    elif sptype.startswith("A"):
        temperature = 56
    elif sptype.startswith("R"):
        temperature = 75
    elif sptype.startswith("G"):
        temperature = 66
    elif sptype.startswith("W"):
        temperature = 35
    elif sptype.startswith("K"):
        temperature = 72
    elif sptype.startswith("N"):
        temperature = 72
    elif sptype.startswith("S"):
        temperature = 72
    elif sptype.startswith("F"):
        temperature = 63
    elif sptype.startswith("DA"):
        temperature = 35
    else:
        # Handle other spectral types or unknown cases
        temperature = 66  # Default temperature

    return temperature

def READ_CASTELLI_SPECTRA(spec_dir = Castelli_data):
    stellar_spectra = [{"temp": None, "spectrum": None, "wavelength": None} for i in range(N_CASTELLI_MODELS)]

    temper = [50000 - i*1000 for i in range(38)] + [13000 - (i - 37)*250 for i in range(38, 76)]
    gindex = [12]*5 + [11]*6 + [10]*53 + [11]*12  # g_effective = (g_index[]-2) *5
    # print(list(zip(temper, gindex)))

    for i in range(len(temper)):
        filename = f"{spec_dir}{os.sep}ckp00_{temper[i]}.fits"
        stellar_spectra[i]["temp"] = temper[i]
        with fits.open(filename) as hdul:
            data = hdul[1].data
            stellar_spectra[i]["spectrum"] = data.field(gindex[i] - 1)
            stellar_spectra[i]["wavelength"] = data.field(0)
    
    return stellar_spectra 

def GET_SPECTRA(spec_dir, data):
    all_spectra = READ_CASTELLI_SPECTRA(spec_dir)
    spectral_FOV = Spectral_FOV()

    wave_min , wave_max = read_parameter_file()
    low_lim = index_greater_than(all_spectra[0]['wavelength'], wave_min)
    high_lim = index_greater_than(all_spectra[0]['wavelength'], wave_max)

    for i in range(len(data)): # repeats over all frames i
        spectral_FOV.frame.append(str(i+1))
        spectral_flux = []
        spectral_wavelength = []
        scale_per_star = []
        photons_per_star = []

        _, d, frame_corner = zip(data[i])

        frame_corner= frame_corner[0]
        # print(f"stars)Frame {i+1} has {len(d[0])} stars, and frame corners = {frame_corner}")
        min_ra, min_dec = frame_corner.min(axis=0)
        max_ra, max_dec = frame_corner.max(axis=0)

        spectral_FOV.frame_size.append([min_ra, min_dec, max_ra, max_dec])
        
        if d[0]:
            c = list(zip(*d[0]))
            ra, dec = c[0], c[1]
            spectral_FOV.ra.append(ra)
            spectral_FOV.dec.append(dec)
            spectral_type = c[7]

            if (len(spectral_type)>1):
                # print(f' Frame {i+1}) The spectra of stars in the FOV are:')
                for j in range(len(spectral_type)):     # repeats over all stars in the FOV in frame i
                    t_index = GET_STAR_TEMP(spectral_type[j])
                    stellar_spectra = StellarSpectrum()
                    data1 = all_spectra[t_index]
                    # stellar_spectra.temperature = float(data1['temp'])
                    # print(f"{j+1}) Spectral type: {spectral_type[j]};   Temperature index: {t_index}   ;   Temperature= {stellar_spectra.temperature}")
                    # print (f"wavelength: {stellar_spectra.wavelength} \nSpectra: {stellar_spectra.spectrum}", end="\n \n")
                    wavelengths = data1['wavelength']
                    flux = data1['spectrum']
                    scale, photons = GET_SCALE_FACTOR(j, c, wavelengths, flux )

                    spectral_wavelength = wavelengths[low_lim:high_lim]
                    spectral_flux.append(flux[low_lim:high_lim])
                    scale_per_star.append(scale)
                    photons_per_star.append(photons[low_lim:high_lim])
                spectral_FOV.spectra_per_star.append(spectral_flux)
                spectral_FOV.wavelength.append(spectral_wavelength)
                spectral_FOV.scale.append(scale_per_star)
                spectral_FOV.photons.append(photons_per_star)

            else:
                # print(f' Frame {i +1})  The spectra of star in the FOV is:')
                t_index = GET_STAR_TEMP(spectral_type[0])
                stellar_spectra = StellarSpectrum()
                data1 = all_spectra[t_index]
                # stellar_spectra.temperature = float(data1['temp'])
                # stellar_spectra.wavelength = np.array(data1['wavelength'][low_lim:high_lim])
                # stellar_spectra.spectrum = np.array(data1['spectrum'][low_lim:high_lim])
                # print(f" Spectral type: {spectral_type[0]}; Temperature index: {t_index}  ; Temperature= {stellar_spectra.temperature}")
                # print (f"wavelength: {stellar_spectra.wavelength} \nSpectra: {stellar_spectra.spectrum}", end="\n \n")
                wavelengths = data1['wavelength']
                flux = data1['spectrum']
                scale, photons = GET_SCALE_FACTOR(0, c, wavelengths, flux )

                spectral_wavelength = wavelengths[low_lim:high_lim]
                spectral_flux.append(flux[low_lim:high_lim])
                scale_per_star.append(scale)
                photons_per_star.append(photons[low_lim:high_lim])

                spectral_FOV.spectra_per_star.append(spectral_flux)
                spectral_FOV.wavelength.append(spectral_wavelength)
                spectral_FOV.scale.append(scale_per_star)
                spectral_FOV.photons.append(photons_per_star)


        else:
            # print('Frame',i +1,'is EMPTY', end="\n \n")
            spectral_FOV.wavelength.append([0])
            spectral_FOV.spectra_per_star.append([[0]])
            spectral_FOV.ra.append([0])
            spectral_FOV.dec.append([0])
            spectral_FOV.scale.append([0])
            spectral_FOV.photons.append([[0]])
    
    return spectral_FOV

def GET_SCALE_FACTOR(j, c, waveL_range, stellar_spectra):

    # print(c)
    V_mag, E_B_V= c[4][j], c[6][j]
    # Check if E_B_V is NaN
    if math.isnan(E_B_V):
        E_B_V = 0

    if V_mag == 0:
        scale = 0
        tot_photons =[0] #.append(0)
    else:
        vindex = index_greater_than(waveL_range, 5450)
        vflux = stellar_spectra[vindex]
        # bindex = index_greater_than(waveL_range, 4360)
        # bflux = stellar_spectra[bindex]  
        # # print (f'bflux:{bflux}, vflux:{vflux}')
        # b_mag = -2.5 * math.log10(bflux / 6.61)
        # v_mag = -2.5 * math.log10(vflux / 3.64)
        # B_V = hipstar['B_mag'] - hipstar['V_mag']
        # ebv = B_V - (b_mag - v_mag)
        # ebv = max(ebv, 0)
        # if parallax > 0:
        #     distance = 1000/parallax  #distance in parsec
        # else:
        #     distance = 1e6  

        
        cross_sec = pd.read_csv(dust_c_section, delimiter=r'\s+', names=['wavelength', 'c_section'])

        scale = 3.64e-9 * pow(10, -0.4 * (V_mag - 3.1 * E_B_V))  / vflux #* 4 * math.pi
        tau = cross_sec['c_section'] * E_B_V * gas_to_dust

# scale = scale /distance**2
# 3.336 x 10^{-19} x lambda^{2} x (4pi)^{-1}

        tot_photons = stellar_spectra * scale * np.exp(-tau) * ERG_TO_PHOT * waveL_range

        # if hipstar['HD_NO'] in [158926, 160578]:
        #     print(f"{hipstar['HIP_NO']} {hipstar['sp_type']} {stellar_spectra[sindex]['filename']} {sindex} {hipstar['V_mag']} {bflux} {vflux} {hipstar['scale']} {stellar_spectra[sindex]['spectrum'][windex]}")
    # print(scale, tot_photons)
    return scale, tot_photons


# # Example usage----------------------------------

# spectral_type = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9']

# for x in spectral_type:#:
#     hipline = {"sp_type": x, "temperature": 0}
#     temperature = GET_STAR_TEMP(hipline['sp_type'])
#     print(f"sptype: {x}, Temperature: {temperature}")

#     all_spectra = READ_CASTELLI_SPECTRA(Castelli_data)
#     stellar_spectra = StellarSpectrum()
#     data1 = all_spectra[temperature]
#     # print(data1)
#     stellar_spectra.temperature = float(data1['temp'])
#     stellar_spectra.wavelength = np.array(data1['wavelength'])
#     stellar_spectra.spectrum = np.array(data1['spectrum']) #*stellar_spectra.wavelength[w]**2 /2.99792458e18

#     tot_photons = []
#     for w in  range (len(stellar_spectra.wavelength)):  # change to ergs cm^{-2} s^{-1} A^{-1} by multiplying c / lambda^2 , also conversion #photon : erg* lanbda / hc = 
#         c = 2.99792458e18 # speed of light in A/s
#         V_mag = 2.5
#         ebv= 1
#         vindex = index_greater_than(stellar_spectra.wavelength , 5450)
#         vflux = stellar_spectra.spectrum[vindex]
#         scale = 3.64e-9 * pow(10, -0.4 * (V_mag - 3.1 * ebv)) * 4 * math.pi / vflux
#         photon_number = stellar_spectra.spectrum[w] * scale * ERG_TO_PHOT  * 2.99792458e18 /stellar_spectra.wavelength[w] 
#         tot_photons.append(photon_number)

#     # print(stellar_spectra.temperature , list(zip(stellar_spectra.wavelength, stellar_spectra.spectrum)))

    # # for i in range(len(stellar_spectra.wavelength)):
    # #     stellar_spectra.spectrum[i] = 3.336e-19 * (4*np.pi)**(-1) * (stellar_spectra.wavelength[i])**2
    # # print(list(zip(stellar_spectra.wavelength, stellar_spectra.spectrum)))
    # # print (stellar_spectra.temperature,'\n',
    # #         stellar_spectra.wavelength,'\n',
    # #         stellar_spectra.spectrum)

    # low_UV = index_greater_than(stellar_spectra.wavelength, 100)
    # high_UV = index_greater_than(stellar_spectra.wavelength, 3800)
    # low_vis = index_greater_than(stellar_spectra.wavelength, 3800)
    # high_vis = index_greater_than(stellar_spectra.wavelength, 7500)


    # # Create the initial figure and axes
    # fig, ax1 = plt.subplots()

    # # # Plot the stellar spectrum on the primary y-axis (left)
    # # color1 = 'tab:blue'
    # # ax1.set_xlabel(r'Wavelength (Å)')
    # # ax1.set_ylabel(r'Star Surface Flux (#photons/s/cm²/A/str)', color=color1)
    # # ax1.plot(stellar_spectra.wavelength[low_UV:high_vis], tot_photons[low_UV:high_vis], color=color1, label='Photon flux')
    # # ax1.tick_params(axis='y', labelcolor=color1)
    # # ax1.legend(loc='upper left')

    # # Create a second y-axis (right) and plot the stellar spectra
    # ax2 = ax1.twinx()  # Create a twin axis sharing the same x-axis
    # color2 = 'grey'
    # ax2.set_ylabel(r'Star Surface Flux (ergs/s/cm²/A)', color=color2)  # Secondary y-axis label
    # ax2.plot(stellar_spectra.wavelength[low_UV:high_vis], stellar_spectra.spectrum[low_UV:high_vis],"--" , linewidth = 1 , color=color2, label='Energy distr.')
    # ax2.tick_params(axis='y', labelcolor=color2)
    # ax2.legend(loc='upper right')

    # # Add a title to the plot
    # plt.title(f'SED for {hipline["sp_type"]}')
    # plt.legend()

    # Save the plot to a file
    # plt.savefig(fr'B_spectra\{hipline["sp_type"]}.png')

    # Show the plot
    # plt.show()


#     import pandas as pd
#     df = pd.DataFrame(list(zip(stellar_spectra.wavelength[low_UV:high_UV],  tot_photons[low_UV:high_UV])), columns=['Wavelength', 'Spectrum'])
#     # Write the DataFrame to a CSV file
#     df.to_csv(f'B_spectra\\unscaled_{hipline["sp_type"]}.csv', index=False)

#-----------------------------------------------------------------------------------------------------------

# # TEST STARS...

# # HD 57061 - HIP 35415 ( o type)
# # HD 122451 (HADAR) - HIP 68702 (B)
# # HD 172167 (VEGA) - HIP 91262 (A)
# # HD 61421 (PROCYON) - HIP 37279 (F)

# df = pd.read_csv("hip_std.dat", header=None,
#                     sep = '|', skipinitialspace=True).iloc[:, [1, 5, 8, 9, 11, 32, 34, 37, 38, 76]]
# df.columns = ['hip', 'mag', 'ra_deg', 'de_deg', 'trig_parallax','B_mag', 'V_mag', 'B-V', 'e_B-V', 'Spectral_type']


# df['mar_size'] = 2*(10 - df['mag'])
# temp = []
# for i in range(len(df)):
#     temp.append(GET_STAR_TEMP(df["Spectral_type"][i]))
#     # print(i, temp)
# df["temp"] = temp
# df['distance'] = 1000/df["trig_parallax"]
# df['B_V'] = df['B_mag'] - df['V_mag']
# print(df)

# # Read the .dat file with whitespace as the delimiter using regex '\s+'
# filename = r'Castelli\crossec0.dat'
# df_C_section = pd.read_csv(filename, delimiter=r'\s+', names=['wavelength', 'c_section'])
# # print(df_C_section.head())
# filename2 = r'Castelli\crossec1.dat'
# new_df = pd.read_csv(filename2, delimiter=r'\s+', names=['wavelength', 'c_section'])
# # Create an interpolation function
# # interpolator = interp1d(df_C_section['wavelength'], df_C_section['c_section'], kind='linear', fill_value="extrapolate")

# plt.plot(df_C_section['wavelength'], df_C_section['c_section'], label='C_section data')
# plt.xlabel('Wavelength (Å)')
# plt.ylabel('Cross-section')
# plt.xscale('log')
# plt.yscale('log')
# plt.title('Cross-section data')
# plt.legend()
# plt.plot(new_df['wavelength'], new_df['c_section'],'--', label='Interpolated data')
# plt.show()

#     # if not os.path.exists(filename2):
#     #     print(f"File {filename} does not exist ")
#     #     new_c_sections = interpolator(stellar_spectra.wavelength)
#     #     new_c_sections = np.where(new_c_sections < min (df_C_section['c_section']),  min (df_C_section['c_section']), new_c_sections)
#     #     new_df = pd.DataFrame({'wavelength': stellar_spectra.wavelength, 'c_section': new_c_sections})
#     #     plt.plot(new_df['wavelength'], new_df['c_section'],'--', label='Interpolated data')
#     #     plt.show()

#     #     # Step 6: Save the new data to a file
#     #     new_df.to_csv(filename2, sep='\t', index=False, header=False, float_format='%.8e')

#     #     print(f"New cross-section data saved to {filename2}")

#     # else:
#     #     print(f"File {filename} exists ")
#     #     new_df = pd.read_csv(filename2, delimiter=r'\s+', names=['wavelength', 'c_section'])


# ebv_array = [0.140, 0.03, 0.009, 0.00]

# j = 3
# for j in range (4):
#     all_spectra = READ_CASTELLI_SPECTRA(Castelli_data)
#     stellar_spectra = StellarSpectrum()
#     temperature = df['temp'][j]
#     data1 = all_spectra[temperature]
#     stellar_spectra.temperature = float(data1['temp'])
#     stellar_spectra.wavelength = np.array(data1['wavelength'])
#     stellar_spectra.spectrum = np.array(data1['spectrum'])
#     print(data1)

#     low_UV = index_greater_than(stellar_spectra.wavelength, 1150)
#     high_UV = index_greater_than(stellar_spectra.wavelength, 3100)
#     low_vis = index_greater_than(stellar_spectra.wavelength, 3800)
#     high_vis = index_greater_than(stellar_spectra.wavelength, 7500)

#     vindex = index_greater_than(stellar_spectra.wavelength, 5450)
#     bindex = index_greater_than(stellar_spectra.wavelength, 4360)
#     bflux = stellar_spectra.spectrum[bindex]
#     vflux = stellar_spectra.spectrum[vindex]
#     # print (f'bflux:{bflux}, vflux:{vflux}')
#     b_mag = -2.5 * math.log10(bflux / 6.61)
#     v_mag = -2.5 * math.log10(vflux / 3.64)
#     print(vflux,bflux,v_mag,b_mag)
#     print(df['B-V'][j], b_mag- v_mag)

#     if df['B_mag'].isna().any():
#         ebv = df['B-V'][j] - (b_mag - v_mag)
#     else:
#         ebv = df['B_V'][j] - (b_mag - v_mag)
#     ebv = max(ebv, 0)
#     print(ebv)
#     scale = 3.64e-9 * pow(10, -0.4 * (df["mag"][j] - 3.1 * ebv_array[j] )) / vflux #df['e_B-V'][j]
#     Tau = new_df['c_section'] * ebv_array[j] * gas_to_dust
#     print(scale, Tau)
#     tot_flux = stellar_spectra.spectrum  *scale * np.exp(-Tau)
#     tot_photons = stellar_spectra.spectrum * scale * np.exp(-Tau) * ERG_TO_PHOT * stellar_spectra.wavelength

#     # tau = 3.1 * ebv / 1.0863
#     # scale = 3.64e-9 * pow(10, -0.4 * (df["mag"][j] )) * np.exp(tau) / vflux
#     # tot_flux2 = stellar_spectra.spectrum  *scale
#     # print(list(zip(stellar_spectra.wavelength, tot_flux, tot_flux2)))


#     file_path = fr'B_spectra\{df['hip'][j]}_spectra.txt'

#     # Read the file into a DataFrame
#     df2 = pd.read_csv(file_path, delimiter=r'\s+', header=None, names=['wavelength', 'flux'])
#     low_UV2 = index_greater_than(df2['wavelength'], 1150)
#     high_UV2 = index_greater_than(df2['wavelength'], 3100)

#     # Create the initial figure and axes
#     fig, ax1 = plt.subplots()

#     # Plot the stellar spectrum on the primary y-axis (left)
#     color1 = 'black'
#     ax1.set_xlabel(r'Wavelength (Å)')
#     ax1.set_ylabel(r'Star Energy Flux at earth (ergs/s/cm²/A)', color=color1)
#     ax1.plot(stellar_spectra.wavelength[low_UV:high_UV], tot_flux[low_UV:high_UV], color= color1, label='Energy flux')
#     ax1.plot(df2['wavelength'][low_UV2:high_UV2], df2['flux'][low_UV2:high_UV2], color = 'grey', label='Energy flux from CADS')
#     # ax1.plot(stellar_spectra.wavelength[low_UV:high_vis], tot_photons[low_UV:high_vis], color=color1, label='Photon flux')
#     ax1.tick_params(axis='y', labelcolor=color1)
#     ax1.legend(loc='upper left')

#     # Create a second y-axis (right) and plot the stellar spectra
#     ax2 = ax1.twinx()  # Create a twin axis sharing the same x-axis
#     color2 = 'red'
#     ax2.set_ylabel(r'Star photon Flux (#photons/s/cm²/A)', color=color2)  # Secondary y-axis label
#     ax2.plot(stellar_spectra.wavelength[low_UV:high_UV], tot_photons[low_UV:high_UV], '--', color = color2, label='Photon flux', linewidth = 0.7)
#     # ax2.plot(stellar_spectra.wavelength[low_UV:high_vis], stellar_spectra.spectrum[low_UV:high_vis],"--" , linewidth = 1 , color=color2, label='Energy distr.')
#     ax2.tick_params(axis='y', labelcolor=color2)
#     ax2.legend(loc='upper right')

#     # Add a title to the plot
#     plt.title(f'SED for Star {df['hip'][j]}')
#     plt.legend()


#     plt.savefig(fr'B_spectra\{df["hip"][j]}_spectrum.png', dpi = 250)
#     # Show the plot
#     plt.show()




