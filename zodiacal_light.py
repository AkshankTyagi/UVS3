
from astropy.coordinates import SkyCoord, get_sun
from astropy.time import Time
import astropy.units as u
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from configparser import ConfigParser
import time


# from star_spectrum import GET_STAR_TEMP
from Params_configparser import get_folder_loc
from star_data import is_point_in_polygon
from star_spectrum import index_greater_than

folder_loc, params_file = get_folder_loc()
# print(" diffused data")

def read_parameter_file(filename= params_file):
    file_loc_set = 'Params_0'
    param_set = 'Params_1'
    
    config = ConfigParser()
    config.read(filename)

    Sol_spec_file = config.get(file_loc_set, 'Sol_spectra_file')
    Zod_dist_file = config.get(file_loc_set, 'Zod_dist_table')
    min_lim = float(config.get(param_set, 'limit_min'))
    max_lim = float(config.get(param_set, 'limit_max'))

    return Sol_spec_file,Zod_dist_file, min_lim, max_lim


def pointing_geometry(target_coords, obstime):
    """
    Compute heliocentric elongation and ecliptic latitude of a target.

    Parameters
        ra_deg , dec_deg (Array float): Target Right Ascension  and Declination in degrees (ICRS).
        obstime (astropy.time.Time): Observation time.

    Returns
        elongation (arrayfloat): Heliocentric elongation (deg) = angular separation between Sun and target along ecliptic.
        beta (array float): Heliocentric ecliptic latitude (deg) of target.
    """
    if not isinstance(obstime, Time):  # convert to astropy Time if needed
            obstime = Time(obstime)

    ra_arr, dec_arr = list(zip(*target_coords))
    # print(list(zip(*target_coords)))

    # Build a single SkyCoord with array inputs (fast)
    target = SkyCoord(ra=ra_arr * u.deg, dec=dec_arr * u.deg, frame="icrs")

    # Get Sun once (vectorized if obstime is array, but here we assume single time)
    sun_icrs = get_sun(obstime)

    # Transform both to ecliptic frame (vectorized)
    target_ecl = target.barycentrictrueecliptic
    sun_ecl = sun_icrs.barycentrictrueecliptic

    # Compute longitude difference and wrap to [0,180]
    # astropy Angle arithmetic is vectorized; result .deg yields numpy array
    delta_lon = np.abs((target_ecl.lon - sun_ecl.lon).wrap_at(180 * u.deg).deg)
    beta = np.abs(target_ecl.lat.deg)

    return np.asarray(delta_lon), np.asarray(beta)
    # elongation_arr =[]
    # beta_arr =[]
    # for coord in target_coords:
    #     ra_deg, dec_deg = coord

    #     # Target in ICRS
    #     target = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")
    #     # Sun position at obstime
    #     sun_icrs = get_sun(obstime)

    #     # Convert both to true ecliptic coordinates
    #     target_ecl = target.barycentrictrueecliptic
    #     sun_ecl = sun_icrs.barycentrictrueecliptic

    #     # Longitude difference (wrap to [0, 180])
    #     delta_lon = abs((target_ecl.lon - sun_ecl.lon).wrap_at(180*u.deg))
    #     elongation = delta_lon.deg
    #     # Ecliptic latitude of target
    #     beta = abs(target_ecl.lat.deg)

    #     elongation_arr.append(elongation)
    #     beta_arr.append(beta)

    # return elongation_arr, beta_arr


def read_zodiacal_spectrum(spec_file):
    """
    Reads a solar spectrum file in the CADS zodiacal light format.
    
    Parameters
        spec_file (str): Path to the spectrum file.
    
    Returns
        wave (np.ndarray): Wavelength array [Angstroms].
        flux (np.ndarray): Flux array [photons/cm²/s/sr/Å].
    """
    wave = []
    flux = []

    with open(spec_file, 'r') as f:
        for line in f:
            line = line.strip()
            # skip comments and blank lines
            if line.startswith('#') or line == '':
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            wave.append(float(parts[0]))
            flux.append(float(parts[1]))

    wave = np.array(wave)
    flux = np.array(flux)
    return wave, flux


def read_zodiacal_distribution(zod_file):
    """
    Reads the 2D zodiacal light distribution file (Leinert et al.)

    Args:
        zod_file (str): Path to the zodiacal light distribution file.

    Returns:
        zod_array (np.ndarray): 2D array of zodiacal intensities (n_lat x n_lon).
        hecl_lon (np.ndarray): Helio-ecliptic longitude grid in degrees.
        hecl_lat (np.ndarray): Helio-ecliptic latitude grid in degrees.
    """
    with open(zod_file, 'r') as f:
        lines = f.readlines()

    # Skip comment lines
    lines = [line for line in lines if not line.strip().startswith('#')]

    # First non-comment line: longitudes
    hecl_lat = np.array([float(x) for x in lines[0].split()])[1:]

    # Remaining lines: each starts with latitude, then intensity values
    hecl_lon = []
    zod_data = []

    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        hecl_lon.append(float(parts[0]))
        zod_data.append([float(x) for x in parts[1:]])

    hecl_lon = np.array(hecl_lon)
    zod_array = np.array(zod_data)*252  # to convert into ph cm-2 s-1 sr-1 A-1 at 5000 A, Multiply by 252     

    return zod_array, hecl_lon, hecl_lat


def scale_zodiacal_spectrum(zod_dist, table_ecl, table_beta, sol_wavelengths, sol_spec, helio_ecl_arr, helio_beta_arr):
    """
    Scale the solar spectrum using the Leinert zodiacal light distribution.

    Args:
        zod_dist (np.ndarray): 2D zodiacal light intensity (lat x lon) at 5000 A.
        table_ecl (np.ndarray): Helio-ecliptic longitude grid (degrees).
        table_beta (np.ndarray): Helio-ecliptic latitude grid (degrees).
        zod_wave (np.ndarray): Wavelength array of solar spectrum (Angstroms).
        sol_spec (np.ndarray): Solar spectrum in photon units.
        helio_ecl_arr (float): Heliocentric-ecliptic longitude of target.
        helio_beta_arr (float): Heliocentric-ecliptic latitude of target.

    Returns:
        np.ndarray: Scaled zodiacal light spectrum.
    """
    _, _, wave_min , wave_max = read_parameter_file()
    low_lim = index_greater_than(sol_wavelengths, wave_min)
    high_lim = index_greater_than(sol_wavelengths, wave_max)

    Zodiacal_spec_arr = []

    # Create interpolator with linear interpolation and extrapolation
    interpolator = RegularGridInterpolator(
        (table_ecl, table_beta),  # grid: latitude, longitude
        zod_dist,
        method='linear',
        bounds_error=False,
        fill_value=None  # None allows extrapolation
    )
    # intensity = interpolator([[abs(160), abs(30)]])[0] # Test point
    # print(f"Zodiacal intensity at helio_ecl=160, helio_beta=30: {intensity} ph/cm²/s/sr/Å")

    for helio_ecl, helio_beta in zip(helio_ecl_arr, helio_beta_arr):

        # Evaluate zodiacal intensity at the target coordinates # Absolute latitude because table is symmetric
        zod_intensity = interpolator([[abs(helio_ecl), abs(helio_beta)]])[0]
        print
        # Find index closest to 5000 Å
        wave_index = np.searchsorted(sol_wavelengths, 5000, side='right') - 1
        wave_index = np.clip(wave_index, 0, len(sol_wavelengths)-1)

        # Scale factor
        zod_scale = zod_intensity / sol_spec[wave_index]
        if zod_intensity < 0:
            print(f"Warning: Negative zodiacal intensity ({zod_intensity}) at helio_ecl={helio_ecl}, helio_beta={helio_beta}. Setting scale factor to zero.")

        # print(f"Solar intensity at 5000 Å: {sol_spec[wave_index]:.4f} ph/cm²/s/sr/Å")
        # print(f"Zodiacal intensity at target: {zod_intensity:.4f} ph/cm²/s/sr/Å")
        # print(f"Scaling factor (at 5000 Å): {zod_scale:.4f}")

        # Apply scaling
        sol_spec_scaled = sol_spec * zod_scale
        Zodiacal_spec_arr.append(sol_spec_scaled[low_lim:high_lim])
    
    return Zodiacal_spec_arr, sol_wavelengths[low_lim:high_lim]


def get_zodiacal_in_FOV( data, time_arr ):
    """
    Calculate zodiacal spectra for points inside each frame and report timing.
    Args:
        data (list): Frame celestial data list.
        time_arr (list or astropy.time.Time): Observation times aligned with frames.
    Returns:
        zodiacal_data (list): list of points_with_spectrum per frame.
        zod_wavelengths (np.ndarray): wavelengths returned by scale_zodiacal_spectrum.
    """
    # print('\nCalculating Zodiacal UV in the FOV (timed).')
    print('Calculating Zodiacal UV in the FOV.')

    # Read the Parameter file to get Data file locations
    spec_file, Zod_dist_file, _, _ = read_parameter_file()
    
    # Read the Solar Spectra File
    wavelength, flux = read_zodiacal_spectrum(spec_file)

    # Read the Zodiacal light Distribution table
    zod_array, table_lon, table_lat = read_zodiacal_distribution(Zod_dist_file)

    zodiacal_data = []
    t0 = time.perf_counter()
    
    for f in range(len(data)):    # f represents frame number
        _, _, frame_corner = zip(data[f])
        frame_corner= frame_corner[0]
        xmin, ymin = frame_corner.min(axis=0)
        xmax, ymax = frame_corner.max(axis=0)

        # Create arrays with 0.1 degree spacing
        x_arr = np.arange(xmin, xmax + 0.1, 0.1)  # include xmax
        y_arr = np.arange(ymin, ymax + 0.1, 0.1)  # include ymax
        X, Y = np.meshgrid(x_arr, y_arr)

        # Now, use is_point_in_polygon to keep only the points inside the rotated FOV
        polygon = frame_corner  # The polygon is defined by the rotated FOV corners
        filtered_points = []
        for ra, dec in zip(X.ravel(), Y.ravel()):
            # print(ra,dec)
            if is_point_in_polygon(ra, dec, polygon):
                filtered_points.append([ra, dec])

        # Calculate the elongation and beta angle for each mesh point
        elong_arr, beta_arr = pointing_geometry(filtered_points, time_arr[f])

        # Calculate the Zodiacal spectra for each point
        scaled_spectrum, zod_wavelengths = scale_zodiacal_spectrum(zod_array, table_lon, table_lat, wavelength, flux, elong_arr, beta_arr)

        points_with_spectrum = [(ra, dec, s) for (ra, dec), s in zip(filtered_points, scaled_spectrum)]
        zodiacal_data.append(points_with_spectrum)

    t1 = time.perf_counter()
    print(f"Total time to calculate zodiacal: {t1 - t0}, per frame: {(t1-t0)/len(data)}")

    return zodiacal_data, zod_wavelengths


def calc_total_zodiacal_flux(diffused_data, wavelength_idx = None):
    data = diffused_data
    c = list(zip(*data))
    tot_flux  = sum(c[2])
    # print(tot_flux)
    if wavelength_idx is not None:
        # print(tot_flux[wavelength_idx])
        return tot_flux[wavelength_idx]
    else:
        return tot_flux
    
def random_scatter_zodiacal_data(diffused_data, wavelength_idx):
    data = diffused_data
    c = list(zip(*data))
    ra, dec, fluxes = c[0], c[1], c[2]
    fluxes = np.array(fluxes)
    # print(fluxes[:, wavelength_idx], fluxes)
    fluxes = fluxes[:, wavelength_idx]

    pixel_size = np.radians(0.1)*np.radians(0.1)

    ra_norm = []
    dec_norm = []
    for i, flux in enumerate(fluxes):
        # if flux <= 0:
        #     continue
        ra_N = np.random.normal(ra[i], 0.07, size= int(1e4*flux*pixel_size))
        dec_N = np.random.normal(dec[i], 0.07, size= int(1e4*flux*pixel_size))
        for j in range(len(ra_N)):
            ra_norm.append(ra_N[j])
            dec_norm.append(dec_N[j])
    
    return ra_norm, dec_norm
