# Functions to animate satellite in orbit and stars data
# Author: Ravi Ram

import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.colors as mc
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
 

from configparser import ConfigParser

from star_spectrum import *
from diffused_data import *
from zodiacal_light import *
from Coordinates import *
from Params_configparser import get_folder_loc

folder_loc, params_file = get_folder_loc()

def read_parameter_file(filename= params_file):
    file_loc_set = 'Params_0'
    param_set = 'Params_1'

    config = ConfigParser()
    config.read(filename)
    global sat_name, Interval, spectra_width, BG_wavelength, height, width, allignment
    
    sat_name = config.get(param_set, 'sat_name')
    allignment = config.get(param_set, 'allignment_with_orbit')
    azm = float(config.get(param_set, 'azm'))
    ele = float(config.get(param_set, 'ele'))
    Interval = float(config.get(param_set, 'interval_bw_Frames'))
    height = float(config.get(param_set, 'height'))
    width = float(config.get(param_set, 'width'))
    spectra_width = float(config.get(param_set, 'longitudinal_spectral_width'))
    BG_wavelength = config.get(param_set, 'BG_wavelength')
    BG_wavelength  = [int(val) for val in BG_wavelength[1:-1].split(',')]

    return azm, ele

def read_components(filename= params_file, param_set = 'Params_2'):
    config = ConfigParser()
    config.read(filename)
    global diffused_bg, Spectra
    solar_marker = config.get(param_set, 'sun')
    lunar_marker = config.get(param_set, 'moon')
    G_plane = config.get(param_set, 'galactic_plane')
    diffused_bg = config.get(param_set, 'diffused_bg')
    Spectra = config.get(param_set, 'Spectra')  
    save_ani = config.get(param_set, 'save_animation')

    return solar_marker, lunar_marker, G_plane,  save_ani 


# print("plotting animation")
# main animate function
def animate(time_arr, state_vectors, celestial_coordinates, sol_position, spectral_fov, diffused_data, zodiacal_data, r ):
    # initiallize 3D earth and satellite view
    def init_orbit(ax):
        global sun, moon, satellite, orbit
        azm, ele = read_parameter_file()
        solar_marker, lunar_marker, G_plane,  _ = read_components()

        # set titles
        title = sat_name + ' Satellite position @ ' + time_arr[0].datetime.strftime('%Y-%m-%d - %H:%M:%S')
       
        ax.set_title(title)        
        
        # set labels
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        # set view
        ax.view_init(elev=ele, azim=azm)
        
        # set correct aspect ratio
        ax.set_box_aspect([1,1,1])
        set_axes_equal_3d(ax)
        
        # set limit
        size = 1.02
        limit = max(max(X), max(Y), max(Z))
        limit_low = min(min(X), min(Y), min(Z))
        ax.set_xlim(size*limit, -size*limit)
        ax.set_ylim(size*limit, -size*limit)
        ax.set_zlim(size*limit, -size*limit) 

        # earth
        # earth_size = r/(X[0]**2+ Y[0]**2+ Z[0]**2)**(1/2)
        # print(r,X[0], Y[0], Z[0], (X[0]**2+ Y[0]**2+ Z[0]**2), earth_size)
        radius = r  # Specify the radius of the sphere
        sphere_x, sphere_y, sphere_z = create_sphere(radius,(0, 0, 0), 100 )

        # ax.scatter(0, 0, 0, marker='o', c='deepskyblue', s= r)
        surf = ax.plot_surface(sphere_x, sphere_y, sphere_z, color='blue', alpha=0.75, label='Earth')
        surf.set_facecolor('deepskyblue')
        # satellite positions as a scatter plot
        satellite = ax.scatter(X[0], Y[0], Z[0], marker='o', c='k', s=2, label = sat_name)
        # orbit path as a dotted line plot
        orbit = ax.plot(X[0], Y[0], Z[0], linewidth=0.9, linestyle='-.', c='k')[0]


        distance = 10000
        if solar_marker == "True":
            sun_ra, sun_dec = sol_position["sun"][0]
            solar_sv = conv_eq_to_cart(sun_ra*15, sun_dec, 1)
            # Create new Sun quiver plot
            sun = ax2.quiver(solar_sv[0] * distance, solar_sv[1] * distance, solar_sv[2] * distance,
                            -solar_sv[0], -solar_sv[1], -solar_sv[2],
                            length=0.3*distance, color='orange', arrow_length_ratio=0.6, 
                            label=f'Sun: {sun_ra * 15:.4f}°RA, {sun_dec:.4f}°Dec')
        else:
            sun = ax2.quiver(0,0,0,0,0,0)
            print("Solar Position marker Not being Displayed")

        if lunar_marker == "True":
            moon_ra, moon_dec = sol_position["moon"][0]
            lunar_sv = conv_eq_to_cart(moon_ra*15, moon_dec, 1)
            # Create new Moon quiver plot
            moon = ax2.quiver(lunar_sv[0] * distance, lunar_sv[1] * distance, lunar_sv[2] * distance,
                        -lunar_sv[0], -lunar_sv[1], -lunar_sv[2],
                        length=0.3*distance , color='grey', arrow_length_ratio=0.6, 
                        label=f'Moon: {moon_ra * 15:.4f}°RA, {moon_dec:.4f}°Dec')
        else:
            moon = ax2.quiver(0,0,0,0,0,0)
            print("Lunar Position marker Not being Displayed")

        
        if G_plane == "True":
            gl = [0, 22, 120]
            gb = [0, 0, 0]
            galactic_ra, galactic_dec = conv_gal_to_eq(gl, gb)
            # print(galactic_ra, galactic_dec)
            sv = conv_eq_to_cart(galactic_ra, galactic_dec, 1)
            x, y, z = sv
            # print(list(zip(x, y, z)))

            dc = find_direction_cosines_plane(list(zip(x, y, z)))
            a, b, c = dc
            d = 0

            xlim = ylim = zlim = (-distance, distance)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)

            num_points = 1000
            # Calculate and plot intercepts on the XY plane (z=0)
            if c != 0:
                x = np.linspace(xlim[0], xlim[1], num_points)
                y = (d - a*x +  distance*c) / b
                mask = (y <=  distance) & (y >= - distance)
                x = x[mask] 
                y = y[mask]     
                z = np.ones_like(x)*- distance
                ax.plot(x, y, z, color='purple' )

                x = np.linspace(xlim[0], xlim[1], num_points)
                y = (d - a*x -  distance*c) / b
                mask = (y <=  distance) & (y >= - distance)
                x = x[mask] 
                y = y[mask]     
                z = np.ones_like(x)* distance
                ax.plot(x,y,z, color='purple' )
                
            if b != 0:
                z = np.linspace(zlim[0], zlim[1], num_points)
                x = (d - c*z  -  distance*b) / a
                mask = (x <=  distance) & (x >= - distance)
                z = z[mask]
                x = x[mask]
                y = np.ones_like(x)* distance
                ax.plot(x, y, z, color='purple' )

                z = np.linspace(zlim[0], zlim[1], num_points)
                x = (d - c*z  +  distance*b) / a
                mask = (x <=  distance) & (x >= - distance)
                z = z[mask]
                x = x[mask]
                y = np.ones_like(x)*- distance
                ax.plot(x, y, z, color='purple' )

            if a != 0:
                y =  np.linspace(ylim[0], ylim[1], num_points)
                z = (d - b*y +  distance*a) / c
                mask = (z <=  distance) & (z >= - distance)

                z = z[mask] 
                y = y[mask]
                x = np.ones_like(y)*- distance
                ax.plot(x, y, z, color='purple', label = 'Galactic plane' )

                y =  np.linspace(ylim[0], ylim[1], num_points)
                z = (d - b*y -  distance*a) / c
                mask = (z >= - distance) & (z<= distance)
                z = z[mask] 
                y = y[mask]     
                x = np.ones_like(y)* distance
                ax.plot(x, y, z, color='purple' )
        else:
            print("Galactic plane markers Not being Displayed")


        ax.legend(loc='center left', bbox_to_anchor=(0.95, 0), fontsize='small') #
        # return
        return ax, satellite, orbit, sun, moon
    
    # init 2D sky view as seen in the velocity direction
    def init_sky(ax):
        global sky,  text

        # set labels
        ax.set_xlabel(r'Right Ascension $^\circ$')
        ax.set_ylabel(r'Declination $^\circ$')
        
        # set titles
        ax.set_title('Sky view in the direction of Instrument FOV')
        
        # get initial frame celestial_coordinates data 
        P, S, corners, Size = get_cles_data_by_frame(0, celestial_coordinates) 

        # set axis limits
        ax.set_xlim(Size[0], Size[2])
        ax.set_ylim(Size[1], Size[3])  
        fOV_area = np.radians(height) * np.radians(width)
        a =  0.1* 1/height * 1/width  #alpha value for scatter plot

        #Scatter plot for Diffused light
        diffused = []

        if diffused_data != [0] or zodiacal_data != [0]:
            zod_data , zod_wavelengths = zodiacal_data
            wave_index = np.searchsorted(zod_wavelengths, BG_wavelength[-1], side='right') - 1
            wave_index = np.clip(wave_index, 0, len(zod_wavelengths)-1)
            colours = ['blue', 'purple']
            loc_ra, loc_dec = random_scatter_data(diffused_data[f'{BG_wavelength[-1]}'][0])
            loc_ra_zod, loc_dec_zod = random_scatter_zodiacal_data(zod_data[0], wave_index)
            zodiacal_wave = ax.scatter(loc_ra_zod, loc_dec_zod, s= 0.04, alpha= a, facecolors=colours[1], label = 'Zodiacal UV')
            diffused_wave = ax.scatter(loc_ra, loc_dec, s= 0.04, alpha= a, facecolors=colours[0], label = 'UV ISRF')
            diffused.append(diffused_wave)
            diffused.append(zodiacal_wave)

            info_diffused = ''
            for wavelength in BG_wavelength:
                wave_index = np.searchsorted(zod_wavelengths, wavelength, side='right') - 1
                wave_index = np.clip(wave_index, 0, len(zod_wavelengths)-1)

                info_line =f"{wavelength} $\\AA$ : {round(calc_total_diffused_flux(diffused_data[f'{wavelength}'][0])*fOV_area, 3)}  |  {round(calc_total_zodiacal_flux(zod_data[0], wave_index)*fOV_area, 3)}\n"
                info_diffused += info_line

            info_text = f"Total Diffused UV Background in FOV \n  $\\lambda$   : ISRF  |  Zod  (# Photons-s\u207B\u00B9cm\u207B\u00B2$\\AA$\u207B\u00B9)\n{info_diffused}"
            text = ax.text(1.1, 0.6, info_text, transform=ax.transAxes, fontsize=7.5, va='center')
            # print(info_text)
        else:
            info_text = 'Diffused UV Background\nNot included'
            text = ax.text(1.1, 0.6, info_text, transform=ax.transAxes, fontsize=7.5, va='center')
    
        # Scatter plot for stars
        if (S[0] == 0.0001) : #no star in the FOV
            sky = ax.scatter(P[0], P[1], s=S[0], facecolors='White')
        else:
            sky = ax.scatter(P[:,0], P[:,1], s=S, facecolors='white')
        
        # plot for FOV boundary
        if allignment != 'False':
            ax.plot(corners[:, 0], corners[:, 1], 'grey', linestyle='--', linewidth = 0.5, label = 'FOV boundary')
            ax.plot([corners[0, 0], corners[3, 0]], [corners[0, 1], corners[3, 1]], 'grey', linestyle='--', linewidth = 0.5)
            ax.set_aspect(2*(Size[2] - Size[0]) / (Size[3] - Size[1]))
        # background_flux = get_flux_ipixel(diffused_BG_wavelength, Size)
        ax.legend(loc='center left', bbox_to_anchor=(1, -0.04), fontsize='small', markerscale=15)
        # return
        return ax, sky , diffused
    
    # init # of Photons plot
    def init_photons(ax):
        global phots
        zod_data , zod_wavelengths = zodiacal_data
        
        # Create a twin Axes sharing the x-axis
        ax_r = ax.twinx() 

        # Put left axis on top of right axis
        ax.set_zorder(2)      # draw ax after ax_r
        ax.patch.set_visible(False)   # make ax background transparent so ax_r grid/labels don't hide it

        ax_r.set_zorder(1)
        ax_r.patch.set_visible(False)  # make right axis patch transparent too

        # Calculate the diffused ISRF spectra
        diffused_isrf = []
        fOV_area = np.radians(height) * np.radians(width)
        for wavelength in BG_wavelength:
            flux = round(calc_total_diffused_flux(diffused_data[f'{wavelength}'][0])*fOV_area, 3)
            diffused_isrf.append(flux)

        # Calculate the zodiacal light spectra
        zodiacal_spectra = np.round(calc_total_zodiacal_flux(zod_data[0])*fOV_area, 3)

        # Call the plotting function for background spectra (Diffuse/Zodiacal)
        ax_r.plot(BG_wavelength, diffused_isrf, marker='o', color='grey', label='Diffused UV ISRF')
        ax_r.plot(zod_wavelengths, zodiacal_spectra, color='black', label='Zodiacal UV')
        
        ax_r.set_ylabel(r'Diffused Background Photons (s\u207B\u00B9 cm\u207B\u00B2 $\AA$\u207B\u00B9)')
        ax_r.yaxis.set_label_position("right")

        # set  up the Stellar SED plot
        # set labels
        ax.set_xlabel(r'Wavelength ($\AA$)')
        ax.set_ylabel('Number of Photons s\u207B\u00B9 cm\u207B\u00B2 $\\AA$\u207B\u00B9')
        ax.set_title('# of Photons from the stars in the Sky view')

        X_wavelength, Y_photons_per_star, ra, dec = get_photons_data_by_frame(0, spectral_fov)
        
        if (X_wavelength[0]!=0):
            wave_min = min(X_wavelength)
            wave_max = max(X_wavelength)
            # max_p = get_max_photon(Y_photons_per_star)
            for i in range(len(Y_photons_per_star)):
                phots = ax.plot(X_wavelength, Y_photons_per_star[i], label = f'ra: {ra[i]:.3f}  ; dec: {dec[i]:.3f}', zorder = 5)
                ax.set_xlim(wave_min, wave_max)
        else:
            wavelengths = np.linspace(1000, 3800, 1000)
            y_zeros = np.zeros_like(wavelengths) 
            phots= ax.plot(wavelengths, y_zeros, color='gray', linestyle='--', label='No stars in Fov', alpha = 0.7)
            ax.set_xlim(min(wavelengths), max(wavelengths))
            ax.set_ylim(-0.1,2)
        ax.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.4)
        
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_r.get_legend_handles_labels()
        # if len(ra)<=10:
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(-0.4, 1.15), markerscale=0.5)

        return ax, ax_r, phots

        # return ax, phots
    
    # init Absorption Spectra plot for all stars in the FOV
    def init_spectra(ax):
        global spectra
        global colors,BtoB_cmap

        X_wavelength, Y_photons_per_star, ra, dec = get_photons_data_by_frame(0, spectral_fov)
        # print(f"init_spec: {spectral_fov.frame_size}")
        FOV_size = spectral_fov.frame_size[0]
        
        # set title
        ax.set_title('Absorption Spectra of each star')
        # set labels
        ax.set_xlabel(r'Wavelength $\AA$')
        ax.set_ylabel(r'Declination $^\circ$')

        # Create a custom colormap (black to blue gradient)
        colors = [(0, 0, 0), (0, 0, 1)]  # Black to blue
        cmap_name = 'black_to_blue'
        BtoB_cmap = mc.LinearSegmentedColormap.from_list(cmap_name, colors)

        if (X_wavelength[0]!=0): #checks for stars in the field of view
            twoD_array = np.zeros((int((FOV_size[3]-FOV_size[1])*100), len(X_wavelength)))
            color_data = get_color_data(twoD_array, X_wavelength, Y_photons_per_star, dec, FOV_size[1])

            # Create the absorption Spectra plot
            spectra = ax.imshow(color_data, cmap=BtoB_cmap, aspect='auto', extent=(min(X_wavelength), max(X_wavelength), FOV_size[3],FOV_size[1]), vmin=0, vmax=1) # , aspect='auto'
            ax.invert_yaxis()
            # print (color_data[:])
        else:
            wavelength = np.linspace(10,3800, 400)
            color_data = np.zeros(( int((FOV_size[3]-FOV_size[1])*100), len(wavelength) ))
            spectra = ax.imshow(color_data, cmap=BtoB_cmap, aspect='auto', extent=(min(wavelength), max(wavelength), FOV_size[3],FOV_size[1]), vmin=0, vmax=1) # , aspect='auto'
            ax.invert_yaxis()

        return ax, spectra

    # initialize plot
    def init():
        global fig, ax2, ax3, ax4, ax_r, ax5
        global orbit, satellite, sky, diffused, phots, spectra, sun, moon
        global X, Y, Z
        global RA, DEC
        
        # position vectors
        X, Y, Z = state_vectors[0], state_vectors[1], state_vectors[2]
        # Sent for figure
        font = {'size'   : 6}
        plt.rc('font', **font)
            
        # Create 2x2 sub plots
        fig = plt.figure( figsize=(12.7, 6.5)) # figsize=(8,6)
        gs = gridspec.GridSpec(2, 2, figure = fig, wspace = 0.6, hspace = 0.3,  width_ratios=[1, 1.7])

        # row 0, col 0
        ax2 = fig.add_subplot(gs[0, 0], projection='3d' )
        # set Earth- satellite layout
        ax2, satellite, orbit, sun, moon = init_orbit(ax2)  

        # row 1, col 0
        ax3 = fig.add_subplot(gs[1, 0], facecolor="black", aspect= 1 )
        # initialize sky
        ax3, sky, diffused = init_sky(ax3)

        # row 0, col 1
        ax4 = fig.add_subplot(gs[0, 1])
        # initialize Photons Plot
        ax4, ax_r, phots = init_photons(ax4)

        # row 1, col 1
        ax5 = fig.add_subplot(gs[1, 1])
        # initialize abs Spectra Plot
        ax5, spectra = init_spectra(ax5)

        # return
        return fig, satellite, orbit, sun, moon, sky, diffused, phots, spectra

    def update(i, satellite, orbit, sun, moon, sky, diffused, phots, spectra):
        # stack as np columns for scatter plot
        xyi, xi, yi, zi = get_pos_data_by_frame(i)
        # print ('frame number',i+1,'- satellite path:', xi, yi, zi)
        title = sat_name + ' Satellite position @ ' + time_arr[i].datetime.strftime('%Y-%m-%d - %H:%M:%S')

        ax2.set_title(title)
        # print("update animation")
        # _offsets3d for scatter
        satellite._offsets3d = ( xi, yi, zi )
        # .set_data() for plot...
        orbit.set_data(xi, yi)
        orbit.set_3d_properties(zi)

        # Remove old Sun and Moon quivers
        # Delete old Sun and Moon quivers if they exist
        # if sun is not None:
        #     sun.remove()
        #     print("deleting sun")
        # if moon is not None:
        #     moon.remove()
        #     print("deleting moon")
        # ax2.get_legend().remove()
        
        # # Update positions for Sun and Moon
        # sun_ra, sun_dec = sol_position["sun"][i]
        # moon_ra, moon_dec = sol_position["moon"][i]
        # solar_sv = conv_eq_to_cart(sun_ra * 15, sun_dec, 1)
        # lunar_sv = conv_eq_to_cart(moon_ra * 15, moon_dec, 1)
        
        # distance = 10000
        # sun_length = 0.1 * distance  # Adjust length as needed
        # moon_length = 0.1 * distance  # Adjust length as needed
        # # Update the legend
        # ax2.legend()

        ax3.clear()
        # get frame data. pos[ra, dec], size
        P, S, corners, Size = get_cles_data_by_frame(i, celestial_coordinates)
        # set labels
        ax3.set_xlabel(r'Right Ascension $^\circ$')
        ax3.set_ylabel(r'Declination $^\circ$')
        #Scatter plot for Diffused light
        diffused = []
        fOV_area = np.radians(height) * np.radians(width)
        a =  0.1#* 1/height * 1/width  #alpha value for scatter plot

        if diffused_data != [0] or zodiacal_data != [0]:
            zod_data , zod_wavelengths = zodiacal_data
            wave_index = np.searchsorted(zod_wavelengths, BG_wavelength[-1], side='right') - 1
            wave_index = np.clip(wave_index, 0, len(zod_wavelengths)-1)

            colours = ['blue', 'purple']
            loc_ra, loc_dec = random_scatter_data(diffused_data[f'{BG_wavelength[-1]}'][i])
            loc_ra_zod, loc_dec_zod = random_scatter_zodiacal_data(zod_data[i], wave_index)
            zodiacal_wave = ax3.scatter(loc_ra_zod, loc_dec_zod, s= 0.04, alpha= a, facecolors=colours[1], label = 'Zodiacal UV')
            diffused_wave = ax3.scatter(loc_ra, loc_dec, s= 0.04, alpha= a, facecolors=colours[0], label = 'UV ISRF') 
            diffused.append(diffused_wave)
            diffused.append(zodiacal_wave)

            info_diffused = ''
            for wavelength in BG_wavelength:
                wave_index = np.searchsorted(zod_wavelengths, wavelength, side='right') - 1
                wave_index =np.clip(wave_index, 0, len(zod_wavelengths)-1)

                zod_tot = calc_total_zodiacal_flux(zod_data[i], wave_index)
                info_line =f"{wavelength} $\\AA$: {round(calc_total_diffused_flux(diffused_data[f'{wavelength}'][i])*fOV_area, 3)}  |  {round(zod_tot*fOV_area, 3)}\n"
                info_diffused += info_line

            info_text = f"Total Diffused UV Background in FOV \n  $\\lambda$   :   ISRF  |  Zod  (# Photons- s\u207B\u00B9cm\u207B\u00B2$\\AA$\u207B\u00B9)\n{info_diffused}"
            ax3.text(1.1, 0.6, info_text, transform=ax3.transAxes, fontsize=7.5, va='center')
            # print(info_text)
        else:
            info_text = 'Diffused Background Not included'
            ax3.text(1.1, 0.6, info_text, transform=ax3.transAxes, fontsize=7.5, va='center')

        # Scatter plot for stars
        if (S[0] == 0.0001) : #no star in the FOV
            sky = ax3.scatter(P[0], P[1], s=S[0], facecolors='White')
        else:
            sky = ax3.scatter(P[:,0], P[:,1], s=S, facecolors='white')
        
        # plot for FOV boundary
        if allignment!= 'False':
            ax3.plot(corners[:, 0], corners[:, 1], 'grey', linestyle='--', linewidth = 0.5, label = 'FOV boundary')
            ax3.plot([corners[0, 0], corners[3, 0]], [corners[0, 1], corners[3, 1]], 'grey', linestyle='--', linewidth = 0.5)
            ax3.set_aspect(2*(Size[2] - Size[0]) / (Size[3] - Size[1]))

        ax3.legend(loc='center left', bbox_to_anchor=(1.0, -0.04), fontsize='small', markerscale=15)

        # change sky limits
        ax3.set_xlim(Size[0], Size[2])
        ax3.set_ylim(Size[1], Size[3])    
        
        # setting up the number of photons vs wavelength plot first the Diffused UV axis
        ax_r.clear()
        ax4.clear()
        # Put left axis on top of right axis
        ax4.set_zorder(2)      # draw ax after ax_r
        ax4.patch.set_visible(False)   # make ax background transparent so ax_r grid/labels don't hide it
        ax_r.set_zorder(1)
        ax_r.patch.set_visible(False)  # make right axis patch transparent too

        # Calculate the diffused ISRF spectra
        diffused_isrf = []
        fOV_area = np.radians(height) * np.radians(width)
        for wavelength in BG_wavelength:
            flux = round(calc_total_diffused_flux(diffused_data[f'{wavelength}'][i])*fOV_area, 3)
            diffused_isrf.append(flux)

        # Calculate the zodiacal light spectra
        zodiacal_spectra = np.round(calc_total_zodiacal_flux(zod_data[i])*fOV_area, 3)

        # Call the plotting function for background spectra (Diffuse/Zodiacal)
        ax_r.plot(BG_wavelength, diffused_isrf, marker='o', color='grey', label='Diffused UV ISRF')
        ax_r.plot(zod_wavelengths, zodiacal_spectra, color='black', label='Zodiacal UV')
        ax_r.set_ylabel('Diffused Background Photons (s\u207B\u00B9 cm\u207B\u00B2 $\\AA$\u207B\u00B9)')
        ax_r.yaxis.set_label_position("right")

        # setting up the number of photons vs wavelength plot for stars
        ax4.set_xlabel(r'Wavelength ($\AA$)')
        ax4.set_ylabel('Number of Photons s\u207B\u00B9 cm\u207B\u00B2 $\\AA$\u207B\u00B9')
        ax4.set_title('# of Photons from the stars in the Sky view')
        # get updated photons data by frame
        X_wavelength, Y_photons_per_star, ra, dec = get_photons_data_by_frame(i, spectral_fov)
        if (X_wavelength[0]!=0):
            wave_min = min(X_wavelength)
            wave_max = max(X_wavelength)
            max_p = get_max_photon(Y_photons_per_star)

            for k in range(len(Y_photons_per_star)):
                ax4.plot(X_wavelength,  Y_photons_per_star[k], label = f'ra: {ra[k]:.3f}  ; dec: {dec[k]:.3f}', zorder = 5) 
                ax4.set_xlim(wave_min, wave_max)

        else:
            wavelengths = np.linspace(1000, 3800, 1000)
            y_zeros = np.zeros_like(wavelengths) 
            phots= ax4.plot(wavelengths, y_zeros, color='gray', linestyle='--', label='No stars in Fov', alpha = 0.7)
            ax4.set_xlim(min(wavelengths), max(wavelengths))
            ax4.set_ylim(-0.1,2)

        ax4.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.3)
        
        # Combine legends from both axes
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax_r.get_legend_handles_labels()

        # if len(ra)<=10:
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(-0.4, 1.15), markerscale=0.5)

        # setting up the absorption spectra plots
        ax5.clear()
        ax5.set_title('Absorption Spectra of each star')
        ax5.set_xlabel(r'Wavelength $\AA$')
        ax5.set_ylabel(r'Declination $^\circ$')

        if (X_wavelength[0]!=0): #checks for stars in the field of view
            twoD_array = np.zeros((int((Size[3]-Size[1])*100), len(X_wavelength)))
            color_data = get_color_data(twoD_array, X_wavelength, Y_photons_per_star, dec, Size[1])

            # Create the absorption Spectra plot
            spectra = ax5.imshow(color_data, cmap=BtoB_cmap, aspect='auto', extent=(min(X_wavelength), max(X_wavelength), Size[3],Size[1]), vmin=0, vmax=1)
            ax5.invert_yaxis()
        else:
            wavelength = np.linspace(10,3800, 400)
            color_data = np.zeros(( int((Size[3]-Size[1])*100), len(wavelength)))
            spectra = ax5.imshow(color_data, cmap=BtoB_cmap, aspect='auto', extent=(min(wavelength), max(wavelength), Size[3],Size[1]), vmin=0, vmax=1)
            ax5.invert_yaxis()
        # return
        return satellite, orbit, sun, moon, sky, diffused, phots, spectra

    # run animation
    def run():
        # plot init
        fig, satellite, orbit, sun, moon, sky, diffused, phots, spectra = init()
        # total no of frames
        frame_count = len(X)
        
        global ani, paused
        paused = False
        fig.canvas.mpl_connect('key_press_event', toggle_pause)

        # create animation using the animate() function
        ani = animation.FuncAnimation(fig, update,
                                      frames=frame_count, interval= Interval, 
                                      fargs=(satellite, orbit, sun, moon, sky, diffused, phots, spectra ),
                                      blit=False, repeat=False)

        # show
        plt.show()
        print("Animation complete")
        
        # save
        _, _, _, save_ani =read_components()

        if save_ani == "True":
            current_date = time_arr[0].datetime.strftime('%d_%m_%Y')
            ani.save(fr'{folder_loc}Output{os.sep}{sat_name}satellite-{current_date}.gif', writer="ffmpeg") #_Large-Field_
            print("Animation Saved")
        else:
            print("Animation not Saved")

        return ani
    
    # Press space bar to pause animation
    def toggle_pause(event, *args, **kwargs):
        global ani, paused
        if event.key == ' ':
            if paused:
                ani.resume()
                print("animation resumed")
            else:
                ani.pause()
                print("animation paused")
            paused = not paused

    # run animation
    run()

    # end-plot-sky
    return

# Set 3D plot axes to equal scale. 
# Required since `ax.axis('equal')` and `ax.set_aspect('equal')` don't work on 3D.
# https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_equal_3d(ax: plt.Axes):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)
    return

# set axis limits
def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])
    return

def create_sphere(radius, center=(0, 0, 0), num_points=1000):
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

# get satellite position data for the given index
def get_pos_data_by_frame(i):
    # pack it like thisfor set_3d_properties   
    xi, yi, zi = X[..., :i+1],  Y[..., :i+1], Z[..., :i+1]
    xy = np.column_stack((xi, yi))
    # return
    return xy, xi, yi, zi

# get celestial coordinates of the stars for the given index
def get_cles_data_by_frame(i, data):
    # select a frame

    frame, d, frame_corner = data[i]
    # frame_corner = frame_corner[0]

    # print(f"plots)Frame {i+1} has {len(d[0])} stars, and frame corners = {frame_corner}")

    min_ra, min_dec = frame_corner.min(axis=0)
    max_ra, max_dec = frame_corner.max(axis=0)

    frame_boundary = [min_ra, min_dec, max_ra, max_dec]

    #  = frame_corner[0]

    # print (frame_boundary)
    # slice into columns
    if d:
        # print(d)
        c = list(zip(*d))
        # print(d[0], c)
        print('Frame',frame+1,'has', len(c[0]),'stars.' )
        # pack it
        #ra, dec, size = np.array(c[0]), np.array(c[1]), np.array(c[2])
        ra, dec, size = c[0], c[1], c[2]
        # stack as np columns for scatter plot
        cles_pos = np.column_stack((ra, dec))
        # Print Stellar Data of the stars in the FOV
        hip, mag, parallax, B_V, Spectral_type = c[3], c[4], c[5], c[6], c[7]
        if (len(c[0])>1):
            print('  The stars in the FOV are:')
            for i in range(len(c[0])):     
                Temp = GET_STAR_TEMP(str(Spectral_type[i]))
                print( f"{str(i+1)}) Hipp number= {str(hip[i])}; Ra & Dec: {str(ra[i])} {str(dec[i])}; Johnson Mag= {str(mag[i])}; Trig Parallax= {str(parallax[i])}; E(B-V)= {str(B_V[i])}; Spectral_Type: {str(Spectral_type[i]).strip()};" , end="\n")

        else:
            print('  The star in the FOV is:')
            Temp = GET_STAR_TEMP(str(Spectral_type[0]))
            print( f"  Hipp number= {str(hip[0])}; Ra & Dec: {str(ra[0])} {str(dec[0])}; Johnson Mag= {str(mag[0])}; Trig Parallax= {str(parallax[0])}; E(B-V)= {str(B_V[0])}; Spectral_Type: {str(Spectral_type[0]).strip()};", end="\n")

        # return
        return cles_pos, size,frame_corner, frame_boundary 
    else:
        print('Frame',frame+1,'is EMPTY', end="\n")
        no_star = [0,0]
        zero_size =(0.0001,)
        return no_star, zero_size, frame_corner, frame_boundary

# get Spectral data of the stars for the given frame index 
def get_spectral_data_by_frame(i, spectral_FOV):
    frame = spectral_FOV.frame[i], 
    Wavelength = spectral_FOV.wavelength[i]
    Spectra_per_star = spectral_FOV.spectra_per_star[i]
    ra = spectral_FOV.ra[i]
    dec = spectral_FOV.dec[i]

    return Wavelength, Spectra_per_star,ra, dec

# get photon number data of the stars for the given frame index 
def get_photons_data_by_frame(i, spectral_FOV):

    frame = spectral_FOV.frame[i], 
    Wavelength = spectral_FOV.wavelength[i]
    photon_per_star = spectral_FOV.photons[i]
    ra = spectral_FOV.ra[i]
    dec = spectral_FOV.dec[i]

    return Wavelength, photon_per_star,ra, dec

# get alpha values or inverse opacity 
def calc_inv_opacity(data, Range, max_val): 
    alpha_val = np.zeros(Range)
    for j in range(Range):
        if (data[j] <= 1):
            alpha_val[j] = (1)
        elif (data[j] >= 0.5 * max_val):
            alpha_val[j] = (max_val - data[j])/(5*max_val)
        elif ( data[j] >= 0.01 * max_val):
            alpha_val[j] = (max_val - data[j])/(4*max_val)
        elif (data[j] >= 0.001 * max_val):
            alpha_val[j] = (max_val - data[j])/(2*max_val)
        elif ( data[j] >= 0.00001 * max_val):
            alpha_val[j] = 0.65
        elif ( data[j] < 0.00001 * max_val):
            alpha_val[j] = 0.7

    return alpha_val

# get observed intensity of photons at each wavelength and convert to RGBA values for cmap 
def calc_obs_color(colors, alpha_val, Range):
    colors2 = np.zeros((Range, 4))
    for j in range(Range): 
        colors2[j][0] = colors[j][0] - (colors[j][0]*alpha_val[j])
        colors2[j][1] = colors[j][1] - (colors[j][1]*alpha_val[j])
        colors2[j][2] = colors[j][2] - (colors[j][2]*alpha_val[j])
        colors2[j][3] = 1 
    return colors2

#calculate Max Photons per wavelength from the stars in the FOV
def get_max_photon(photons_data):
    photons_max =0
    for i in range(len(photons_data)):
        MaxP = max(photons_data[i])
        if (MaxP >photons_max):
            photons_max = MaxP
    return photons_max

# make star spectra graph data from photons data
def get_color_data(data, wavelength, photons_data, dec, min_dec):
    spectra_width 
    width = spectra_width * height/3
    star_row =[]
    for decli in dec:
        star_row.append(int(((decli) - min_dec)*100))
    row_spread = int(width*100)

    photons_max = get_max_photon(photons_data)

    # for each star in the FOV edit color data rows for particular declination of the star
    for i in range(len(dec)): # ith star
        for j, row  in enumerate(data): # edit color data in rows for star i
            if (j >= star_row[i]- row_spread/2) and (j <= star_row[i] + row_spread/2):
                data[j] = row + get_photons_brightness(wavelength,photons_data[i], photons_max)
                # print(j,"row edited to")
    # print(data)
    return data

# from the Number of photons per wavelength recieved from a star, obtain a relative photon ratio that gives the brightnressn of that wavelength
def get_photons_brightness(wavelength, photon_data, photon_max):
    rel = np.zeros(len(wavelength))

    for i, num_photons in enumerate(photon_data):
        if (num_photons <= 0.00001 * photon_max):
            rel[i] = 1 - (photon_max - num_photons)/(photon_max)
        elif (num_photons >= 0.5 * photon_max):
            rel[i] = 1 - (photon_max - num_photons)/(5*photon_max) #5
        elif (num_photons >= 0.01 * photon_max):
            rel[i] = 1 - (photon_max - num_photons)/(3.5*photon_max) #4
        elif (num_photons >= 0.0005 * photon_max):
            rel[i] = 1 - (photon_max - num_photons)/(2.5*photon_max) #2.5
        elif (num_photons >= 0.00001 * photon_max):
            rel[i] = 1 - (photon_max - num_photons)/(1.5*photon_max) #1.5
    return rel
