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


from configparser import ConfigParser

from star_spectrum import *
from diffused_data import *
from view_orbit import get_folder_loc

folder_loc, params_file = get_folder_loc()

def read_parameter_file(filename= params_file, param_set = 'Params_1'):
    config = ConfigParser()
    config.read(filename)
    global sat_name, Interval, spectra_width, BG_wavelength, height
    sat_name = config.get(param_set, 'sat_name')
    azm = float(config.get(param_set, 'azm'))
    ele = float(config.get(param_set, 'ele'))
    Interval = float(config.get(param_set, 'interval_bw_Frames'))
    height = float(config.get(param_set, 'height'))
    spectra_width = float(config.get(param_set, 'longitudinal_spectral_width'))
    BG_wavelength = config.get(param_set, 'BG_wavelength')
    return azm, ele

# main animate function
def animate(time_arr, state_vectors, celestial_coordinates, spectral_fov, diffused_data, r ):
    # initiallize 3D earth and satellite view
    def init_orbit(ax):
        azm, ele = read_parameter_file()

        # set titles
        title = sat_name + ' Satellite position @ ' + time_arr[0].item().strftime('%Y-%m-%d - %H:%M:%S.')        
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
        ax.legend()
        # return
        return ax, satellite, orbit
    
    # init 2D sky view as seen in the velocity direction
    def init_sky(ax):
        global sky, diffused, text
               
        # set labels
        ax.set_xlabel('Right Ascension $^\circ$')
        ax.set_ylabel('Declination $^\circ$')
        
        # set titles
        ax.set_title('Sky view in the direction of velocity vector')
        
        # get initial frame celestial_coordinates data 
        P, S, Size = get_cles_data_by_frame(0, celestial_coordinates) 
        Size = Size[0]
        # print (Size)
        # print(Size)  
        # set axis limits
        ax.set_xlim(Size[0], Size[2])
        ax.set_ylim(Size[1], Size[3])  
        # ax.set_xlim(min(P[:,0]), max(P[:,0]))
        # ax.set_ylim(min(P[:,1]), max(P[:,1]))       

        #Scatter plot for Diffused light
        a =  0.05* 2/height
        loc_ra, loc_dec = random_scatter_data(diffused_data[0])
        diffused = ax.scatter(loc_ra, loc_dec, s= 0.05, alpha= a,  facecolors='Blue')

        tot_phot = calc_total_diffused_flux(diffused_data[0])
        info_text = f"Diffused UV Background\n    at {BG_wavelength} $\AA$\nNum_photons from diffused \n= {round(tot_phot, 3)} s\u207B\u00B9 cm\u207B\u00B2 $\AA$\u207B\u00B9 sr\u207B\u00B9"
        # print(info_text)
        text = ax.text(1.04, 0.6, info_text, transform=ax.transAxes, fontsize=7.5, va='center')

        # Scatter plot for stars
        if (S[0] == 0.0001) : #no star in the FOV
            sky = ax.scatter(P[0], P[1], s=S[0], facecolors='White')
        else:
            sky = ax.scatter(P[:,0], P[:,1], s=S, facecolors='white')
        # print(S)

        # background_flux = get_flux_ipixel(diffused_BG_wavelength, Size)
        
        # return
        return ax, sky , diffused
    
    # init Intrinsic Spectral plot for all stars in the FOV
    # def init_Spectra(ax):
    #     global flux
        
    #     # set labels
    #     ax.set_xlabel('log$_{10}$[ Wavelength ($\AA$)]')
    #     ax.set_ylabel('log$_{10}$[ Flux (FLAM)] + offset')
        
    #     # set titles
    #     ax.set_title('Spectrum of the stars in the Sky view')

    #     X_wavelength, Y_Spectra_per_star, ra, dec = get_spectral_data_by_frame(0, spectra)
    #     _, _, Size = get_cles_data_by_frame(0, celestial_coordinates) 
    #     Size = Size[0]

    #     if (X_wavelength[0]!=0):
    #         # y_offset = [float(float(d) - Size[1]) * (Size[3] - Size[1]) for d in np.array(dec)]
    #         # flux = ax4.plot(np.log10(X_wavelength), np.log10(Y_Spectra_per_star[0]), label = f'{ra[0]},{dec[0]}')
    #         for i in range(len(Y_Spectra_per_star)):
    #             y_offset = (dec[i] - Size[1])*(Size[3] - Size[1]) 
    #             # flux = ax.plot(np.log10(X_wavelength), Y_Spectra_per_star[i], label = f'ra: {ra[i]}  ; dec: {dec[i]}') #+ [y_offset]*len(Y_Spectra_per_star[i])
    #             flux = ax.plot(np.log10(X_wavelength),  np.log10(Y_Spectra_per_star[i]), label = f'ra: {ra[i]}  ; dec: {dec[i]}') # + [y_offset]*len(Y_Spectra_per_star[i])
    #             ax.set_ylim(-1, 11)
    #     else:
    #         wavelengths = np.linspace(100, 4000, 1000)
    #         y_zeros = np.zeros_like(wavelengths)
    #         flux= ax.plot(np.log10(wavelengths), y_zeros, color='gray', linestyle='--', label='y = 0')
    #         ax.set_ylim(-1, 11)
    #     ax.legend()
    #     # ax.clear()

    #     return ax, flux
    
    # init # of Photons plot
    def init_photons(ax):
        global phots
        
        # set labels
        ax.set_xlabel('Wavelength ($\AA$)')
        ax.set_ylabel('Number of Photons s\u207B\u00B9 cm\u207B\u00B2 $\AA$\u207B\u00B9 sr\u207B\u00B9')
        # ax.set_xlabel('Log[Wavelength ($\AA$)]')
        # ax.set_ylabel('log[Number of Photons]')

        # set titles
        ax.set_title('# of Photons from the stars in the Sky view')

        X_wavelength, Y_photons_per_star, ra, dec = get_photons_data_by_frame(0, spectral_fov)
        # _, _, Size = get_cles_data_by_frame(0, celestial_coordinates)  #used in y_offset
        # Size = Size[0]
        
        if (X_wavelength[0]!=0):
            wave_min = min(X_wavelength)
            wave_max = max(X_wavelength)
            max_p = get_max_photon(Y_photons_per_star)
            # y_offset = [float(float(d) - Size[1]) * (Size[3] - Size[1]) for d in np.array(dec)]
            # flux = ax4.plot(np.log10(X_wavelength), np.log10(Y_Spectra_per_star[0]), label = f'{ra[0]},{dec[0]}')
            for i in range(len(Y_photons_per_star)):
                # y_offset = (dec[i] - Size[1])*(Size[3] - Size[1]) 
                # phots = ax.plot(np.log10(X_wavelength), np.log10(Y_photons_per_star[i]), label = f'ra: {ra[i]}  ; dec: {dec[i]}') #+ [y_offset]*len(Y_Spectra_per_star[i])
                phots = ax.plot(X_wavelength, Y_photons_per_star[i], label = f'ra: {ra[i]}  ; dec: {dec[i]}') # + [y_offset]*len(Y_Spectra_per_star[i])
                ax.set_xlim(wave_min, wave_max)
                ax.set_ylim(-1, max_p)
                # ax.set_ylim(0, np.log10(max_p))
        else:
            wavelengths = np.linspace(100, 3800, 1000)
            y_zeros = np.zeros_like(wavelengths) 
            phots= ax.plot(wavelengths, y_zeros, color='gray', linestyle='--', label='No stars in Fov')
            # phots= ax.plot(np.log10(wavelengths), y_zeros, color='gray', linestyle='--', label='No stars in Fov')
            ax.set_xlim(min(wavelengths), max(wavelengths))
            ax.set_ylim(-100, 1e+6)
            # ax.set_ylim(0, 6)

        if len(ra)<=10:
            ax.legend()
        # ax.clear()

        return ax, phots
    
    # init Absorption Spectra plot for all stars in the FOV
    def init_spectra(ax):
        global spectra
        global colors,BtoB_cmap

        X_wavelength, Y_photons_per_star, ra, dec = get_photons_data_by_frame(0, spectral_fov)
        FOV_size = spectral_fov.frame_size[0]
        
        # set title
        ax.set_title('Absorption Spectra of each star')
        # set labels
        ax.set_xlabel('Wavelength $\AA$')
        ax.set_ylabel('Declination $^\circ$')

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
        global fig, ax2, ax3, ax4, ax5
        global orbit, satellite, sky, diffused, phots, spectra
        global X, Y, Z
        global RA, DEC
        
        # position vectors
        X, Y, Z = state_vectors[0], state_vectors[1], state_vectors[2]
        # Sent for figure
        font = {'size'   : 6}
        plt.rc('font', **font)
            
        # Create 2x2 sub plots
        gs = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.3, width_ratios=[1, 1] ) # , , width_ratios=[1, 2]
        # fig and ax
        # fig = plt.figure(figsize=(12,6))

        fig = plt.figure(layout='constrained', figsize=(12.7,6.5)) # figsize=(8,6)
        subfigs = fig.subfigures(2, 2, wspace=0.01, hspace= 0.02, width_ratios=[1, 1]) #, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
        # fig = plt.figure(layout='constrained', figsize=(10, 4))
        # subfigs = fig.subfigures(1, 2, wspace=0.07)
        # print(np.shape(subfigs))

        # axsLeft = subfigs[0].subplots(1, 2, sharey=True)
        # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

        # row 0, col 0
        # ax2 = subfigs[0,0].add_subplot( projection='3d')
        ax2 = fig.add_subplot(gs[0, 0], projection='3d' )
        # set layout
        ax2, satellite, orbit = init_orbit(ax2)  

        # plt.subplots_adjust(left=0.3, right=0.6, bottom=0.1, top=0.9)

        # row 1, col 0
        # ax3 = subfigs[1,0].add_subplot( facecolor="black", aspect= 0.6)
        ax3 = fig.add_subplot(gs[1, 0], facecolor="black", aspect= 0.25 )
        # initialize sky
        ax3, sky, diffused = init_sky(ax3)

        # # row 1, col 0 (Plot removed)
        # ax4 = fig.add_subplot(gs[1, 0])
        # # initialize Spectrum Plot
        # ax4, flux = init_Spectra(ax4)

        # row 0, col 1
        # ax4 = subfigs[0,1].add_subplot()
        ax4 = fig.add_subplot(gs[0, 1])
        # ax4.set_aspect('equal', adjustable='box')
        # initialize Photons Plot
        ax4, phots = init_photons(ax4)

        # row 1, col 1
        # ax5 = subfigs[1,1].add_subplot()
        ax5 = fig.add_subplot(gs[1, 1])
        # initialize Photons Plot
        ax5, spectra = init_spectra(ax5)


        # to avoid subplot title overlap with x-tick
        # fig.tight_layout()
        
        # return
        return fig, satellite, orbit, sky, diffused, phots, spectra

    def update(i, satellite, orbit, sky, diffused, phots, spectra):
        # stack as np columns for scatter plot
        xyi, xi, yi, zi = get_pos_data_by_frame(i)
        # print ('frame number',i+1,'- satellite path:', xi, yi, zi)
        title = sat_name+' Satellite position @ ' + time_arr[i].item().strftime('%Y-%m-%d - %H:%M:%S.')        
        ax2.set_title(title)
        # print("update animation")
        # _offsets3d for scatter
        satellite._offsets3d = ( xi, yi, zi )
        # .set_data() for plot...
        orbit.set_data(xi, yi)
        orbit.set_3d_properties(zi)

        # ax3.clear()
        # get frame data. pos[ra, dec], size
        P, S, Size = get_cles_data_by_frame(i, celestial_coordinates)
        Size = Size[0]
        loc_ra, loc_dec = random_scatter_data(diffused_data[i])
        diffused_offsets = np.column_stack((loc_ra, loc_dec))
        diffused.set_offsets(diffused_offsets)
        # print(S)
        # Update scatter object
        sky.set_offsets(P)
        # print('P is working')
        sky.set_sizes(S)
        # print('S is working')

        tot_phot = calc_total_diffused_flux(diffused_data[i])
        info_text = f"Diffused UV Background\n    at {BG_wavelength} $\AA$\nNum_photons from diffused \n= {round(tot_phot, 3)} s\u207B\u00B9 cm\u207B\u00B2 $\AA$\u207B\u00B9 sr\u207B\u00B9"
        text.set_text(info_text)

        # change sky limits
        ax3.set_xlim(Size[0], Size[2])
        ax3.set_ylim(Size[1], Size[3])    

        
        # setting up the number of photons vs wavelength plot
        ax4.clear()
        ax4.set_xlabel('Wavelength ($\AA$)')
        ax4.set_ylabel('Number of Photons s\u207B\u00B9 cm\u207B\u00B2 $\AA$\u207B\u00B9 sr\u207B\u00B9')
        # ax4.set_xlabel('Log[Wavelength ($\AA$)]')
        # ax4.set_ylabel('log[Number of Photons]')
        ax4.set_title('# of Photons from the stars in the Sky view')
        # get updated photons data by frame
        X_wavelength, Y_photons_per_star, ra, dec = get_photons_data_by_frame(i, spectral_fov)
        if (X_wavelength[0]!=0):
            wave_min = min(X_wavelength)
            wave_max = max(X_wavelength)
            max_p = get_max_photon(Y_photons_per_star)
            # y_offset = [float(float(d) - Size[1]) * (Size[3] - Size[1]) for d in np.array(dec)]
            # flux = ax4.plot(np.log10(X_wavelength), np.log10(Y_Spectra_per_star[0]), label = f'{ra[0]},{dec[0]}')
            for k in range(len(Y_photons_per_star)):
                # y_offset = (dec[i] - Size[1])*(Size[3] - Size[1]) 
                # phots = ax4.plot(np.log10(X_wavelength), np.log10(Y_photons_per_star[k]), label = f'ra: {ra[k]}  ; dec: {dec[k]}') #+ [y_offset]*len(Y_Spectra_per_star[i])
                phots = ax4.plot(X_wavelength,  Y_photons_per_star[k], label = f'ra: {ra[k]}  ; dec: {dec[k]}') # + [y_offset]*len(Y_Spectra_per_star[i])
                ax4.set_xlim(wave_min, wave_max)
                ax4.set_ylim(-1, max_p)
                # ax4.set_ylim(0, np.log10(max_p))
        else:
            wavelengths = np.linspace(100, 3800, 1000)
            y_zeros = np.zeros_like(wavelengths) 
            phots= ax4.plot(wavelengths, y_zeros, color='gray', linestyle='--', label='No stars in Fov')
            # phots= ax4.plot(np.log10(wavelengths), y_zeros, color='gray', linestyle='--', label='No stars in Fov')
            ax4.set_xlim(min(wavelengths), max(wavelengths))
            ax4.set_ylim(-1, 1e+6)
            # ax4.set_ylim(-1, 6)
        if len(ra)<=10:
            ax4.legend()


        # setting up the absorption spectra plots
        ax5.clear()
        ax5.set_title('Absorption Spectra of each star')
        ax5.set_xlabel('Wavelength $\AA$')
        ax5.set_ylabel('Declination $^\circ$')

        if (X_wavelength[0]!=0): #checks for stars in the field of view
            twoD_array = np.zeros((int((Size[3]-Size[1])*100), len(X_wavelength)))
            color_data = get_color_data(twoD_array, X_wavelength, Y_photons_per_star, dec, Size[1])

            # Create the absorption Spectra plot
            spectra = ax5.imshow(color_data, cmap=BtoB_cmap, aspect='auto', extent=(min(X_wavelength), max(X_wavelength), Size[3],Size[1]), vmin=0, vmax=1)
            ax5.invert_yaxis()
        else:
            wavelength = np.linspace(10,3800, 400)
            color_data = np.zeros(( int((Size[3]-Size[1])*100), len(wavelength) ))
            spectra = ax5.imshow(color_data, cmap=BtoB_cmap, aspect='auto', extent=(min(wavelength), max(wavelength), Size[3],Size[1]), vmin=0, vmax=1)
            ax5.invert_yaxis()
        # return
        return satellite, orbit, sky, diffused, phots, spectra
    # Press space bar to pause animation

    # run animation
    def run():
        # plot init
        fig, satellite, orbit, sky, diffused, phots, spectra = init()
        # total no of frames
        frame_count = len(X)
        # print (frame_count)
        
        global ani, paused
        paused = False
        fig.canvas.mpl_connect('key_press_event', toggle_pause)

        # create animation using the animate() function
        ani = animation.FuncAnimation(fig, update,
                                      frames=frame_count, interval= Interval, 
                                      fargs=(satellite, orbit, sky, diffused, phots, spectra ),
                                      blit=False, repeat=False)

        
        # show
        plt.show()
        print("animation complete")
        # save
        # ani.save(f'{folder_loc}Demo_file\{sat_name}satellite_try.gif', writer="ffmpeg")
        # print("saved")
        return ani
    
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
    # print(data[i])
    frame, d, frame_boundary = zip(data[i])
    # print (frame_boundary)
    # slice into columns
    if d[0]:
        c = list(zip(*d[0]))
        print('Frame',frame[0]+1,'has', len(c[0]),'stars.' )
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
                print( f"{str(i+1)}) Hipp_number= {str(hip[i])}; Ra & Dec: {str(ra[i])} {str(dec[i])}; Johnson Mag= {str(mag[i])}; trig Parallax= {str(parallax[i])}; Color(B-V)= {str(B_V[i])}; Spectral_Type: {str(Spectral_type[i])}; T_index: {Temp}" , end="\n")

        else:
            print('  The star in the FOV is:')
            Temp = GET_STAR_TEMP(str(Spectral_type[0]))
            print( f"  Hipp_number= {str(hip[0])}; Ra & Dec: {str(ra[0])} {str(dec[0])}; Johnson Mag= {str(mag[0])}; trig Parallax= {str(parallax[0])}; Color(B-V)= {str(B_V[0])}; Spectral_Type: {str(Spectral_type[0])}; T_index: {Temp}", end="\n")

        # return
        return cles_pos, size, frame_boundary 
    else:
        print('Frame',frame[0]+1,'is EMPTY', end="\n")
        no_star = [0,0]
        zero_size =(0.0001,)
        return no_star, zero_size, frame_boundary

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


        # # get updated spectral data by frame
        # X_wavelength, Y_Spectra_per_star, ra, dec = get_spectral_data_by_frame(i, spectra)

        # # Update Spectral plot
        # if (X_wavelength[0]!=0):
        #     ax4.clear()
        #     # y_offset = [float(float(d) - Size[1]) * (Size[3] - Size[1]) for d in np.array(dec)]
        #     # flux = ax4.plot(np.log10(X_wavelength), np.log10(Y_Spectra_per_star[0]), label = f'{ra[0]},{dec[0]}')
        #     for j in range(len(Y_Spectra_per_star)):
        #         # y_offset = (dec[i] - Size[1])*(Size[3] - Size[1])
        #         # flux = ax4.plot(np.log10(X_wavelength), Y_Spectra_per_star[i], label = f'ra: {ra[i]}  ; dec: {dec[i]}') #+ [y_offset]*len(Y_Spectra_per_star[i])
        #         flux = ax4.plot(np.log10(X_wavelength), np.log10(Y_Spectra_per_star[j]), label = f'ra: {ra[j]}  ; dec: {dec[j]}') #+ [y_offset]*len(Y_Spectra_per_star[i])
        #         ax4.set_ylim(-1, 11)        
        # else:
        #     ax4.clear()
        #     wavelengths = np.linspace(100, 4000, 1000)
        #     y_zeros = np.zeros_like(wavelengths)
        #     flux= ax4.plot(np.log10(wavelengths), y_zeros, color='gray', linestyle='--', label='y = 0')
        #     ax4.set_ylim(-1, 11)
        # ax4.legend()
        # # set labels
        # ax4.set_xlabel('log$_{10}$[ Wavelength ($\AA$)]')
        # ax4.set_ylabel('log$_{10}$[ Flux (FLAM)] + offset')
        # # set title
        # ax4.set_title('Spectrum of the stars in the Sky view')

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

# def get_diffused_in_FOV(wavelength, frame_boundary):
#     loc =[]
#     fluxes = []
#     with open(fr"{folder_loc}diffused_UV_data\flux_{wavelength}.csv", 'r', newline='') as csvfile:
#         csv_reader = csv.reader(csvfile)
#         for i, row in enumerate(csv_reader):
#             for j, flux_location in enumerate(row):
#                 flux, ra, dec = flux_location.split(', ')
#                 flux = float(flux[1:])
#                 ra = float(ra)
#                 dec = float(dec[:-1])
#                 if ra == 'nan':
#                     continue
#                 elif (ra >= frame_boundary[0]and  ra <= frame_boundary[2]):
#                     if (dec >= frame_boundary[1] and  dec <= frame_boundary[3]):
#                         loc.append((ra,dec))
#                         fluxes.append(flux)
#                         # print(i, j, ra, dec, flux)           
    
#     return loc, fluxes

# if n == 1:  # only 1 star in the Fov
#     spectra = ax.add_subplot( )
#     ax_pos = spectra
#     # photons_max = max(Y_photons_per_star[0])
#     #calculate alpha value from stars flux of photons, for each wavelength
#     alpha_val = calc_inv_opacity(Y_photons_per_star[0], len(Y_photons_per_star[0]), photons_max)  
    
#     # Calculates observance of a wavelength by its flux of photons
#     colors2 = calc_obs_color(colors, alpha_val, len(X_wavelength))
#     flux_cmap = mc.ListedColormap(colors2)

#     spectra.imshow(color_data[:].reshape(1, -1), cmap=flux_cmap, aspect='auto', extent=(wave_min, wave_max, Size[3],Size[1]))
#     spectra.set_xlabel(' Wavelength ($\AA$)')
#     spectra.set_ylabel(f'Star {i+1}')
# else:
#     spectra = ax5.subplots(n, 1, sharex=True)

#     for i, axs in enumerate(spectra): #plot spectra for each star 1 by 1
#         # photons_max = max(Y_photons_per_star[i])
#         alpha_val = calc_inv_opacity(Y_photons_per_star[i],len(Y_photons_per_star[i]), photons_max)
#         colors2 = calc_obs_color(colors, alpha_val, len(X_wavelength))
#         flux_cmap = mc.ListedColormap(colors2)       

#         spectra[i] =axs.imshow(color_data[:].reshape(1, -1), cmap=flux_cmap, aspect='auto', extent=(wave_min, wave_max, min(dec), max(dec)))
#         if (i == n-1):
#             axs.set_xlabel(' Wavelength ($\AA$)')
#         axs.set_ylabel(f'Star {i+1}: RA= {ra[i]}')
#         ax_pos.append(axs)
# Add colorbar
# norm = mc.Normalize(vmin=wave_min, vmax=wave_max)
# scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=mc.ListedColormap(colors))
# scalar_mappable.set_array([])  # Optional: set an empty array to the ScalarMappable
# ax.colorbar(scalar_mappable, orientation='horizontal', ax = ax_pos, label='Wavelength ($\AA$)')