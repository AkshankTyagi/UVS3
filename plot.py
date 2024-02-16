# Functions to animate satellite in orbit and stars data
# Author: Ravi Ram

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

from configparser import ConfigParser
config = ConfigParser()

from star_spectrum import *

def read_parameter_file(filename='init_parameter.txt', param_set = 'Params_1'):
    config.read(filename)
    global sat_name, Interval
    sat_name = config.get(param_set, 'sat_name')
    azm = float(config.get(param_set, 'azm'))
    ele = float(config.get(param_set, 'ele'))
    Interval = float(config.get(param_set, 'interval_bw_Frames'))
    return azm, ele

# main animate function
def animate(time_arr, state_vectors, celestial_coordinates, r ):
    # init 3D earth and satellite view
    def init_orbit(ax):
        azm, ele = read_parameter_file(filename='init_parameter.txt', param_set = 'Params_1')

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
        ax.scatter(0, 0, 0, marker='o', c='deepskyblue', s=r)
        # satellite positions as a scatter plot
        satellite = ax.scatter(X[0], Y[0], Z[0], marker='o', c='k', s=2)
        # orbit path as a dotted line plot
        orbit = ax.plot(X[0], Y[0], Z[0], linewidth=0.9, linestyle='-.', c='k')[0] 

        # return
        return ax, satellite, orbit
    
    # init 2D sky view as seen in the velocity direction
    def init_sky(ax):
        global sky
               
        # set labels
        ax.set_xlabel('Right Ascension $^\circ$')
        ax.set_ylabel('Declination $^\circ$')
        
        # set titles
        ax.set_title('Sky view in the direction of velocity vector')
        
        # get initial frame celestial_coordinates data 
        P, S, Size = get_cles_data_by_frame(0, celestial_coordinates) 
        Size = Size[0]
        # print(Size)  
        # set axis limits
        ax.set_xlim(Size[0], Size[2])
        ax.set_ylim(Size[1], Size[3])  
        # ax.set_xlim(min(P[:,0]), max(P[:,0]))
        # ax.set_ylim(min(P[:,1]), max(P[:,1]))       
        
        # Scatter plot
        if (S[0] == 0.0001) :
            sky = ax.scatter(P[0], P[1], s=S[0], facecolors='White')
        else:
            sky = ax.scatter(P[:,0], P[:,1], s=S, facecolors='Red')
        
        # return
        return ax, sky    
    
    # initialize plot
    def init():
        global fig, ax2, ax3
        global orbit, satellite, sky
        global X, Y, Z
        global RA, DEC
        
        # position vectors
        X, Y, Z = state_vectors[0], state_vectors[1], state_vectors[2]
        # Sent for figure
        font = {'size'   : 6}
        plt.rc('font', **font)
            
        # Create 2x2 sub plots
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) # , wspace=0.5, hspace=0.5, , width_ratios=[1, 2]

        # fig and ax
        fig = plt.figure(figsize=(12,6)) # figsize=(8,6)
        # row 0, col 0
        ax2 = fig.add_subplot(gs[0, 0], projection='3d' )
        # set layout
        ax2, satellite, orbit = init_orbit(ax2)        
        # row 0, col 1
        ax3 = fig.add_subplot(gs[0, 1], facecolor="black")
        
        # initialize sky
        ax3, sky = init_sky(ax3)

        # to avoid subplot title overlap with x-tick
        fig.tight_layout()
        
        # return
        return fig, satellite, orbit, sky

    def update(i, satellite, orbit, sky):
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
        
        # get frame data. pos[ra, dec], size
        P, S, Size = get_cles_data_by_frame(i, celestial_coordinates)
        Size = Size[0]
        # print(S)
        # Update scatter object
        sky.set_offsets(P)
        # print('P is working')
        sky.set_sizes(S)
        # print('S is working')   

        # change sky limits
        ax3.set_xlim(Size[0], Size[2])
        ax3.set_ylim(Size[1], Size[3])    
        # ax3.set_xlim(min(P[:,0]), max(P[:,0]))
        # ax3.set_ylim(min(P[:,1]), max(P[:,1]))
        
        # return
        return satellite, orbit, sky, 

    # run animation
    def run():
        # plot init
        fig, satellite, orbit, sky = init()
        # total no of frames
        frame_count = len(X)
        # print (frame_count)
        # create animation using the animate() function
        ani = animation.FuncAnimation(fig, update,
                                      frames=frame_count, interval= Interval, 
                                      fargs=(satellite, orbit, sky, ),
                                      blit=False, repeat=False)
        # save
        #ani.save('satellite.gif', writer="ffmpeg")
        # show
        plt.show()
        return ani
    
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
        hip, mag, parallax, B_V, Spectral_type = c[3], c[4], c[5],c[6], c[7]
        if (len(c[0])>1):
            print('  The stars in the FOV are:')
            for i in range(len(c[0])):     
                print( f"{str(i+1)}) Hipp_number= {str(hip[i])}; Ra & Dec: {str(ra[i])} {str(dec[i])}; Johnson Mag= {str(mag[i])}; trig Paraalax= {str(parallax[i])}; Color index(B-V)= {str(B_V[i])}; Spectral_Type: {str(Spectral_type[i])}" , end="\n")
                # Temp = GET_STAR_TEMP(str(Spectral_type[i]))
                # print('  Temperature Index of star: ', Temp)
        else:
            print('  The star in the FOV is:')
            print( f"  Hipp_number= {str(hip[0])}; Ra & Dec: {str(ra[0])} {str(dec[0])}; Johnson Mag= {str(mag[0])}; trig Paraalax= {str(parallax[0])}; Color index(B-V)= {str(B_V[0])}; Spectral_Type: {str(Spectral_type[0])}" , end="\n")
            # print('  Hipp_number='+str(hip[0])+'; Ra & Dec:'+str(ra[0])+' '+str(dec[0])+'; Johnson Mag='+str(mag[0])+'; trig Paraalax='+str(parallax[0])+'; Color index(B-V)='+str(B_V[0])+'; Spectral_Type:'+str(Spectral_type[0]) )
            # Temp = GET_STAR_TEMP(str(Spectral_type[0]))
            # print('  Temperature Index of star: ', Temp)
        # return
        return cles_pos, size, frame_boundary 
    else:
        print('Frame',frame[0]+1,'is EMPTY', end="\n")
        no_star = [0,0]
        zero_size =(0.0001,)
        return no_star, zero_size, frame_boundary


