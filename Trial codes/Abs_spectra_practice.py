# Plot Absorption spectrum multiple on single imshow plot


# import matplotlib.pyplot as plt
# import numpy as np

# # Define x and y values
# x = np.linspace(-5, 5, 1000)
# y1 = x**2
# y2 = np.sin(x)

# # Create a figure and axis object
# fig, ax = plt.subplots()

# # Plot the two lines
# ax.plot(x, y1, color='blue')
# ax.plot(x, y2, color='green')

# # Shade the region between the lines only where y1 > 0 and y2 < 0
# ax.fill_between(x, y1, y2, where=(y1 > y2) & (y2 < 0), facecolor='gray', alpha=0.5)

# # Show the plot
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # Define x values and a function y(x)
# x = np.linspace(0, 2*np.pi, 100)
# y = np.sin(x)

# # Create a figure and axis object
# fig, ax = plt.subplots()

# # Plot the line
# ax.plot(x, y, 'k-')

# # Create a horizontal line above the sine curve
# upper_bound = np.ones_like(x)

# # Fill the region above the sine curve with a light color
# ax.fill_between(x, y, upper_bound, where=(y >= 0), color='lightblue')

# # Set the limits of the x and y axis
# ax.set_xlim(0, 2*np.pi)
# ax.set_ylim(-1.2, 1.2)

# # Show the plot
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # Define x values and two functions y1(x) and y2(x)
# x = np.linspace(0, 2*np.pi, 100)
# y1 = np.sin(x)
# y2 = np.cos(x)

# # Create a figure and axis object
# fig2, ax2 = plt.subplots()

# # Plot the two lines
# ax2.plot(x, y1, 'k-')
# ax2.plot(x, y2, 'k-')

# # Fill the region between the two lines with a lower bound
# ax2.fill_between(x, y1, y2, where=(y1 >= y2), color='lightblue')

# # Set the limits of the x and y axis
# ax2.set_xlim(0, 2*np.pi)
# ax2.set_ylim(-1.2, 1.2)

# # Show the plot
# plt.show()

# import matplotlib.pyplot as plt
# from matplotlib.figure import Figure
# import numpy as np


# def example_plot(ax, fontsize=12, hide_labels=False):
#     pc = ax.pcolormesh(np.random.randn(30, 30), vmin=-2.5, vmax=2.5)
#     if not hide_labels:
#         ax.set_xlabel('x-label', fontsize=fontsize)
#         ax.set_ylabel('y-label', fontsize=fontsize)
#         ax.set_title('Title', fontsize=fontsize)
#     return pc

# np.random.seed(19680808)

# # gridspec inside gridspec
# fig = plt.figure(layout='constrained', figsize=(7, 14))
# subfigs = fig.subfigures(1, 2, wspace=0.07)

# subfigs[0].figure.set_size_inches(6, 6)
# axsLeft = subfigs[0].add_subplot( )
# subfigs[0].set_facecolor('0.75')

# pc = example_plot(axsLeft)
# subfigs[0].suptitle('Left plots', fontsize='x-large')
# subfigs[0].colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')

# subfigs[1].figure.set_size_inches(6, 6)
# axsRight = subfigs[1].subplots(4, 1, sharex=True)
# for nn, ax in enumerate(axsRight):
#     # ax.figure.set_size_inches(6, 4)
#     pc = example_plot(ax, hide_labels=True)
#     if nn == 3:
#         ax.set_xlabel('xlabel')
#     if nn == 1:
#         ax.set_ylabel('ylabel')

# subfigs[1].set_facecolor('0.85')
# subfigs[1].colorbar(pc, shrink=0.6, ax=axsRight[0:4])
# subfigs[1].suptitle('Right plots', fontsize='x-large')

# fig.suptitle('Figure suptitle', fontsize='xx-large')

# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Define the colors for the colormap
colors = [(0, 0, 0), (0, 0, 1)]  # Black to Blue

# Create a colormap object
flux_cmap = LinearSegmentedColormap.from_list("BlackToBlue", colors)

# Assuming dec is a list or array containing the declination values
dec_min = 21
dec_max = 25
colors = plt.cm.rainbow(np.linspace(0, 1,  2000))
flux_cmap = mc.ListedColormap(colors[:200])
for j in range(len(colors)): 
    colors[j][0] = 0
    colors[j][1] = 0
    colors[j][2] = 0
    colors[j][3] = 1 

# Create a 2D array of zeros with the desired size
data = np.zeros((dec_max - dec_min, 3800 - 10))
for j, row in enumerate(data):
    print(j, row)

# # Create a figure and axis
# fig, ax = plt.subplots()

# Plot the black pseudocolor plot
# im = ax.imshow(data, cmap= flux_cmap, extent=(10, 3800, dec_min, dec_max), aspect='auto', vmin=0, vmax=1)

# # Add labels and colorbar
# plt.xlabel('Wavelength')
# plt.ylabel('Declination')
# plt.colorbar(im, label='Intensity')

# plt.show()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Define the colors for the colormap (black to blue gradient)
colors = [(0, 0, 0), (0, 0, 1)]  # Black to blue

# Create the colormap
cmap_name = 'black_to_blue'
custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)

# Plot example using the custom colormap
x = np.linspace(0, 2, 100)
y = np.zeros(100)
y[3:19] = 1
X, Y = np.meshgrid(x**2, y)
Z = X*Y  # Example data

spectra = plt.imshow(Z, cmap=custom_cmap, vmin=0, vmax=1)
plt.colorbar(label='Value')
plt.title('Custom Colormap: Black to Blue')
plt.show()
