# Test code to plot the Absorption spectra after dispersing light through a prism
# 
# Multiple Imshow Plots with Separate Y Axes

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

# def wavelength_to_rgba(wavelength, gamma=0.8):
#     ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
#     This converts a given wavelength of light to an 
#     approximate RGBA color value. The wavelength must be given
#     in nanometers in the range from 380 nm through 750 nm
#     (789 THz through 400 THz).

#     Based on code by Dan Bruton
#     http://www.physics.sfasu.edu/astro/color/spectra.html
#     Additionally alpha value set to 0.5 outside range
#     '''
#     wavelength = float(wavelength)
#     if wavelength >= 380 and wavelength <= 750:
#         A = np.random.random()
#     else:
#         A=0.5
#     if wavelength < 380:
#         wavelength = 380.
#     if wavelength >750:
#         wavelength = 750.
#     if wavelength >= 380 and wavelength <= 440:
#         attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
#         R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
#         G = 0.0
#         B = (1.0 * attenuation) ** gamma
#     elif wavelength >= 440 and wavelength <= 490:
#         R = 0.0
#         G = ((wavelength - 440) / (490 - 440)) ** gamma
#         B = 1.0
#     elif wavelength >= 490 and wavelength <= 510:
#         R = 0.0
#         G = 1.0
#         B = (-(wavelength - 510) / (510 - 490)) ** gamma
#     elif wavelength >= 510 and wavelength <= 580:
#         R = ((wavelength - 510) / (580 - 510)) ** gamma
#         G = 1.0
#         B = 0.0
#     elif wavelength >= 580 and wavelength <= 645:
#         R = 1.0
#         G = (-(wavelength - 645) / (645 - 580)) ** gamma
#         B = 0.0
#     elif wavelength >= 645 and wavelength <= 750:
#         attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
#         R = (1.0 * attenuation) ** gamma
#         G = 0.0
#         B = 0.0
#     else:
#         R = 0.0
#         G = 0.0
#         B = 0.0
#     return (R,G,B,A)

# clim=(350,780)
# norm = plt.Normalize(*clim)
# wl = np.arange(clim[0],clim[1]+1,2)
# colorlist = list(zip(norm(wl),[wavelength_to_rgba(w) for w in wl]))
# spectralmap = matplotlib.colors.LinearSegmentedColormap.from_list("spectrum", colorlist)

# fig, axs = plt.subplots(1, 1, figsize=(8,4), tight_layout=True)

# wavelengths = np.linspace(200, 1000, 1000)
# spectrum = (5 + np.sin(wavelengths*0.1)**2) * np.exp(-0.00002*(wavelengths-600)**2)

# plt.plot(wavelengths, spectrum, color='darkred' )
# plt.figure(facecolor="black")

# y = np.linspace(0, 6, 100)
# X,Y = np.meshgrid(wavelengths, y)

# extent=(np.min(wavelengths), np.max(wavelengths), np.min(y), np.max(y))

# # Create a custom colormap for black
# black = np.array([0, 0, 0, 1]).reshape(1, -1)  # Black color
# black_cmap = matplotlib.colors.ListedColormap(black)

# # Plot the black background
# plt.imshow(np.zeros_like(X), cmap=black_cmap, extent=extent, aspect='auto')

# # Plot the spectral colors with transparency
# plt.imshow(X, clim=clim, extent=extent, cmap=spectralmap, aspect='auto')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Intensity')

# plt.fill_between(wavelengths, spectrum, 8, color='w')
# plt.savefig('WavelengthColors.png', dpi=200)
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.colors

# # Number of plots
# n = 3

# # Create some data
# x = np.linspace(0, 10, 100)
# y = np.random.rand(100, n)

# # Create a custom colormap with transparency
# colors = plt.cm.viridis(np.linspace(0, 1, 256))
# colors[:, -1] = np.random.rand(256)  # Set transparency linearly
# colors[0, :] = [0, 0, 0, 1]  # Set first color to black
# cmap_with_alpha = matplotlib.colors.ListedColormap(colors)

# # Create subplots
# fig, axs = plt.subplots(n, 1, sharex=True, figsize=(8, 2*n))

# # Plot data on each subplot
# for i in range(n):
#     im = axs[i].imshow(y[:, i].reshape(1, -1), aspect='auto', cmap=cmap_with_alpha)
#     axs[i].set_yticks([0])
#     axs[i].set_yticklabels([f'Y{i+1}'])

# # Add labels and title
# plt.xlabel('X')
# plt.suptitle('Multiple Imshow Plots with Separate Y Axes and Transparency')
# plt.colorbar(im, ax=axs.ravel().tolist(), label='Value')
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.colors

# # Number of plots
# n = 4

# # Create some data
# x = np.tile(np.linspace(20, 4000, 1000),(n,1))
# y = np.tile(np.linspace( 0, 1,1000),(n,1))
# print(np.shape(y))

# # Create a custom colormap with transparency
# colors = plt.cm.rainbow(np.linspace(0, 1, 256))
# print(colors)
# # colors[:, -1] = np.random.random(256)  # Set transparency randomly from 1 to 0
# # print(colors)

# # print(np.array(cmap_with_alpha))

# # Create subplots
# fig, axs = plt.subplots(n, 1, sharex=True, figsize=(8, 2*n))


# # Plot data on each subplot
# for i in range(n):
#     colors[:, -1] = np.random.random(256) 
#     print(x[i, :])
#     print(colors)
#     cmap_with_alpha = matplotlib.colors.ListedColormap(colors)
#     print(cmap_with_alpha)
#     im = axs[i].imshow(x[i, :].reshape(1, -1), aspect='auto', cmap=cmap_with_alpha, vmin=0, vmax=1)
#     axs[i].set_yticks([0])
#     axs[i].set_yticklabels([f'Y{i+1}'])

# # Add labels and title
# colors = plt.cm.viridis(np.linspace(0, 1, 256))

# plt.xlabel('X')
# plt.suptitle('Multiple Imshow Plots with Separate Y Axes and Transparency')
# plt.colorbar(im, ax=axs.ravel().tolist(), label='Value')
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.colors

# # Number of plots
# n = 3

# # Generate some data (flux values for each wavelength)
# wavelengths = np.tile(np.linspace(350, 780, 1000),(n, 1))
# flux_data = (wavelengths - 350)/ (780-350)

# colors = plt.cm.rainbow(np.linspace(0, 1, 256))
# # Create custom RGBA colormap based on flux values
# def flux_to_rgba(i, flux):
#     colors_copy = colors
#     colors_copy[i, -1] = flux
#     return colors_copy[i]
#  # Blue color with varying alpha (transparency)

# # Create custom colormap

# # Create subplots
# fig, axs = plt.subplots(n, 1, sharex=True, figsize=(8, 4*n))

# # Plot data on each subplot
# for i in range(n):
#     flux = np.random.random(256)
#     print(i)
#     colors2 = [flux_to_rgba(j, flux[j]) for j in range(0,256)]
#     # print (colors2)
#     flux_cmap = matplotlib.colors.ListedColormap(colors2)
#     im = axs[i].imshow(flux_data[i, :].reshape(1, -1), cmap=flux_cmap, aspect='auto', extent=(350, 780, 0, 1))
#     axs[i].set_ylim(0, 1)  # Set y-axis limits to 0 and 1
#     axs[i].set_ylabel(f'Plot {i+1}')


# # Add labels and title
# plt.xlabel('Wavelength (nm)')
# plt.suptitle('Multiple Imshow Plots with Separate Y Axes')
# plt.colorbar(im, ax=axs.ravel().tolist(), label='Value')
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors

# Number of plots
n = 3

# Generate some data (flux values for each wavelength)
wavelengths = np.tile(np.linspace(350, 780, 350),(n, 1))
flux_data = (wavelengths - 350)/ (780-350)

# Create subplots
fig, axs = plt.subplots(n, 1, sharex=True, figsize=(8, 4*n))

# Plot data on each subplot
for i in range(n):
    # Generate random transparency levels for each color
    alpha_values = np.random.rand(350)
    
    # Create a custom colormap based on transparency levels
    colors = plt.cm.rainbow(np.linspace(0, 1, 1000))
    colors = colors[0:350]
    for j in range(350):
        colors[j][0] = colors[j][0] - (colors[j][0]*alpha_values[j])
        colors[j][1] = colors[j][1] - (colors[j][1]*alpha_values[j])
        colors[j][2] = colors[j][2] - (colors[j][2]*alpha_values[j])
        colors[j][3] = 1  # Set alpha value
    flux_cmap = matplotlib.colors.ListedColormap(colors)
    
    # Plot the data with the custom colormap
    im = axs[i].imshow(flux_data[i, :].reshape(1, -1), cmap=flux_cmap, aspect='auto', extent=(350, 780, 0, 1))
    axs[i].set_ylim(0, 1)  # Set y-axis limits to 0 and 1
    axs[i].set_ylabel(f'Plot {i+1}')

# Add labels and title
    
norm = matplotlib.colors.Normalize(vmin=350, vmax=780)
colors = plt.cm.rainbow(np.linspace(0, 1, 1000))
colors = colors[0:350]
scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=matplotlib.colors.ListedColormap(colors))
scalar_mappable.set_array([])  # Optional: set an empty array to the ScalarMappable
plt.xlabel('Wavelength (nm)')
plt.suptitle('Multiple Imshow Plots with Separate Y Axes')
plt.colorbar(scalar_mappable,ax = axs, label='Wavelength (nm)')


# Create a colorbar

# Set the ticks and labels for the colorbar
# ticks = np.linspace(350, 780, 8)
# cb.set_ticks(ticks)
# cb.set_ticklabels([f'{t:.0f}' for t in ticks])

plt.show()
