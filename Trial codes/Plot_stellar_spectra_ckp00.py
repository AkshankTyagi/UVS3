# @D and 3D Plot of Spectra for 4 Stars with Flux vs wavelength and Declination

import matplotlib.pyplot as plt

def index_greater_than(lst, value):
    for i, elem in enumerate(lst):
        if elem >= value:
            return i
    return None  

with open(r'C:\Users\Akshank Tyagi\Documents\GitHub\UV-Sky-Simulations\ckp00\t40000g50p00.dat',"r") as a:
    wavelength = []
    Surface_Flux = []
    lines = a.readlines()

# Process the lines as needed
for line in lines:
    dfg = line.strip()
    lamda = dfg.split(' ')[0]
    wavelength.append(float(lamda))
    dfg1 = dfg[len(lamda):].strip()
    Surface_Flux.append(float(dfg1.split(' ')[0]))



low_UV = index_greater_than(wavelength, 100)
high_UV = index_greater_than(wavelength, 3800)
low_vis = index_greater_than(wavelength, 3800)
high_vis = index_greater_than(wavelength, 7500)

# print(wavelength, Surface_Flux)
print (len(wavelength))
fig, ax = plt.subplots()
ax.plot(wavelength[low_UV:high_UV], Surface_Flux[low_UV:high_UV], color='blue', label = r'ckp00_3500')
# ax.plot(wavelength[low_vis:high_vis], Surface_Flux[low_vis:high_vis], color='blue', label = r'50000g50p00')
# ax.plot(wavelength, Surface_Flux, color='blue', label = r't19000g40p00')

ax.set_xlabel(r'Wavelength- A')
ax.set_ylabel(r'Surface Flux')
# ax.set_xscale('log')
# ax.set_yscale('log')

# plt.savefig( 'C:\\Users\\Akshank Tyagi\\Desktop\\ckp00_3500.png' )
# plt.show()

# for line in a:
#         # print(line[7]), print(line[8]), print(line[9])
#         wavelength.append(line.split('     ')[0])
#         Surface_Flux.append(line.split('   ')[1])
# 


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Sample data for four stars
def get_data(temp):
    with open(f'C:\\Users\\Akshank Tyagi\\Documents\\GitHub\\UV-Sky-Simulations\\ckp00\\t{str(temp)}g50p00.dat',"r") as a:
        wavelength = []
        Surface_Flux = []
        lines = a.readlines()

    # Process the lines as needed
    for line in lines:
        dfg = line.strip()
        lamda = dfg.split(' ')[0]
        wavelength.append(float(lamda))
        dfg1 = dfg[len(lamda):].strip()
        Surface_Flux.append(float(dfg1.split(' ')[0]))
    return wavelength[low_UV:high_vis], Surface_Flux[low_UV:high_vis]

wavelength_star1, flux_star1 = get_data(5000)
declination_star1 = 20

wavelength_star2, flux_star2 = get_data(30000)
declination_star2 = 30

wavelength_star3, flux_star3 = get_data(15000)
declination_star3 = 45

wavelength_star4, flux_star4 = get_data(50000)
declination_star4 = 50

#----------------------------------------------------------

# Creating a 2D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot()

# flux = [flux_star1, flux_star2, flux_star3, flux_star4]
# print (flux[0], flux[1])
# for i in range(len(flux)):
#     flux_star = flux[i]
#     print(i,'\n',flux_star)
#     flux = ax.plot(wavelength_star1, flux_star, color='black', label = f'star 1') #'

# # Plotting the spectra
ax.plot(wavelength_star1, flux_star1,  label='Star 1') # declination_star1,
ax.plot(wavelength_star2, flux_star2, label='Star 2') # declination_star2,
ax.plot(wavelength_star3, flux_star3, label='Star 3') #, declination_star3,
ax.plot(wavelength_star4, flux_star4, label='Star 4') #, declination_star4

# Adding labels and title
ax.set_xlabel('Wavelength')
ax.set_ylabel('Flux')
# ax.set_zlabel('Declination')
# ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('2D Plot of Spectra for 4 Stars')
ax.legend()

# Display the plot
# plt.show()

#----------------------------------------------------------

# Creating a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the spectra
ax.plot(wavelength_star1, flux_star1, declination_star1, label='Star 1')
ax.plot(wavelength_star2, flux_star2, declination_star2, label='Star 2')
ax.plot(wavelength_star3, flux_star3, declination_star3, label='Star 3')
ax.plot(wavelength_star4, flux_star4, declination_star4, label='Star 4')

# Adding labels and title
ax.set_xlabel('Log(Wavelength)')
ax.set_ylabel('Flux')
ax.set_zlabel('Declination')
ax.view_init(elev=90, azim=-90)
# ax.set_yscale('log')
ax.set_title('3D Plot of Spectra for 4 Stars (Log Scale)')
ax.legend()

# Display the plot
plt.show()
# Assuming Size contains [xmin, xmax, ymin, ymax] for the plot



# 152 and 105
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# # Sample data for four stars
# wavelength_star1 = [4000, 4100, 4200, 4300, 4400]
# flux_star1 = [0.5, 0.6, 0.7, 0.8, 0.9]
# declination_star1 = 20

# wavelength_star2 = [4000, 4100, 4200, 4300, 4400]
# flux_star2 = [0.4, 0.5, 0.6, 0.7, 0.8]
# declination_star2 = 30

# wavelength_star3 = [4000, 4100, 4200, 4300, 4400]
# flux_star3 = [0.6, 0.7, 0.8, 0.9, 1.0]
# declination_star3 = 40

# wavelength_star4 = [4000, 4100, 4200, 4300, 4400]
# flux_star4 = [0.7, 0.8, 0.9, 1.0, 1.1]
# declination_star4 = 50



