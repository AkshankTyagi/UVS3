# plots a 3D spherical celestial sphere and plots orbital plane and pole for that plane with great circle passing through the poles

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a spherical grid
theta = np.linspace(0, np.pi, 100)  # Latitude from 0 to pi (north to south)
phi = np.linspace(0, 2 * np.pi, 100)  # Longitude from 0 to 2pi (east to west)
theta, phi = np.meshgrid(theta, phi)

# Radius of the sphere
r = 1

# Convert spherical coordinates to Cartesian for grid
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

# Set up the plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the grid lines of the sphere (transparent surface)
ax.plot_wireframe(x, y, z, color='black', alpha=0.5, rstride=5, cstride=5,  linewidth=0.5)

# Define the normal vector to the plane of the circle
# Let's take an arbitrary normal vector for the plane, e.g., (1, 1, 1)
normal_vector = np.array([1/2, 1, 1])
normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize the vector

# Calculate the two poles where the normal vector intersects the sphere
pole_1 = normal_vector * r
pole_2 = -normal_vector * r

# Plot the poles
ax.scatter([pole_1[0], pole_2[0]], [pole_1[1], pole_2[1]], [pole_1[2], pole_2[2]], color='red', s=100, label="Poles")
# ax.scatter(, pole_2[1], pole_2[2], color='red', s=100, label="Pole 2")

# Generate points for the great circle (geodesic)
# Parametrize the great circle using the normal vector
t = np.linspace(0, 2 * np.pi, 100)  # Parameter for the circle

# Create an orthogonal basis for the great circle in the plane of the normal vector
# Find two perpendicular vectors to the normal vector
v1 = np.cross(normal_vector, [1, 0, 0])  # Cross product with a vector along x-axis
if np.linalg.norm(v1) == 0:  # Handle the case when normal is parallel to the x-axis
    v1 = np.cross(normal_vector, [0, 1, 0])
v1 = v1 / np.linalg.norm(v1)  # Normalize

v2 = np.cross(normal_vector, v1)  # Ensure v2 is perpendicular to both v1 and normal
v2 = v2 / np.linalg.norm(v2)  # Normalize

# Parametrize the great circle using v1 and v2
circle_points = np.cos(t)[:, None] * v1 + np.sin(t)[:, None] * v2

# Plot the great circle
ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color='r', lw=2, label="Orbital plane")

# Generate points for the second great circle passing through the poles
# The second great circle will lie in the plane formed by the normal vector and another arbitrary vector
t_poles = np.linspace(0, 2 * np.pi, 100)
v3 = np.cross(normal_vector, v1)  # Another vector in the plane of the poles
v3 = v3 / np.linalg.norm(v3)  # Normalize

# Parametrize the second great circle
second_circle_points = np.cos(t_poles)[:, None] * normal_vector + np.sin(t_poles)[:, None] * v3

# Plot the second great circle passing through the poles
ax.plot(second_circle_points[:, 0], second_circle_points[:, 1], second_circle_points[:, 2], alpha=0.5, color='blue', lw=2, label="Great Circle through Poles")

# Generate the third great circle passing through the poles
# This great circle will lie in another perpendicular plane
v4 = np.cross(normal_vector, v3)  # A third vector orthogonal to both normal_vector and v3
v4 = v4 / np.linalg.norm(v4)  # Normalize

third_circle_points = np.cos(t_poles)[:, None] * normal_vector + np.sin(t_poles)[:, None] * v4

# Plot the third great circle passing through the poles
ax.plot(third_circle_points[:, 0], third_circle_points[:, 1], third_circle_points[:, 2], alpha=0.5, color='blue', lw=2)

# Generate points for the fourth great circle passing through the poles
# Find a vector orthogonal to v1, v3, and v4
v5 = 2*v2+ 2* v1  # A vector orthogonal to both v1 and v3
v5 = v5 / np.linalg.norm(v5)  # Normalize
print (v5)
# Parametrize the fourth great circle
fourth_circle_points = np.cos(t_poles)[:, None] * normal_vector + np.sin(t_poles)[:, None] * v5

# Plot the fourth great circle passing through the poles
ax.plot(fourth_circle_points[:, 0], fourth_circle_points[:, 1], fourth_circle_points[:, 2], alpha=0.5, color='blue', lw=2)


# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set equal aspect ratio for better visualization
ax.set_box_aspect([1, 1, 1])

# Add a legend
ax.legend()

# Show the plot
plt.title("Great Circle with Poles Marked")
plt.show()
