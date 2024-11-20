# 3D plot of the Celestial sphere and the Satellite vectors used to calculate Angle b/w normal and RA, chi angle

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to draw a unit sphere (celestial sphere)
def plot_sphere(ax):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_wireframe(x, y, z, color='c', alpha=0.3, linewidth=0.5)

# Function to plot a vector from the origin
def plot_vector(ax, vec, label, color, start_point=np.array([0, 0, 0]), al = 1):
    ax.quiver(start_point[0], start_point[1], start_point[2],
              vec[0], vec[1], vec[2], color=color, label=label, alpha = al)

# Function to compute the angle between two vectors in radians
def angle_between_vectors(vec1, vec2):
    # Dot product of the vectors
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    # Compute the cosine of the angle using dot product formula
    cos_theta = dot_product / (magnitude_vec1 * magnitude_vec2)
    # Return the angle in radians using arccos (make sure the value is in range [-1, 1])
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

# Set up 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot celestial sphere
plot_sphere(ax)

# Define position vector (R), velocity vector (V), and normal vector (N)
a = -0.5
b = 0.5
c = -0.5
V = np.array([a, b, c])  # Example velocity vector
R = np.array([1, 0.5, - (1*a + 0.5*b)/c])  # Example position vector
N = np.cross(R, V)  # Orbital normal vector (perpendicular to R and V)
V = V / np.linalg.norm(V) 
R = R / np.linalg.norm(R) 
N = N / np.linalg.norm(N)  # Normalize the normal vector

x = 0.3 # Weighting factor for the linear combination of R and V
P = x*V + (1-x)* R #The Pointing vector givenn by the linear combination of R and V

# Plot vectors R, V, and N
plot_vector(ax, R, 'R (Position)', 'r')  # Red for position vector
# plot_vector(ax, P, 'P (Pointing Vector)', 'c')  # Cyan for pointing vector
plot_vector(ax, V, 'V (Velocity)', 'b')  # Green for velocity vector
t = np.linspace(0, 2 * np.pi, 100) 
circle_points = np.cos(t)[:, None] * R/5 + np.sin(t)[:, None] * V/5
ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], '--', color='black', lw=0.4, label="Satellite Orbit")
plot_vector(ax, V/5, 'Sat Velocity from orbit', 'b', start_point = R/5 , al = 0.3 )  
circle_points = np.cos(t)[:, None] * R + np.sin(t)[:, None] * V
ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color='r', lw=2, label="Orbital plane")
plot_vector(ax, N, 'N (Normal)', 'black')    # Blue for normal vector


# Calculate the point where P intersects the celestial sphere
V_intersect = P / np.linalg.norm(P)  # Normalize V{to get a unit vector

# Tangential RA and Dec vectors at the point where V intersects the sphere
# RA: A vector in the xy-plane (tangential along constant declination)
RA = np.array([-V_intersect[1], V_intersect[0], 0])  # Tangent in xy-plane
RA = RA / np.linalg.norm(RA)  # Normalize RA to unit length

# Dec: A vector in the vertical plane perpendicular to RA (tangential along constant RA)
Dec = np.cross(V_intersect, RA)  # Perpendicular to both V_intersect and RA
Dec = Dec / np.linalg.norm(Dec)  # Normalize Dec to unit length

# Plot RA and Dec vectors starting from the point where V intersects the celestial sphere
plot_vector(ax, V_intersect, 'P (Pointing Vector)', 'purple', start_point=np.array([0, 0, 0]))
plot_vector(ax, RA, 'RA (Right Ascension)', 'orange', start_point=V_intersect)
plot_vector(ax, Dec, 'Dec (Declination)', 'brown', start_point=V_intersect)

# Set plot limits
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])

# Add labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend( bbox_to_anchor=(0.3, 0.9), prop={'size': 7})


# Set equal aspect ratio for the plot
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=-45, azim=91)


# angle_rad = angle_between_vectors(N, Dec)
# angle_deg = np.degrees(angle_rad)

angle_rad = angle_between_vectors(N, RA)
angle_deg = np.degrees(angle_rad)
print(f"Angle between N and RA vector: {angle_deg:.2f} degrees ({angle_rad:.5f} radians )")
print(f"Angle between N and Dec vector: {angle_deg - 90:.2f} degrees ({angle_rad - np.pi/2:.5f} radians )")


plt.title('Celestial Sphere with Tangential RA and Dec Vectors')
plt.show()


