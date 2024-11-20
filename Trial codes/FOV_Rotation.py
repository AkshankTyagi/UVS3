# Plots the rotated boundaries of the FOV and filters data within the FOV boundaries

import numpy as np
import matplotlib.pyplot as plt

def get_frame_boundaries(w, h, x, y, chi=0):
    # Convert chi to radians for rotation
    chi_rad = np.deg2rad(chi)

    # Calculate the four corners of the unrotated FOV (xmin, ymin, etc.)
    xmin = float(x) - float(w) / 2.0
    xmax = float(x) + float(w) / 2.0
    ymin = float(y) - float(h) / 2.0
    ymax = float(y) + float(h) / 2.0
    
    # Correct for boundary limits
    # xmin = 0 if xmin < 0 else xmin
    # xmax = 360 if xmax > 360 else xmax
    # ymin = -90 if ymin < -90 else ymin
    # ymax = 90 if ymax > 90 else ymax
    
    # Define the unrotated corners as (RA, Dec) pairs
    corners = np.array([[xmin, ymin],  # bottom-left
                        [xmin, ymax],  # top-left
                        [xmax, ymax], # top-right
                        [xmax, ymin]]) # bottom-right

    # Translate the FOV center to the origin for rotation
    translated_corners = corners - np.array([x, y])

    # Apply 2D rotation matrix to each corner (rotation around z-axis)
    rotation_matrix = np.array([[np.cos(chi_rad), -np.sin(chi_rad)],
                                [np.sin(chi_rad),  np.cos(chi_rad)]])
    
    rotated_corners = np.dot(translated_corners, rotation_matrix.T)

    # Translate the corners back to their original position
    rotated_corners += np.array([x, y])

    # Ensure RA stays within [0, 360] and Dec stays within [-90, 90]
    # for ra in rotated_corners[:, 0]:
    #     if ra < 0:
    #         ra = 0
    #     elif ra > 360:
    #         ra = 360
    # # rotated_corners[:, 0] = rotated_corners[:, 0] %360
    # rotated_corners[:, 0] = np.clip(rotated_corners[:, 0], 0, 360)
    # rotated_corners[:, 1] = np.clip(rotated_corners[:, 1], -90, 90)

    radius = np.sqrt((w/2)**2 + (h/2)**2)

    # Return both the unrotated and rotated corners
    return corners, rotated_corners, radius
    # return rotated_corners 

def plot_fov_boundaries(w, h, ra, dec, chi):
    # Get unrotated and rotated corners
    unrotated_corners, rotated_corners, radius = get_frame_boundaries(w, h, ra, dec, chi)
    
    # Add the first corner at the end to close the loop
    unrotated_corners = np.vstack([unrotated_corners, unrotated_corners[0]])
    rotated_corners = np.vstack([rotated_corners, rotated_corners[0]])
    # unrotated_corners[[1, 2, 3]] = unrotated_corners[[2, 3, 1]]
    # rotated_corners[[2, 3]] = rotated_corners[[3, 2]]
    
    # Enable interactive mode
    # plt.ion()

    # Plotting the unrotated and rotated boundaries
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(unrotated_corners[:, 0], unrotated_corners[:, 1], 'r-', label='Unrotated FOV')
    ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], 'b-', label=f'Rotated FOV by {chi}°')

    # Mark the center of the FOV
    ax.scatter([ra], [dec], color='green', marker='o', label='FOV Center')

    # Plot the circumscribing circle
    circle = plt.Circle((ra, dec), radius, color='green', fill=False, linestyle='--', label='Circumscribing Circle')
    ax.add_artist(circle)

    # Add labels and legend
    ax.set_xlabel('RA (degrees)')
    ax.set_ylabel('Dec (degrees)')
    ax.set_title('Field of View Boundaries (Unrotated vs Rotated)')
    ax.legend()

    # Set the aspect ratio to be equal
    ax.set_aspect('equal', 'box')

    # Display the plot
    plt.grid(True)
    plt.show()

def is_point_in_polygon(x, y, poly):
    """ Check if a point (x, y) is inside a polygon defined by `poly` vertices. """
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        # print (i, x, y, ":",p1x, p1y, p2x, p2y)
        if y >= min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        # print(xints)
                    if p1x == p2x or x <= xints:
                        # print (i, x, y, ":",p1x, p1y, p2x, p2y, xints, not inside)
                        # print(x, xints, not inside)
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def filter_by_fov(w, h, ra, de, chi): 
    # Frame field of view
    # Get valid boundaries  
    # width, height, _, _ = read_parameter_file()
    
    # Get the rotated corners of the FOV
    _, rotated_corners, _ = get_frame_boundaries(w, h, ra, de, chi)
    
    # Calculate min and max RA and Dec from rotated corners
    min_ra, min_dec = rotated_corners.min(axis=0)
    max_ra, max_dec = rotated_corners.max(axis=0)

    # Extract useful columns
    # mdf = mdf[['ra_deg', 'de_deg', 'mar_size', 'hip', 'mag', 'trig_parallax', 'B-V', 'Spectral_type']]
    
    # # Filter data within the boundaries
    # q = 'ra_deg >= @min_ra & ra_deg <= @max_ra & de_deg >= @min_dec & de_deg <= @max_dec' 
    # mdf_filtered = mdf.query(q)

    # # Convert rotated corners to a list of tuples for polygon testing
    # polygon = [tuple(corner) for corner in rotated_corners]
    
    # Apply polygonal filtering
    # mdf_filtered = mdf_filtered[mdf_filtered.apply(lambda row: is_point_in_polygon(row['ra_deg'], row['de_deg'], polygon), axis=1)  ]
    
    # Return filtered data and frame boundaries
    frame_boundaries = [min_ra, min_dec, max_ra, max_dec]
    return rotated_corners, frame_boundaries #mdf_filtered,



# Example usage
w, h = 10, 20  # Width and height of FOV in degrees
ra, dec = 58, 30  # Center of FOV (RA, Dec)
chi = 20  # Rotation angle in degrees


plot_fov_boundaries(w, h, ra, dec, chi)

rotated_corners, frame_boundaries = filter_by_fov(w, h, ra, dec, chi)
polygon = [tuple(corner) for corner in rotated_corners]
print(polygon)
i = 58
j = 20
print(i, j,is_point_in_polygon(i, j, polygon))
# x = np.linspace(ra-w/2-1, ra+w/2+1, 13)
# y = np.linspace(dec-h/2-1, dec+h/2+1, 23)
# # x = [53, 58, 63]
# # y = [20, 30, 40]
# for i in x:
#     for j in y:
#         print(i, j,is_point_in_polygon(i, j, polygon))