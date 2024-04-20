import osmnx as ox
import shapely
import numpy as np


import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def transform_sd2cart(sd_path_2d, reference_path_2d):
    # Interpolate reference path
    s_values = np.cumsum(np.sqrt(np.sum(np.diff(reference_path_2d, axis=0)**2, axis=1)))
    s_values = np.insert(s_values, 0, 0)
    interp_x = interp1d(s_values, reference_path_2d[:, 0], kind='linear')
    interp_y = interp1d(s_values, reference_path_2d[:, 1], kind='linear')

    # Initialize Cartesian path
    cartesian_path = []

    for s, d in sd_path_2d:
        # Find closest point on reference path and its tangent
        closest_point_x = interp_x(s)
        closest_point_y = interp_y(s)

        # Compute the direction (tangent) at the closest point
        if s == s_values[-1]:  # If at end of path, take previous segment's direction
            dx = reference_path_2d[-1, 0] - reference_path_2d[-2, 0]
            dy = reference_path_2d[-1, 1] - reference_path_2d[-2, 1]
        else:
            dx = interp_x(s + 1) - closest_point_x
            dy = interp_y(s + 1) - closest_point_y

        # Normalize the direction vector
        norm = np.sqrt(dx**2 + dy**2)
        dx /= norm
        dy /= norm

        # Calculate the perpendicular direction
        perp_dx = -dy
        perp_dy = dx

        # Calculate the final Cartesian coordinates
        final_x = closest_point_x + d * perp_dx
        final_y = closest_point_y + d * perp_dy

        cartesian_path.append((final_x, final_y))

    return np.array(cartesian_path)

# Visualization
def visualize_paths(reference_path, *sd_paths):
    plt.figure(figsize=(10, 10))
    plt.plot(reference_path[:, 0], reference_path[:, 1], label="Reference Path", color='black')

    for idx, sd_path in enumerate(sd_paths):
        cart_path = transform_sd2cart(sd_path, reference_path)
        plt.plot(cart_path[:, 0], cart_path[:, 1], label=f"SD Path {idx+1}")

    plt.legend()
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("SD Paths Transformed to Cartesian Coordinates")
    plt.grid(True)
    plt.show()

# Assuming your reference path is already processed and stored in `reference_path_2d`
# For example:
# reference_path_2d = np.column_stack((x, y))

# Define your sd_paths
sd_path_1 = [(0,0), (20, 0), (70, 25), (300, 25)]
sd_path_2 = [(0,0), (100, 0), (170, 25), (300, 25)]
sd_path_3 = [(0,0), (200, 0), (270, 25), (300, 25)]

# Visualize
# visualize_paths(reference_path_2d, sd_path_1, sd_path_2, sd_path_3)

G = ox.graph_from_place(
    "Boğaziçi Üniversitesi Güney Yerleşkesi", 
    network_type='drive'
)
route = [538400209, 269419454, 269419457] # OSM node ids for South Campus ramp

# Visualize the whole network
# ox.plot_graph(G)

# Visualize the route over the network
# ox.plot.plot_graph_route(G, route)

# osmnx uses shapely library to represent edge.geometry
# We need to extract raw coordinates of the road
edges = ox.utils_graph.get_route_edge_attributes(G, route)
mls = shapely.MultiLineString([e['geometry'] for e in edges])
path = shapely.line_merge(mls)

xc, yc = path.xy # in degree coordinates

# Conversion from longitude-latitude degree coordinates to meters
# ---
# The distance between two latitudes is always 111120m.
# The distance between two meridians is a function of latitude.
# Normalized to the starting point (South Campus Gate).
# ---
x = (np.array(xc) - xc[0]) * np.cos(np.deg2rad(yc[0])) * 111120
y = (np.array(yc) - yc[0]) * 111120