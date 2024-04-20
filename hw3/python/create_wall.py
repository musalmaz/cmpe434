import numpy as np
import math

def angle_between(p1, p2):
    """Calculate the angle in radians between two points."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.atan2(dy, dx)

def distance_between(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def create_wall_segment(mid_point, angle, length, height, thickness, color):
    """Generate XML for a single wall segment."""
    return (
        f"<body pos='{mid_point[0]} {mid_point[1]} {mid_point[2]}' euler='0 0 {math.degrees(angle)}'>"
        f"<geom type='box' size='{length / 2} {thickness / 2} {height / 2}' rgba='{color}'/>"
        "</body>"
    )

def create_adjusted_inner_path(path_points, offset):
    """Adjust path points for the inner wall to prevent overlap."""
    adjusted_points = []
    N = len(path_points)

    for i in range(N):
        p0 = path_points[i - 1]  # Previous point
        p1 = path_points[i]      # Current point
        p2 = path_points[(i + 1) % N]  # Next point

        # Calculate angles
        angle1 = angle_between(p0, p1)
        angle2 = angle_between(p1, p2)

        # Calculate offsets for each point
        offset1 = [p1[0] - offset * math.sin(angle1), p1[1] + offset * math.cos(angle1)]
        offset2 = [p1[0] - offset * math.sin(angle2), p1[1] + offset * math.cos(angle2)]

        # Midpoint of the offsets
        mid_offset = [(offset1[0] + offset2[0]) / 2, (offset1[1] + offset2[1]) / 2]

        adjusted_points.append(mid_offset)

    return adjusted_points

def create_walls_xml(path_points, wall_height=1.0, wall_thickness=0.1, wall_color='1 0 0 1', offset=0.5):
    """Create XML for walls (both inner and outer) along a given path."""
    outer_wall_xml = []
    inner_wall_xml = []
    inner_path = create_adjusted_inner_path(path_points, offset)

    for i in range(len(path_points)):
        start_point_outer = path_points[i]
        end_point_outer = path_points[(i + 1) % len(path_points)]
        start_point_inner = inner_path[i]
        end_point_inner = inner_path[(i + 1) % len(inner_path)]

        wall_length_outer = distance_between(start_point_outer, end_point_outer)
        wall_angle_outer = angle_between(start_point_outer, end_point_outer)
        wall_length_inner = distance_between(start_point_inner, end_point_inner)
        wall_angle_inner = angle_between(start_point_inner, end_point_inner)

        # Compute the position of the outer and inner walls
        mid_point_outer = [(start_point_outer[0] + end_point_outer[0]) / 2, (start_point_outer[1] + end_point_outer[1]) / 2, wall_height / 2]
        mid_point_inner = [(start_point_inner[0] + end_point_inner[0]) / 2, (start_point_inner[1] + end_point_inner[1]) / 2, wall_height / 2]

        # Create XML for outer and inner walls
        outer_wall_xml.append(create_wall_segment(mid_point_outer, wall_angle_outer, wall_length_outer, wall_height, wall_thickness, wall_color))
        inner_wall_xml.append(create_wall_segment(mid_point_inner, wall_angle_inner, wall_length_inner, wall_height, wall_thickness, wall_color))

    return outer_wall_xml, inner_wall_xml

# Define the path
path_points = [(1, 1), (5, 5), (10, 10), (5, 15), (0, 20), (-5, 25), (0, 40), (10, 30), (15, 10), (0,0)]


# Create wall XML segments for both inner and outer walls
outer_walls_xml, inner_walls_xml = create_walls_xml(path_points)

# Construct the full XML
full_xml = f"""<mujoco>
<worldbody>
{"".join(outer_walls_xml)}
{"".join(inner_walls_xml)}
</worldbody>
</mujoco>"""

# Write to XML file
with open("race_track", "w") as file:
    file.write(full_xml)
