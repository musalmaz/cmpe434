import matplotlib.pyplot as plt
from config import *
from functions import *
import math

CONFIG =  CONFIG()

class PurePursuit():
    def __init__(self, circle_radius):
        self.circle_radius = circle_radius # as meters

    def line_circle_intersection(self, circle_center, line_start, line_end):
        circle_center = np.array(circle_center)
        line_start = np.array(line_start)
        line_end = np.array(line_end)

        d = line_end - line_start
        f = line_start - circle_center

        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - self.circle_radius**2

        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return []  # No intersection

        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        intersections = []
        for t in [t1, t2]:
            if 0 <= t <= 1:  # Check if intersection is within segment
                intersections.append(line_start + t * d)

        return intersections
    
    def find_goal_point(self, current_position, lastFoundIndex=0):
        for i in range(lastFoundIndex, len(CONFIG.PATH) - 1):
            intersections = self.line_circle_intersection(
                current_position, 
                CONFIG.PATH[i], 
                CONFIG.PATH[i + 1]
            )

            if intersections:
                # Pick the intersection closest to the end of the current path segment
                if len(intersections) == 1:
                    return intersections[0], i  # Single intersection
                else:
                    # Multiple intersections, pick the one closest to the end of the segment
                    if np.linalg.norm(intersections[0] - CONFIG.PATH[i + 1]) < np.linalg.norm(intersections[1] - CONFIG.PATH[i + 1]):
                        return intersections[0], i
                    else:
                        return intersections[1], i

        print("NO INTERSECTION FOUND")
        return [0, 0], lastFoundIndex  # Return default if no intersection found

    
    def calculate_abs_target_angle(self, current_position, goal_point):
        # Extract the x and y coordinates
        current_x, current_y = current_position
        goal_x, goal_y = goal_point

        # Calculate the relative position of the goal point
        delta_x = goal_x - current_x
        delta_y = goal_y - current_y

        # Calculate the angle in radians 
        angle_radians = math.atan2(delta_y, delta_x)

        # Normalize the angle to be in the range of 0 to 2 * pi
        if angle_radians < 0:
            angle_radians += 2 * math.pi

        return angle_radians
    
    def find_steering_angle(self, absTargetAngle, currentHeading):
        # Calculate the difference in angles
        minAngle = absTargetAngle - currentHeading  

        # Normalize the angle to be within -pi to pi
        while minAngle > math.pi:
            minAngle -= 2 * math.pi
        while minAngle < -math.pi:
            minAngle += 2 * math.pi

        return minAngle
    
    
    def update(self,current_pose, current_orient, lastFoundIndex_):
        goal_point, lastFoundIndex_ = self.find_goal_point(current_pose, lastFoundIndex_)
        required_angle = get_angle(current_pose, goal_point)
        steering_angle =  angle_diff(required_angle, current_orient)
        print("GOAL POINT : ", goal_point)
        return steering_angle, goal_point
    

