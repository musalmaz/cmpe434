import numpy as np
from config import *
from functions import *

CONFIG =  CONFIG()

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0

        self.current_index = 0

        self.current_orientation = 0

        self.dt = 0.01

    def update_error(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative
    
    def is_waypoint_reached(self, current_position):
        target_point = CONFIG.PATH[self.current_index]
        distance_to_target = distance(current_position, target_point)
        if(np.abs(distance_to_target) < CONFIG.ACCEPTANCE_RADIUS):
            return True 
        
    def update(self, current_position):
        target_point = self.get_target_point()
        required_angle = get_angle(current_position, target_point)
        steering_error = angle_diff(required_angle, self.current_orientation)
        applied_orient = self.update_error(steering_error, self.dt)
        return applied_orient
    
    def get_target_point(self):
        # Convert to numpy array for vectorized operations
        path = np.array(CONFIG.PATH)

        target_point = path[self.current_index]
        print("target point : ", target_point)
        return target_point