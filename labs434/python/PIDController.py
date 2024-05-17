import numpy as np

def distance(p1, p2): 
    return np.sqrt(np.sum(np.square(p1 - p2)))

def get_angle(p1, p2):
    return (np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) + 2*np.pi) % (2*np.pi)

def angle_diff(angle1, angle2):
    diff = angle1 - angle2
    diff = (diff + np.pi) % (2*np.pi) - np.pi
    return diff

def normalize_steering(steerig):
    max_steering = 4.0
    min_steering = -4.0

    if steerig < min_steering:
        steerig = min_steering
    if steerig > max_steering:
        steerig = max_steering
    return steerig

class PIDController:
    def __init__(self):
        self.kp = 5
        self.ki = 0
        self.kd = -0.1
        self.integral = 0
        self.prev_error = 0

        self.current_index = 0

        self.current_orientation = 0

        self.dt = 0.1

        self.acceptance_radius = 1

    def update_error(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative
    
    def is_waypoint_reached(self, current_position, path):
        target_point = path[self.current_index]
        distance_to_target = distance(current_position, target_point)
        if(np.abs(distance_to_target) < self.acceptance_radius):
            return True 
        
    def update(self, current_position, path):
        target_point = self.get_target_point(path)
        required_angle = get_angle(current_position, target_point)
        steering_error = angle_diff(required_angle, self.current_orientation)
        applied_orient = self.update_error(steering_error, self.dt)
        return applied_orient
    
    def get_target_point(self, path):
        # Convert to numpy array for vectorized operations
        path = np.array(path)

        target_point = path[self.current_index]
        # print("target point : ", target_point)
        return target_point