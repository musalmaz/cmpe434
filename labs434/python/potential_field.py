import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Constants
k_att = 30  # Attractive constant
k_rep = 3  # Repulsive constant
obstacle_range = 2.0  # Influence range of obstacles

def compute_attractive_force(current_pos, goal_pos):
    """ Compute the attractive force towards the goal. """
    return k_att * (goal_pos - current_pos)

def compute_repulsive_force(current_pos, obstacles):
    """ Compute the repulsive force away from obstacles. """
    force = np.zeros(2)
    for obs in obstacles:
        obs_vector = current_pos - obs
        dist_to_obs = np.linalg.norm(obs_vector)
        if dist_to_obs < obstacle_range:
            force += k_rep * (1/dist_to_obs - 1/obstacle_range) * (1/dist_to_obs**2) * (obs_vector/dist_to_obs)
    return force

def potential_field_steering(obstacles, car_pos, car_heading, goal_pos):
    """ Calculate the net force and determine the steering angle. """
    attractive_force = compute_attractive_force(car_pos, goal_pos)
    repulsive_force = compute_repulsive_force(car_pos, obstacles)
    net_force = attractive_force + repulsive_force
    
    # Compute the steering angle; assuming the heading is along the x-axis
    steering_angle = np.arctan2(net_force[1], net_force[0]) - car_heading
    return np.rad2deg(steering_angle)

# Simulation parameters
car_pos = np.array([2.0, 2.0])
car_heading = np.deg2rad(200)
goal_pos = np.array([4, 6])
obstacles = np.array([[0, 2], [0, 6], [3,3],[2,0], [2, 3], [3, 3], [5, 5]])
speed = 1  # Constant speed
dt = 0.1  # Time step for the simulation

# Setting up the plot
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
car_dot, = ax.plot([], [], 'bo', label='Car')
goal_dot, = ax.plot([], [], 'go', label='Goal')
obstacles_dot = ax.scatter(obstacles[:, 0], obstacles[:, 1], color='red', s=100, label='Obstacles')
path_line, = ax.plot([], [], 'k--', label='Path')
ax.legend()

def init():
    car_dot.set_data([], [])
    goal_dot.set_data([], [])
    path_line.set_data([], [])
    return car_dot, goal_dot, path_line

def update(frame):
    global car_pos, car_heading
    steering_angle = potential_field_steering(obstacles, car_pos, car_heading, goal_pos)
    car_heading += np.radians(steering_angle)
    car_pos += np.array([np.cos(car_heading), np.sin(car_heading)]) * speed * dt
    
    car_dot.set_data(car_pos[0], car_pos[1])
    goal_dot.set_data(goal_pos[0], goal_pos[1])
    old_x, old_y = path_line.get_data()
    new_x = np.append(old_x, car_pos[0])
    new_y = np.append(old_y, car_pos[1])
    path_line.set_data(new_x, new_y)
    
    return car_dot, goal_dot, path_line

ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 200), init_func=init, blit=True, interval=50)
plt.show()
