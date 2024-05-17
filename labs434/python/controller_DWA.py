import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

# Parameters
class DWAConfig:
    def __init__(self):
        # Robot (or car in this case) configuration
        self.max_speed = 3.0  # maximum speed [m/s]
        self.min_speed = -1.0  # minimum speed [m/s]
        self.max_yawrate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.min_yawrate = -40.0 * math.pi / 180.0  # [rad/ss]
        self.max_accel = 0.2  # maximum acceleration [m/ss]
        self.max_dyawrate = 0.2  # maximum change in yaw rate [rad/ss]
        self.v_resolution = 0.01  # velocity resolution [m/s]
        self.yawrate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # time tick [s]
        self.predict_time = 3.0  # predict time in the future [s]
        self.to_goal_cost_gain = 20  # cost-to-goal weighting
        self.speed_cost_gain = 1.0  # speed cost weighting
        self.obstacle_cost_gain = 40.0  # cost for obstacles
        self.robot_radius = 0.3  # robot radius [m]
        self.obstacle_proximity_threshold = 0.5 
        self.heading_cost_gain = 4.0

# Motion Model
def motion_model(x, u, dt):
    # x = [x(m), y(m), theta(rad), v(m/s)]
    theta = x[2] + u[1] * dt  # Update the heading based on yaw rate and time step
    x_new = x[3] * np.cos(theta) * dt  # Calculate new x position
    y_new = x[3] * np.sin(theta) * dt  # Calculate new y position
    return np.array([x[0] + x_new, x[1] + y_new, theta, u[0]])  # Return the new state


# Dynamic Window Approach
def dwa_control(x, config, goal, ob):
    dw = calc_dynamic_window(x, config)
    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)
    return u, trajectory

# Calculate Dynamic Window
def calc_dynamic_window(x, config):
    # Dynamic window from robot specs
    Vs = [config.min_speed, config.max_speed, config.min_yawrate, config.max_yawrate]
    # Dynamic window based on current state
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[2] - config.max_dyawrate * config.dt,
          x[2] + config.max_dyawrate * config.dt]
    # Combine windows
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
    return dw

# Calculate the best trajectory
def calc_control_and_trajectory(x, dw, config, goal, ob):
    x_init = np.array([x])  # Ensure x_init is a 2D array
    min_cost = float('inf')
    best_u = [0.0, 0.0]
    best_trajectory = x_init
    # Evaluate all combinations in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for w in np.arange(dw[2], dw[3], config.yawrate_resolution):
            trajectory = x_init.copy()
            for _ in range(int(config.predict_time / config.dt)):
                new_state = motion_model(trajectory[-1, :], [v, w], config.dt)
                trajectory = np.vstack((trajectory, new_state))
            # Calculate cost for this trajectory
            cost = calculate_cost(trajectory, config, goal, ob)
            if cost < min_cost:
                min_cost = cost
                best_u = [v, w]
                best_trajectory = trajectory
    return best_u, best_trajectory

# Calculate cost for trajectory
def calculate_cost(trajectory, config, goal, obstacles):
    cost = 0
    # Distance to goal cost
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    goal_distance = np.sqrt(dx**2 + dy**2)
    cost += config.to_goal_cost_gain * goal_distance

    # Obstacle proximity cost
    min_obstacle_distance = np.min([np.sqrt((obstacle[0] - trajectory[-1, 0])**2 + (obstacle[1] - trajectory[-1, 1])**2) for obstacle in obstacles])
    if min_obstacle_distance < config.obstacle_proximity_threshold:
        cost += config.obstacle_cost_gain / min_obstacle_distance
    
    # Heading error cost
    current_heading = np.arctan2(dy, dx)
    heading_difference = np.abs(current_heading - trajectory[-1, 2])  # Assuming the last column in trajectory is heading
    heading_difference = np.minimum(heading_difference, 2 * np.pi - heading_difference)  # Smallest angle difference
    cost += config.heading_cost_gain * heading_difference

    # Speed cost
    current_speed = trajectory[-1, 3]  # Assuming the fourth column in trajectory is speed
    if hasattr(config, 'target_speed'):
        speed_difference = np.abs(current_speed - config.target_speed)
        cost += config.speed_cost_gain * speed_difference
    else:
        cost += config.speed_cost_gain * current_speed  # Cost increases with speed

    return cost


def plot_robot_trajectory(trajectory, goal, obstacles, u):
    # Setup the plot
    fig, ax = plt.subplots()
    # Plot trajectory using only the first two columns for x and y positions
    ax.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory', color='blue')
    # Plot starting position
    ax.scatter(trajectory[0, 0], trajectory[0, 1], color='black', s=100, zorder=5, label='Start')
    # Plot robot positions along the trajectory
    for i in range(len(trajectory)):
        circle = plt.Circle((trajectory[i, 0], trajectory[i, 1]), 0.2, color='gray', alpha=0.5, zorder=3)
        ax.add_patch(circle)
    # Plot obstacles
    ax.scatter(obstacles[:, 0], obstacles[:, 1], color='red', s=50, zorder=4, label='Obstacles')
    # Plot goal
    ax.scatter(goal[0], goal[1], color='green', s=100, zorder=5, label='Goal')
    # Annotations for control inputs (only if you want to annotate the same input throughout)
    #for i in range(len(trajectory)):
      #  ax.annotate(f"v={u[0]:.2f}, w={u[1]:.2f}", (trajectory[i, 0], trajectory[i, 1]), textcoords="offset points", xytext=(0,10), ha='center')
    # Setting graph properties
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.legend()
    ax.grid(True)
    plt.axis('equal')
    plt.show()



# # Example usage
# config = DWAConfig()
# x = np.array([0, 0, 0, 0])  # Initial state [x(m), y(m), theta(rad), v(m/s)]
# goal = np.array([1, 10])  # Goal position [x(m), y(m)]
# obstacles = np.array([[1, 2], [3, 4], [5, 6]])  # Obstacles positions [[x(m), y(m)], ...]

# while 1:

#     u, trajectory = dwa_control(x, config, goal, obstacles)
#     x = motion_model(x, u, config.dt)
#     print(trajectory)
#     print(u)
#     plot_robot_trajectory(trajectory, goal, obstacles, u)
# print("Optimal velocity and yaw rate:", u)
# plt.plot(trajectory[:, 0], trajectory[:, 1])
# plt.scatter(obstacles[:,0], obstacles[:,1], color='red') # Obstacle positions
# plt.scatter(goal[0], goal[1], color='green') # Goal position
# plt.grid(True)
# plt.show()

# Example usage
config = DWAConfig()
x = np.array([0, 0, 0, 0])  # Initial state [x(m), y(m), theta(rad), v(m/s)]
goal = np.array([3, 4])  # Goal position [x(m), y(m)]
obstacles = np.array([[1, 3],
                      [-2, 3],
                      [3, -3]])
path_points = []
def update(frame):
    global x, path_points  # Use the global state and path variables

    u, trajectory = dwa_control(x, config, goal, obstacles)
    path_points.append(x[:2])  # Append current position to path
    path = trajectory[:, :2]
    x = motion_model(x, u, config.dt)  # Update the robot's state

    # Clear the current plot
    ax.clear()

    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Trajectory')

    # Plot the accumulated path as a line
    ax.scatter(path[:, 0], path[:, 1], color='black', s=50, label='Path')

    # Plot obstacles
    ax.scatter(obstacles[:, 0], obstacles[:, 1], color='red', s=50, label='Obstacles')

    # Plot the goal
    ax.scatter(goal[0], goal[1], color='green', s=100, label='Goal')

    # Plot the robot's current position as a circle
    robot_circle = plt.Circle((x[0], x[1]), 0.2, color='blue', fill=True)
    ax.add_patch(robot_circle)

    # Setting graph properties each time since the plot is cleared
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    ax.set_xlim([np.min(obstacles[:,0]) - 1, np.max(obstacles[:,0]) + 1])
    ax.set_ylim([np.min(obstacles[:,1]) - 1, np.max(obstacles[:,1]) + 1])

fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=np.arange(1000), repeat=False)
plt.show()
