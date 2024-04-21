import numpy as np
import matplotlib.pyplot as plt

# Parameters
class DWAConfig:
    def __init__(self):
        # Robot (or car in this case) configuration
        self.max_speed = 3.0  # maximum speed [m/s]
        self.min_speed = 0.0  # minimum speed [m/s]
        self.max_yawrate = 0.38  # maximum yaw rate [rad/s]
        self.min_yawrate = -0.38  # minimum yaw rate [rad/s]
        self.max_accel = 0.2  # maximum acceleration [m/ss]
        self.max_dyawrate = 0.1  # maximum change in yaw rate [rad/ss]
        self.v_resolution = 0.01  # velocity resolution [m/s]
        self.yawrate_resolution = 0.01  # yaw rate resolution [rad/s]
        self.dt = 0.1  # time tick [s]
        self.predict_time = 3.0  # predict time in the future [s]
        self.to_goal_cost_gain = 0.15  # cost-to-goal weighting
        self.speed_cost_gain = 0.1  # speed cost weighting
        self.obstacle_cost_gain = 1.0  # cost for obstacles
        self.robot_radius = 0.5  # robot radius [m]

# Motion Model
def motion_model(x, u, dt):
    # x = [x(m), y(m), theta(rad), v(m/s)]
    theta = x[2] + u[1] * dt
    x_new = x[3] * np.cos(theta) * dt
    y_new = x[3] * np.sin(theta) * dt
    return np.array([x[0] + x_new, x[1] + y_new, theta, u[0]])

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
def calculate_cost(trajectory, config, goal, ob):
    # Define costs here
    # For example, penalize distance to goal and proximity to obstacles
    cost = 0
    # Distance to goal cost
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    goal_distance = np.sqrt(dx**2 + dy**2)
    cost += config.to_goal_cost_gain * goal_distance
    # Add more costs based on your requirements
    return cost

# Example usage
config = DWAConfig()
x = np.array([0, 0, np.pi/4, 0])  # Initial state [x(m), y(m), theta(rad), v(m/s)]
goal = np.array([1, 10])  # Goal position [x(m), y(m)]
obstacles = np.array([[1, 2], [3, 4], [5, 6]])  # Obstacles positions [[x(m), y(m)], ...]

u, trajectory = dwa_control(x, config, goal, obstacles)
print("Optimal velocity and yaw rate:", u)
plt.plot(trajectory[:, 0], trajectory[:, 1])
plt.scatter(obstacles[:,0], obstacles[:,1], color='red') # Obstacle positions
plt.scatter(goal[0], goal[1], color='green') # Goal position
plt.grid(True)
plt.show()
