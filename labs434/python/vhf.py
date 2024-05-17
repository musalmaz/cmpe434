import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the VFH algorithm functions (assuming the latest refined versions provided are in use)
def calculate_polar_histogram(obstacles, car_pos, num_sectors=90, sector_angle=4, max_distance=3):
    histogram = np.zeros(num_sectors)
    for obstacle in obstacles:
        dx, dy = obstacle - car_pos
        distance = np.hypot(dx, dy)
        if distance < max_distance:
            angle = np.degrees(np.arctan2(dy, dx))
            index = int((angle % 360) / sector_angle)
            histogram[index] += (1 - distance / max_distance)
    return histogram

def find_best_sector(histogram, goal_direction, car_heading, num_sectors=90, sector_angle=4):
    goal_sector = int((goal_direction % 360) / sector_angle)
    heading_sector = int((car_heading % 360) / sector_angle)
    candidate_sectors = np.arange(goal_sector - 3, goal_sector + 4) % num_sectors
    best_sector = min(candidate_sectors, key=lambda s: histogram[s])
    return (best_sector - heading_sector) * sector_angle

def vfh(obstacles, car_pos, car_heading, goal_pos):
    goal_direction = np.degrees(np.arctan2(goal_pos[1] - car_pos[1], goal_pos[0] - car_pos[0]))
    histogram = calculate_polar_histogram(obstacles, car_pos)
    steering_angle = find_best_sector(histogram, goal_direction, car_heading)
    return steering_angle

# # Simulation setup

obstacles = np.array([[0, 2], [0, 6]])
car_pos = np.array([2.0, 4.0])
car_heading = np.rad2deg(0 )
goal_pos = np.array([4, 6])
speed = 1  # Constant speed
dt = 0.1  # Time step for the simulation

fig, ax = plt.subplots()
ax.set_xlim(0, 20)
ax.set_ylim(0, 15)
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
    steering_angle = vfh(obstacles, car_pos, car_heading, goal_pos)
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
