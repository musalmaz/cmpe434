import time

import mujoco
import mujoco.viewer

import cmpe434_utils
from dynamic_window_approach import *

def get_angle(p1, p2):
    return (np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) + 2*np.pi) % (2*np.pi)

def distance(p1, p2): 
    return np.sqrt(np.sum(np.square(p1 - p2)))

def get_car_state(data, m):
    car_pos = data.qpos.copy()[:2]
    car_vel = data.qvel.copy()[:2]
    car_orient = get_angle([0, 0], car_vel[:2])

    return car_pos, car_orient, car_vel

scene, scene_assets = cmpe434_utils.get_model('scenes/empty_floor.xml')
robot, robot_assets = cmpe434_utils.get_model('models/mushr_car/model.xml')
# robot, robot_assets = cmpe434_utils.get_model('models/mujoco_car/model.xml')
# robot, robot_assets = cmpe434_utils.get_model('models/skydio_x2/x2.xml')

# Add the robot to the scene.
scene.include_copy(robot)

# Combine all assets into a single dictionary.
all_assets = {**scene_assets, **robot_assets}

m = mujoco.MjModel.from_xml_string(scene.to_xml_string(), assets=all_assets)
d = mujoco.MjData(m)

paused = False # Global variable to control the pause state.

# Pressing SPACE key toggles the paused state. 
# You can define other keys for other actions here.
def key_callback(keycode):
  if chr(keycode) == ' ':
    global paused
    paused = not paused
distance_threshold = 0.5
box_size = (0.1, 0.1, 0.1)
obstacle_rgba = (0.8, 0.3, 0.3, 1.0)
wall_rgba = (0.8, 1, 0.5, 0)
obstacle_geom_ids = 0

# Define the wall array to form a closed room around the origin
wall = np.array([
    [-5, -5],
    [-5, -4],
    [-5, -3],
    [-5, -2],
    [-5, -1],
    [-5, 0],
    [-5, 1],
    [-5, 2],
    [-5, 3],
    [-5, 4],
    [-5, 5],
    [-4, 5],
    [-3, 5],
    [-2, 5],
    [-1, 5],
    [0, 5],
    [1, 5],
    [2, 5],
    [3, 5],
    [4, 5],
    [5, 5],
    [5, 4],
    [5, 3],
    [5, 2],
    [5, 1],
    [5, 0],
    [5, -1],
    [5, -2],
    [5, -3],
    [5, -4],
    [5, -5],
    [4, -5],
    [3, -5],
    [2, -5],
    [1, -5],
    [0, -5],
    [-1, -5],
    [-2, -5],
    [-3, -5],
    [-4, -5]
])

wall_radius = 0.4
wall_height = 1
obstacle_radius = 0.3
# Define obstacles as cylinders with position, radius, and height
obstacles = np.array([[1, 3],
                      [-2, 3],
                      [3, -3]])
# [
#     {"pos": [1, 3], "radius": 0.5, "height": 1.0},
#     {"pos": [-2, 3], "radius": 0.4, "height": 0.8},
#     {"pos": [3, 3], "radius": 0.3, "height": 0.6}
# ]
obstacles_with_wall = np.concatenate((obstacles, wall))
with mujoco.viewer.launch_passive(m, d) as viewer:
    velocity = d.actuator("throttle_velocity")
    steering = d.actuator("steering")

    # Set up wall
    for i, point in enumerate(wall):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[wall_radius, wall_height, 0],
            pos=[point[0], point[1], 0],
            mat=np.eye(3).flatten(),
            rgba=wall_rgba
        )

    # Set up obstacles
    for j, obstacle in enumerate(obstacles):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[len(wall) + j],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[obstacle_radius, wall_height, 0],
            pos=[obstacle[0], obstacle[1], 0],
            mat=np.eye(3).flatten(),
            rgba=obstacle_rgba
        )
    # Update the total number of geometries
    viewer.user_scn.ngeom = len(wall) + len(obstacles)
    viewer.sync()

    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < 100:
        step_start = time.time()

        if not paused:
            goal = np.array([3, 4])
            delta_time = time.time() - step_start
            current_position, car_orient, current_velocity = get_car_state(d, m)
            current_speed = np.sqrt(current_velocity[0] ** 2 + current_velocity[1] ** 2)

            x = np.array([current_position[0], current_position[1], car_orient, current_speed, 0.0])

            u, predicted_trajectory = dwa_control(x, config, goal, obstacles_with_wall, delta_time)
            # main(obstacles_with_wall,3,4, RobotType.rectangle)
            print("applied values : ", u[0], u[1])

            velocity.ctrl = u[0] # update velocity control value
            steering.ctrl = u[1] # update steering control value

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            distance_to_goal = distance(current_position, goal)
            if distance_to_goal < distance_threshold:
                break


        # Rudimentary time keeping, will drift relative to wall clock.
        # time_until_next_step = m.opt.timestep - (time.time() - step_start)
        # if time_until_next_step > 0:
        #   time.sleep(time_until_next_step)
