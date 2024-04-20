import time
import numpy as np


import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import cmpe434_utils

from PIDController import *

from purePursit import *

from functions import *

from config import *
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Controller and Velocity settings')

# Add arguments
parser.add_argument('--controller', type=int, choices=[4, 5], required=True, help='Select your controller (4 or 5)')
parser.add_argument('--velocity', type=int, choices=range(0, 11), required=True, help='Set the velocity control (0-10)')

# Parse arguments
args = parser.parse_args()

CONTROLLER = args.controller
VELOCITY = args.velocity

CONFIG = CONFIG()

def get_car_state(data, m):
    car_pos = data.qpos.copy()[:2]
    car_vel = data.qvel.copy()[:2]
    car_orient = get_angle([0, 0], car_vel[:2])

    return car_pos, car_orient, car_vel

paused = False # Global variable to control the pause state.

# Pressing SPACE key toggles the paused state. 
# You can define other keys for other actions here.
def key_callback(keycode):
  if chr(keycode) == ' ':
    global paused
    paused = not paused

scene, scene_assets = cmpe434_utils.get_model(CONFIG.floor_path)
robot, robot_assets = cmpe434_utils.get_model(CONFIG.car_model_path)
# scene, scene_assets = cmpe434_utils.get_model('scenes/empty_floor.xml')
# robot, robot_assets = cmpe434_utils.get_model('models/mushr_car/model.xml')
# robot, robot_assets = cmpe434_utils.get_model('models/mujoco_car/model.xml')
# robot, robot_assets = cmpe434_utils.get_model('models/skydio_x2/x2.xml')

# Add the robot to the scene.
scene.include_copy(robot)

# Combine all assets into a single dictionary.
all_assets = {**scene_assets, **robot_assets}

m = mujoco.MjModel.from_xml_string(scene.to_xml_string(), assets=all_assets)
d = mujoco.MjData(m)

# 6 0 -0.1 fena değil
# Inıtialize the gains 5 0 -0.1
kp_steering = 5
ki_steering = 0
kd_steering = -0.1
# Initialize PID controllers
####################### Hız 1 iken 5 0 -0.1 iyi
pid_steering = PIDController(kp_steering, ki_steering, kd_steering)

look_ahead_distance = 1.5
pure_pursuit = PurePursuit(look_ahead_distance)

# Store position and velocity data
position_data = []
velocity_data = []

fixed_timestep = pid_steering.dt
m.opt.timestep = fixed_timestep

lastFoundIndex = 0
box_geom_ids = 0

with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:

  # viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
  velocity = d.actuator("throttle_velocity")
  steering = d.actuator("steering")

  # # Initialize the target point sphere
  # target_geom_id = len(viewer.user_scn.geoms)  # Assign a new unique ID based on the number of existing geometries
  # mujoco.mjv_initGeom(
  #     viewer.user_scn.geoms[target_geom_id - 1],
  #     type=mujoco.mjtGeom.mjGEOM_SPHERE,
  #     size=CONFIG.box_size,  # Define the size of the sphere
  #     pos=[0, 0, 0],  # Set an initial position
  #     mat=np.eye(3).flatten(),
  #     rgba=CONFIG.target_color  # Define the color of the sphere
  # )

  # # Ensure that the total number of geometries is updated
  # viewer.user_scn.ngeom = len(viewer.user_scn.geoms) 

  start = time.time()
  prev_time = 0
  # box_geom_ids = []
  viewer.user_scn.ngeom = 0

  while viewer.is_running() and time.time() - start < 1000:
    step_start = time.time()

    if not paused:
        current_position, car_orient, current_velocity = get_car_state(d, m)
        position_data.append(current_position)
        velocity_data.append(current_velocity)
        print("Current position : ", current_position)
        # print("current vel : ", current_velocity)

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[box_geom_ids],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,  # Replace with GEOM_BOX if needed
            size=CONFIG.box_size,
            pos=[current_position[0], current_position[1], 0],  # Initial position (updated later)
            mat=np.eye(3).flatten(),
            rgba=CONFIG.box_rgba
        )

        curr_time = time.time()
        applied_orient  = 0
        if (CONTROLLER == 4):
          pid_steering.dt = curr_time - prev_time
          prev_time = curr_time
          pid_steering.current_orientation = car_orient
          applied_orient = pid_steering.update(current_position)

        if (CONTROLLER == 5):
          applied_orient, goal_point = pure_pursuit.update(current_position, car_orient, lastFoundIndex)
          # mujoco.mjv_initGeom(
          #   viewer.user_scn.geoms[target_geom_id - 1],
          #   type=mujoco.mjtGeom.mjGEOM_SPHERE,  # Replace with GEOM_BOX if needed
          #   size=CONFIG.box_size,
          #   pos=[goal_point[0], goal_point[1], 0],  # Initial position (updated later)
          #   mat=np.eye(3).flatten(),
          #   rgba=CONFIG.target_color
          # )
          box_geom_ids += 1
        

        velocity.ctrl =  VELOCITY 
        steering.ctrl = applied_orient  # update steering control value

        # print("APPLied orient : ", applied_orient)
        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(m, d)
        box_geom_ids += 1
        viewer.user_scn.ngeom = box_geom_ids
        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        if (CONTROLLER == 4):
          if(pid_steering.is_waypoint_reached(current_position)):
            print("WAYPOINT REACHED")
            pid_steering.current_index += 1
          if(pid_steering.current_index == len(CONFIG.PATH)):
            print("PATH IS COMPLETED")
            break

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)

# write data to files
write_data_to_file(position_data, CONFIG.position_data_path)
write_data_to_file(velocity_data, CONFIG.velocity_data_path)

# Convert to numpy arrays
path1_np = np.array(CONFIG.PATH)
path2_np = np.array(position_data)

# Create plot
plt.figure(figsize=(10, 8))

# Plot path 1
plt.plot(path1_np[:, 0], path1_np[:, 1], label='Original Path', marker='o')

# Plot path 2
plt.plot(path2_np[:, 0], path2_np[:, 1], label='Traversed Path', marker='x')

# Set plot title and labels
plt.title('Paths Visualization')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# Show legend
plt.legend()

# Show grid
plt.grid(True)

# Show the plot
plt.show()

