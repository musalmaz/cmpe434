import time
import random
import numpy as np
import scipy as sp

import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

import cmpe434_utils
import cmpe434_dungeon

import a_star

from PIDController import *
from potential_field import *

kp_steering = 5
ki_steering = 0
kd_steering = -0.1
# Initialize PID controllers
####################### Hız 1 iken 5 0 -0.1 iyi
pid_steering = PIDController()

def normalize_steering(steering):
    max_steering = 3.0
    if steering > max_steering:
        steering =  max_steering
    if steering < -max_steering:
        steering = -max_steering
    return steering

def get_car_state(data):
    # car_pos = data.qpos.copy()[:2]
    car_pos = data.body("buddy").xpos[:2]
    car_vel = data.qvel.copy()[:2]
    car_orient = get_angle([0, 0], car_vel[:2])
    car_speed = np.sqrt(car_vel[0]**2 + car_vel[1]**2)

    return car_pos, car_orient, car_speed


def adjust_heading_to_waypoint(car_pos, car_orient, waypoint):
    # Calculate the vector to the waypoint
    direction_vector = waypoint - car_pos
    # Calculate the desired heading (angle with respect to the x-axis)
    desired_heading = np.arctan2(direction_vector[1], direction_vector[0])
    
    # Assume car_orient is in radians and aligned with the x-axis (0 radians)
    # Calculate the necessary rotation
    rotation_needed = desired_heading - car_orient
    
    # Normalize the angle to be within -pi to pi for consistent behavior
    rotation_needed = (rotation_needed + np.pi) % (2 * np.pi) - np.pi
    
    return rotation_needed

def obstacles_within_radius(obstacles, current_position, radius):
    """
    Finds the obstacles that are within a given radius around the current position.

    Parameters:
    - obstacles (List[Tuple[float, float]]): List of tuples representing obstacle coordinates.
    - current_position (Tuple[float, float]): Tuple representing the current position (x, y).
    - radius (float): Threshold radius to filter obstacles.

    Returns:
    - List[Tuple[float, float]]: List of obstacles that are within the specified radius.
    """
    obstacles_in_radius = []
    for obstacle in obstacles:
        distance = ((obstacle[0] - current_position[0])**2 + (obstacle[1] - current_position[1])**2)**0.5
        if distance <= radius:
            obstacles_in_radius.append(obstacle)

    return obstacles_in_radius

def interpolate_points(p1, p2, num_points):
    """ Interpolates num_points between two given points p1 and p2. """
    # Include the start point and exclude the end point since it will be included by the next segment
    return [(p1[0] + (p2[0] - p1[0]) * i / (num_points + 1), p1[1] + (p2[1] - p1[1]) * i / (num_points + 1)) 
            for i in range(1, num_points + 1)]


# Pressing SPACE key toggles the paused state. 
# You can define other keys for other actions here.
def key_callback(keycode):
    if chr(keycode) == ' ':
        global paused
        paused = not paused

paused = False # Global variable to control the pause state.
walls = [] # store all obstacles
start_pose = []
end_pose = []

def create_scenario():

    scene, scene_assets = cmpe434_utils.get_model('scenes/empty_floor.xml')
    # robot, robot_assets = cmpe434_utils.get_model('models/mujoco_car/model.xml')
    # robot, robot_assets = cmpe434_utils.get_model('models/skydio_x2/x2.xml')

    tiles, rooms, connections = cmpe434_dungeon.generate(3, 2, 8)

    for index, r in enumerate(rooms):
        (xmin, ymin, xmax, ymax) = cmpe434_dungeon.find_room_corners(r)
        scene.worldbody.add('geom', name='R{}'.format(index), type='plane', size=[(xmax-xmin)+1, (ymax-ymin)+1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[(xmin+xmax), (ymin+ymax), 0])

    for pos, tile in tiles.items():
        if tile == "#":
            scene.worldbody.add('geom', type='box', size=[1, 1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[pos[0]*2, pos[1]*2, 0])
            walls.append([pos[0]*2 + 0.5, pos[1]*2 + 0.5])
            walls.append([pos[0]*2 + 0.5, pos[1]*2 -0.5])
            walls.append([pos[0]*2 - 0.5, pos[1]*2 + 0.5])
            walls.append([pos[0]*2 -0.5, pos[1]*2 -0.5])

    # scene.worldbody.add('geom', type='plane', size=[(xmax-xmin)/2+0.1, (ymax-ymin)/2+0.1, 0.01], rgba=[0.8, 0.6, 0.4, 1],  pos=[(xmin+xmax)/2, (ymin+ymax)/2, 0])

    # scene.worldbody.add('geom', type='box', size=[0.1, (ymax-ymin)/2+0.1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[xmin, (ymin+ymax)/2, 0.1])
    # scene.worldbody.add('geom', type='box', size=[0.1, (ymax-ymin)/2+0.1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[xmax, (ymin+ymax)/2, 0.1])
    # scene.worldbody.add('geom', type='box', size=[(xmax-xmin)/2+0.1, 0.1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[(xmin+xmax)/2, ymin, 0.1])
    # scene.worldbody.add('geom', type='box', size=[(xmax-xmin)/2+0.1, 0.1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[(xmin+xmax)/2, ymax, 0.1])

    # Add the robot to the scene.
    robot, robot_assets = cmpe434_utils.get_model('models/mushr_car/model.xml')
    start_pos = random.choice([key for key in tiles.keys() if tiles[key] == "."])
    final_pos = random.choice([key for key in tiles.keys() if tiles[key] == "." and key != start_pos])

    start_pose.append([2*start_pos[0], 2*start_pos[1]])
    end_pose.append([2*final_pos[0], 2*final_pos[1]])

    scene.worldbody.add('site', name='start', type='box', size=[0.5, 0.5, 0.01], rgba=[0, 0, 1, 1],  pos=[start_pos[0]*2, start_pos[1]*2, 0])
    scene.worldbody.add('site', name='finish', type='box', size=[0.5, 0.5, 0.01], rgba=[1, 0, 0, 1],  pos=[final_pos[0]*2, final_pos[1]*2, 0])

    # create obstables for each room
    for i, room in enumerate(rooms):
        obs_pos = random.choice([tile for tile in room if tile != start_pos and tile != final_pos])
        scene.worldbody.add('geom', name='Z{}'.format(i), type='cylinder', size=[0.2, 0.05], rgba=[0.8, 0.0, 0.1, 1],  pos=[obs_pos[0]*2, obs_pos[1]*2, 0.08])

    start_yaw = random.randint(0, 359)
    robot.find("body", "buddy").set_attributes(pos=[start_pos[0]*2, start_pos[1]*2, 0.1], euler=[0, 0, start_yaw])

    scene.include_copy(robot)

    # Combine all assets into a single dictionary.
    all_assets = {**scene_assets, **robot_assets}

    return scene, all_assets

def execute_scenario(scene, ASSETS=dict()):

    m = mujoco.MjModel.from_xml_string(scene.to_xml_string(), assets=all_assets)
    d = mujoco.MjData(m)

    rooms = [m.geom(i).id for i in range(m.ngeom) if m.geom(i).name.startswith("R")]
    obstacles = [m.geom(i).id for i in range(m.ngeom) if m.geom(i).name.startswith("Z")]

    uniform_direction_dist = sp.stats.uniform_direction(2)
    obstacle_direction = [[x, y, 0] for x,y in uniform_direction_dist.rvs(len(obstacles))]

    unused = np.zeros(1, dtype=np.int32)

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:

        # velocity = m.actuator("throttle_velocity")
        # steering = m.actuator("steering")

        velocity = d.actuator("throttle_velocity")
        steering = d.actuator("steering")

        x_pos =  []
        y_pos = []
        for i in walls:
            x_pos.append(i[0])
            y_pos.append(i[1])
        planner = a_star.AStarPlanner(x_pos, y_pos, 1, 1)
        rx, ry = planner.planning(start_pose[0][0], start_pose[0][1], end_pose[0][0], end_pose[0][1])
        path = []
        for i in range(len(rx)):
            path.append([rx[len(rx) -1 - i], ry[len(rx) -1 - i]])
        print(path)

        plt.plot(rx, ry, "-r")
        plt.plot(x_pos, y_pos, ".k")
        plt.plot(start_pose[0][0], start_pose[0][1], "og")
        plt.plot(end_pose[0][0], end_pose[0][1], "xb")
        plt.grid(True)
        plt.axis("equal")
        plt.pause(0.001)
        plt.show()

        print("Start : ", start_pose, "End : ",end_pose)

        new_path = []

        # Number of points to add between each pair of path points
        num_inter_points = 3  # You can adjust this number based on your specific needs

        # Construct the new path with interpolated points
        for i in range(len(path) - 1):
            # Add the current point
            new_path.append(path[i])
            # Add interpolated points between the current point and the next point
            new_path.extend(interpolate_points(path[i], path[i+1], num_inter_points))

        # Don't forget to add the last point from the original path
        new_path.append(path[-1])


        # Put cylinder on path
        for j, p in enumerate(new_path):
            if j < len(viewer.user_scn.geoms):  # Check if there is a geometry available
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[j],
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[0.1, 0.1, 0],  # Adjusted size
                    pos=[p[0], p[1], 0.00001],  # Adjust z position if needed
                    mat=np.eye(3).flatten(),
                    rgba=(0.8, 0.3, 0.3, 1.0)
                )
        viewer.user_scn.ngeom = len(new_path)
        viewer.sync()


        car_x_pose = []
        car_y_pose = []
        sphere_geom_ids = 0
        prev_time = 0
        car_pos, car_orient, car_speed = get_car_state(d)
        car_x_pose.append(car_pos[0])
        car_y_pose.append(car_pos[1])

        print(start_pose, end_pose)

        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        heading_corrected = False
        prev_applied_steering = 0
        # max_yaw_rate = 0.8
        while viewer.is_running() and time.time() - start < 300:
            step_start = time.time()

            if not paused:
                car_pos, car_orient, car_speed = get_car_state(d)
                # if not heading_corrected: 
                #     rotation_needed = adjust_heading_to_waypoint(car_pos, car_orient, path[1])
                #     # Check if the rotation needed is greater than the acceptable threshold
                #     while abs(rotation_needed) > np.pi / 20:  
                #         # Set controls
                #         velocity.ctrl = 0.2  # Constant velocity
                #         steering.ctrl = rotation_needed  # Set steering control value
                        
                #         # Update simulation to apply the steering
                #         mujoco.mj_step(m, d)  # Assuming this steps the physics engine
                        
                #         # Update the car state to get the new orientation
                #         car_pos, car_orient, car_speed = get_car_state(d)
                        
                #         # Recalculate the rotation needed after the update
                #         rotation_needed = adjust_heading_to_waypoint(car_pos, car_orient, path[1])
                        
                #     heading_corrected = True


                dynamic_obstacles = []
                # velocity.ctrl = 4.0 # update velocity control value
                # steering.ctrl = 20.0 # update steering control value

                # obstable update
                for i, x in enumerate(obstacles):
                    dx = obstacle_direction[i][0]
                    dy = obstacle_direction[i][1]

                    px = m.geom_pos[x][0]
                    py = m.geom_pos[x][1]
                    pz = 0.02

                    nearest_dist = mujoco.mj_ray(m, d, [px, py, pz], obstacle_direction[i], None, 1, -1, unused)

                    if nearest_dist >= 0 and nearest_dist < 0.4:
                        obstacle_direction[i][0] = -dy
                        obstacle_direction[i][1] = dx

                    m.geom_pos[x][0] = m.geom_pos[x][0]+dx*0.001
                    m.geom_pos[x][1] = m.geom_pos[x][1]+dy*0.001
                    dynamic_obstacles.append([m.geom_pos[x][0], m.geom_pos[x][1]])

                ###############
                curr_time = time.time()
               
                end = time.time()
                delta_t = curr_time -prev_time
                prev_time = curr_time

                possible_obstacles = dynamic_obstacles + walls

                ####### INNER LOOP
                goal_pos_loop = path[pid_steering.current_index]
                obstacle_near = obstacles_within_radius(possible_obstacles, car_pos, 2)
                dynamic_obstacle_near = obstacles_within_radius(dynamic_obstacles, car_pos, 1)
                if obstacle_near and dynamic_obstacle_near and pid_steering.current_index > 1:
                    while True:
                        dynamic_obstacle_near = obstacles_within_radius(dynamic_obstacles, car_pos, 1)
                        car_pos, car_orient, car_speed = get_car_state(d)
                        if not paused:
                            steering_angle = potential_field_steering(obstacle_near, car_pos, car_orient, goal_pos_loop)
                            steering_angle = np.radians(steering_angle)
                            # steering_angle = steering_angle % np.pi
                            steering_angle = normalize_steering(steering_angle)
                            print("Current, goal : ", car_pos, goal_pos_loop)
                            velocity.ctrl = 1.0 # update velocity control value
                            steering.ctrl = steering_angle # update steering control value
                            mujoco.mj_step(m, d)

                            # Pick up changes to the physics state, apply perturbations, update options from GUI.
                            viewer.sync()
                        if distance(car_pos, goal_pos_loop) < 1:
                            print("INNER LOOP COMPLETED............. ")
                            break
                        if not dynamic_obstacle_near :
                            break
                            
                
                pid_steering.dt = delta_t
                pid_steering.current_orientation = car_orient
                applied_steering = pid_steering.update(car_pos, path)
                applied_steering = normalize_steering(applied_steering)
                # applied_steering = applied_steering % np.pi
                print(applied_steering)
                # if prev_applied_steering != 0 and abs(applied_steering - prev_applied_steering) > max_yaw_rate :
                #     applied_steering = prev_applied_steering

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                velocity.ctrl = 1.0 # update velocity control value
                steering.ctrl = applied_steering # update steering control value
                mujoco.mj_step(m, d)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()
                ###########################
                if(pid_steering.is_waypoint_reached(car_pos, path)):
                    print("WAYPOINT REACHED")
                    pid_steering.current_index += 1
                if(pid_steering.current_index == len(path)):
                    print("PATH IS COMPLETED")
                    break
                if(pid_steering.current_index == len(path) - 1):
                    pid_steering.acceptance_radius = 0.1

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    return m, d

if __name__ == '__main__':
    scene, all_assets = create_scenario()
    execute_scenario(scene, all_assets)

