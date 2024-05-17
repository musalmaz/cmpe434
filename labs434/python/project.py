import time
import random

import mujoco
import mujoco.viewer

import cmpe434_utils
import cmpe434_dungeon

import numpy as np


from PIDController import *
# from controller_DWA import *
import a_star
from dynamic_window_approach import *
# from vhf import *
from rrt import *
from PIDController import *

kp_steering = 5
ki_steering = 0
kd_steering = -0.1
# Initialize PID controllers
####################### Hız 1 iken 5 0 -0.1 iyi
pid_steering = PIDController()

def get_angle(p1, p2):
    return (np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) + 2*np.pi) % (2*np.pi)

def distance(p1, p2): 
    return np.sqrt(np.sum(np.square(p1 - p2)))

def interpolate_points(p1, p2, num_points):
    """ Interpolates num_points between two given points p1 and p2. """
    # Include the start point and exclude the end point since it will be included by the next segment
    return [(p1[0] + (p2[0] - p1[0]) * i / (num_points + 1), p1[1] + (p2[1] - p1[1]) * i / (num_points + 1)) 
            for i in range(1, num_points + 1)]

def unit_vector(vector):
    """Ensure the vector is a unit vector and 3D."""
    if len(vector) == 2:
        vector = np.append(vector, 0)
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """Calculate the angle in radians between 3D vectors 'v1' and 'v2'."""
    if len(v1) == 2:
        v1 = np.append(v1, 0)
    if len(v2) == 2:
        v2 = np.append(v2, 0)
    
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def normalize_steering(steering):
    max_steering = 3
    if steering > max_steering:
        steering =  max_steering
    if steering < -max_steering:
        steering = -max_steering
    return steering
    
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


def get_car_state(data):
    # car_pos = data.qpos.copy()[:2]
    car_pos = data.body("buddy").xpos[:2]
    car_vel = data.qvel.copy()[:2]
    car_orient = get_angle([0, 0], car_vel[:2])
    car_speed = np.sqrt(car_vel[0]**2 + car_vel[1]**2)

    return car_pos, car_orient, car_speed

# Pressing SPACE key toggles the paused state. 
# You can define other keys for other actions here.
def key_callback(keycode):
    if chr(keycode) == ' ':
        global paused
        paused = not paused

paused = False # Global variable to control the pause state.
obstacles = [] # store all obstacles
start_pose = []
end_pose = []

pid_controller = PIDController()
# dwa_config = DWAConfig()
config = Config()
config.robot_type = RobotType.rectangle

def create_scenario():

    scene, scene_assets = cmpe434_utils.get_model('scenes/empty_floor.xml')
    # robot, robot_assets = cmpe434_utils.get_model('models/mujoco_car/model.xml')
    # robot, robot_assets = cmpe434_utils.get_model('models/skydio_x2/x2.xml')

    tiles, rooms, connections = cmpe434_dungeon.generate(3, 2, 8)

    for index, r in enumerate(rooms):
        (xmin, ymin, xmax, ymax) = cmpe434_dungeon.find_room_corners(r)
        scene.worldbody.add('geom', name='R{}'.format(index), type='plane', size=[(xmax-xmin)+1, (ymax-ymin)+1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[(xmin+xmax), (ymin+ymax), 0])
        # print("Size R : ", [(xmax-xmin)+1, (ymax-ymin)+1, 0.1], [(xmin+xmax), (ymin+ymax), 0])
        # obstacles.append([2*((xmax-xmin)+1), 2*((ymax-ymin)+1)])
        # obstacles.append([2*(xmin+xmax), 2*(ymax-ymin)])
        # obstacles.append([2*(xmin+xmax), 2*(ymin+ymax)])
        # obstacles.append([2*(xmax-xmin), 2*(ymin+ymax)])
        # obstacles.append([2*(xmin+xmax), 2*(ymin+ymax)])

    for pos, tile in tiles.items():
        if tile == "#":
            scene.worldbody.add('geom', type='box', size=[1, 1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[pos[0]*2, pos[1]*2, 0])
            # print("### : ", [pos[0]*2, pos[1]*2, 0])
            obstacles.append([pos[0]*2 + 0.5, pos[1]*2 + 0.5])
            obstacles.append([pos[0]*2 + 0.5, pos[1]*2 -0.5])
            obstacles.append([pos[0]*2 - 0.5, pos[1]*2 + 0.5])
            obstacles.append([pos[0]*2 -0.5, pos[1]*2 -0.5])

    # scene.worldbody.add('geom', type='plane', size=[(xmax-xmin)/2+0.1, (ymax-ymin)/2+0.1, 0.01], rgba=[0.8, 0.6, 0.4, 1],  pos=[(xmin+xmax)/2, (ymin+ymax)/2, 0])

    # scene.worldbody.add('geom', type='box', size=[0.1, (ymax-ymin)/2+0.1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[xmin, (ymin+ymax)/2, 0.1])
    # scene.worldbody.add('geom', type='box', size=[0.1, (ymax-ymin)/2+0.1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[xmax, (ymin+ymax)/2, 0.1])
    # scene.worldbody.add('geom', type='box', size=[(xmax-xmin)/2+0.1, 0.1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[(xmin+xmax)/2, ymin, 0.1])
    # scene.worldbody.add('geom', type='box', size=[(xmax-xmin)/2+0.1, 0.1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[(xmin+xmax)/2, ymax, 0.1])

    # Add the robot to the scene.
    robot, robot_assets = cmpe434_utils.get_model('models/mushr_car/model.xml')
    start_pos = random.choice([key for key in tiles.keys() if tiles[key] == "."])
    final_pos = random.choice([key for key in tiles.keys() if tiles[key] == "."])
    
    start_pose.append([2*start_pos[0], 2*start_pos[1]])
    end_pose.append([2*final_pos[0], 2*final_pos[1]])

    # print("start end pos : ", start_pos, final_pos)

    scene.worldbody.add('site', name='start', type='box', size=[0.5, 0.5, 0.01], rgba=[0, 0, 1, 1],  pos=[start_pos[0]*2, start_pos[1]*2, 0])
    scene.worldbody.add('site', name='finish', type='box', size=[0.5, 0.5, 0.01], rgba=[1, 0, 0, 1],  pos=[final_pos[0]*2, final_pos[1]*2, 0])

    start_yaw = random.randint(0, 359)
    robot.find("body", "buddy").set_attributes(pos=[start_pos[0]*2, start_pos[1]*2, 0.1], euler=[0, 0, start_yaw])

    scene.include_copy(robot)

    # Combine all assets into a single dictionary.
    all_assets = {**scene_assets, **robot_assets}

    return scene, all_assets

def execute_scenario(scene, ASSETS=dict()):

    m = mujoco.MjModel.from_xml_string(scene.to_xml_string(), assets=all_assets)
    d = mujoco.MjData(m)

    # m.opt.timestep = 0.1

    # print(obstacles)
    # print(start_pose, end_pose)

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:

        # velocity = m.actuator("throttle_velocity")
        # steering = m.actuator("steering")

        velocity = d.actuator("throttle_velocity")
        steering = d.actuator("steering")

        for i in range(100):
            mujoco.mj_step(m, d)
            viewer.sync()

        x_pos =  []
        y_pos = []
        for i in obstacles:
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
                    pos=[p[0], p[1], 0.05],  # Adjust z position if needed
                    mat=np.eye(3).flatten(),
                    rgba=(0.8, 0.3, 0.3, 1.0)
                )
        viewer.user_scn.ngeom = len(new_path)
        viewer.sync()

        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        # start_wait = True
        path_index = 1
        waypoint_acceptance_radius = 0.2
        car_x_pose = []
        car_y_pose = []
        sphere_geom_ids = 0
        prev_time = 0
        car_pos, car_orient, car_speed = get_car_state(d)
        car_x_pose.append(car_pos[0])
        car_y_pose.append(car_pos[1])
        x = np.array([car_pos[0], car_pos[1], car_orient, car_speed, 1])

        prev_applied_steering = 0
        max_yaw_rate = 0.3

        while viewer.is_running() and time.time() - start < 300:
            # if start_wait :
            #     time.sleep(30)
            #     start_wait = False
            step_start = time.time()

            if not paused:

                car_pos, car_orient, car_speed = get_car_state(d)
                car_x_pose.append(car_pos[0])
                car_y_pose.append(car_pos[1])
                # print("Cars states : ", car_pos, car_speed)

                # mujoco.mjv_initGeom(
                #     viewer.user_scn.geoms[sphere_geom_ids],
                #     type=mujoco.mjtGeom.mjGEOM_SPHERE,  # Replace with GEOM_BOX if needed
                #     size=(0.05, 0.05, 0.05),
                #     pos=[car_pos[0], car_pos[1], 0],  # Initial position (updated later)
                #     mat=np.eye(3).flatten(),
                #     rgba=(0, 0.5, 0.5, 1)
                # )
                # sphere_geom_ids += 1
                # viewer.user_scn.ngeom = len(new_path) + sphere_geom_ids
                
                curr_time = time.time()
               
                end = time.time()
                delta_t = curr_time -prev_time
                prev_time = curr_time
                
                obstacle_near = obstacles_within_radius(obstacles, car_pos, 3)  
                # start_node = Node(car_pos[0], car_pos[1], car_orient)
                # goal_node = Node(path[path_index][0], path[path_index][1], 0)
                # print("GOAL NODE : ", path[path_index][0], path[path_index][1])
                # obstacle_with_size = []
                # for i in obstacle_near:
                #     obstacle_with_size.append((i[0], i[1], 0.5))

                # result = RRT(start_node, goal_node, obstacle_with_size)
                # if result is not None:
                #     path_x, path_y = result
                #     path_gen = []
                #     for i in range(len(path_x)):
                #         path_gen.append([path_x[i], path_y[i]])
                #     print("GEN PATH : ", path_gen)  

                pid_steering.dt = delta_t
                pid_steering.current_orientation = car_orient
                applied_orient = pid_steering.update(car_pos, path)
                applied_steering = normalize_steering(applied_orient)
                if prev_applied_steering != 0 and abs(applied_steering - prev_applied_steering) > max_yaw_rate :
                    applied_steering = prev_applied_steering
                print(applied_steering)
                
                # plot_path(start_node, goal_node, path_gen, nodes, obstacle_with_size)


                # x[0] = car_pos[0]
                # x[1] = car_pos[1]
                
                # u, predicted_trajectory = dwa_control(x, config, path[path_index], obstacle_near, curr_time -prev_time)
                # x = motion(x, u, config.dt)
                # path_small = predicted_trajectory[:, :2]
                # print(path_small)
                # print("goal",path[path_index], "current loc: ", car_pos)
                # print(u)
                # print()
                # print("samll path : ", path_small[:5])
                # theta = x[2] + u[1] * delta_t  # Update the heading based on yaw rate and time step
                # main(x, end - step_start, obstacle_near,path[path_index], robot_type=RobotType.rectangle)
                # applied_steering = normalize_steering(-steering_angle)
                # applied_throttle = normalize_steering(throttle)
                # applied_vel = normalize_steering(u[0])
                # print(car_orient, theta, u[1] * delta_t)
                # if distance(car_pos, path[path_index]) < waypoint_acceptance_radius:
                #     print("WP REACHED............. ")
                #     path_index += 1
                
               
                velocity.ctrl = 1 # update velocity control value
                steering.ctrl = applied_orient # update steering control value
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(m, d)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()
                prev_applied_steering = applied_steering
                if(pid_steering.is_waypoint_reached(car_pos, path)):
                    print("WAYPOINT REACHED")
                    pid_steering.current_index += 1
                if(pid_steering.current_index == len(path)):
                    print("PATH IS COMPLETED")
                    break
                if(pid_steering.current_index == len(path) - 1):
                    pid_steering.acceptance_radius = 0.1
                # viewer.render()  # Make sure to render updates
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        plt.plot(car_x_pose, car_y_pose, "-p")
        plt.plot(rx, ry, "-r")
        plt.plot(x_pos, y_pos, ".k")
        plt.plot(start_pose[0][0], start_pose[0][1], "og")
        plt.plot(end_pose[0][0], end_pose[0][1], "xb")
        plt.grid(True)
        plt.axis("equal")
        plt.pause(0.001)
        plt.show()
    
    return m, d

if __name__ == '__main__':
    scene, all_assets = create_scenario()
    execute_scenario(scene, all_assets)
    

