import time
import random

import mujoco
import mujoco.viewer

import cmpe434_utils
import cmpe434_dungeon

import numpy as np

def get_angle(p1, p2):
    return (np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) + 2*np.pi) % (2*np.pi)

def distance(p1, p2): 
    return np.sqrt(np.sum(np.square(p1 - p2)))

def get_car_state(data):
    car_pos = data.qpos.copy()[:2]
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

def create_scenario():

    scene, scene_assets = cmpe434_utils.get_model('scenes/empty_floor.xml')
    # robot, robot_assets = cmpe434_utils.get_model('models/mujoco_car/model.xml')
    # robot, robot_assets = cmpe434_utils.get_model('models/skydio_x2/x2.xml')

    tiles, rooms, connections = cmpe434_dungeon.generate(3, 2, 8)

    for index, r in enumerate(rooms):
        (xmin, ymin, xmax, ymax) = cmpe434_dungeon.find_room_corners(r)
        scene.worldbody.add('geom', name='R{}'.format(index), type='plane', size=[(xmax-xmin)+1, (ymax-ymin)+1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[(xmin+xmax), (ymin+ymax), 0])
        print("Size R : ", [(xmax-xmin)+1, (ymax-ymin)+1, 0.1], [(xmin+xmax), (ymin+ymax), 0])

    for pos, tile in tiles.items():
        if tile == "#":
            scene.worldbody.add('geom', type='box', size=[1, 1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[pos[0]*2, pos[1]*2, 0])
            print("### : ", [pos[0]*2, pos[1]*2, 0])

    # scene.worldbody.add('geom', type='plane', size=[(xmax-xmin)/2+0.1, (ymax-ymin)/2+0.1, 0.01], rgba=[0.8, 0.6, 0.4, 1],  pos=[(xmin+xmax)/2, (ymin+ymax)/2, 0])

    # scene.worldbody.add('geom', type='box', size=[0.1, (ymax-ymin)/2+0.1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[xmin, (ymin+ymax)/2, 0.1])
    # scene.worldbody.add('geom', type='box', size=[0.1, (ymax-ymin)/2+0.1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[xmax, (ymin+ymax)/2, 0.1])
    # scene.worldbody.add('geom', type='box', size=[(xmax-xmin)/2+0.1, 0.1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[(xmin+xmax)/2, ymin, 0.1])
    # scene.worldbody.add('geom', type='box', size=[(xmax-xmin)/2+0.1, 0.1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[(xmin+xmax)/2, ymax, 0.1])

    # Add the robot to the scene.
    robot, robot_assets = cmpe434_utils.get_model('models/mushr_car/model.xml')
    start_pos = random.choice([key for key in tiles.keys() if tiles[key] == "."])
    final_pos = random.choice([key for key in tiles.keys() if tiles[key] == "."])

    print("start end pos : ", start_pos, final_pos)

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

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:

        # velocity = m.actuator("throttle_velocity")
        # steering = m.actuator("steering")

        velocity = d.actuator("throttle_velocity")
        steering = d.actuator("steering")

        # Check all available attributes in the MjModel object
        print(dir(m))

        # If 'geom_names' is directly accessible, we can use it directly
        if hasattr(m, 'geom_names'):
            geom_names = [name for name in m.geom_names]
            print("Geometry names:", geom_names)
        else:
            print("geom_names attribute is not directly accessible. Please check the correct attribute.")


        # Assuming m is your MjModel object and d is your MjData object
        geom_names = [m.names[m.geom_nameadr[i]:m.geom_nameadr[i]+m.geom_namesz[i]].decode('utf-8') for i in range(m.ngeom)]
        r_indexes = [i for i, name in enumerate(geom_names) if name.startswith('R')]

        print("Indexes of 'R' geometries:", r_indexes)

        for idx in r_indexes:
            # Access static properties from MjModel
            geom_type = m.geom_type[idx]
            geom_size = m.geom_size[idx*3:(idx+1)*3]  # Assuming each geom_size has 3 components
            geom_rgba = m.geom_rgba[idx*4:(idx+1)*4]  # Assuming RGBA has 4 components
            
            # Access dynamic properties from MjData
            geom_xpos = d.geom_xpos[idx*3:(idx+1)*3]  # Current position in the simulation

            print(f"Geometry {geom_names[idx]}:")
            print(f"  Type: {geom_type}, Size: {geom_size}, RGBA: {geom_rgba}")
            print(f"  Current Position: {geom_xpos}")

        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < 300:
            step_start = time.time()

            if not paused:
                velocity.ctrl = 1.0 # update velocity control value
                steering.ctrl = 4.0 # update steering control value

                car_pos, car_orient, car_speed = get_car_state(d)
                # print("Cars states : ", car_pos, car_speed)

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(m, d)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    return m, d

if __name__ == '__main__':
    scene, all_assets = create_scenario()
    execute_scenario(scene, all_assets)

