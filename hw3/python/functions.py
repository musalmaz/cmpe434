import numpy as np

def write_data_to_file(data_list, file_name):
    with open(file_name, 'w') as file:
        for data in data_list:
            x1, x2 = data
            file.write(f"data : {data} \n")

def quat_to_euler(quad):
    w = quad[0]
    x = quad[1]
    y = quad[2]
    z = quad[3]
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2>+1.0,+1.0,t2)
    #t2 = +1.0 if t2 > +1.0 else t2

    t2 = np.where(t2<-1.0, -1.0, t2)
    #t2 = -1.0 if t2 < -1.0 else t2
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))

    return X, Y, Z 


def get_angle(p1, p2):
    return (np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) + 2*np.pi) % (2*np.pi)

def angle_diff(angle1, angle2):
    diff = angle1 - angle2
    diff = (diff + np.pi) % (2*np.pi) - np.pi
    return diff

def distance(p1, p2): 
    return np.sqrt(np.sum(np.square(p1 - p2)))

# def create_video_from_frames(images, output_file, fps):
#     # Determine the width and height from the first image
#     frame = cv2.imread(images[0])
#     height, width, layers = frame.shape

#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
#     out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

#     for image in images:
#         frame = cv2.imread(image)
#         out.write(frame)  # Write out frame to video

#     out.release()
#     print(f"Video saved as {output_file}")

# def calculate_velocity_error(current_velocity, target_velocity):
#     current_speed = np.linalg.norm(current_velocity)
#     return target_velocity - current_speed

# def calculate_steering_error(car_orient, current_position, target_point):
#     # Convert orientation quaternion to euler angles to find yaw
#     car_yaw = car_orient

#     # Calculate the required heading to the target point
#     direction_to_target = np.arctan2(target_point[1] - current_position[1], target_point[0] - current_position[0])

#     # Steering error is the difference in current yaw and the required heading
#     error = direction_to_target - car_yaw
#     return error - np.pi