import pyrealsense2_beta as rs
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def save_imu_data(bag_file, gyro_file, accel_file):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False)
    config.enable_stream(rs.stream.gyro)
    config.enable_stream(rs.stream.accel)

    pipeline.start(config)
    playback = pipeline.get_active_profile().get_device().as_playback()
    playback.set_real_time(False)

    imu_data = {'gyro': [], 'accel': []}
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                print("No more frames available.")
                break

            for frame in frames:
                if frame.is_motion_frame():
                    motion = frame.as_motion_frame()
                    stream_name = motion.get_profile().stream_name()
                    data = {'timestamp': motion.get_timestamp(), 'data': motion.get_motion_data()}
                    if stream_name == "Gyro":
                        imu_data['gyro'].append(data)
                    elif stream_name == "Accel":
                        imu_data['accel'].append(data)
    finally:
        pipeline.stop()

    np.save(gyro_file, [
        {'angular_velocity': {'x': entry['data'].x, 'y': entry['data'].y, 'z': entry['data'].z}} for entry in
        imu_data['gyro']
    ])
    np.save(accel_file, [
        {'linear_acceleration': {'x': entry['data'].x, 'y': entry['data'].y, 'z': entry['data'].z}} for entry in
        imu_data['accel']
    ])

    print(f"Saved {len(imu_data['gyro'])} gyro frames.")
    print(f"Saved {len(imu_data['accel'])} accel frames.")

def load_imu_data(gyro_file, accel_file):
    gyro_data = np.load(gyro_file, allow_pickle=True)
    accel_data = np.load(accel_file, allow_pickle=True)

    gyro_t = np.arange(len(gyro_data))
    gyro_values = np.array([[v['x'], v['y'], v['z']] for v in (d['angular_velocity'] for d in gyro_data)])

    accel_t = np.arange(len(accel_data))
    accel_values = np.array([[v['x'], v['y'], v['z']] for v in (d['linear_acceleration'] for d in accel_data)])

    return {
        'gyro': {'t': gyro_t, 'data': gyro_values},
        'accel': {'t': accel_t, 'data': accel_values}
    }

def calculate_trajectory(imu_data):
    gyro_t, gyro_data = imu_data['gyro']['t'], imu_data['gyro']['data']
    accel_t, accel_data = imu_data['accel']['t'], imu_data['accel']['data']
    scale = 0.00000006

    all_t = np.sort(np.union1d(gyro_t, accel_t))

    position = np.zeros(3)
    velocity = np.zeros(3)
    orientation = R.identity()

    trajectory = []
    for i in range(1, len(all_t)):
        dt = all_t[i] - all_t[i - 1]

        gyro_interp = np.array([np.interp(all_t[i], gyro_t, gyro_data[:, dim]) for dim in range(3)])
        accel_interp = np.array([np.interp(all_t[i], accel_t, accel_data[:, dim]) for dim in range(3)])

        orientation *= R.from_rotvec(gyro_interp * dt)

        accel_global = orientation.apply(accel_interp) - np.array([0.0, 0.0, -9.81])
        velocity += accel_global * dt
        position += velocity * dt

        trajectory.append(position.copy())

    return np.array(trajectory) * scale

def generate_point_cloud(bag_file, max_distance=10):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    align = rs.align(rs.stream.color)

    pipeline.start(config)
    frames = align.process(pipeline.wait_for_frames())
    pipeline.stop()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        raise ValueError("Failed to retrieve frames from the bag file.")

    # Konwertowanie ramki głębi i koloru na tablice numpy
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Elementy aparatu
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy

    height, width = depth_image.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Konwertuj wartości głębokości na punkty 3D
    z = depth_image / 1000.0  # metry
    x = -(x - cx) * z / fx
    y = (y - cy) * z / fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    valid = (z > 0) & (z < max_distance)
    points = points[valid.reshape(-1)]

    color_image = cv2.resize(color_image, (width, height), interpolation=cv2.INTER_LINEAR)
    colors = color_image.reshape(-1, 3)[valid.reshape(-1)] / 255.0

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    return cloud

def create_trajectory_line_set(trajectory, radius=0.01):
    line_meshes = []
    for start, end in zip(trajectory[:-1], trajectory[1:]):
        direction = (end - start) / np.linalg.norm(end - start)
        rotation_axis = np.cross([0, 0, 1], direction)
        rotation_angle = np.arccos(np.dot([0, 0, 1], direction))
        rotation_matrix = R.from_rotvec(
            rotation_angle * rotation_axis / (np.linalg.norm(rotation_axis) + 1e-8)).as_matrix()

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=np.linalg.norm(end - start))
        cylinder.rotate(rotation_matrix, center=[0, 0, 0])
        cylinder.translate(start)
        cylinder.paint_uniform_color([0.0, 1.0, 0.0])

        line_meshes.append(cylinder)

    combined_mesh = o3d.geometry.TriangleMesh()
    for mesh in line_meshes:
        combined_mesh += mesh
    return combined_mesh

def create_coordinate_axes(size=1.0):

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
    return axes

if __name__ == "__main__":
    bag_file = r"d435i_walking.bag"
    gyro_file = r"gyro_data.npy"
    accel_file = r"accel_data.npy"

    save_imu_data(bag_file, gyro_file, accel_file)
    imu_data = load_imu_data(gyro_file, accel_file)
    trajectory = calculate_trajectory(imu_data)

    # Apply necessary rotations to align the trajectory and board
    rotation_x_matrix = -R.from_euler('x', 0, degrees=True).as_matrix()  # Rotate 90 degrees around X-axis
    rotation_y_matrix = R.from_euler('y', 0, degrees=True).as_matrix()  # Rotate 180 degrees around Y-axis
    rotation_z_matrix = R.from_euler('z', 0, degrees=True).as_matrix()  # Rotate -180 degrees around Z-axis

    new_axes_rotation = rotation_z_matrix @ rotation_y_matrix @ rotation_x_matrix
    trajectory = (new_axes_rotation @ trajectory.T).T

    cloud = generate_point_cloud(bag_file, max_distance=10)
    cloud.rotate(new_axes_rotation, center=np.array([0, 0, 0]))

    # Ruch cloudem
    cloud.translate(np.array([-0.2, 0.0, 1.0]))

    trajectory_mesh = create_trajectory_line_set(trajectory, radius=0.01)

    axes = create_coordinate_axes(size=1.0)

    o3d.visualization.draw_geometries([cloud, trajectory_mesh, axes])
