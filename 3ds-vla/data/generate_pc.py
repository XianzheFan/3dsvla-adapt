import os
import numpy as np
import json
from PIL import Image
import open3d as o3d
import tqdm

DEFAULT_RGB_SCALE_FACTOR = 256000.0
DEFAULT_GRAY_SCALE_FACTOR = {np.uint8: 100.0,
                             np.uint16: 1000.0,
                             np.int32: DEFAULT_RGB_SCALE_FACTOR}
DEPTH_SCALE = 2**24 - 1
def sample_farthest_points(points, num_samples):
    """
    Perform farthest point sampling (FPS) on a point cloud.

    Args:
        points (np.ndarray): Input point cloud of shape (N, D).
        num_samples (int): Number of points to sample (K).

    Returns:
        np.ndarray: Indices of the sampled points of shape (K,).
        np.ndarray: Sampled points of shape (K, D).
    """
    N, D = points.shape

    # Initialize arrays
    sampled_indices = np.zeros(num_samples, dtype=np.int32)
    distances = np.full(N, np.inf)  # Initially, all distances are infinite

    # Randomly select the first point (or start with the first point)
    sampled_indices[0] = np.random.randint(0, N)
    selected_point = points[sampled_indices[0]]

    for i in range(1, num_samples):
        # Compute distances from the selected point to all other points
        dist = np.sum((points - selected_point) ** 2, axis=1)  # Squared Euclidean distance

        # Update the minimum distance for each point
        distances = np.minimum(distances, dist)

        # Select the farthest point
        sampled_indices[i] = np.argmax(distances)
        selected_point = points[sampled_indices[i]]

    # Gather sampled points
    sampled_points = points[sampled_indices]

    return sampled_indices, sampled_points
def image_to_float_array(image, scale_factor=None):
    """Recovers the depth values from an image.

    Reverses the depth to image conversion performed by FloatArrayToRgbImage or
    FloatArrayToGrayImage.

    The image is treated as an array of fixed point depth values.  Each
    value is converted to float and scaled by the inverse of the factor
    that was used to generate the Image object from depth values.  If
    scale_factor is specified, it should be the same value that was
    specified in the original conversion.

    The result of this function should be equal to the original input
    within the precision of the conversion.

    Args:
        image: Depth image output of FloatArrayTo[Format]Image.
        scale_factor: Fixed point scale factor.

    Returns:
        A 2D floating point numpy array representing a depth image.

    """
    image_array = np.array(image)
    image_dtype = image_array.dtype
    image_shape = image_array.shape

    channels = image_shape[2] if len(image_shape) > 2 else 1
    assert 2 <= len(image_shape) <= 3
    if channels == 3:
        # RGB image needs to be converted to 24 bit integer.
        float_array = np.sum(image_array * [65536, 256, 1], axis=2)
        if scale_factor is None:
            scale_factor = DEFAULT_RGB_SCALE_FACTOR
    else:
        if scale_factor is None:
            scale_factor = DEFAULT_GRAY_SCALE_FACTOR[image_dtype.type]
        float_array = image_array.astype(np.float32)
    scaled_array = float_array / scale_factor
    return scaled_array
def generate_point_cloud(depth_map):
    """
    Generate a 3D point cloud from an RGB image, depth map, and camera parameters.
    
    Args:
        rgb_image (torch.Tensor): RGB image of shape (3, H, W).
        depth_map (torch.Tensor): Depth map of shape (1, H, W).
        camera_intrinsics (torch.Tensor): Intrinsic matrix (3x3).
        camera_extrinsics (torch.Tensor): Extrinsic matrix (4x4).
    
    Returns:
        torch.Tensor: Point cloud of shape (N, 3) in world coordinates.
    """
    camera_intrinsics = np.array([[-923.1524421, 0, 336],[0, -923.1524421,  336],[0,0,1]])
    
    camera_extrinsics = np.array([[ 1.19209290e-07, -4.22617942e-01, -9.06307936e-01,  1.34999919e+00],
 [-1.00000000e+00, -5.96046448e-07,  1.49011612e-07, 3.71546562e-08],
 [-5.66244125e-07 , 9.06307936e-01, -4.22617912e-01,  1.57999933e+00],
 [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])
     
    # Ensure inputs are tensors
    H, W = depth_map.shape

    # Create pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))  # Pixel coordinates
    u = u.flatten()  # Flatten for vectorized operations
    v = v.flatten()

    # Flatten depth map
    depth = depth_map.flatten()

    # Filter out pixels with zero depth
    valid = depth > 0
    u, v, depth = u[valid], v[valid], depth[valid]

    # Intrinsic parameters
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    # Back-project to 3D camera coordinates
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    # Combine into camera-space 3D points
    camera_points = np.stack((x, y, z, np.ones_like(x)), axis=1)  # Shape: (N, 4)

    # Transform to world coordinates using the extrinsic matrix
    world_points = camera_extrinsics @ camera_points.T  # Shape: (4, N)
    world_points = world_points.T[:, :3]  # Shape: (N, 3)

    return world_points
def depth_to_depthscale(depth_dir):
        front_depth = image_to_float_array(
                            
                                Image.open(depth_dir),
                            DEPTH_SCALE)
        near = 0.009999999776482582 
        far = 4.5
        front_depth_m = near + front_depth * (far - near)
        return front_depth_m
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='./3ds-vla/data/train_data')
args = parser.parse_args()
data_dir = args.data_dir
data_item_list = os.listdir(data_dir)
for data_item in tqdm.tqdm(data_item_list):
    
    json_dir = os.path.join(data_dir, data_item)
    with open(json_dir, 'r') as f:
        data = json.load(f)
    image_filename = data['image']
    depth_filename = image_filename.replace('front_rgb', 'front_depth')
    depth_filename = depth_filename.replace('rgb', 'depth')
    depth_m = depth_to_depthscale(depth_filename)
    
    image = Image.open(image_filename).convert('RGB')
    image_arr = np.array(image)

    # depth_m = Image.fromarray(depth_m)
    point_cloud = generate_point_cloud(depth_m) #451584x3
    # xmin, ymin, zmin, xmax, ymax, zmax = [-1.5, -1.5, 0.76, 1.5, 1.5, 3.0]
    xmin, ymin, zmin, xmax, ymax, zmax = [-1.5, -3, 0.76, 3.0, 2.0, 3.0]
    min_bound = np.array([xmin, ymin, zmin])
    max_bound = np.array([xmax, ymax, zmax])
    mask = np.all(point_cloud[:, :3] > min_bound, axis=1)
    point_cloud = point_cloud[mask]
    mask = np.all(point_cloud[:, :3] < max_bound, axis=1)
    point_cloud = point_cloud[mask]
    # #fps sampling
    _,point_cloud = sample_farthest_points(point_cloud, 2304)
    # point_cloud = np.vstack((point_cloud, contact_point)) 

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(point_cloud)
    # colors = np.ones_like(point_cloud)  # Creates an array of [1.0, 1.0, 1.0] for all points (white)

    # # Set the last point to red color
    # colors[-1] = [1.0, 0.0, 0.0]  # Red color (R=1, G=0, B=0)

    # # Assign the colors to the point cloud
    # pc.colors = o3d.utility.Vector3dVector(colors)
    out_dir = image_filename.replace('front_rgb', 'front_pc')
    out_dir = out_dir.replace('png', 'ply')
    if not os.path.exists('/'.join(out_dir.split('/')[:-1])):
        os.makedirs('/'.join(out_dir.split('/')[:-1]))
    print(out_dir)
    
    # Save to PLY file
    o3d.io.write_point_cloud(out_dir, pc)
