import os
import pickle
import numpy as np
#from plyfile import PlyData, PlyElement
from pyrep.objects import VisionSensor
from PIL import Image, ImageDraw
from rlbench.backend.utils import image_to_float_array, rgb_handles_to_mask
data_dir = '/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris-2/RLBench/train_data0819_672/2024-8-19-4-15-10-62-45'
RADIUS = 4
from PIL import Image, ImageDraw
COLOR = 'red'
DEPTH_SCALE = 2**24 - 1
from rlbench.backend import utils
#with open(os.path.join(data_dir,'low_dim_obs.pkl'), 'rb') as f:
#    demo = pickle.load(f)
# print(demo[98].gripper_pose//0.01, demo[121].gripper_pose//0.01,demo[121].misc['front_camera_extrinsics'],demo[98].misc['front_camera_intrinsics'])
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
    _,H, W = depth_map.shape

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
def generate_pointcloud(depth_m):
    point_cloud = generate_point_cloud(depth_m) #451584x3
    
    xmin, ymin, zmin, xmax, ymax, zmax = [-1.5, -1.5, 0.76, 1.5, 1.5, 3.0]
    min_bound = np.array([xmin, ymin, zmin])
    max_bound = np.array([xmax, ymax, zmax])
    mask = np.all(point_cloud[:, :3] > min_bound, axis=1)
    point_cloud = point_cloud[mask]
    
    mask = np.all(point_cloud[:, :3] < max_bound, axis=1)
    point_cloud = point_cloud[mask]
    
    # #fps sampling
    _,point_cloud = sample_farthest_points(point_cloud, 2304)
    return point_cloud
def rotation_matrix_to_quaternion2(cur_target,matrix_z,matrix_y,matrix_x=None):
    # Ensure matrix is 3x3
    # matrix_z /= 100
    matrix_x = np.cross(matrix_y, matrix_z)
    matrix_x /= np.linalg.norm(matrix_x)
    # matrix_y = np.cross(matrix_x, matrix_z)
    # matrix_y /= np.linalg.norm(matrix_y)
    matrix = np.eye(3)
    matrix[:,2] = matrix_z
    matrix[:,1] = matrix_y
    matrix[:,0] = matrix_x
    m = matrix
    
    trace = np.trace(m)
    
    if trace > 0:
        # Calculate quaternion w, x, y, z
        S = np.sqrt(trace + 1.0) * 2  # S = 4 * w
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2  # S = 4 * x
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2  # S = 4 * y
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2  # S = 4 * z
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
    if x * cur_target[0] < 0:
        return np.array([-x, -y, -z, -w]), m
    else:
        return np.array([x, y, z, w]), m
def depth_to_depthscale(depth_dir, data_item):
    front_depth = image_to_float_array(
                        
                            Image.open(depth_dir),
                        DEPTH_SCALE)
    near = data_item.misc['front_camera_near']
    far = data_item.misc['front_camera_far']
    front_depth_m = near + front_depth * (far - near)
    return front_depth_m

def worldpoint_to_pixel(data_item):
    camera_extrinsics = data_item.misc['front_camera_extrinsics']
    camera_intrinsics = data_item.misc['front_camera_intrinsics']

    point_world = np.ones(4)
    point_world[:3] = data_item.gripper_pose[:3]
    point_cam =  np.linalg.inv(camera_extrinsics) @ point_world
    # print(final_postion_cam)
    # final_postion_cam2 = [-point_cam[1],-point_cam[2],final_postion_cam[0]]
    pixel_xy = np.dot(camera_intrinsics[:3,:3],point_cam[:3])
    pixel_xy = (pixel_xy/pixel_xy[2])[:2] # (x,y,r^2,r)
    final_pixel = (int(pixel_xy[0]),  int(pixel_xy[1]))
    print(final_pixel)
    return final_pixel

def pixel_to_worldpoint(depth_dir, data_item,x,y):
    depth_m = depth_to_depthscale(depth_dir,data_item)
    # print(depth_m[x,y],data_item.misc['front_camera_extrinsics'].shape,data_item.misc['front_camera_extrinsics'].shape,np.matmul(data_item.misc['front_camera_extrinsics'],data_item.misc['front_camera_extrinsics']))
    point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        depth_m,
                        data_item.misc['front_camera_extrinsics'],
                        data_item.misc['front_camera_intrinsics'])
    return point_cloud[x,y]

def pixel_to_worldpoint2(depth_in_meters, front_camera_extrinsics,front_camera_intrinsics,x,y):
    #print(depth_in_meters.shape,front_camera_extrinsics.shape,front_camera_extrinsics.shape)
    point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        depth_in_meters,
                        front_camera_extrinsics,
                        front_camera_intrinsics)
    # print('point_cloud',point_cloud)
    return point_cloud[y,x]
    # print(point_cloud.shape,point_cloud[x,y],data_item.gripper_pose[:3])
    # H, W, _ = point_cloud.shape
    # points = point_cloud.reshape(-1, 3)  # Flatten to Nx3
    # points[-1] = data_item.gripper_pose[:3]
    # colors = np.full((points.shape[0], 3), [255, 255, 255], dtype=np.uint8)
    # colors[-1] = [255, 0, 0]

    # points_with_colors = np.hstack([points, colors])

    # # Convert to a list of tuples (x, y, z, r, g, b)
    # points_with_colors_tuple = [tuple(point) for point in points_with_colors]

    # # Create a PlyElement for the vertices with color
    # vertex = np.array(points_with_colors_tuple, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    #                                                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    # vertex_element = PlyElement.describe(vertex, 'vertex')

    # # Write the point cloud to a .ply file
    # ply_data = PlyData([vertex_element])
    # ply_data.write('point_cloud.ply')
    # print(tuple(data_item.gripper_pose[:3]) in points_tuple)

def find_nearest_bright_red_point(x, y):
    mask_img =  np.array(Image.open(os.path.join(data_dir,'front_mask','0.png')).convert('RGB'))
    # mask_img =  np.array(Image.open(os.path.join(data_dir,'front_mask','0.png')))
    # print(mask_img)
    # exit()
    # mask_to_rgb_coded_handles(mask_img)
    # 计算红色主导性：红色通道减去绿色和蓝色通道的平均值
    red_dominance = mask_img[:, :, 0] - (mask_img[:, :, 1] + mask_img[:, :, 2]) / 2
    # print(red_dominance)
    # Create a mask for regions where the mask is not [255, 255, 255]
    not_white_mask = np.all(mask_img != [255, 255, 255], axis=-1)

    # Apply this mask to the red dominance values
    red_dominance_filtered = np.where(not_white_mask, red_dominance, -np.inf)

    # Find the maximum red dominance value
    max_red_dominance = np.max(red_dominance_filtered)-6
    # print('----',max_red_dominance,red_dominance[473,240])
    
    # 找到红色主导性最高区域的掩码
    max_red_region_mask = (red_dominance_filtered >= max_red_dominance)
    gray_mask = (max_red_region_mask.astype(np.uint8)) * 255

    # # 将数组转换为Pillow图像
    mask_image = Image.fromarray(gray_mask)

    # # 保存为PNG文件
    mask_image.save(os.path.join(data_dir,'front_mask','0_mask.png'))
    # exit()
    # 获取这个区域的所有坐标
    max_red_region_points = np.column_stack(np.where(max_red_region_mask))
    # print(max_red_region_points)
    # print(mask_img[292,671],mask_img[139,294])
    if max_red_region_mask[x, y]:
        return y,x # 如果xy在最红的区域内，直接返回
    
    # 如果不在，计算区域内离xy最近的点
    distances = np.sqrt((max_red_region_points[:, 1] - y)**2 + (max_red_region_points[:, 0] - x)**2)
    
    # 找到最小距离的索引
    min_index = np.argmin(distances)
    print(x,y,tuple(max_red_region_points[min_index][::-1]))
    # exit()
    # # 返回距离最近的红色主导点
    return tuple(max_red_region_points[min_index][::-1])
    # print(max_red_region_points)
    # exit()
    # return tuple(bright_red_points[min_index][::-1])
#y, x = worldpoint_to_pixel(demo[80])
# print(demo[80].gripper_touch_forces,demo[0].gripper_touch_forces,demo[80].joint_forces[6],demo[0].joint_forces[6])
#rgb_img =  Image.fromarray(np.array(Image.open(os.path.join(data_dir,'front_rgb','0.png')).convert('RGB')))
#draw = ImageDraw.Draw(rgb_img)
#draw.ellipse((y - RADIUS, x - RADIUS, 
#              y + RADIUS, x + RADIUS), fill=COLOR)
#rgb_img.save(os.path.join(data_dir,'front_rgb','0_draw.png'))
#world_point = pixel_to_worldpoint(os.path.join(data_dir,'front_depth','0.png'),demo[0],x,y)
'''
if no contact, gripper_touch_forces = [ 1.12984657e-04 -3.10348805e-05 -1.82940846e-03  1.12530637e-04
  3.29974573e-05  1.83088181e-03] or joint_forces[6] < 0.1
  if contact, gripper_touch_forces = [-0.02713364 -0.02041738 -0.23257861 -0.02604654 -0.04021612 -0.31030497] or joint_forces[6] > 0.1
'''

# rgb_img =  Image.fromarray(np.array(Image.open(os.path.join(data_dir,'front_rgb','98.png')).convert('RGB')))
# draw = ImageDraw.Draw(rgb_img)
# draw.ellipse((y - RADIUS, x - RADIUS, 
#               y + RADIUS, x + RADIUS), fill=COLOR)
# rgb_img.save(os.path.join(data_dir,'front_rgb','98_draw.png'))

def goto_original_pose():
    #pos = [2.78500736e-1,-8.15216638e-3,1.47193456]
    pos=[0.278500736, -0.00815216638, 1.4]
    quat = [0,1,0,0]
    return pos, quat
def quaternion_to_rotation_matrix(q):
    """
    Converts a quaternion into a 3x3 rotation matrix.
    
    Args:
    q: A list or array of 4 elements representing the quaternion (w, x, y, z)
    
    Returns:
    A 3x3 rotation matrix as a numpy array
    """
    x, y, z, w = q
    
    # Compute the elements of the rotation matrix
    R = np.array([[1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
                  [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
                  [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]])
    
    return R

def prepose_before_contact(pred_p,pred_q):
    rotmat = quaternion_to_rotation_matrix(pred_q)
    #print(rotmat)
    z_axis = rotmat[:, 2]
    # print('z_axis:',z_axis)
    new_p = pred_p-0.1*z_axis #move along z-axis for 3 centimeters
    #new_p_2 = pred_p+0.01*z_axis
    return new_p

def postpose_before_contact(pred_p,pred_q,task_name):
    rotmat = quaternion_to_rotation_matrix(pred_q)
    #print(rotmat)
    z_axis = rotmat[:, 2]
    # print('z_axis:',z_axis)
    if 'slide' in task_name:
        print('+++++++++++++++++++++++++++++++++++')
        if pred_p[2] < 0.77:
            new_p = pred_p-0.02*z_axis
        else:
            new_p = pred_p
    elif 'water' in task_name:
        print('++++++++++++++++++++++++++++++++++++0.002')
        new_p = pred_p+0.002*z_axis
    elif 'clock' in task_name:
        print('++++++++++++++++++++++++++++++++++++0.008')
        new_p = pred_p+0.008*z_axis
        new_p[2] = 1.032
    elif 'rope' in task_name:
        print('++++++++++++++++++++++++++++++++++++0.008')
        new_p = pred_p+0.0*z_axis
    else:
        new_p = pred_p+0.008*z_axis #move along z-axis for 3 centimeters
    #new_p_2 = pred_p+0.01*z_axis
    return new_p

def find_nearest_non_yellow_pixel_in_circle(image, center, radius=20):
    """
    Find the nearest non-yellow pixel within a 20-pixel radius of the given center point.
    
    Args:
    - image: PIL Image
    - center: Tuple (x, y) - The center point
    - radius: The radius of the circle (default 20 pixels)

    Returns:
    - Tuple (x, y): Coordinates of the nearest non-yellow pixel, or None if all are yellow
    """
    width, height = image.size
    x0, y0 = center
    pixels = np.array(image)  # Convert image to NumPy array for fast pixel access
    nearest_pixel = None
    min_distance = float('inf')

    # Loop over the bounding box of the circle
    for x in range(max(0, x0 - radius), min(width, x0 + radius + 1)):
        for y in range(max(0, y0 - radius), min(height, y0 + radius + 1)):
            # Check if the pixel is within the circle
            if (x - x0) ** 2 + (y - y0) ** 2 <= radius ** 2:
                # Get the pixel color
                pixel_color = tuple(pixels[y, x][:3])  # Get RGB values
                
                # If the pixel is not yellow, check the distance
                if not is_yellow_rgb(pixel_color):
                    distance = (x - x0) ** 2 + (y - y0) ** 2  # Squared distance
                    if distance < min_distance:
                        min_distance = distance
                        nearest_pixel = (x, y)
    
    return nearest_pixel

def is_yellow_rgb(color):
    """
    Check if a given color in RGB is within the yellow range.
    
    Args:
    - color: Tuple (R, G, B)
    
    Returns:
    - Boolean: True if the color is yellow, False otherwise
    """
    r, g, b = color
    # Define the yellow color range in RGB
    # Yellow has high Red and Green values, and low Blue values
    #lower_yellow = (180, 180, 0)
    #upper_yellow = (255, 255, 150)
    lower_yellow = (160, 120, 50)
    upper_yellow = (225, 190, 110)
    
    return all(lower_yellow[i] <= color[i] <= upper_yellow[i] for i in range(3))
def is_green_rgb(color):
    """
    Check if a given color in RGB is within the yellow range.
    
    Args:
    - color: Tuple (R, G, B)
    
    Returns:
    - Boolean: True if the color is yellow, False otherwise
    """
    r, g, b = color
    # Define the yellow color range in RGB
    # Yellow has high Red and Green values, and low Blue values
    background_green = (10, 60, 60)
    # upper_yellow = (255, 255, 150)
    
    return all(color[i]<=background_green[i] for i in range(3))
def is_grey_rgb(color):
    """
    Check if a given color in RGB is within the yellow range.
    
    Args:
    - color: Tuple (R, G, B)
    
    Returns:
    - Boolean: True if the color is yellow, False otherwise
    """
    r, g, b = color
    # Define the yellow color range in RGB
    # Yellow has high Red and Green values, and low Blue values
    ground_grey = (180, 180, 180)
    # upper_yellow = (255, 255, 150)
    
    return all(color[i]>=ground_grey[i] for i in range(3))


def find_nearest_non_yellowgreengrey_pixel_in_circle(image, center, radius=20):
    """
    Find the nearest non-yellow pixel within a 20-pixel radius of the given center point.
    
    Args:
    - image: PIL Image
    - center: Tuple (x, y) - The center point
    - radius: The radius of the circle (default 20 pixels)

    Returns:
    - Tuple (x, y): Coordinates of the nearest non-yellow pixel, or None if all are yellow
    """
    width, height = image.size
    x0, y0 = center
    pixels = np.array(image)  # Convert image to NumPy array for fast pixel access
    nearest_pixel = None
    min_distance = float('inf')

    # Loop over the bounding box of the circle
    for x in range(max(0, x0 - radius), min(width, x0 + radius + 1)):
        for y in range(max(0, y0 - radius), min(height, y0 + radius + 1)):
            # Check if the pixel is within the circle
            if (x - x0) ** 2 + (y - y0) ** 2 <= radius ** 2:
                # Get the pixel color
                pixel_color = tuple(pixels[y, x][:3])  # Get RGB values
                
                # If the pixel is not yellow, check the distance
                if not is_yellow_rgb(pixel_color) and not is_green_rgb(pixel_color) :
                    distance = (x - x0) ** 2 + (y - y0) ** 2  # Squared distance
                    if distance < min_distance:
                        min_distance = distance
                        nearest_pixel = (x, y)
    
    return nearest_pixel
def is_surrounding_valid(mask_img, pixel, death_in_meters=None, radius=2):
    """
    Check if the surrounding circle of a given pixel is not white.
    
    Args:
    - mask_img: The binary mask image (numpy array)
    - pixel: The center pixel (x, y)
    - radius: The radius of the surrounding circle to check

    Returns:
    - bool: True if the surrounding circle is not white, False otherwise
    """
    x, y = pixel
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if 0 <= x + dx < mask_img.shape[1] and 0 <= y + dy < mask_img.shape[0]:
                if mask_img[y + dy, x + dx] == 255:
                    return False
    # print(np.unique(death_in_meters))
    # assert(0)
    if death_in_meters is not None:
        if death_in_meters[y,x] < 4.0:
            print(death_in_meters[y,x])
            return True
        else:
            return False
    else:
        return True

# image_dir = '/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris/LLaMA-Adapter-v3-Demo-release_336_large_Chinese_ori/save/2024-09-28_07-07-51_exp-909_train_data_v1_0909_touch2d_box_targetbox_0912/water_plants/images/image_water_plant_0_0_draw.png'
def find_nearest_segmentation_pixel_in_circle(image_dir, center, death_in_meters, radius=20):
    """
    Find the nearest non-yellow pixel within a 20-pixel radius of the given center point.
    
    Args:
    - image: PIL Image
    - center: Tuple (x, y) - The center point
    - radius: The radius of the circle (default 20 pixels)

    Returns:
    - Tuple (x, y): Coordinates of the nearest non-yellow pixel, or None if all are yellow
    """
    seg_dir = image_dir[:-8]+'seg.png'
    mask_img =  np.array(Image.open(seg_dir).convert('L'))
    Image.open(seg_dir).convert('L').save('./mask.png')
    
    object_pixels = np.argwhere(mask_img < 255)
    
    if len(object_pixels) == 0:
        return None, None

    # Compute distances from the given pixel to all object pixels
    
    distances = np.linalg.norm(object_pixels - (center[1],center[0]), axis=1)
    # print(distances)
    # Sort object pixels by distance
    sorted_indices = np.argsort(distances)
    # return [object_pixels[0][1], object_pixels[0][0]]
    # Iterate over sorted pixels and check if the surrounding area is valid
    for idx in sorted_indices:
        # print(idx)
        # print(distances[idx],distances)
        nearest_pixel = [object_pixels[idx][1], object_pixels[idx][0]]
        
        if is_surrounding_valid(mask_img, nearest_pixel,death_in_meters):
            return nearest_pixel

    # Find the index of the nearest object pixel
    # nearest_idx = np.argmin(distances)
    # nearest_pixel = [object_pixels[nearest_idx][1],object_pixels[nearest_idx][0]]
    # print(mask_img[nearest_pixel[0],nearest_pixel[1]],center,nearest_pixel)
    # return nearest_pixel
# image_dir = '/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris/LLaMA-Adapter-v3-Demo-release_336_large_Chinese_ori/save/2024-09-29_07-45-40_train_data_v1_0909_touch2d_box_target3dpoint_0928/water_plants/images/image_water_plant_1_2_target_point.png'
def find_nearest_segmentation_pixel_in_circle_target(image_dir, center, radius=20):
    """
    Find the nearest non-yellow pixel within a 20-pixel radius of the given center point.
    
    Args:
    - image: PIL Image
    - center: Tuple (x, y) - The center point
    - radius: The radius of the circle (default 20 pixels)

    Returns:
    - Tuple (x, y): Coordinates of the nearest non-yellow pixel, or None if all are yellow
    """
    seg_dir = '_'.join(image_dir.split('_')[:-2])+'_seg_target.png'
    mask_img =  np.array(Image.open(seg_dir).convert('L'))
    Image.open(seg_dir).convert('L').save('./mask.png')
    
    object_pixels = np.argwhere(mask_img < 255)
    
    if len(object_pixels) == 0:
        return None, None

    # Compute distances from the given pixel to all object pixels
    
    distances = np.linalg.norm(object_pixels - (center[1],center[0]), axis=1)
    # print(distances)
    # Sort object pixels by distance
    sorted_indices = np.argsort(distances)
    # return [object_pixels[0][1], object_pixels[0][0]]
    # Iterate over sorted pixels and check if the surrounding area is valid
    for idx in sorted_indices:
        # print(idx)
        # print(distances[idx],distances)
        nearest_pixel = [object_pixels[idx][1], object_pixels[idx][0]]
        
        if is_surrounding_valid(mask_img, nearest_pixel):
            img = Image.open(image_dir)
            draw = ImageDraw.Draw(img)
            draw.ellipse((nearest_pixel[0] - 3, nearest_pixel[1] - 3, nearest_pixel[0] + 3, nearest_pixel[1] + 3), fill='blue')
            img.save(image_dir[:-4]+'_adj.png')
            return nearest_pixel
            

    # Find the index of the nearest object pixel
    # nearest_idx = np.argmin(distances)
    # nearest_pixel = [object_pixels[nearest_idx][1],object_pixels[nearest_idx][0]]
    # print(mask_img[nearest_pixel[0],nearest_pixel[1]],center,nearest_pixel)
    # return nearest_pixel
# new_pos = find_nearest_segmentation_pixel_i/n_circle_target(image_dir,[116,467])
# img = Image.open(image_dir)
# draw = ImageDraw.Draw(img)
# draw.ellipse((new_pos[0] - 3, new_pos[1] - 3, new_pos[0] + 3, new_pos[1] + 3), fill='blue')
# img.save('./new_pos.png')