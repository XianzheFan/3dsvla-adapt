import os
import re
import pickle
import json
import numpy as np
import sys
import shutil
from PIL import Image, ImageDraw
from PIL import Image
from pyrep.objects import VisionSensor
DEPTH_SCALE = 2**24 - 1
#sys.path.append('/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris/LLaMA-Adapter-v3-Demo-release_336_large_Chinese_ori/YARR/yarr/utils/')
from rlbench.backend.utils import image_to_float_array, rgb_handles_to_mask
#from load_pickle import find_nearest_bright_red_point
front_camera_extrinsics = np.array([[ 1.19209290e-07, -4.22617942e-01, -9.06307936e-01,  1.34999919e+00],
 [-1.00000000e+00, -5.96046448e-07,  1.49011612e-07, 3.71546562e-08],
 [-5.66244125e-07 , 9.06307936e-01, -4.22617912e-01,  1.57999933e+00],
 [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])
front_camera_intrinsics = np.array([[-461.57622105 ,   0.    ,      168.],[0.  ,       -461.57622105 , 168.],[0,0,1]])
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
def is_surrounding_valid(mask_img, pixel, radius=3):
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
    return True
def find_nearest_segmentation_pixel_in_circle(image_dir, center, radius=20):
    """
    Find the nearest non-yellow pixel within a 20-pixel radius of the given center point.
    
    Args:
    - image: PIL Image
    - center: Tuple (x, y) - The center point
    - radius: The radius of the circle (default 20 pixels)

    Returns:
    - Tuple (x, y): Coordinates of the nearest non-yellow pixel, or None if all are yellow
    """
    center.reverse()
    front_dir = '/'.join(image_dir.split('/')[:-1])
    img_list = os.listdir(front_dir)
    for img_item in sorted(img_list):
        if 'buzz' in image_dir:
            if 'segmask' in img_item:
                break
        else:
            if 'segmask' in img_item and 'target' in img_item:
                break
    seg_dir = os.path.join(front_dir,img_item)
    mask_img =  np.array(Image.open(seg_dir).convert('L'))
    img = Image.open(image_dir)
    draw = ImageDraw.Draw(img)
    
    object_pixels = np.argwhere(mask_img < 255)
    draw.ellipse((center[0] - 3, center[1] - 3, center[0] + 3, center[1] + 3), fill='blue')
    img.save(os.path.join(front_dir,'draw_before_0929.png'))
    if mask_img[center[1],center[0]] != 255:
        return center
    # print(mask_img[center[1],center[0]],front_dir)
    # assert(0)
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
            draw.ellipse((nearest_pixel[0] - 3, nearest_pixel[1] - 3, nearest_pixel[0] + 3, nearest_pixel[1] + 3), fill='red')
            img.save(os.path.join(front_dir,'draw_adj_0929.png'))
            return nearest_pixel
def depth_to_depthscale(depth_dir, data_item):
    front_depth = image_to_float_array(
                        
                            Image.open(depth_dir),
                        DEPTH_SCALE)
    near = 0.009999999776482582 
    far = 4.5
    front_depth_m = near + front_depth * (far - near)
    return front_depth_m
def pixel_to_worldpoint(depth_dir, data_item,x,y):
    depth_m = depth_to_depthscale(depth_dir,data_item)
    
    point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        depth_m,
                        front_camera_extrinsics,
                        front_camera_intrinsics)
    
    return point_cloud[x,y]
def discretize_1(q):
    q = [element / 0.01 for element in q]
    q = [round(element) for element in q]
    return q

def discretize_2(q):
    q = [element / 0.02 for element in q]
    q = [round(element) for element in q]
    return q

def format_prompt_question(data_item,task_name):
    task= data_item['variation_description']
    # print(' '.join(task_name.split('_')))
    # assert(0)
    gripper_pos_now = data_item['gripper_pos_n-1']
    gripper_open_now = data_item['gripper_open_n-1']
    box_data = data_item['box_data']
    target_box_data = data_item['target_box_data']
    
    #if isinstance(gripper_pos_now, np.ndarray):
    #    gripper_pos_now = gripper_pos_now.tolist()
    #if isinstance(gripper_open_now, np.ndarray):
    #    gripper_open_now = gripper_open_now.tolist()

    
    if gripper_pos_now is not None:
        pos_left=gripper_pos_now['left'][:3]
        pos_right=gripper_pos_now['right'][:3]
        rot_left=gripper_pos_now['left'][3:]
        rot_right=gripper_pos_now['right'][3:]
        rot_left = discretize_1(rot_left)
        rot_right = discretize_1(rot_right)
        pos_left = discretize_1(pos_left)
        pos_right = discretize_1(pos_right)
        gripper_left_open_now = gripper_open_now['left']
        gripper_right_open_now = gripper_open_now['right']

        #gripper_pos_now_str = ', '.join(map(str, gripper_pos_now))
        pos_left_str = ', '.join(map(str, pos_left))
        pos_right_str = ', '.join(map(str, pos_right))
        rot_left_str = ', '.join(map(str, rot_left))
        rot_right_str = ', '.join(map(str, rot_right))
        # prompt=("Below is an instruction that describes a task. "
        #         "Write a response that appropriately completes the request.\n\n"
        #         "### Instruction:\nI fail to {task}. Correct the gripper's position, rotation and open-status of completing the task. "
        #         "The gripper position now is [{pos_str}]. The gripper rotation quaternion now is [{rot_str}]. The gripper open-status now is {gripper_open_now} \n\n"
        #         "### Response:").format_map({'task': task, 'pos_str': pos_str, 'rot_str': rot_str, 'gripper_open_now': gripper_open_now})
        if target_box_data:
            
            
            if 'rope' in task_name:
                
                l_target_top_left = [target_box_data['l_top_left'][1],target_box_data['l_top_left'][0]]
                l_target_bottom_right = [target_box_data['l_bottom_right'][1],target_box_data['l_bottom_right'][0]]
                l_target_center_2d = [(l_target_top_left[0]+l_target_bottom_right[0])//2,(l_target_top_left[1]+l_target_bottom_right[1])//2]
                l_target_center_2d.reverse()
                image_path = os.path.join('/'.join(data_item['image'].split('/')[:-1]),'rgb_0000.png')
                depth_dir = os.path.join('/'.join(image_path.split('/')[:-2]),'front_depth','depth_0000.png')
                left_world_point = discretize_1(pixel_to_worldpoint(depth_dir, data_item,l_target_center_2d[1],l_target_center_2d[0]))
                
                r_target_top_left = [target_box_data['r_top_left'][1],target_box_data['r_top_left'][0]]
                r_target_bottom_right = [target_box_data['r_bottom_right'][1],target_box_data['r_bottom_right'][0]]
                r_target_center_2d = [(r_target_top_left[0]+r_target_bottom_right[0])//2,(r_target_top_left[1]+r_target_bottom_right[1])//2]
                r_target_center_2d.reverse()
                image_path = os.path.join('/'.join(data_item['image'].split('/')[:-1]),'rgb_0000.png')
                depth_dir = os.path.join('/'.join(image_path.split('/')[:-2]),'front_depth','depth_0000.png')
                right_world_point = discretize_1(pixel_to_worldpoint(depth_dir, data_item,r_target_center_2d[1],r_target_center_2d[0]))
                prompt=(
                        "Determine the gripper's position, orientation, and operational state for the next step in the task: {task}. The current left gripper position is [{pos_left_str}], left gripper rotation is [{rot_left_str}], left gripper open state is {gripper_left_open_now}. ").format_map({'task': ' '.join(task_name.split('_')), 'pos_left_str': pos_left_str, 'rot_left_str': rot_left_str, 'gripper_left_open_now': gripper_left_open_now})
                prompt+=(
                        "The current right gripper position is [{pos_right_str}], right gripper rotation is [{rot_right_str}], right gripper open state is {gripper_right_open_now}. The target position is within or near the point: {target_point},{target_point2} ").format_map({'pos_right_str': pos_right_str, 'rot_right_str': rot_right_str, 'gripper_right_open_now': gripper_right_open_now,'target_point':left_world_point,'target_point2':right_world_point})
                
            else:
                image_path = os.path.join('/'.join(data_item['image'].split('/')[:-1]),'rgb_0000.png')
                target_top_left = [target_box_data['top_left'][1],target_box_data['top_left'][0]]
                target_bottom_right = [target_box_data['bottom_right'][1],target_box_data['bottom_right'][0]]
                
                target_center_2d = [(target_top_left[0]+target_bottom_right[0])//2,(target_top_left[1]+target_bottom_right[1])//2]
                
                target_center_2d = find_nearest_segmentation_pixel_in_circle(image_path, target_center_2d, radius=20)
            
                target_center_2d.reverse()
                img = Image.open(image_path)
                draw = ImageDraw.Draw(img)
                
                
                draw.ellipse((target_center_2d[0] - 3, target_center_2d[1] - 3, target_center_2d[0] + 3, target_center_2d[1] + 3), fill='blue')
                img.save(os.path.join('/'.join(image_path.split('/')[:-2]),'front_depth','draw_before_1025.png'))
                   
            
                depth_dir = os.path.join('/'.join(image_path.split('/')[:-2]),'front_depth','depth_0000.png')
                world_point = discretize_1(pixel_to_worldpoint(depth_dir, data_item,target_center_2d[1],target_center_2d[0]))
                

                prompt=(
                        "Determine the gripper's position, orientation, and operational state for the next step in the task: {task}. The current left gripper position is [{pos_left_str}], left gripper rotation is [{rot_left_str}], left gripper open state is {gripper_left_open_now}. ").format_map({'task': ' '.join(task_name.split('_')), 'pos_left_str': pos_left_str, 'rot_left_str': rot_left_str, 'gripper_left_open_now': gripper_left_open_now})
                prompt+=(
                        "The current right gripper position is [{pos_right_str}], right gripper rotation is [{rot_right_str}], right gripper open state is {gripper_right_open_now}. The target position is within or near the point: {target_point} ").format_map({'pos_right_str': pos_right_str, 'rot_right_str': rot_right_str, 'gripper_right_open_now': gripper_right_open_now,'target_point':world_point})
            
        else:
            
            prompt=(
                    "Determine the gripper's position, orientation, and operational state for the next step in the task: {task}. The current left gripper position is [{pos_left_str}], left gripper rotation is [{rot_left_str}], left gripper open state is {gripper_left_open_now}. ").format_map({'task': ' '.join(task_name.split('_')), 'pos_left_str': pos_left_str, 'rot_left_str': rot_left_str, 'gripper_left_open_now': gripper_left_open_now})
            prompt+=(
                    "The current right gripper position is [{pos_right_str}], right gripper rotation is [{rot_right_str}], right gripper open state is {gripper_right_open_now}. ").format_map({'pos_right_str': pos_right_str, 'rot_right_str': rot_right_str, 'gripper_right_open_now': gripper_right_open_now})
            
    else:
        
        if box_data:
            if 'rope' not in task_name:
                top_left = [box_data['top_left'][1],box_data['top_left'][0]]
                bottom_right = [box_data['bottom_right'][1],box_data['bottom_right'][0]]
                if 'slide' not in task:
                    prompt=(
                            "Determine the gripper's position, orientation, and operational state for the next step in the task: {task}. The position is within or near the bounding box: {tl}, {br}").format_map({'task': ' '.join(task_name.split('_')), 'tl':top_left,'br':bottom_right})
                else:
                    image_path = os.path.join('/'.join(data_item['image'].split('/')[:-1]),'0.png')
                    target_top_left = [target_box_data['top_left'][1],target_box_data['top_left'][0]]
                    target_bottom_right = [target_box_data['bottom_right'][1],target_box_data['bottom_right'][0]]
                    target_center_2d = [(target_top_left[0]+target_bottom_right[0])//2,(target_top_left[1]+target_bottom_right[1])//2]
                    target_center_2d = find_nearest_segmentation_pixel_in_circle(image_path, target_center_2d, radius=20)
                    prompt=(
                            "Determine the gripper's position, orientation, and operational state for the next step in the task: {task}. The position is within or near the bounding box: {tl}, {br}. The target is opposite to: {target_pixel}").format_map({'task': ' '.join(task_name.split('_')), 'tl':top_left,'br':bottom_right,'target_pixel':[target_center_2d[1],target_center_2d[0]]})
            else:
                l_top_left = [box_data['l_top_left'][1],box_data['l_top_left'][0]]
                l_bottom_right = [box_data['l_bottom_right'][1],box_data['l_bottom_right'][0]]
                r_top_left = [box_data['r_top_left'][1],box_data['r_top_left'][0]]
                r_bottom_right = [box_data['r_bottom_right'][1],box_data['r_bottom_right'][0]]
                prompt=(
                                "Determine the gripper's position, orientation, and operational state for the next step in the task: {task}. The position is within or near the bounding box: {tl}, {br}, {tl2}, {br2}").format_map({'task': ' '.join(task_name.split('_')), 'tl':l_top_left,'br':l_bottom_right, 'tl2':r_top_left,'br2':r_bottom_right})
                
        else:
            prompt=(
                "Determine the gripper's position, orientation, and operational state for the next step in the task: {task}.").format_map({'task': task})
    #prompt=(
    #prompt=(
    #            "Determine the gripper's position, orientation, and operational state for the next step in the task: {task}.").format_map({'task': task})
    
    return prompt

def format_prompt_answer(data_item,task_name):
    gripper_pose_future = data_item['gripper_pos_n']
    gripper_open_future = data_item['gripper_open_n']

    # if isinstance(gripper_pose_future, np.ndarray):
    #     gripper_pose_future = gripper_pose_future.tolist()
    # if isinstance(gripper_open_future, np.ndarray):
    #     gripper_open_future = gripper_open_future.tolist()
    if len(gripper_pose_future['left'])==6:
        pos_left=gripper_pose_future['left'][:2]
        rot_left=gripper_pose_future['left'][2:]
        pos_right=gripper_pose_future['right'][:2]
        rot_right=gripper_pose_future['right'][2:]
    else:
        pos_left=gripper_pose_future['left'][:3]
        rot_left=gripper_pose_future['left'][3:]
        pos_right=gripper_pose_future['right'][:3]
        rot_right=gripper_pose_future['right'][3:]
        pos_left = discretize_1(pos_left)
        pos_right = discretize_1(pos_right)
    gripper_left_open_future = gripper_open_future['left']
    gripper_right_open_future = gripper_open_future['right']
    rot_left = discretize_1(rot_left)
    rot_right = discretize_1(rot_right)

    
    
    pos_left_str = ', '.join(map(str, pos_left))
    pos_right_str = ', '.join(map(str, pos_right))
    rot_left_str = ', '.join(map(str, rot_left))
    rot_right_str = ', '.join(map(str, rot_right))
    prompt=("The left gripper position is [{pos_left_str}], the left gripper rotation quaternion is [{rot_left_str}], the left gripper open status is {gripper_left_open_future}. ").format_map({'pos_left_str': pos_left_str, 'rot_left_str': rot_left_str, 'gripper_left_open_future': gripper_left_open_future})
    prompt += ("The right gripper position is [{pos_right_str}], the right gripper rotation quaternion is [{rot_right_str}], the right gripper open status is {gripper_right_open_future}. ").format_map({'pos_right_str': pos_right_str, 'rot_right_str': rot_right_str, 'gripper_right_open_future': gripper_right_open_future})
    
    return prompt

def find_episode_folders(root_path):
    episode_folders = []
    episode_pattern = re.compile(r'episode\d+')

    for dirpath, dirnames, filenames in os.walk(root_path):
        for dirname in dirnames:
            if episode_pattern.match(dirname):
                episode_folders.append(os.path.join(dirpath, dirname))

    return episode_folders

def get_image_paths(image_folder):
    image_paths = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    
    #img_idx = [int(os.path.basename(path).split('.')[0]) for path in image_paths]
    img_idx = []
    valid_image_paths = []
    for path in image_paths:
        try:
            idx = int(os.path.basename(path).split('.')[0])
            img_idx.append(idx)
            valid_image_paths.append(path)
        except (ValueError, IndexError):
            continue
    image_paths = valid_image_paths
            

    img_idx.sort()
    image_paths = [os.path.join(image_folder, str(idx)+'.png') for idx in img_idx]
    #image_paths.sort()
    return image_paths

def idx_filter(image_paths, gripper_pos, gripper_open, touch_force,joint_force):
    idxs= [int(os.path.basename(path).split('.')[0]) for path in image_paths]
    #print_flag=0
    #if image_paths[0]=="/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris-2/RLBench/train_data0819_672/2024-8-19-4-15-10-31-87/front_rgb/0.png":
    #    print_flag=1
    #    print(image_paths)
    if isinstance(gripper_pos, np.ndarray):
        gripper_pos = gripper_pos.tolist()
    if isinstance(gripper_open, np.ndarray):
        gripper_open = gripper_open.tolist()

    pos=[pos[:3] for pos in gripper_pos]
    rot=[pos[3:] for pos in gripper_pos]
    rot = [discretize_2(i) for i in rot]
    pos = [discretize_2(i) for i in pos]

    pop_idxs=[]
    #if print_flag:
    #    print(pos[1])
    #    print(rot[1])
    #    print(pos[2])
    #    print(rot[2])

    for i in range(len(idxs)-1):
        pos_diff = sum(abs(a-b) for a,b in zip(pos[i], pos[i+1]))
        rot_diff = sum(abs(a-b) for a,b in zip(rot[i], rot[i+1]))
        if idxs[i]-idxs[i+1]==-1 and pos_diff<5 and rot_diff<5:
            #if print_flag:
            #    print(idxs[i])
            pop_idxs.append(i)
            #if print_flag:
            #    print(i)
    for i in reversed(pop_idxs):
        idxs.pop(i)
        image_paths.pop(i)
        gripper_pos.pop(i)
        gripper_open.pop(i)
        touch_force.pop(i)
        joint_force.pop(i)
        pos.pop(i)
        rot.pop(i)
    task_name = image_paths[0].split('/')[-3].split('-')[0]
    if 'lamp_on' in task_name or 'push_button' in task_name:
        
        START_IDX = len(idxs)-2
        # last_id = image_paths[-1].split('/')[-1].split('.')[0]
        # print(last_id)
        gripper_pos[len(idxs)-1][2] -= 0.03
    elif 'usb' in task_name:
        START_IDX = len(idxs)-2
    elif 'slide_block' in task_name:
        START_IDX = len(idxs)-2
    elif 'stack' in task_name:
        START_IDX = len(idxs)-2
    else:
        START_IDX = len(idxs)-1
    
    if 'jar' in task_name or 'phone' in task_name:
        pos_diff_threshold = 2
    else:
        pos_diff_threshold = 5
    if 'clock' in task_name :
        rot_diff_threshold = 3
        pos_diff_threshold = 3
    else:
        rot_diff_threshold = 7

    for i in range(START_IDX,0,-1):
        pos_diff = sum(abs(a-b) for a,b in zip(pos[i], pos[i-1]))
        rot_diff = sum(abs(a-b) for a,b in zip(rot[i], rot[i-1]))

        print(i,pos_diff,rot_diff)
        if pos_diff< pos_diff_threshold and rot_diff<rot_diff_threshold:
            force_norm_i = np.linalg.norm(touch_force[i])
            force_norm_i_1 = np.linalg.norm(touch_force[i-1])
            if force_norm_i<0.01 and force_norm_i_1<0.01:
                if abs(joint_force[i])< abs(joint_force[i-1]):
                    for lst in (idxs,image_paths, gripper_pos, gripper_open, touch_force, joint_force, pos, rot):
                        lst.pop(i)
                    #if print_flag:
                    #    print(i)
                else:
                    for lst in (idxs,image_paths, gripper_pos, gripper_open, touch_force, joint_force, pos, rot):
                        lst.pop(i-1)
                    #if print_flag:
                    #    print(i-1)
            elif force_norm_i < force_norm_i_1:
                for lst in (idxs,image_paths, gripper_pos, gripper_open, touch_force, joint_force, pos, rot):
                    lst.pop(i)
                #if print_flag:
                #    print(i)
            else:
                for lst in (idxs,image_paths, gripper_pos, gripper_open, touch_force, joint_force, pos, rot):
                    lst.pop(i-1)
                #if print_flag:
                #    print(i-1)
    
    return image_paths, gripper_pos, gripper_open, touch_force,joint_force

def idx_touch_filter(image_paths, gripper_pos, gripper_open, touch_force,joint_force):
    assert len(image_paths)==len(gripper_pos)==len(gripper_open)==len(touch_force)==len(joint_force)
    #print_flag=0
    #if image_paths[0]=="/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris-2/RLBench/train_data0819_672/2024-8-19-4-15-10-31-87/front_rgb/0.png":
    #    print_flag=1
    #    print(image_paths)
    #bool_list=[False]*len(image_paths)
    #for i in range(len(image_paths)):
    #    force_norm = np.linalg.norm(touch_force[i])
    #    if force_norm<0.01 and abs(joint_force[i])<0.08:
    #        bool_list[i]=True
    bool_list=[True]*len(image_paths)
    for i in range(len(image_paths)):
        force_norm = np.linalg.norm(touch_force[i])
        if force_norm>=0.01:
            bool_list[i]=False
    task_name = image_paths[0].split('/')[-3].split('-')[0]
        
    #print(image_paths)
    #print(bool_list)
    bool_list_copy=bool_list.copy()
    for i in range(len(image_paths)):
        # 边界情况处理
        if i == 0:  # 第一个元素
            if bool_list_copy[i+1]:
                if abs(joint_force[i]) >= 0.08:
                    bool_list[i] = False
        elif i == len(image_paths) - 1:  # 最后一个元素
            if bool_list_copy[i-1]:
                if abs(joint_force[i]) >= 0.08:
                    bool_list[i] = False
        else:  # 中间元素
            if bool_list_copy[i-1] and bool_list_copy[i+1]:
                if abs(joint_force[i]) >= 0.08:
                    bool_list[i] = False
    if 'slide_block' in task_name:
        # print(bool_list)
        # assert(0)
        return image_paths, gripper_pos, gripper_open, touch_force,joint_force,bool_list   
    #print(bool_list)
    #if print_flag:
    #    print(bool_list)
    poped_idxs=[]
    #print(image_paths)
    for i in reversed(range(len(image_paths))):
        #print("i:",i)
        if bool_list[i] and i!=len(image_paths)-1 and any(not x for x in bool_list[i+1:]):
            j=i+1
            #print(j)
            #print(len(image_paths))
            while j<len(image_paths) and bool_list[j]:
                if j not in poped_idxs:
                    #print(image_paths[j])
                    #print(touch_force[j])
                    #print(joint_force[j])
                    poped_idxs.append(j)
                j+=1
    #print(poped_idxs)
    # if 'clock' in task_name:
    #     print(bool_list)
    #     assert(0)
    print(poped_idxs)
    for i in sorted(poped_idxs, reverse=True):
        for lst in (image_paths, gripper_pos, gripper_open, touch_force, joint_force,bool_list):
            lst.pop(i)
    return image_paths, gripper_pos, gripper_open, touch_force,joint_force,bool_list       

def worldpoint_to_pixel_left(data_item):
    camera_extrinsics = data_item.misc['front_camera_extrinsics']
    camera_intrinsics = data_item.misc['front_camera_intrinsics']

    point_world = np.ones(4)
    point_world[:3] = data_item.left.gripper_pose[:3]
    point_cam =  np.linalg.inv(camera_extrinsics) @ point_world
    # print(final_postion_cam)
    # final_postion_cam2 = [-point_cam[1],-point_cam[2],final_postion_cam[0]]
    pixel_xy = np.dot(camera_intrinsics[:3,:3],point_cam[:3])
    pixel_xy = (pixel_xy/pixel_xy[2])[:2] # (x,y,r^2,r)
    final_pixel = (int(pixel_xy[0]),  int(pixel_xy[1]))
    #print(final_pixel)
    return final_pixel
def worldpoint_to_pixel_right(data_item):
    camera_extrinsics = data_item.misc['front_camera_extrinsics']
    camera_intrinsics = data_item.misc['front_camera_intrinsics']

    point_world = np.ones(4)
    point_world[:3] = data_item.right.gripper_pose[:3]
    point_cam =  np.linalg.inv(camera_extrinsics) @ point_world
    # print(final_postion_cam)
    # final_postion_cam2 = [-point_cam[1],-point_cam[2],final_postion_cam[0]]
    pixel_xy = np.dot(camera_intrinsics[:3,:3],point_cam[:3])
    pixel_xy = (pixel_xy/pixel_xy[2])[:2] # (x,y,r^2,r)
    final_pixel = (int(pixel_xy[0]),  int(pixel_xy[1]))
    #print(final_pixel)
    return final_pixel

def find_nearest_bright_red_point(x, y,image_path):
    mask_path = image_path.replace('front_rgb', 'front_mask')
    mask_path = os.path.dirname(mask_path) + '/0.png'
    mask_img =  np.array(Image.open(mask_path).convert('RGB'))
    if x > 672:
        print(x)
        x = 671
    if y > 672:
        print(y)
        y = 671
    #mask_img =  np.array(Image.open(os.path.join(data_dir,'front_mask','0.png')).convert('RGB'))
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
    #print('----',max_red_dominance,red_dominance[473,240])
    
    # 找到红色主导性最高区域的掩码
    max_red_region_mask = (red_dominance_filtered >= max_red_dominance)
    gray_mask = (max_red_region_mask.astype(np.uint8)) * 255

    # # 将数组转换为Pillow图像
    mask_image = Image.fromarray(gray_mask)

    # # 保存为PNG文件
    #mask_image.save(os.path.join(data_dir,'front_mask','0_mask.png'))
    mask_image.save(mask_path.replace('.png','_mask.png'))
    # exit()
    # 获取这个区域的所有坐标
    max_red_region_points = np.column_stack(np.where(max_red_region_mask))
    #print(max_red_region_points)
    #print(mask_img[292,671],mask_img[139,294])
    if max_red_region_mask[x, y]:
        return y,x # 如果xy在最红的区域内，直接返回
    
    # 如果不在，计算区域内离xy最近的点
    distances = np.sqrt((max_red_region_points[:, 1] - y)**2 + (max_red_region_points[:, 0] - x)**2)
    
    # 找到最小距离的索引
    min_index = np.argmin(distances)
    #print(x,y,tuple(max_red_region_points[min_index][::-1]))
    # exit()
    # # 返回距离最近的红色主导点
    return tuple(max_red_region_points[min_index][::-1])
    # print(max_red_region_points)
    # exit()
    # return tuple(bright_red_points[min_index][::-1])
        

def get_dataset(root_path):
    #episode_folders = find_episode_folders(root_path)
    episode_folders = []
    failed_episodes = ['2024-8-19-4-15-11-100-78', '2024-8-19-4-15-11-6-13','2024-8-19-4-15-11-37-84','2024-8-19-4-15-11-54-54',
                       '2024-8-19-4-15-11-41-64','2024-8-19-4-15-11-68-7','2024-8-19-4-15-11-75-41','2024-8-19-4-15-11-44-71','2024-8-19-4-15-11-69-17',
                       '2024-8-19-4-15-11-3-51','2024-8-19-4-15-11-53-94','2024-8-19-4-15-11-21-67','2024-8-19-4-15-11-3-81','2024-8-19-4-15-11-45-73',
                       '2024-8-19-4-15-11-44-26','2024-8-19-4-15-11-95-53','2024-8-19-4-15-11-92-54','2024-8-19-4-15-12-80-32','close_fridge-2024-9-4-15-56-40-89-81',
                       'close_drawer-2024-9-4-15-56-40-36-39','close_jar-2024-9-5-8-20-21-75-1','close_jar-2024-9-5-8-20-20-45-91','close_drawer-2024-9-4-15-56-40-45-83',
                       'close_jar-2024-9-5-8-20-21-53-25','close_jar-2024-9-5-8-20-19-82-17','turn_tap-2024-9-6-6-22-27-32-51','place_wine_at_rack_location-2024-9-6-6-22-26-43-85',
                       'close_jar-2024-9-5-8-20-19-8-71','close_jar-2024-9-5-8-20-19-8-71','close_jar-2024-9-5-8-20-19-78-75','close_drawer-2024-9-4-15-56-40-39-6',
                       'open_fridge-2024-9-9-11-41-32-55-93','stack_blocks-2024-9-9-3-22-11-25-96','water_plants-2024-9-9-12-16-45-19-25','phone_on_base-2024-10-18-7-48-59-55-45',
                       'pour_from_cup_to_cup-2024-10-18-7-49-0-100-29','beat_the_buzz-2024-10-18-7-48-59-87-7','pour_from_cup_to_cup-2024-10-18-7-49-0-1-88','take_umbrella_out_of_umbrella_stand-2024-10-18-7-49-0-95-66',
                       'beat_the_buzz-2024-10-18-7-48-59-25-51','put_plate_in_colored_dish_rack-2024-10-18-7-49-0-89-87','change_clock-2024-10-18-7-49-0-21-24','take_off_weighing_scales-2024-10-18-7-49-0-5-17']
    CAT_LIST = {'lamp_on':'little green light switch','slide_block_to_target':'block','light_bulb_in':'small grey circle','push_button':'round button','turn_tap':'small cross-shaped cross on faucet','stack_blocks':'blocks' ,'place_wine_at_rack_location':'wine','turn on the light':'little green light switch', 'open_fridge': 'fridge','put_rubbish_in_bin':'small white rubbish','open_microwave':'small handle on microwave','close_jar': 'small grey circle on the table', 'close_fridge': 'fridge', 'close_microwave': 'microwave with the whole body', 'drawer': 'drawer', 'close_laptop_lid': 'laptop', 'seat': 'toilet seat with two parts', 'close_box':'box with two parts', 'put_toilet_roll': 'the smallest cylindrical toilet roll that is on the box', 'take_usb':'small usb','lamp_on':'little green light switch', 'unplug_charger':'the smallest black charger on vertical board', 'water_plants':'watering can','sweep':'the dark brown long pole with two parts'}
    CAT_LIST.update({'straighten_rope':'rope','beat_the_buzz':'orange handle','change_clock':'wooden button','phone_on_base':'grey phone near the red base on the desk','place_hanger_on_rack':'handle of hanger','press_switch':'switch button', 'put_knife_on_chopping_board':'knife','put_plate_in_colored_dish_rack':'white dish','take_frame_off_hanger':'frame','take_off_weighing_scales':'red pepper','take_umbrella_out_of_umbrella_stand':'umbrella_handle'})
    failes =[]
    idx = 0
    for task in sorted(os.listdir(root_path)):
        # if '2024-' not in item:
        #     continue
        # item = '2024-8-19-4-15-11-93-7'

        if '.' not in task:
            for episode in os.listdir(os.path.join(root_path,task,'all_variations','episodes')):
                episode_folders.append(os.path.join(root_path,task,'all_variations','episodes',episode))

    ann=[]
    # episode_folders = ['/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris-2/RLBench/train_data0819_672/straighten_rope-2024-10-18-7-49-0-76-13']
    for episode_folder in episode_folders:
        # episode_folder = '/new/algo/user8/lixiaoqi/cloris-2/RLBench/train_data1018/put_plate_in_colored_dish_rack-2024-10-18-7-49-0-28-14'
        print(episode_folder)
        # if 'box' not in episode_folder:
        #     continue
        task_name = episode_folder.split('/')[-4]
        
        image_folder = os.path.join(episode_folder, 'front_rgb')
        with open(os.path.join(episode_folder, 'low_dim_obs.pkl'), 'rb') as f:
            demo = pickle.load(f)
        with open(os.path.join(episode_folder, 'variation_descriptions.pkl'), 'rb') as f:
            variation_description = pickle.load(f)
        # image_paths = get_image_paths(image_folder)
        # image_idx= [int(os.path.basename(path).split('.')[0]) for path in image_paths]
        key_frame_idx = []
        for i in range(len(demo)):
            if i == len(demo)-1:
                key_frame_idx.append(i)
            elif i == 0:
                key_frame_idx.append(i)
            elif (demo[i].left.gripper_open != demo[i-1].left.gripper_open and demo[i-1].right.gripper_open == 0) or (demo[i].right.gripper_open != demo[i-1].right.gripper_open and demo[i-1].left.gripper_open == 0):
                if ('tray' in task_name) and (i-10) < key_frame_idx[-1]:
                    key_frame_idx.pop(-1)
                    key_frame_idx.append(i)
                else:
                    
                    key_frame_idx.append(i)
            elif (all(abs(value) < 0.1 for value in demo[i].left.joint_velocities) and all(abs(value) < 0.1 for value in demo[i].right.joint_velocities)):
                if (sum(abs(demo[i].left.gripper_pose-demo[key_frame_idx[-1]].left.gripper_pose))) > 0.04 and (i-10) > key_frame_idx[-1]:
                    if ('tray' in task_name or 'laptop' in task_name):
                        if demo[i].right.gripper_open==0 or demo[i].left.gripper_open==0:
                            
                            key_frame_idx.append(i)
                    else:
                        
                        key_frame_idx.append(i)
            else:
                continue
        if 'ball' in task_name or 'laptop' in task_name or 'rope' in task_name:
            
            key_frame_idx.pop(1)
        if 'rope' in task_name:
            
            key_frame_idx.pop(-1)
        image_paths = []
        for image_idx in key_frame_idx:
            source_path = os.path.join(image_folder,'rgb_{}.png'.format(str(image_idx).zfill(4)))
            if not os.path.exists(image_folder+'_key_1204_4'):
                os.makedirs(image_folder+'_key_1204_4')
            image_paths.append(source_path)
            target_path = os.path.join(image_folder+'_key_1204_4', os.path.basename(source_path))
            
            shutil.copy(source_path, target_path)
        
        image_idx = key_frame_idx
        
        
        gripper_left_pos = [demo[idx].left.gripper_pose for idx in image_idx]
        gripper_right_pos = [demo[idx].right.gripper_pose for idx in image_idx]
        gripper_left_open=[demo[idx].left.gripper_open for idx in image_idx]
        gripper_right_open=[demo[idx].right.gripper_open for idx in image_idx]
        bool_list = [False]*len(image_paths)
        bool_list[0] = True

        

        assert len(image_paths)==len(gripper_left_pos)==len(gripper_right_pos)==len(gripper_left_open)==len(gripper_right_open)==len(bool_list)
        
        # print(bool_list)
        for i in range(len(image_paths)):
            count = 0
            if i==len(image_paths)-1:
                break 
            elif i==0 or (bool_list[i] and not bool_list[i+1]):
                # print('-----2d')
                count += 1
                image_idx=int(os.path.basename(image_paths[i+1]).split('_')[1].split('.')[0])
                box_data = None
                target_box_data = None
                y_left,x_left = worldpoint_to_pixel_left(demo[image_idx])
                y_right,x_right = worldpoint_to_pixel_right(demo[image_idx])
                
                image_list = os.listdir('/'.join(image_paths[i].split('/')[:-1]))
                for item in sorted(image_list):
                    if '1022' in item and 'json' in item and 'target' not in item:
                        
                        with open(os.path.join(episode_folder, 'front_rgb',item), 'rb') as f:
                            box_data = json.load(f)
                            
                            break
                if 'rope' in task_name:
                    box_data = {"l_top_left": [y_left-50, x_left-50], "l_bottom_right": [y_left+50, x_left+50],"r_top_left": [y_right-50, x_right-50], "r_bottom_right": [y_right+50, x_right+50]}
                
                pos_left=[x_left,y_left]+gripper_left_pos[i+1].tolist()[3:]
                pos_right=[x_right,y_right]+gripper_right_pos[i+1].tolist()[3:]
                pos = {'left':pos_left, 'right':pos_right}
                gripper_open = {'left':gripper_left_open[i+1],'right':gripper_right_open[i+1]}
                target_box_data = None
                
                rgb_img = Image.open(image_paths[i])
                draw = ImageDraw.Draw(rgb_img)
                draw.ellipse((y_left - 3, x_left - 3, 
                            y_left + 3, x_left + 3), fill='red')
                draw.ellipse((y_right - 3, x_right - 3, 
                            y_right + 3, x_right + 3), fill='blue')
                rgb_img.save(image_paths[i].replace('.png','_draw3.png'))
                
                data_item = {'image': image_paths[i], 'gripper_pos_n': pos, 'gripper_open_n': gripper_open, 'variation_description': variation_description[0], 'gripper_pos_n-1': None, 'gripper_open_n-1': None,'box_data':box_data,'target_box_data':target_box_data}
                
            else:
                # print('-----3d')
                target_box_data = None
                target_json = 0
                if 'rope' not in task_name:
                    for item in sorted(image_list):
                        if 'target_1022' in item and 'json' in item:
                            target_json += 1
                            
                            with open(os.path.join(episode_folder, 'front_rgb',item), 'rb') as f:
                                target_box_data = json.load(f)
                                
                                break
                else:
                    target_box_data = {}
                    for item in sorted(image_list):
                        if 'target_1022' in item and 'json' in item and '_0_' in item:
                            target_json += 1
                            
                            with open(os.path.join(episode_folder, 'front_rgb',item), 'rb') as f:
                                target_box_data_json = json.load(f)
                                
                                target_box_data['l_top_left'] = target_box_data_json['top_left']
                                target_box_data['l_bottom_right'] = target_box_data_json['bottom_right']
                            continue
                        if 'target_1022' in item and 'json' in item and '_1_' in item:
                            target_json += 1
                            
                            with open(os.path.join(episode_folder, 'front_rgb',item), 'rb') as f:
                                target_box_data_json = json.load(f)
                                
                                target_box_data['r_top_left'] = target_box_data_json['top_left']
                                target_box_data['r_bottom_right'] = target_box_data_json['bottom_right']
                            break
                    
                                
                gripper_pos = {'left':gripper_left_pos[i+1],'right':gripper_right_pos[i+1]}
                gripper_open = {'left':gripper_left_open[i+1],'right':gripper_right_open[i+1]}
                gripper_pos_n_1 = {'left':gripper_left_pos[i],'right':gripper_right_pos[i]}
                gripper_open_n_1 = {'left':gripper_left_open[i],'right':gripper_right_open[i]}
                data_item = {'image': image_paths[i], 'gripper_pos_n': gripper_pos, 'gripper_open_n': gripper_open, 'variation_description': variation_description[0], 'gripper_pos_n-1': gripper_pos_n_1, 'gripper_open_n-1': gripper_open_n_1,'box_data':None,'target_box_data':target_box_data}
                
                
            question = format_prompt_question(data_item,task_name)
            
            answer = format_prompt_answer(data_item,task_name)
            # print(question,answer)
            # assert(0)
            data_item_json = {}
            data_item_json['image'] = data_item['image']
            data_item_json['question'] = question
            data_item_json['answer'] = answer
            
            name=task_name+'_'+data_item['image'].split('/')[-3]+'_'+data_item['image'].split('/')[-1].split('.')[0]
            # print(name,question)
            # assert(0)
            #print(data_item)
            with open(f'/new/algo/user8/lixiaoqi/cloris/LLaMA-Adapter-v3-Demo-release_336_large_Chinese_ori/data/train_data_touch2d_box_target_3dpoint_5task_dual_1209/{name}.json', 'w') as f:
                json.dump(data_item_json, f)
        # assert(0)
    #for item in failes:
    #    print(item)
    #print(idx)
    return ann

if __name__ == '__main__':
    root_path = '/new/algo/user8/lixiaoqi/cloris-2/peract_bimanual/rlbench_data'
    output_path = './train_data_touch2d_box_target_3dpoint_5task_dual_1209'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # ann = get_dataset(root_path)
    ann = get_dataset(root_path)
    # ann = get_dataset(root_path2)
    # print(len(ann))
    # #print(ann[:10])
    # #print(x_max, y_max, z_max)      0.5253432989120483 0.41891759634017944 1.4720853567123413                                                        
    # #print(x_min, y_min, z_min)      -0.12893009185791016 -0.46705663204193115 0.7693771123886108
    
    # for idx,item in enumerate(ann):
    #     name=item['image'].split('/')[-3]+'_'+item['image'].split('/')[-1].split('.')[0]
    #     with open(f'data/rlbench_dataset/{name}.json', 'w') as f:
    #         json.dump(item, f)
