import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import json
import llama.utils
from llama import Tokenizer
import copy
import torchvision.transforms as transforms
import pandas as pd
import random
import os
import re
import multiprocessing
from data.get_dataset import get_dataset
import pytorch3d.ops as torch3d_ops
import open3d as o3d
import numpy as np
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
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

# create data
#transform_train = transforms.Compose([
#    transforms.RandomResizedCrop(size=(336, 336), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC),  # 3 is bicubic
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

transform_train = transforms.Compose([
    transforms.Resize(size=(336, 336), interpolation=BICUBIC),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
transform_train_depth = transforms.Compose([
    transforms.Resize(size=(336, 336), interpolation=BICUBIC),  # 3 is bicubic
    transforms.ToTensor()])
def generate_point_cloud(rgb_image, depth_map):
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
class FinetuneDatasetReal(Dataset):
    def __init__(self, max_words=30, tokenizer_path=None,data_config=None):

        #ann1 = json.load(open('llava_instruct_150k_single_turn.json'))
        #ann2 = json.load(open('alpaca_gpt4_data.json'))
        #ann3 = json.load(open('alpaca_gpt4_data_zh.json'))
        #ann4 = json.load(open('alpaca_data_zh_51k.json'))
        #self.ann = get_dataset('/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris-2/RLBench/train_data0819_672')
        ann = []
        # assert(0)
        dataset_dir = data_config
        for file in os.listdir(dataset_dir):
            if file.endswith(".json"):
                data = json.load(open(os.path.join(dataset_dir, file)))
                ann.append(data)
        #print(f"total length: {len(ann)}")
        self.ann = ann
        self.lang_type = ['EN'] * len(self.ann)
        print(f"total length: {len(self)}")
        self.transform = transform_train
        self.transform_train_depth = transform_train_depth
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        #assert(0)
        data_item = self.ann[index]
        #if 'image' in data_item.keys():
        #    filename = data_item['image'].replace('/data0/data', '/mnt/petrelfs/share_data/hanjiaming')
        #    question = data_item['conversations'][0]['value']
        #    answer = data_item['conversations'][1]['value']
            # < fill path substitution logics here>
            # filename = url.replace("/data0/data/coco/", "/mnt/petrelfs/leimeng/datasets/coco/")

        #    image = Image.open(filename).convert('RGB')
        #    image = self.transform(image)
        #    format_instruction = question
        #    format_input = None
        #else:
        #    image = torch.zeros(3, 336, 336)
        #    format_instruction = data_item['instruction'],
        #    format_input = data_item['input']
        #    answer = data_item['output']
        #input1 = llama.utils.format_prompt(format_instruction, format_input, self.lang_type[index])
        image_filename = data_item['image']
        image = Image.fromarray(np.array(Image.open(image_filename).convert('RGB'))[144:144+336, 152:152+336,:])
        image = self.transform(image)

        input1=llama.utils.format_prompt(data_item['question'], None)
        # input1 = data_item['question']
        answer=data_item['answer']
        
        input2 = input1 + answer
        
        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, image, data_item['question'], None
class FinetuneDataset(Dataset):
    def __init__(self, max_words=30, tokenizer_path=None,data_config=None):

        #ann1 = json.load(open('llava_instruct_150k_single_turn.json'))
        #ann2 = json.load(open('alpaca_gpt4_data.json'))
        #ann3 = json.load(open('alpaca_gpt4_data_zh.json'))
        #ann4 = json.load(open('alpaca_data_zh_51k.json'))
        #self.ann = get_dataset('/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris-2/RLBench/train_data0819_672')
        ann = []
        # assert(0)
        dataset_dir = data_config
        for file in os.listdir(dataset_dir):
            if file.endswith(".json"):
                data = json.load(open(os.path.join(dataset_dir, file)))
                ann.append(data)
        #print(f"total length: {len(ann)}")
        self.ann = ann
        self.lang_type = ['EN'] * len(self.ann)
        print(f"total length: {len(self)}")
        self.transform = transform_train
        self.transform_depth = transform_train_depth
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
    def depth_to_depthscale(self,depth_dir):
        front_depth = image_to_float_array(
                            
                                Image.open(depth_dir),
                            DEPTH_SCALE)
        near = 0.009999999776482582 
        far = 4.5
        front_depth_m = near + front_depth * (far - near)
        return front_depth_m
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        #assert(0)
        data_item = self.ann[index]
        #if 'image' in data_item.keys():
        #    filename = data_item['image'].replace('/data0/data', '/mnt/petrelfs/share_data/hanjiaming')
        #    question = data_item['conversations'][0]['value']
        #    answer = data_item['conversations'][1]['value']
            # < fill path substitution logics here>
            # filename = url.replace("/data0/data/coco/", "/mnt/petrelfs/leimeng/datasets/coco/")

        #    image = Image.open(filename).convert('RGB')
        #    image = self.transform(image)
        #    format_instruction = question
        #    format_input = None
        #else:
        #    image = torch.zeros(3, 336, 336)
        #    format_instruction = data_item['instruction'],
        #    format_input = data_item['input']
        #    answer = data_item['output']
        #input1 = llama.utils.format_prompt(format_instruction, format_input, self.lang_type[index])
        image_filename = data_item['image']
        depth_filename = image_filename.replace('front_rgb', 'front_depth')
        # depth_m = self.depth_to_depthscale(depth_filename)
        image = Image.open(image_filename).convert('RGB')
        image_arr = np.array(image)
        image = self.transform(image)
        # depth_m = Image.fromarray(depth_m)
        # point_cloud = generate_point_cloud(image_arr,depth_m) #451584x3
        
        # point_cloud_ori = point_cloud
        # xmin, ymin, zmin, xmax, ymax, zmax = [-1.5, -1.5, 0.76, 1.5, 1.5, 3.0]
        # min_bound = np.array([xmin, ymin, zmin])
        # max_bound = np.array([xmax, ymax, zmax])
        # mask = np.all(point_cloud[:, :3] > min_bound, axis=1)
        # point_cloud = point_cloud[mask]
        # mask = np.all(point_cloud[:, :3] < max_bound, axis=1)
        # point_cloud = point_cloud[mask]
        # fps sampling
        # _,point_cloud = sample_farthest_points(point_cloud, 2304)
        
        pc_dir = image_filename.replace('front_rgb', 'front_pc')
        pc_dir = pc_dir.replace('png', 'ply')
        pc = o3d.io.read_point_cloud(pc_dir)
        point_cloud = torch.from_numpy(np.asarray(pc.points)).float()
        
        # point_cloud,_=torch3d_ops.sample_farthest_points(points=point_cloud, K=2304)
        # point_cloud = torch.tensor(point_cloud).cuda().unsqueeze(0)
 
        input1=llama.utils.format_prompt(data_item['question'], None)
        # input1 = data_item['question']
        answer=data_item['answer']
        pos = [float(x)/100 for x in answer.split("position is [")[1].split("]")[0].split(",")]
        quat = [float(x)/100 for x in answer.split("quaternion is [")[1].split("]")[0].split(",")]
        if quat[0] < 0:
            quat = [-q for q in quat]
            
        open_status_match = re.search(r"open status is (\d+(\.\d+)?)", answer)
        open_status = [float(open_status_match.group(1)) if open_status_match else 0.0]
        
        input2 = input1 + answer
        # print(input1, answer)
        # assert(0)
        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        
        return input2, labels, input2_mask, image, data_item['question'],point_cloud


class PretrainDataset(Dataset):
    def __init__(self, config_path, transform, max_words=30, tokenizer_path=None):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        images, captions = [], []
        for meta_path in self.config['META']:
            images_this_meta, captions_this_meta = [], []
            for chunk in pd.read_csv(meta_path, sep='\t', lineterminator='\n', chunksize=10 ** 6):
                images_this_meta.extend(chunk['url'].tolist())
                captions_this_meta.extend(chunk['caption'].tolist())
            print(f"{meta_path}: len {len(images_this_meta)}")
            images.extend(images_this_meta)
            captions.extend(captions_this_meta)

        self.data_list = []
        for x, y in zip(images, captions):
            self.data_list.append({'url': x, 'caption': y})
        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        image_path, caption = sample['url'], sample['caption']
        if isinstance(caption, list):
            caption = random.choice(caption)
        caption = str(caption)

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        format_instruction = "Generate caption of this image"
        input1 = llama.utils.format_prompt(format_instruction, None)
        input2 = input1 + caption

        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, image