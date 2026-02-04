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
import time
import traceback
import multiprocessing as mp
from typing import Dict, Any, List, Optional, Tuple
import tensorflow as tf
import tensorflow_datasets as tfds
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


# --------------------------
# Utils: SE(3) / point cloud
# --------------------------
def _quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """q: (4,) xyzw"""
    x, y, z, w = q.astype(np.float64)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy+zz),     2*(xy-wz),       2*(xz+wy)],
        [2*(xy+wz),         1 - 2*(xx+zz),   2*(yz-wx)],
        [2*(xz-wy),         2*(yz+wx),       1 - 2*(xx+yy)],
    ], dtype=np.float64)
    return R

def _pose_to_T(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Return 4x4 transform matrix."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_xyzw_to_rotmat(quat_xyzw)
    T[:3, 3] = position.astype(np.float64)
    return T

def _backproject_depth_to_points(
    depth_m: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    stride: int = 2,
) -> np.ndarray:
    """
    depth_m: (H, W) float, meters (or consistent unit).
    Return: (N, 3) points in camera frame.
    """
    assert depth_m.ndim == 2
    H, W = depth_m.shape
    us = np.arange(0, W, stride, dtype=np.float64)
    vs = np.arange(0, H, stride, dtype=np.float64)
    u, v = np.meshgrid(us, vs)
    z = depth_m[v.astype(np.int64), u.astype(np.int64)].astype(np.float64)

    valid = np.isfinite(z) & (z > 0)
    u = u[valid]
    v = v[valid]
    z = z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)
    return pts

def sample_farthest_points_np(points: np.ndarray, num_samples: int) -> np.ndarray:
    """Naive FPS in numpy (O(NK)). Use only after aggressive subsampling."""
    N = points.shape[0]
    if N <= num_samples:
        return points

    # init
    sampled_idx = np.zeros((num_samples,), dtype=np.int64)
    distances = np.full((N,), np.inf, dtype=np.float64)

    sampled_idx[0] = np.random.randint(0, N)
    last = points[sampled_idx[0]]

    for i in range(1, num_samples):
        d = np.sum((points - last) ** 2, axis=1)
        distances = np.minimum(distances, d)
        sampled_idx[i] = int(np.argmax(distances))
        last = points[sampled_idx[i]]

    return points[sampled_idx]

def _maybe_fast_subsample(points: np.ndarray, cap: int = 20000) -> np.ndarray:
    """Speed helper before FPS."""
    if points.shape[0] <= cap:
        return points
    idx = np.random.choice(points.shape[0], size=cap, replace=False)
    return points[idx]


# --------------------------
# RLDS -> training sample worker
# --------------------------
def _list_rlds_subfolders(root: str) -> List[str]:
    subs = []
    for sub in os.listdir(root):
        p = os.path.join(root, sub)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "dataset_info.json")):
            subs.append(p)
    subs = list(sorted(subs))
    if not subs:
        raise FileNotFoundError(f"[RLDS] No TFDS subfolders under: {root} (expect each contains dataset_info.json)")
    return subs

def _make_tfds_dataset_from_dirs(root: str, train: bool) -> tf.data.Dataset:
    subfolders = _list_rlds_subfolders(root)
    builder = tfds.builder_from_directories(subfolders)

    read_config = tfds.ReadConfig(
        shuffle_seed=int(time.time() * 1e6),
        skip_prefetch=True,
        num_parallel_calls_for_interleave_files=1,
        # num_parallel_calls_for_interleave_files=tf.data.AUTOTUNE,
        interleave_cycle_length=4,
        # interleave_cycle_length=48,
    )

    if "val" not in builder.info.splits:
        split = "train[:95%]" if train else "train[95%:]"
    else:
        split = "train" if train else "val"

    ds = builder.as_dataset(
        split=split,
        shuffle_files=train,
        read_config=read_config,
        decoders={"steps": tfds.decode.SkipDecoding()},
    )
    return ds

def _ignore_tfds_errors(ds: tf.data.Dataset):
    """
    Mirror your logic: skip NotFound/DataLoss, optionally create empty file to bypass TFDS bug.
    """
    import re
    file_path_matcher = re.compile(r"}} (.+); No such")
    it = iter(ds)
    sample_count = 0
    missing_count = 0

    while True:
        try:
            ret = next(it)
            sample_count += 1
            yield ret
        except StopIteration:
            break
        except tf.errors.NotFoundError as e:
            missing_count += 1
            m = file_path_matcher.search(str(e))
            file_path = m.group(1) if m else "UNKNOWN"
            rate = missing_count / max(sample_count + missing_count, 1)
            print(f"[RLDS] tfds missing file {file_path}, missing rate {rate:.4f}")
            if m and file_path != "UNKNOWN":
                try:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, "a"):
                        pass
                except Exception as ee:
                    print(f"[RLDS] failed to create empty file for {file_path}: {ee}")
            continue
        except tf.errors.DataLossError as e:
            missing_count += 1
            rate = missing_count / max(sample_count + missing_count, 1)
            print(f"[RLDS] tfds data corruption, missing rate {rate:.4f}: {e}")
            continue
        except tf.errors.FailedPreconditionError as e:
            if "dataset was expected to contain" in str(e):
                break
            raise

def _cycle_iter(ds: tf.data.Dataset):
    while True:
        for x in ds:
            yield x

def _decode_rgb_bytes(rgb_bytes: Any) -> np.ndarray:
    """
    rgb_bytes could be tf.string bytes or already uint8. We decode robustly.
    Expect output: (H, W, 3) uint8.
    """
    # If already numeric array:
    if isinstance(rgb_bytes, np.ndarray) and rgb_bytes.dtype == np.uint8:
        if rgb_bytes.ndim == 3:
            return rgb_bytes
    t = tf.io.decode_image(rgb_bytes, channels=3, expand_animations=False, dtype=tf.uint8)
    return t.numpy()

def _decode_depth_bytes(depth_bytes: Any) -> np.ndarray:
    """
    Your RLDS depth encoding: 4-channel uint8 -> bitcast float32.
    Return: (H, W) float32
    """
    t = tf.io.decode_image(depth_bytes, channels=4, expand_animations=False, dtype=tf.uint8)
    depth = tf.bitcast(t, tf.float32).numpy()
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 3 and depth.shape[-1] == 4:
        # some datasets might store float bytes differently; fallback:
        # but your pipeline expects bitcast already. We keep first channel if weird.
        depth = depth[..., 0]
    return depth.astype(np.float32)

def _extract_env_config(traj: Dict[str, Any]) -> Dict[str, Any]:
    """
    traj['environment_config'] is a tf.string encoding a python literal dict.
    Your code uses eval(..., {'array': np.array}).
    """
    s = traj["environment_config"].numpy().decode()
    return eval(s, {"array": np.array})

def _pick_intrinsics(env_cfg: Dict[str, Any], default_fx_fy_cx_cy=(322.6666666667, 322.6666666667, 128.0, 128.0)) -> Tuple[float,float,float,float]:
    """
    Try find camera intrinsics in env_cfg['camera_info'] (if present), else fallback.
    Adjust here if your env_cfg schema differs.
    """
    fx, fy, cx, cy = default_fx_fy_cx_cy
    cams = env_cfg.get("camera_info", [])
    # common patterns: cam['name'] in {'front','rgb0','camera0'}
    for cam in cams:
        name = str(cam.get("name", ""))
        if name in ("front", "rgb0", "camera0", "cam0"):
            intr = cam.get("intrinsics", None)
            if intr and all(k in intr for k in ("fx","fy","cx","cy")):
                return float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
    return float(fx), float(fy), float(cx), float(cy)

def _pick_extrinsics_world_T_cam(env_cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    If env_cfg has camera pose, build world_T_cam. Otherwise return None (camera frame).
    """
    cams = env_cfg.get("camera_info", [])
    for cam in cams:
        name = str(cam.get("name", ""))
        if name in ("front", "rgb0", "camera0", "cam0"):
            pos = cam.get("position", None)
            quat = cam.get("orientation", None)
            if pos is not None and quat is not None:
                pos = np.asarray(pos, dtype=np.float64)
                quat = np.asarray(quat, dtype=np.float64)
                if quat.shape[-1] == 4:
                    return _pose_to_T(pos, quat)
    return None

def _format_action_answer(pos_xyz: np.ndarray, quat_xyzw: np.ndarray, open_status: float) -> str:
    """
    Keep the same pattern your FinetuneDataset parses:
    - "position is [..]"
    - "quaternion is [..]"
    - "open status is .."
    """
    # match your 1/100 scaling convention (optional). Here we keep raw meters; adjust if your parser expects /100.
    pos_s = ", ".join([f"{x:.6f}" for x in pos_xyz.tolist()])
    quat_s = ", ".join([f"{x:.6f}" for x in quat_xyzw.tolist()])
    return f"position is [{pos_s}] quaternion is [{quat_s}] open status is {open_status:.3f}\n"


def _rlds_worker_loop(
    queue: "mp.Queue",
    root: str,
    train: bool,
    use_depth: bool,
    image_key: str = "rgb0",
    depth_key: str = "depth0",
    pc_num: int = 2304,
    sample_stride: int = 2,
    max_traj_samples: int = 1,
):
    """
    In a spawned process:
    - build tfds dataset
    - iterate trajectories
    - sample timesteps
    - produce dict: {question, answer, rgb(uint8), pc(float32 Nx3 or None)}
    """
    try:
        # Avoid TF grabbing GPUs in worker
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            pass

        ds = _make_tfds_dataset_from_dirs(root, train=train)
        opt = tf.data.Options()
        opt.threading.private_threadpool_size = 1
        opt.threading.max_intra_op_parallelism = 1
        try:
            opt.autotune.enabled = False
            opt.autotune.cpu_budget = 1
        except Exception:
            try:
                opt.experimental_optimization.autotune = False
            except Exception:
                pass
        try:
            opt.experimental_optimization.map_parallelization = False
        except Exception:
            pass
        ds = ds.with_options(opt)
        ds = tf.data.Dataset.from_generator(
            lambda: _ignore_tfds_errors(ds),
            output_signature=ds.element_spec
        )
        it = _cycle_iter(ds)

        while True:
            traj = next(it)
            traj_np = tfds.as_numpy(traj)

            # filter
            if not bool(traj_np.get("valid", True)):
                continue
            if "success" in traj_np and not bool(traj_np["success"]):
                continue

            # steps
            steps = tfds.as_numpy(traj["steps"])  # keep as structured np arrays/dicts

            # env_config and captions
            try:
                env_cfg = _extract_env_config(traj)
                captions = env_cfg.get("captions", None)
                if not captions:
                    # fallback
                    captions = ["pick up the object"]
            except Exception:
                env_cfg = {}
                captions = ["pick up the object"]

            # observation arrays
            obs = steps["observation"]
            act = steps["action"]

            if image_key not in obs:
                # fallback heuristic
                # choose first rgb key that exists
                rgb_candidates = [k for k in obs.keys() if "rgb" in k]
                if not rgb_candidates:
                    continue
                rgb_k = sorted(rgb_candidates)[0]
            else:
                rgb_k = image_key

            if use_depth:
                if depth_key not in obs:
                    depth_candidates = [k for k in obs.keys() if "depth" in k]
                    if not depth_candidates:
                        continue
                    depth_k = sorted(depth_candidates)[0]
                else:
                    depth_k = depth_key
            else:
                depth_k = None

            # command eef pose keys (mirror your _check_fk usage)
            # adjust here if your dataset uses different names
            pos_k = "command_eef_position" if "command_eef_position" in act else ("eef_position" if "eef_position" in act else None)
            quat_k = "command_eef_orientation" if "command_eef_orientation" in act else ("eef_orientation" if "eef_orientation" in act else None)
            grip_k = "gripper_action" if "gripper_action" in act else None

            if pos_k is None or quat_k is None:
                continue

            T = len(act[pos_k])
            if T < 2:
                continue

            # sample some timesteps from this trajectory
            for _ in range(max_traj_samples):
                t = random.randint(0, T - 2)

                # decode rgb
                rgb_bytes = obs[rgb_k][t]
                rgb = _decode_rgb_bytes(rgb_bytes)

                # decode depth + build pc
                pc = None
                if use_depth and depth_k is not None:
                    try:
                        depth_bytes = obs[depth_k][t]
                        depth_m = _decode_depth_bytes(depth_bytes)  # (H,W)

                        fx, fy, cx, cy = _pick_intrinsics(env_cfg)
                        cam_pts = _backproject_depth_to_points(depth_m, fx, fy, cx, cy, stride=sample_stride)

                        world_T_cam = _pick_extrinsics_world_T_cam(env_cfg)
                        if world_T_cam is not None:
                            ones = np.ones((cam_pts.shape[0], 1), dtype=np.float64)
                            cam_h = np.concatenate([cam_pts.astype(np.float64), ones], axis=1)  # (N,4)
                            world = (world_T_cam @ cam_h.T).T[:, :3]
                            pc = world.astype(np.float32)
                        else:
                            pc = cam_pts.astype(np.float32)

                        # speed: subsample -> FPS
                        pc = _maybe_fast_subsample(pc, cap=20000)
                        pc = sample_farthest_points_np(pc, pc_num).astype(np.float32)
                    except Exception:
                        pc = None

                # action -> answer
                pos = np.asarray(act[pos_k][t], dtype=np.float64)
                quat = np.asarray(act[quat_k][t], dtype=np.float64)
                if quat.shape[-1] != 4:
                    continue

                # enforce quat w >= 0 if you want (match your existing behavior)
                if quat[-1] < 0:
                    quat = -quat

                if grip_k is not None:
                    g = float(act[grip_k][t])
                    # your comment: 1 open, -1 close
                    open_status = 1.0 if g > 0 else 0.0
                else:
                    open_status = 0.0

                question = random.choice(captions)
                answer = _format_action_answer(pos, quat, open_status)

                queue.put({
                    "question": str(question),
                    "answer": str(answer),
                    "rgb": rgb,          # uint8 HWC
                    "pc": pc,            # float32 Nx3 or None
                })

    except KeyboardInterrupt:
        return
    except Exception:
        print("[RLDS worker] crashed:\n", traceback.format_exc())
        return


class FinetuneDatasetRLDS(Dataset):
    """
    3dsvla-compatible dataset:
      returns (input2, labels, input2_mask, image, question, point_cloud)
    Data source: RLDS TFDS directories via builder_from_directories.
    Implementation: a spawned worker process continuously produces samples into a Queue.
    """

    def __init__(
        self,
        rlds_root: str,
        tokenizer_path: str,
        max_words: int = 512,
        train: bool = True,
        use_depth: bool = True,
        pc_num: int = 2304,
        prefetch: int = 8,
        sample_stride: int = 2,
        max_traj_samples: int = 1,
        transform=None,
    ):
        super().__init__()
        self.rlds_root = rlds_root
        self.max_words = int(max_words)
        self.train = bool(train)
        self.use_depth = bool(use_depth)
        self.pc_num = int(pc_num)
        self.prefetch = int(prefetch)
        self.sample_stride = int(sample_stride)
        self.max_traj_samples = int(max_traj_samples)

        self.tokenizer = Tokenizer(model_path=tokenizer_path)

        # Use the same image transform as FinetuneDataset if not provided.
        # Expect the caller to pass transform_train from existing code.
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

        # IMPORTANT: use spawn context to avoid TF + fork issues
        self._ctx = mp.get_context("spawn")
        self._queue: mp.Queue = self._ctx.Queue(maxsize=self.prefetch)
        self._worker: mp.Process = self._ctx.Process(
            target=_rlds_worker_loop,
            args=(
                self._queue,
                self.rlds_root,
                self.train,
                self.use_depth,
            ),
            kwargs=dict(
                image_key="rgb0",
                depth_key="depth0",
                pc_num=self.pc_num,
                sample_stride=self.sample_stride,
                max_traj_samples=self.max_traj_samples,
            ),
            daemon=True,
        )
        self._worker.start()

    def __len__(self):
        # RLDS is essentially infinite stream; keep large
        return int(1e9)

    def __getitem__(self, index):
        sample = self._queue.get()

        question = sample["question"]
        answer = sample["answer"]

        # image
        rgb = sample["rgb"]
        img = Image.fromarray(rgb).convert("RGB")
        img = self.transform(img)

        # point cloud
        pc = sample.get("pc", None)
        if pc is None:
            point_cloud = None
        else:
            point_cloud = torch.from_numpy(pc).float()

        # tokenize (same as FinetuneDataset)
        input1 = llama.utils.format_prompt(question, None)
        input2 = input1 + answer

        input1_ids = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2_ids = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)

        padding = self.max_words - input2_ids.shape[0]
        if padding > 0:
            input2_ids = torch.cat((input2_ids, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2_ids = input2_ids[:self.max_words]

        labels = input2_ids.clone()
        labels[:len(input1_ids)] = -1

        input2_mask = input2_ids.ge(0)
        label_mask = labels.ge(0)

        input2_ids[~input2_mask] = 0
        labels[~label_mask] = 0

        input2_mask = input2_mask.float()

        return input2_ids, labels, input2_mask, img, question, point_cloud

    def __del__(self):
        try:
            if hasattr(self, "_worker") and self._worker.is_alive():
                self._worker.terminate()
        except Exception:
            pass


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