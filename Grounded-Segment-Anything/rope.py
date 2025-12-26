import cv2
import numpy as np
from skimage.morphology import skeletonize
lower_bound = np.array([0, 100, 100])     # HSV 下界
upper_bound = np.array([10, 255, 255])
seg_dir = '/new/algo/user8/lixiaoqi/cloris/LLaMA-Adapter-v3-Demo-release_336_large_Chinese_ori/save/2024-12-02_05-26-21_train_data_touch2d_box_target_3dpoint_v1_1024_moretasks_v2_1127_train_oppo_loss/straighten_rope/images/image_straighten_rope_0_0_seg.png'
hsv_mask = cv2.cvtColor(cv2.imread(seg_dir), cv2.COLOR_BGR2HSV)
binary_mask = cv2.inRange(hsv_mask, lower_bound, upper_bound)
    # 调用前述逻辑寻找二值图像中绳子的两端点
# points = np.column_stack(np.where(binary_mask > 0))
# if points.shape[0] < 2:
#     raise ValueError("无法检测到足够的绳子像素，检查颜色范围或图像内容")

# # 寻找两端点（欧几里得距离最大的点对）
# max_distance = 0
# endpoints = (points[0], points[0])

# for i in range(len(points)):
#     for j in range(i + 1, len(points)):
#         dist = np.sum((points[i] - points[j]) ** 2)
#         if dist > max_distance:
#             max_distance = dist
#             endpoints = (points[i], points[j])
def is_endpoint(mask, y, x):
    """
    Check if a point (y, x) in the mask is an endpoint (has exactly one neighbor).

    Args:
        mask: 2D boolean array representing the skeletonized rope mask.
        y, x: Coordinates of the point to check.

    Returns:
        True if the point is an endpoint, False otherwise.
    """
    # Extract a 3x3 neighborhood
    neighbors = mask[max(0, y-1):y+2, max(0, x-1):x+2]
    # Count the number of True values in the neighborhood (excluding the center point itself)
    num_neighbors = np.sum(neighbors) - mask[y, x]
    return num_neighbors == 1  # Endpoint has exactly one neighbor
# (y1, x1), (y2, x2) = endpoints
# print((y1, x1), (y2, x2))
mask = binary_mask > 0
print(mask.shape)
skeleton = skeletonize(mask)
points = np.column_stack(np.where(skeleton))
# print(points)
endpoints = []
for y, x in points:
    # Extract a 3x3 neighborhood
    neighbors = skeleton[max(0, y-1):y+4, max(0, x-1):x+4]
    # Count neighbors (exclude the point itself)
    num_neighbors = np.sum(neighbors) - skeleton[y, x]
    if num_neighbors == 1:  # Endpoint has exactly one neighbor
        endpoints.append((y, x))
print(endpoints)
exit()
# 寻找端点：度为1的点
endpoints = []
for y, x in points:
    # 检查 (y, x) 的邻域像素数量
    neighbors = skeleton[max(0, y-1):y+2, max(0, x-1):x+2]
    if np.sum(neighbors) == 2:  # 自身像素加1，所以邻居总和为2
        endpoints.append((y, x))

# 检查是否找到两个端点
if len(endpoints) != 2:
    raise ValueError("未能找到准确的两端点，请检查输入掩码")

print(endpoints[0], endpoints[1])