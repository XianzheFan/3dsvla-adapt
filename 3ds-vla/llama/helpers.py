# Parametric Networks for 3D Point Cloud Classification
# import pytorch3d.ops as torch3_ops
import torch
import torch.nn as nn
import pytorch3d.ops as torch3d_ops
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx
# def sample_farthest_points(points, num_samples):
#     """
#     Perform farthest point sampling (FPS) on a batched point cloud.

#     Args:
#         points (torch.Tensor): Input point cloud of shape (B, N, D).
#         num_samples (int): Number of points to sample (K).

#     Returns:
#         torch.Tensor: Indices of the sampled points of shape (B, K).
#         torch.Tensor: Sampled points of shape (B, K, D).
#     """
#     B, N, D = points.shape

#     # Initialize arrays
#     sampled_indices = torch.zeros((B, num_samples), dtype=torch.long, device=points.device)
#     distances = torch.full((B, N), float("inf"), device=points.device)  # Initially, all distances are inf

#     # Randomly select the first point for each batch
#     sampled_indices[:, 0] = torch.randint(0, N, (B,), device=points.device)
#     selected_points = points[torch.arange(B), sampled_indices[:, 0]]  # Shape: (B, D)

#     for i in range(1, num_samples):
#         # Compute distances from the selected point to all other points
#         dist = torch.sum((points - selected_points.unsqueeze(1)) ** 2, dim=2)  # Shape: (B, N)

#         # Update the minimum distance for each point
#         distances = torch.minimum(distances, dist)

#         # Select the farthest point for each batch
#         sampled_indices[:, i] = torch.argmax(distances, dim=1)
#         selected_points = points[torch.arange(B), sampled_indices[:, i]]  # Shape: (B, D)

#     # Gather sampled points
#     sampled_points = points[torch.arange(B).unsqueeze(1), sampled_indices]  # Shape: (B, K, D)

#     return  sampled_points,sampled_indices

# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        _, fps_idx = torch3d_ops.sample_farthest_points(points=xyz, K=self.group_num)
        fps_idx = torch.randint(0, N, (B, self.group_num)).long()
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)
        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)
        # lc_xyz&lc_x is cernter point&feature, knn_xyz&knn_x is neighbor point&feature
        return lc_xyz, lc_x, knn_xyz, knn_x, fps_idx, knn_idx


# Local Geometry Aggregation
class LGA(nn.Module):
    def __init__(
        self, out_dim, alpha, beta, block_num, dim_expansion, type, adapter_layer=0
    ):
        super().__init__()
        self.type = type
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta)
        if dim_expansion == 1:
            expand = 2
        elif dim_expansion == 2:
            expand = 1

        self.adapter_layer = adapter_layer
        self.linear2 = []
        for i in range(block_num):
            self.linear2.append(
                Linear2Layer(out_dim, bias=True, adapter_layer=adapter_layer)
            )
        self.linear2 = nn.Sequential(*self.linear2)

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):

        # Normalization
        if self.type == "mn40":
            mean_xyz = lc_xyz.unsqueeze(dim=-2)
            std_xyz = torch.std(knn_xyz - mean_xyz)
            knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        elif self.type == "scan":
            knn_xyz = knn_xyz.permute(0, 3, 1, 2)
            knn_xyz -= lc_xyz.permute(0, 2, 1).unsqueeze(-1)
            knn_xyz /= torch.abs(knn_xyz).max(dim=-1, keepdim=True)[0]
            knn_xyz = knn_xyz.permute(0, 2, 3, 1)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        # concate the knn_x with the max of dim 2

        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)
        ##take neighbor to do
        # Linear
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_x = knn_x.reshape(B, -1, G, K)

        # Geometry Extraction
        knn_x_w = self.geo_extract(knn_xyz, knn_x)
        # Linear
        for layer in self.linear2:
            knn_x_w = layer(knn_x_w)

        return knn_x_w


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0]
        return lc_x


# Linear layer 1
class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(Linear1Layer, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
            ),
            nn.BatchNorm1d(out_channels),
            self.act,
        )

    def forward(self, x):
        return self.net(x)


# Linear Layer 2
class Linear2Layer(nn.Module):
    def __init__(
        self, in_channels, kernel_size=1, groups=1, bias=True, adapter_layer=0
    ):
        super(Linear2Layer, self).__init__()
        basic_dim = 32
        self.act = nn.ReLU(inplace=True)
        if adapter_layer == 2:
            self.net1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=int(32),
                    kernel_size=kernel_size,
                    groups=groups,
                    bias=bias,
                ),
                nn.BatchNorm2d(int(32)),
                self.act,
            )
            self.net2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=int(32),
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    bias=bias,
                ),
                nn.BatchNorm2d(in_channels),
            )
        else:
            self.net1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=int(in_channels / 2),
                    kernel_size=kernel_size,
                    groups=groups,
                    bias=bias,
                ),
                nn.BatchNorm2d(int(in_channels / 2)),
                self.act,
            )
            self.net2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=int(in_channels / 2),
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    bias=bias,
                ),
                nn.BatchNorm2d(in_channels),
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


# PosE for Local Geometry Extraction
class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, knn_xyz, knn_x):
        B, _, G, K = knn_xyz.shape  
        feat_dim = self.out_dim // (self.in_dim * 2)  

        feat_range = torch.arange(feat_dim).float().to(knn_xyz.device)
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)  
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)  
        cos_embed = torch.cos(div_embed)
        position_embed = torch.cat([sin_embed, cos_embed], -1)
        position_embed = position_embed.permute(
            0, 1, 4, 2, 3
        ).contiguous()  
        position_embed = position_embed.view(B, self.out_dim, G, K)
        knn_x_w = knn_x + position_embed

        return knn_x_w


class EncP(nn.Module):
    def __init__(
        self,
        in_channels,
        input_points,
        num_stages,
        embed_dim,
        k_neighbors,
        alpha,
        beta,
        LGA_block,
        dim_expansion,
        type,
    ):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        # Raw-point Embedding
        self.raw_point_embed = Linear1Layer(in_channels, self.embed_dim, bias=False)
        # for name, param in self.raw_point_embed.named_parameters():
        #     with torch.no_grad():  # Disable gradient computation for conversion
        #         param.copy_(param.to(dtype=torch.float32))
        self.FPS_kNN_list = nn.ModuleList() 
        self.LGA_list = nn.ModuleList()  
        self.Pooling_list = nn.ModuleList() 

        out_dim = self.embed_dim
        group_num = self.input_points
        group_num_dict = {'0':1804,'1':1048,'2':576}
        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # if i == self.num_stages-1:
            #     out_dim = 1024
            # else:
            out_dim = out_dim * dim_expansion[i]
            # print(out_dim,group_num)
            # group_num = group_num // 2
            group_num = group_num_dict[str(i)]
            
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.LGA_list.append(
                LGA(
                    out_dim,
                    self.alpha,
                    self.beta,
                    LGA_block[i],
                    dim_expansion[i],
                    type,
                    adapter_layer=i,
                )
            )

            self.Pooling_list.append(Pooling(out_dim))
        

    def forward(self, xyz, x):

        # Raw-point Embedding
        
        x = self.raw_point_embed(x) 
        
        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x, _, _ = self.FPS_kNN_list[i](
                xyz, x.permute(0, 2, 1)
            )
            # Local Geometry Aggregation
            knn_x_w = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x)
            # Pooling
            
            x = self.Pooling_list[i](knn_x_w)
        #     print(x.shape)
        # assert(0)
            

        return xyz, x


class Point_PN_scan(nn.Module):
    def __init__(
        self,
        in_channels=3,
        class_num=15,
        input_points=2304,
        num_stages=3,
        embed_dim=96,
        beta=100,
        alpha=1000,
        LGA_block=[2, 1, 1, 1],
        dim_expansion=[2, 2, 2, 1],
        type="mn40",
        k_neighbors=81,
    ):
        super().__init__()
        # Parametric Encoder
        self.EncP = EncP(
            in_channels,
            input_points,
            num_stages,
            embed_dim,
            k_neighbors,
            alpha,
            beta,
            LGA_block,
            dim_expansion,
            type,
        )
        self.out_channels = embed_dim
        for i in dim_expansion:
            self.out_channels *= i

    def forward(self, x, xyz):
        xyz, x = self.EncP(xyz, x)
        return xyz, x
import numpy as np
import torch
import yaml
from easydict import EasyDict

LENGTH = 77
TRANS = -1.6

RESOLUTION = 336
TRANS = -1.6


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     :param angle: [3] or [b, 3]
     :return
        rotmat: [3] or [b, 3, 3]
    source
    https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    """
    if len(angle.size()) == 1:
        x, y, z = angle[0], angle[1], angle[2]
        _dim = 0
        _view = [3, 3]
    elif len(angle.size()) == 2:
        b, _ = angle.size()
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
        _dim = 1
        _view = [b, 3, 3]

    else:
        assert False

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zero = z.detach() * 0
    one = zero.detach() + 1
    zmat = torch.stack(
        [cosz, -sinz, zero, sinz, cosz, zero, zero, zero, one], dim=_dim
    ).reshape(_view)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack(
        [cosy, zero, siny, zero, one, zero, -siny, zero, cosy], dim=_dim
    ).reshape(_view)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack(
        [one, zero, zero, zero, cosx, -sinx, zero, sinx, cosx], dim=_dim
    ).reshape(_view)

    rot_mat = xmat @ ymat @ zmat
    return rot_mat


def distribute(depth, _x, _y, size_x, size_y, image_height, image_width):
    """
    Distributes the depth associated with each point to the discrete coordinates (image_height, image_width) in a region
    of size (size_x, size_y).
    :param depth:
    :param _x:
    :param _y:
    :param size_x:
    :param size_y:
    :param image_height:
    :param image_width:
    :return:
    """

    assert size_x % 2 == 0 or size_x == 1
    assert size_y % 2 == 0 or size_y == 1
    batch, _ = depth.size()
    epsilon = torch.tensor([1e-12], requires_grad=False, device=depth.device)
    _i = torch.linspace(
        -size_x / 2, (size_x / 2) - 1, size_x, requires_grad=False, device=depth.device
    )
    _j = torch.linspace(
        -size_y / 2, (size_y / 2) - 1, size_y, requires_grad=False, device=depth.device
    )
    extended_x = (
        _x.unsqueeze(2).repeat([1, 1, size_x]) + _i
    ) 
    extended_y = (
        _y.unsqueeze(2).repeat([1, 1, size_y]) + _j
    )  

    extended_x = extended_x.unsqueeze(3).repeat(
        [1, 1, 1, size_y]
    )  
    extended_y = extended_y.unsqueeze(2).repeat(
        [1, 1, size_x, 1]
    ) 

    extended_x.ceil_()
    extended_y.ceil_()

    value = (
        depth.unsqueeze(2).unsqueeze(3).repeat([1, 1, size_x, size_y])
    ) 

    # all points that will be finally used
    masked_points = (
        (extended_x >= 0)
        * (extended_x <= image_height - 1)
        * (extended_y >= 0)
        * (extended_y <= image_width - 1)
        * (value >= 0)
    )

    true_extended_x = extended_x
    true_extended_y = extended_y

    # to prevent error
    extended_x = extended_x % image_height
    extended_y = extended_y % image_width

    distance = torch.abs(
        (extended_x - _x.unsqueeze(2).unsqueeze(3))
        * (extended_y - _y.unsqueeze(2).unsqueeze(3))
    )
    weight = masked_points.float() * (
        1 / (value + epsilon)
    ) 
    weighted_value = value * weight

    weight = weight.view([batch, -1])
    weighted_value = weighted_value.view([batch, -1])

    coordinates = (extended_x.view([batch, -1]) * image_width) + extended_y.view(
        [batch, -1]
    )
    coord_max = image_height * image_width
    true_coordinates = (
        true_extended_x.view([batch, -1]) * image_width
    ) + true_extended_y.view([batch, -1])
    true_coordinates[~masked_points.view([batch, -1])] = coord_max
    weight_scattered = torch.zeros(
        [batch, image_width * image_height], device=depth.device
    ).scatter_add(1, coordinates.long(), weight)

    masked_zero_weight_scattered = weight_scattered == 0.0
    weight_scattered += masked_zero_weight_scattered.float()

    weighed_value_scattered = torch.zeros(
        [batch, image_width * image_height], device=depth.device
    ).scatter_add(1, coordinates.long(), weighted_value)

    return weighed_value_scattered, weight_scattered


def points2depth(points, image_height, image_width, size_x=4, size_y=4):
    """
    :param points: [B, num_points, 3]
    :param image_width:
    :param image_height:
    :param size_x:
    :param size_y:
    :return:
        depth_recovered: [B, image_width, image_height]
    """

    epsilon = torch.tensor([1e-12], requires_grad=False, device=points.device)
    coord_x = (points[:, :, 0] / (points[:, :, 2] + epsilon)) * (
        image_width / image_height
    )  
    coord_y = points[:, :, 1] / (points[:, :, 2] + epsilon)  

    batch, total_points, _ = points.size()
    depth = points[:, :, 2]  
    _x = ((coord_x + 1) * image_height) / 2
    _y = ((coord_y + 1) * image_width) / 2

    weighed_value_scattered, weight_scattered = distribute(
        depth=depth,
        _x=_x,
        _y=_y,
        size_x=size_x,
        size_y=size_y,
        image_height=image_height,
        image_width=image_width,
    )

    depth_recovered = (weighed_value_scattered / weight_scattered).view(
        [batch, image_height, image_width]
    )

    return depth_recovered


def points2grid(points, resolution=518, depth=8):
    """Quantize each point cloud to a 3D grid.
    Args:
        points (torch.tensor): of size [B, _, 3]
    Returns:
        grid (torch.tensor): of size [B * self.num_views, depth, resolution, resolution]
    """

    batch, pnum, _ = points.shape

    pmax, pmin = points.max(dim=1)[0], points.min(dim=1)[0]
    pcent = (pmax + pmin) / 2
    pcent = pcent[:, None, :]
    prange = (pmax - pmin).max(dim=-1)[0][:, None, None]
    points = (points - pcent) / prange * 2.0
    points[:, :, :2] = points[:, :, :2] * 0.8

    depth_bias = 0.2
    _x = (points[:, :, 0] + 1) / 2 * resolution
    _y = (points[:, :, 1] + 1) / 2 * resolution
    _z = ((points[:, :, 2] + 1) / 2 + depth_bias) / (1 + depth_bias) * (depth - 2)

    _x.ceil_()
    _y.ceil_()
    z_int = _z.ceil()

    _x = torch.clip(_x, 1, resolution - 2)
    _y = torch.clip(_y, 1, resolution - 2)
    _z = torch.clip(_z, 1, depth - 2)

    return _x, _y


def points2pos(points, length, size_x=4, size_y=4, args=None):
    """
    :param points: [B, num_points, 3]
    :param size_x:
    :param size_y:
    :return:
        depth_recovered: [B, image_width, image_height]
    """

    direction = torch.tensor(args.pos_cor, requires_grad=False, device=points.device)
    direction = direction / torch.norm(direction)

    center = torch.mean(points, dim=1)
    relative_positions = points - center.unsqueeze(1)

    projections = torch.matmul(relative_positions, direction)

    min_projections = (
        torch.min(projections, dim=1)[0].unsqueeze(1).repeat([1, projections.size(1)])
    )
    max_projections = (
        torch.max(projections, dim=1)[0].unsqueeze(1).repeat([1, projections.size(1)])
    )
    normalized_projections = (projections - min_projections) / (
        max_projections - min_projections
    )

    final_projections = normalized_projections * length

    _i = torch.linspace(
        -size_x / 2,
        (size_x / 2) - 1,
        size_x,
        requires_grad=False,
        device=final_projections.device,
    )
    final_projections = final_projections + _i 
    # to prevent error
    final_projections = final_projections % length

    return final_projections


def points2pos_2d(points, image_height, image_width, size_x=4, size_y=4, args=None):
    """
    :param points: [B, num_points, 3]
    :param image_width:
    :param image_height:
    :param size_x:
    :param size_y:
    :return:
        depth_recovered: [B, image_width, image_height]
    """
    epsilon = torch.tensor([1e-12], requires_grad=False, device=points.device)
    coord_x = (points[:, :, 0] / (points[:, :, 2] + epsilon)) * (
        image_width / image_height
    ) 
    coord_y = points[:, :, 1] / (points[:, :, 2] + epsilon)  
    batch, total_points, _ = points.size()
    depth = points[:, :, 2] 
    _x = ((coord_x + 1) * image_height) / 2
    _y = ((coord_y + 1) * image_width) / 2

    assert size_x % 2 == 0 or size_x == 1
    assert size_y % 2 == 0 or size_y == 1
    _i = torch.linspace(
        -size_x / 2, (size_x / 2) - 1, size_x, requires_grad=False, device=depth.device
    )
    _j = torch.linspace(
        -size_y / 2, (size_y / 2) - 1, size_y, requires_grad=False, device=depth.device
    )
    extended_x = (
        _x.unsqueeze(2).repeat([1, 1, size_x]) + _i
    )  
    extended_y = (
        _y.unsqueeze(2).repeat([1, 1, size_y]) + _j
    ) 

    extended_x = extended_x.unsqueeze(3).repeat(
        [1, 1, 1, size_y]
    ) 
    extended_y = extended_y.unsqueeze(2).repeat(
        [1, 1, size_x, 1]
    )  

    # to prevent error
    extended_x = extended_x % image_height
    extended_y = extended_y % image_width

    return extended_x.squeeze(), extended_y.squeeze()


# source: https://discuss.pytorch.org/t/batched-index-select/9115/6
def batched_index_select(inp, dim, index):
    """
    input: B x * x ... x *
    dim: 0 < scalar
    index: B x M
    """
    views = [inp.shape[0]] + [1 if i != dim else -1 for i in range(1, len(inp.shape))]
    expanse = list(inp.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(inp, dim, index)


def point_fea_img_fea(point_fea, point_coo, h, w):
    """
    each point_coo is of the form (x*w + h). points not in the canvas are removed
    :param point_fea: [batch_size, num_points, feat_size]
    :param point_coo: [batch_size, num_points]
    :return:
    """
    assert len(point_fea.shape) == 3
    assert len(point_coo.shape) == 2
    assert point_fea.shape[0:2] == point_coo.shape

    coo_max = ((h - 1) * w) + (w - 1)
    mask_point_coo = (point_coo >= 0) * (point_coo <= coo_max)
    point_coo *= mask_point_coo.float()
    point_fea *= mask_point_coo.float().unsqueeze(-1)

    bs, _, fs = point_fea.shape
    point_coo = point_coo.unsqueeze(2).repeat([1, 1, fs])
    img_fea = torch.zeros([bs, h * w, fs], device=point_fea.device).scatter_add(
        1, point_coo.long(), point_fea
    )

    return img_fea


def distribute_img_fea_points(img_fea, point_coord):
    """
    :param img_fea: [B, C, H, W]
    :param point_coord: [B, num_points], each coordinate  is a scalar value given by (x * W) + y
    :return
        point_fea: [B, num_points, C], for points with coordinates outside the image, we return 0
    """
    B, C, H, W = list(img_fea.size())
    img_fea = img_fea.permute(0, 2, 3, 1).view([B, H * W, C])

    coord_max = ((H - 1) * W) + (W - 1)
    mask_point_coord = (point_coord >= 0) * (point_coord <= coord_max)
    mask_point_coord = mask_point_coord.float()
    point_coord = mask_point_coord * point_coord
    point_fea = batched_index_select(inp=img_fea, dim=1, index=point_coord.long())
    point_fea = mask_point_coord.unsqueeze(-1) * point_fea
    return point_fea


class PCViews:
    """For creating images from PC based on the view information. Faster as the
    repeated operations are done only once whie initialization.
    """

    def __init__(self):

        self.num_views = 6
        ##6view projection
        _views = np.asarray(
            [
                [[0 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
                [[1 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
                [[2 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
                [[3 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
                [[0, -np.pi / 2, np.pi / 2], [0, 0, TRANS]],
                [[0, np.pi / 2, np.pi / 2], [0, 0, TRANS]],
            ]
        )
        # #1view projection
        # _views = np.asarray(
        #     [
        #         [[0 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
        #         [[0 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
        #         [[0 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
        #         [[0 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
        #         [[0 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
        #         [[0 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
        #     ]
        # )

        angle = torch.tensor(_views[:, 0, :]).float().cuda()
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        self.translation = torch.tensor(_views[:, 1, :]).float().cuda()
        self.translation = self.translation.unsqueeze(1)

    def get_pos(self, points, args):
        """Get image based on the prespecified specifications.

        Args:
            points (torch.tensor): of size [B, _, 3]
        Returns:
            img (torch.tensor): of size [B * self.num_views, RESOLUTION,
                RESOLUTION]
        """

        b, _, _ = points.shape
        v = self.translation.shape[0]

        _points = self.point_transform(
            points=torch.repeat_interleave(points, v, dim=0),
            rot_mat=self.rot_mat.repeat(b, 1, 1),
            translation=self.translation.repeat(b, 1, 1),
        )

        pos_x = points2pos(points=_points, length=LENGTH, size_x=1, size_y=1, args=args)

        return pos_x, _points, self.rot_mat, self.translation

    def get_pos_2d(self, points, args=None):
        """Get image based on the prespecified specifications.

        Args:
            points (torch.tensor): of size [B, _, 3]
        Returns:
            img (torch.tensor): of size [B * self.num_views, RESOLUTION,
                RESOLUTION]
        """

        b, _, _ = points.shape
        v = self.translation.shape[0]

        _points = self.point_transform(
            points=torch.repeat_interleave(points, v, dim=0),
            rot_mat=self.rot_mat.repeat(b, 1, 1),
            translation=self.translation.repeat(b, 1, 1),
        )
        pos_x, pos_y = points2pos_2d(
            points=_points,
            image_height=RESOLUTION, 
            image_width=RESOLUTION,
            size_x=1,
            size_y=1,
            args=args,
        )
        return (
            pos_x,
            pos_y,
            _points,
        )  

    @staticmethod
    def point_transform(points, rot_mat, translation):  
        """
        :param points: [batch, num_points, 3]
        :param rot_mat: [batch, 3]
        :param translation: [batch, 1, 3]
        :return:
        """
        rot_mat = rot_mat.to(points.device)
        translation = translation.to(points.device)
        points = torch.matmul(points, rot_mat)
        points = points - translation
        return points


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == "_base_":
                with open(new_config["_base_"], "r") as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, "r") as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config
