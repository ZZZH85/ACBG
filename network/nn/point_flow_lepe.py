import torch
import torch.nn as nn
import torch.nn.functional as F
from  .mynn import  Norm2d
import matplotlib.pyplot as plt

import numpy as np
# class LePEAttention(nn.Module):
#     def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
#                  qk_scale=None):
#         super().__init__()
#         self.dim = dim
#         self.dim_out = dim_out or dim
#         self.resolution = resolution
#         self.split_size = split_size  #split_size: 如果idx为0或1，表示图像分割的大小
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5
#         if idx == -1:
#             H_sp, W_sp = self.resolution, self.resolution
#         elif idx == 0:
#             H_sp, W_sp = self.resolution, self.split_size
#         elif idx == 1:
#             W_sp, H_sp = self.resolution, self.split_size
#         else:
#             print("ERROR MODE", idx)
#             exit(0)
#         self.H_sp = H_sp
#         self.W_sp = W_sp
#         stride = 1
#         self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
#
#         self.attn_drop = nn.Dropout(attn_drop)
#
#     # def im2cswin(self, x):
#     #     B, N, C = x.shape
#     #     H = W = int(np.sqrt(N))
#     #     x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
#     #     x = img2windows(x, self.H_sp, self.W_sp)
#     #     x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
#     #     return x
#
#     def get_lepe(self, x, func):
#         B, N, C = x.shape
#         H = W = int(np.sqrt(N))
#         x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
#
#         H_sp, W_sp = self.H_sp, self.W_sp
#         x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
#         x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'
#
#         lepe = func(x)  ### B', C, H', W'
#         lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
#
#         x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
#         return x, lepe
#
#     def forward(self, q,k,v):
#         """
#             x: B L C
#             """
#         # q, k, v = qkv[0], qkv[1], qkv[2]
#         # q, k, v = qkv[0], qkv[1], qkv[2]
#
#         v, lepe = self.get_lepe(v, self.get_v)
#
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
#         attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v) + lepe
#         x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, self.dim)  # B head N N @ B head N C
#
#     # def forward(self, qkv):
#     #     """
#     #     x: B L C
#     #     """
#     #     q, k, v = qkv[0], qkv[1], qkv[2]
#
#         # ### Img2Window
#         # H = W = self.resolution
#         # B, L, C = q.shape
#         # assert L == H * W, "flatten img_tokens has wrong size"
#
#         # q = self.im2cswin(q)
#         # k = self.im2cswin(k)
#
#         # v, lepe = self.get_lepe(v, self.get_v)
#         #
#         # q = q * self.scale
#         # attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
#         # attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
#         # attn = self.attn_drop(attn)
#         #
#         # x = (attn @ v) + lepe
#         # x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C
#
#         ### Window2Img
#         # x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C
#
#         return x


class LePEAttention(nn.Module):
    def __init__(self, dim, dim_out=None, attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.attn_drop = nn.Dropout(attn_drop)

    def get_lepe(self, x, func):
        B, N, C = x.shape
        x = x.unsqueeze(2)
        lepe = func(x)
        lepe = lepe.squeeze(2)
        lepe = lepe.reshape(-1, C, N).permute(0, 2, 1).contiguous()
        x = x.reshape(-1, C, N).permute(0, 2, 1).contiguous()
        return x, lepe

    # def get_lepe(self, x, func):
    #     B, L, C = x.shape
    #     H = int(np.sqrt(L))
    #     W = L // H
    #     x = x.reshape(B, C, H, W)
    #     lepe = func(x)
    #     lepe = lepe.reshape(-1, C, H * W).permute(0, 2, 1).contiguous()
    #     x = x.reshape(-1, C, H * W).permute(0, 2, 1).contiguous()
    #     return x, lepe

    def forward(self, q,k,v):
        """
        x: B L C
        """
        # q, k, v = qkv[0], qkv[1], qkv[2]
        # q, k, v = q.transpose(1, 2),k.transpose(1,2),v.transpose(1, 2)
        B, N, C = q.shape
        v, lepe = self.get_lepe(v, self.get_v)

        attn = (q @ k.transpose(-2, -1))
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, N, C)  # B L C

        return x




def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs,align_corners=False)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    return point_indices, point_coords


class PointMatcher(nn.Module):
    """
        Simple Point Matcher
    """
    def __init__(self, dim, kernel_size=3):
        super(PointMatcher, self).__init__()
        self.match_conv = nn.Conv2d(dim*2, 1, kernel_size, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_high, x_low = x
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)#low是语义信息low，分辨率是更高的，将高分辨率特征图两倍下采样
        certainty = self.match_conv(torch.cat([x_high, x_low], dim=1))#concate之后进过一层卷积压缩成1通道
        return self.sigmoid(certainty)   #一通道segmoid


class PointFlowModuleWithMaxAvgpool(nn.Module):
    def __init__(self, in_planes,  dim=64, maxpool_size=8, avgpool_size=8, matcher_kernel_size=3,
                  edge_points=64):
        super(PointFlowModuleWithMaxAvgpool, self).__init__()
        self.dim = dim
        self.point_matcher = PointMatcher(dim, matcher_kernel_size)
        self.down_h = nn.Conv2d(in_planes, dim, 1)
        self.down_l = nn.Conv2d(in_planes, dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.maxpool_size = maxpool_size
        self.avgpool_size = avgpool_size
        self.edge_points = edge_points
        self.max_pool = nn.AdaptiveMaxPool2d((maxpool_size, maxpool_size), return_indices=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((avgpool_size, avgpool_size))
        self.edge_final = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=3, padding=1, bias=False),
            Norm2d(in_planes),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes, out_channels=1, kernel_size=3, padding=1, bias=False)
        )
        self.lepe = LePEAttention(in_planes)
        # self.output_map = nn.Conv2d(dim, 1, 7, 1, 3)

    def forward(self, x):
        x_high, x_low = x # [1,512,16,16]  [1,512 ,32,32]
        stride_ratio = x_low.shape[2] / x_high.shape[2]   #对应比例
        x_high_embed = self.down_h(x_high)
        x_low_embed = self.down_l(x_low)  #两个特征统一成dim通道  [1,256,16,16]  [1,256,32,32]
        N, C, H, W = x_low.shape
        N_h, C_h, H_h, W_h = x_high.shape

        certainty_map = self.point_matcher([x_high_embed, x_low_embed])#saliency map Ml  # [8,1,16,16]
        avgpool_grid = self.avg_pool(certainty_map) #平均池化，池化成8×8大小 # [8,1,8,8]
        _, _, map_h, map_w = certainty_map.size()
        avgpool_grid = F.interpolate(avgpool_grid, size=(map_h, map_w), mode="bilinear", align_corners=True)#池化后8×8大小，差值成x_high大小   # [8,1,16,16]

        # edge part
        x_high_edge = x_high - x_high * avgpool_grid #边界 Flb  # [1,512,16,16]
        edge_pred = self.edge_final(x_high_edge)#卷积BN、RELU、卷积，执行完1通道，与x_high尺寸一样 # [8,1,16,16]

        # d = edge_pred[0][0].cpu().detach().numpy()
        # plt.title("edge")
        # plt.subplot(1, 2, 2)
        # plt.axis('off')
        # plt.imshow(d,cmap='gray')
        # plt.show()

        point_indices, point_coords = get_uncertain_point_coords_on_grid(edge_pred, num_points=self.edge_points)
        sample_x = point_indices % W_h * stride_ratio  #坐标
        sample_y = point_indices // W_h * stride_ratio
        low_edge_indices = sample_x + sample_y * W


        low_sample_x = point_indices % W_h   #坐标
        low_sample_y = point_indices // W_h

        # points = [(x, y) for x, y in zip(sample_x, sample_y)]

        # point_low_edge_indices = point_indices.unsqueeze(1).expand(-1, C, -1).long()

        low_edge_indices = low_edge_indices.unsqueeze(1).expand(-1, C, -1).long()

        #选点进行cross attention
        high_edge_feat = point_sample(x_high, point_coords)
        low_edge_feat = point_sample(x_low, point_coords)


        #cross attention
        # affinity_edge = torch.bmm(high_edge_feat.transpose(2, 1), low_edge_feat).transpose(2, 1)#32×32
        # affinity = self.softmax(affinity_edge)
        # high_edge_feat = torch.bmm(affinity, high_edge_feat.transpose(2, 1)).transpose(2, 1)#矩阵乘法


        #lepe
        lepe = self.lepe(low_edge_feat,high_edge_feat,high_edge_feat)

        fusion_edge_feat = lepe + low_edge_feat

        # residual part
        # maxpool_grid, maxpool_indices = self.max_pool(certainty_map)
        # maxpool_indices = maxpool_indices.expand(-1, C, -1, -1)
        # maxpool_grid = F.interpolate(maxpool_grid, size=(map_h, map_w), mode="bilinear", align_corners=True)
        # x_indices = maxpool_indices % W_h * stride_ratio
        # y_indices = maxpool_indices // W_h * stride_ratio
        # low_indices = x_indices + y_indices * W
        # low_indices = low_indices.long()
        # x_high = x_high + maxpool_grid*x_high
        # flattened_high = x_high.flatten(start_dim=2)#B,C,W*H
        # high_features = flattened_high.gather(dim=2, index=maxpool_indices.flatten(start_dim=2)).view_as(maxpool_indices)#2,C,8,8
        # flattened_low = x_low.flatten(start_dim=2)
        # low_features = flattened_low.gather(dim=2, index=low_indices.flatten(start_dim=2)).view_as(low_indices)
        # feat_n, feat_c, feat_h, feat_w = high_features.shape
        # high_features = high_features.view(feat_n, -1, feat_h*feat_w)
        # low_features = low_features.view(feat_n, -1, feat_h*feat_w)
        # affinity = torch.bmm(high_features.transpose(2, 1), low_features).transpose(2, 1)
        # affinity = self.softmax(affinity)  # b, n, n
        # high_features = torch.bmm(affinity, high_features.transpose(2, 1)).transpose(2, 1)
        # fusion_feature = high_features + low_features
        # mp_b, mp_c, mp_h, mp_w = low_indices.shape
        # low_indices = low_indices.view(mp_b, mp_c, -1)

        final_features = x_low.reshape(N, C, H*W).scatter(2, low_edge_indices, fusion_edge_feat).view(N, C, H, W)
        # final_features = final_features.scatter(2, low_indices, fusion_feature).view(N, C, H, W)
        # output_map=self.output_map(final_features)
        # return final_features, edge_pred,low_sample_x,low_sample_y
        return final_features, edge_pred


