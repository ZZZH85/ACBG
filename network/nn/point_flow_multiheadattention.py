import torch
import torch.nn as nn
import torch.nn.functional as F
from  .mynn import  Norm2d
import matplotlib.pyplot as plt
import math

class MultiheadAttention(nn.Module):
    # n_heads：多头注意力的数量
    # hid_dim：每个词输出的向量维度
    def __init__(self, hid_dim, n_heads, dropout):
        super(MultiheadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # 强制 hid_dim 必须整除 h
        assert hid_dim % n_heads == 0
        # 定义 W_q 矩阵
        self.w_q = nn.Linear(hid_dim, hid_dim)
        # 定义 W_k 矩阵
        self.w_k = nn.Linear(hid_dim, hid_dim)
        # 定义 W_v 矩阵
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):
        # K: [64,10,300], batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
        # V: [64,10,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        # Q: [64,12,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        bsz = query.shape[0]
        Q = self.w_q(query.transpose(1, 2))
        K = self.w_k(key.transpose(1, 2))
        V = self.w_v(value.transpose(1, 2))
        # 这里把 K Q V 矩阵拆分为多组注意力，变成了一个 4 维的矩阵
        # 最后一维就是是用 self.hid_dim // self.n_heads 来得到的，表示每组注意力的向量长度, 每个 head 的向量长度是：300/6=50
        # 64 表示 batch size，6 表示有 6组注意力，10 表示有 10 词，50 表示每组注意力的词的向量长度
        # K: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # V: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # Q: [64,12,300] 拆分多组注意力 -> [64,12,6,50] 转置得到 -> [64,6,12,50]
        # 转置是为了把注意力的数量 6 放到前面，把 10 和 50 放到后面，方便下面计算
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        # 第 1 步：Q 乘以 K的转置，除以scale
        # [64,6,12,50] * [64,6,50,10] = [64,6,12,10]
        # attention：[64,6,12,10]
        self.scale = self.scale.to(query.device)
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 把 mask 不为空，那么就把 mask 为 0 的位置的 attention 分数设置为 -1e10
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # 第 2 步：计算上一步结果的 softmax，再经过 dropout，得到 attention。
        # 注意，这里是对最后一维做 softmax，也就是在输入序列的维度做 softmax
        # attention: [64,6,12,10]
        attention = self.do(torch.softmax(attention, dim=-1))

        # 第三步，attention结果与V相乘，得到多头注意力的结果
        # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]
        # x: [64,6,12,50]
        x = torch.matmul(attention, V)

        # 因为 query 有 12 个词，所以把 12 放到前面，把 5 和 60 放到后面，方便下面拼接多组的结果
        # x: [64,6,12,50] 转置-> [64,12,6,50]
        x = x.permute(0, 2, 1, 3).contiguous()
        # 这里的矩阵转换就是：把多组注意力的结果拼接起来
        # 最终结果就是 [64,12,300]
        # x: [64,12,6,50] -> [64,12,300]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        x = x.transpose(1, 2)
        return x



# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, n_head):
#         super(MultiHeadAttention, self).__init__()
#         self.n_head = n_head
#         self.head_dim = d_model // n_head
#         self.d_model = d_model
#
#         self.w_qs = nn.Linear(d_model, d_model)
#         self.w_ks = nn.Linear(d_model, d_model)
#         self.w_vs = nn.Linear(d_model, d_model)
#
#         self.fc = nn.Linear(d_model, d_model)
#
#     def forward(self, high_f, low_f):
#         # input:(8,512,32)  512表示特征维度.
#         batch_size, M, _ = high_f.size()
#         _, N, _ = low_f.size()
#
#         high_f = high_f.transpose(1, 2)
#         low_f =  low_f.transpose(1, 2)
#
#         # batch_size, M, _ = high_f.size()
#         # _, N, _ = low_f.size()
#
#         # Linear projections of Q, K, and V   (bs,32,8,64)  8个head
#         q = self.w_qs(high_f).view(batch_size, -1, self.n_head, self.head_dim)
#         k = self.w_ks(low_f).view(batch_size, -1, self.n_head, self.head_dim)
#         v = self.w_vs(low_f).view(batch_size, -1, self.n_head, self.head_dim)
#
#         # Transpose to perform batch-wise attention
#         q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
#
#         # Calculate attention scores
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         attn_scores = F.softmax(attn_scores, dim=-1)
#
#         # Apply attention to values
#         output = torch.matmul(attn_scores, v)  #[8,8,32,64]
#
#         # Reshape and concatenate the outputs from all attention heads
#         # output = output.transpose(1, 2).contiguous().view(batch_size, M, -1)
#         output = output.contiguous().view(batch_size, M, -1)  #[8,512,32]
#         output = self.fc(output)
#
#         return output

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
        # self.multiheadattention = MultiHeadAttention(in_planes,8)
        self.multiheadattention  = MultiheadAttention(hid_dim=in_planes, n_heads=8, dropout=0.1)
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
        # high_edge_feat = torch.bmm(affinity, high_edge_feat.transpose(2, 1)).transpose(2, 1)#矩阵乘法 (bs,512,32)
        # fusion_edge_feat = high_edge_feat + low_edge_feat



        #multihead attention
        # fusion_edge_feat = self.multiheadattention(high_edge_feat,low_edge_feat)

        # attention = MultiheadAttention(hid_dim=300, n_heads=6, dropout=0.1)
        attention = self.multiheadattention (low_edge_feat, high_edge_feat, high_edge_feat)
        fusion_edge_feat = attention + low_edge_feat


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


