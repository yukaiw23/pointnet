import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from typing import Optional
from torch import Tensor
from functools import partial


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

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
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
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
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint: number of points sampled in FPS 最远点采样
        radius: search radius in local region
        nsample: how many points in each local region
        
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]; sample points
    new_xyz = index_points(xyz, fps_idx) # use farthest-point-sample find index of sampling points
    
    #query ball finds 'nsample'个 points that are within a radius to 'new_xyz' from 'xyz'
    idx = query_ball_point(radius, nsample, xyz, new_xyz) 
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)


    #如果一个点上有新的特征，凭借其与新的特征；否则只返回旧特征
    if points is not None:
        grouped_points = index_points(points, idx)
        
        #特征数据+坐标数据
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


'''sample+group+pointnet
首先通过sample_and_group找到局部group，然后对group中每一个点做MLP， 最后进行局部MaxPooling
获得局部的 全局特征'''
class PointNetSetAbstraction(nn.Module):
    ''' npoint: points sampled in FPS
        nsample: how many points in each local region
    '''
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.res_mlps = nn.ModuleList()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.res_mlps.append(BottleNeck(out_channel,out_channel))
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points,mlp):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
       
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
       
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        
        #对局部group中每一个点做MLP,1x1的2d 卷积把每一个group作为一个通道，一个npoint个通道   
        # last_channel = in_channel
        # i=0
        # while i < len(mlp):
        #     model = BottleNeck(last_channel,mlp[i])
        #         # print(last_channel,i)
        #     if torch.cuda.is_available():
        #         model.cuda()
        #     new_points = model(new_points)
        #     last_channel = mlp[i]
        #     i=i+1
        # last_channel = in_channel
        for i, conv in enumerate(self.mlp_convs):
            
            bn = self.mlp_bns[i]
            bottleneck = self.res_mlps[i]
            if i == 0:
            	
            	new_points =  F.relu(bn(bottleneck(conv(new_points))))
            else:
            
            	new_points = F.relu(bn(conv(new_points)))
            last_channel = mlp[i]
            
            ''' Attach residual mlp behind certain layer of MLP in SA'''
            # if i <2:
            #     model = BottleNeck(last_channel,mlp[i])
            #     # print(last_channel,i)
            #     if torch.cuda.is_available():
            #         model.cuda()
            #         new_points = model(new_points)
            #         last_channel = mlp[i]
             
                
        '''After all MLP in SA,  attach bottleneck residual MLP here: 1X1 -> 3X3 -> 1x1 ''' 
        #model = BottleNeck(last_channel,last_channel)
        
        #if torch.cuda.is_available():
          #  model.cuda()
        #new_points = model(new_points)
        ''' '''
        
        #最后做 maxpoling，得到局部的全局特征
        new_points = torch.max(new_points, 2)[0]
        # print('points shape:', new_points.shape)
        new_xyz = new_xyz.permute(0, 2, 1)
        
        return new_xyz, new_points

'''MSG:Multi-Scale Group, various radius,
对不于同的半径做ball query,并将各个特种拼接'''
class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.res_mlps = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bottleneck = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bottleneck.append(BottleNeck(out_channel, out_channel))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
            self.res_mlps.append(bottleneck)
    def forward(self, xyz, points,mlp_list):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # 
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
            # print('xyz shape:' ,xyz.shape, 'points shape:', points.shape)
        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                #拼接特征数据以及坐标数据
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            # print('grouped_points shape:',grouped_points.shape)
            #pointnet,1x1 2d conv
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bottle = self.res_mlps[i][j]
                bn = self.bn_blocks[i][j]
               
                if j ==1:
                     grouped_points =  F.relu(bn(bottle(conv(grouped_points))))
                else:
                     grouped_points = F.relu(bn(conv(grouped_points)))
                
                # last_channel = mlp_list[i][j]
             
                ''' BottleNeck Residual block here'''
            # model = BottleNeck(last_channel,last_channel)      
            # if torch.cuda.is_available():
            #     model.cuda()
            # grouped_points = model(grouped_points)
                
                
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            # print('radius:',radius, 'newpoint shape:',new_points.shape)
            new_points_list.append(new_points)
 
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        # print('points shape MSG:', new_points_concat.shape)
        return new_xyz, new_points_concat

'''Residual MLP '''
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        X = torch.unsqueeze(X,3)
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        Y = torch.squeeze(F.relu(Y),3)
        return Y

'''segmentation '''
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
        **kwargs
    ):

        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            norm(out_features),
            act(),
        )

   
        
class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut
        
    def forward(self, x):
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        x += res
        return x
Conv1X1BnReLU = partial(ConvNormAct, kernel_size=1)
Conv3X3BnReLU = partial(ConvNormAct, kernel_size=3)  

class BottleNeck(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, coeff: int = 2):
        reduced_features = out_features // coeff
        # reduced_features = in_features * coeff ## Inverted Redisual
        super().__init__(
            nn.Sequential(
                ResidualAdd(
                    nn.Sequential(
                        # wide -> narrow
                        #Conv1X1BnReLU(in_features, reduced_features),
                        # narrow -> narrow
                        #Conv3X3BnReLU(reduced_features, reduced_features),
                        # narrow -> wide
                        #Conv1X1BnReLU(reduced_features, out_features, act=nn.Identity),
                        
                        Conv3X3BnReLU(in_features, in_features),
                        
                        Conv1X1BnReLU(in_features, reduced_features),
                        Conv1X1BnReLU(reduced_features, out_features),
                        
                        Conv3X3BnReLU(out_features, out_features,act=nn.Identity),
                    ),
                    shortcut=Conv1X1BnReLU(in_features, out_features)
                    if in_features != out_features
                    else None,
                ),
                #nn.ReLU(),
            )
        )
    
