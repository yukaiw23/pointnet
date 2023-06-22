#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: wyk
"""

import torch
from torch import nn
from torch.nn import functional as F



class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=True, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=2, padding=1, stride=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=2, padding=1,stride=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=1)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        X = torch.unsqueeze(X,3)
        Y = F.relu(self.bn1(self.conv1(X)))
        # print(Y.shape)
        Y = self.bn2(self.conv2(Y))
        # print(Y.shape)
        if self.conv3:
            X = self.conv3(X)
        # Y += X
        Y = torch.squeeze(F.relu(Y),3)
        # print(Y.shape)
        return Y
    
    
blk1 = Residual(320,320, use_1x1conv=True)   
# #  4: xxx; 128: in channel; 32: nsample; 
X1 = torch.rand(4, 320, 512)
Y1 = blk1(X1)
# print(Y1.shape)

# 4 320 512
# 4 640 128
# 4 1024 1

# blk2 = Residual(256,256, use_1x1conv=True) 

# X2 = torch.rand(4,256,128,1)
# Y2 = blk2(X2)
# print(Y2.shape)

# blk3 = Residual(1024,1024, use_1x1conv=True)
# X3 = torch.rand(4,1024,1,1)
# Y3 = blk3(X3)
# print(Y3.shape)

from typing import Optional
from torch import Tensor
from functools import partial

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


Conv1X1BnReLU = partial(ConvNormAct, kernel_size=1)
Conv3X3BnReLU = partial(ConvNormAct, kernel_size=3)


class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut
        
    def forward(self, x) :
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        x += res
        return x
    
class BottleNeck(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, reduction: int = 4):
        reduced_features = out_features // reduction
        super().__init__(
            nn.Sequential(
                ResidualAdd(
                    nn.Sequential(
                        # wide -> narrow
                        Conv1X1BnReLU(in_features, reduced_features),
                        # narrow -> narrow
                        Conv3X3BnReLU(reduced_features, reduced_features),
                        # narrow -> wide
                        Conv1X1BnReLU(reduced_features, out_features, act=nn.Identity),
                    ),
                    shortcut=Conv1X1BnReLU(in_features, out_features)
                    if in_features != out_features
                    else None,
                ),
                nn.ReLU(),
            )
        )
    
model = BottleNeck(256,256)
        
if torch.cuda.is_available():
    model.cuda()

X=torch.randn(4,256,128,1).cuda()
out = model(X)
