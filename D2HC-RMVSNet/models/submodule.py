#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Hongwei Yi (hongweiyi@pku.edu.cn)

import torch.nn as nn
import torch
import numpy as np

def conv(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.ReLU(inplace=True)
    )

def convbn(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.BatchNorm2d(out_channels),
        #nn.SyncBatchNorm(out_channels),
        #nn.LeakyReLU(0.0,inplace=True)
        nn.ReLU(inplace=True)
    )

def convgnrelu(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True, group_channel=8):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.GroupNorm(int(max(1, out_channels / group_channel)), out_channels),
        nn.ReLU(inplace=True)
    )

# def conv3d(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
#     return nn.Sequential(
#         nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
#         nn.LeakyReLU(0.0,inplace=True)
#     )


def conv3dgn(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.GroupNorm(1, 1), 
        nn.LeakyReLU(0.0,inplace=True)
    )

def conv3d(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.BatchNorm3d(out_channels),
        nn.LeakyReLU(0.0,inplace=True)
    )

def resnet_block(in_channels,  kernel_size=3, dilation=[1,1], bias=True):
    return ResnetBlock(in_channels, kernel_size, dilation, bias=bias)

def resnet_block_bn(in_channels,  kernel_size=3, dilation=[1,1], bias=True):
    return ResnetBlockBn(in_channels, kernel_size, dilation, bias=bias)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias),
            nn.LeakyReLU(0.0, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
        )
    def forward(self, x):
        out = self.stem(x) + x
        return out

class ResnetBlockBn(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(ResnetBlockBn, self).__init__()
        self.stem = nn.Sequential(
            convbn(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], bias=bias),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
        )
    def forward(self, x):
        out = self.stem(x) + x
        return out

##### Define weightnet-3d
def volumegatelight(in_channels, kernel_size=3, dilation=[1,1], bias=True):
    return nn.Sequential(
        #MSDilateBlock3D(in_channels, kernel_size, dilation, bias),
        conv3d(in_channels, 1, kernel_size=1, stride=1, bias=bias),
        conv3d(1, 1, kernel_size=1, stride=1)
     )

def volumegatelightgn(in_channels, kernel_size=3, dilation=[1,1], bias=True):
    return nn.Sequential(
        #MSDilateBlock3D(in_channels, kernel_size, dilation, bias),
        conv3dgn(in_channels, 1, kernel_size=1, stride=1, bias=bias),
        conv3dgn(1, 1, kernel_size=1, stride=1)
     )

##### Define gatenet
def gatenetbn(bias=True):
    return nn.Sequential(
        convbn(32, 16, kernel_size=3, stride=1, pad=1, dilation=1, bias=bias),
        resnet_block_bn(16, kernel_size=1),
        nn.Conv2d(16, 1, kernel_size=1, padding=0),
        nn.Sigmoid()
    )

def pillarnetbn(bias=True):
    return nn.Sequential(
        nn.Linear(192, 32, bias=bias),
        nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE,inplace=True),
        nn.Linear(32, 2),
    )


class ResnetBlockGn(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias, group_channel=8):
        super(ResnetBlockGn, self).__init__()
        self.stem = nn.Sequential(
            convgnrelu(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], bias=bias, group_channel=group_channel), 
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
            nn.GroupNorm(int(max(1, in_channels / group_channel)), in_channels),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.stem(x) + x
        out = self.relu(out)
        return out

def resnet_block_gn(in_channels,  kernel_size=3, dilation=[1,1], bias=True, group_channel=8):
    return ResnetBlockGn(in_channels, kernel_size, dilation, bias=bias, group_channel=group_channel)

def gatenet(gn=True, in_channels=32, bias=True): # WORK 
    if gn:
        return nn.Sequential(
            #nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, dilation=1, padding=1, bias=bias), # in_channels=64
            #nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE,inplace=True),
            convgnrelu(in_channels, 4, kernel_size=3, stride=1, dilation=1, bias=bias), # 4: 10G,8.6G; 
            resnet_block_gn(4, kernel_size=1),
            nn.Conv2d(4, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, dilation=1, padding=1, bias=bias), # in_channels=64
            nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE,inplace=True),
            resnet_block(16, kernel_size=1),
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
def gatenet_m4(gn=True, in_channels=32, bias=True):
    if gn:
        return nn.Sequential(
            #nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, dilation=1, padding=1, bias=bias), # in_channels=64
            #nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE,inplace=True),
            convgnrelu(in_channels, 8, kernel_size=3, stride=1, dilation=1, bias=bias), # 4: 10G,8.6G; 
            resnet_block_gn(8, kernel_size=1),
            nn.Conv2d(8, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, dilation=1, padding=1, bias=bias), # in_channels=64
            nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE,inplace=True),
            resnet_block(16, kernel_size=1),
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

def pillarnet(in_channels=192, bias=True): # origin_pillarnet: 192, 96, 48, 24
    return nn.Sequential(
        nn.Linear(in_channels, 32, bias=bias),
        nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE,inplace=True),
        nn.Linear(32, 2),
    )
