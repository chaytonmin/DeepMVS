import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
import sys
from copy import deepcopy

# Multi-scale feature extractor && Coarse To Fine Regression Module 

class FeatureNetHigh(nn.Module): #Original Paper Setting
    def __init__(self):
        super(FeatureNetHigh, self).__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        
        self.conv7 = ConvBnReLU( 32, 32, 5, 2, 2)
        self.conv8 = ConvBnReLU(32, 32, 3, 1, 1)
        
        self.conv9 = ConvBnReLU(32, 64, 5, 2, 2)
        self.conv10 = ConvBnReLU(64, 64, 3, 1, 1)
        
        self.conv11 = ConvBnReLU(64, 64, 5, 2, 2)
        self.conv12 = ConvBnReLU(64, 64, 3, 1, 1)
        
        self.feature1 = nn.Conv2d(32, 32, 3, 1, 1)
        
        self.feature2 = nn.Conv2d(32, 32, 3, 1, 1)
        
        self.feature3 = nn.Conv2d(64, 64, 3, 1, 1)
        
        self.feature4 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.conv6(self.conv5(x))
        feature1 = self.feature1(x)
        x = self.conv8(self.conv7(x))
        feature2 = self.feature2(x)
        x = self.conv10(self.conv9(x))
        feature3 = self.feature3(x)
        x = self.conv12(self.conv11(x))
        feature4 = self.feature4(x)
        return [feature1, feature2, feature3, feature4]


class FeatureNetHighGN(nn.Module): #Original Paper Setting
    def __init__(self):
        super(FeatureNetHighGN, self).__init__()
        self.inplanes = 32

        self.conv0 = ConvGnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvGnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvGnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvGnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvGnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvGnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvGnReLU(32, 32, 3, 1, 1)
        
        self.conv7 = ConvGnReLU( 32, 32, 5, 2, 2)
        self.conv8 = ConvGnReLU(32, 32, 3, 1, 1)
        
        self.conv9 = ConvGnReLU(32, 64, 5, 2, 2)
        self.conv10 = ConvGnReLU(64, 64, 3, 1, 1)
        
        self.conv11 = ConvGnReLU(64, 64, 5, 2, 2)
        self.conv12 = ConvGnReLU(64, 64, 3, 1, 1)
        
        self.feature1 = nn.Conv2d(32, 32, 3, 1, 1)
        
        self.feature2 = nn.Conv2d(32, 32, 3, 1, 1)
        
        self.feature3 = nn.Conv2d(64, 64, 3, 1, 1)
        
        self.feature4 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.conv6(self.conv5(x))
        feature1 = self.feature1(x)
        x = self.conv8(self.conv7(x))
        feature2 = self.feature2(x)
        x = self.conv10(self.conv9(x))
        feature3 = self.feature3(x)
        x = self.conv12(self.conv11(x))
        feature4 = self.feature4(x)
        return [feature1, feature2, feature3, feature4]

class RegNetUS0_Coarse2Fine(nn.Module):
    def __init__(self, origin_size=False, dp_ratio=0.0, image_scale=0.25):
        super(RegNetUS0_Coarse2Fine, self).__init__()
        self.origin_size = origin_size
        self.image_scale = image_scale

        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(32, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(128, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(97, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(49, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))
        
        self.prob1 = nn.Conv3d(41, 1, 1,bias=False)
        self.dropout1 = nn.Dropout3d(p=dp_ratio)
        self.prob2 = nn.Conv3d(49, 1, 1,bias=False)
        self.dropout2 = nn.Dropout3d(p=dp_ratio)
        self.prob3 = nn.Conv3d(97, 1, 1,bias=False)
        self.dropout3 = nn.Dropout3d(p=dp_ratio)
        self.prob4 = nn.Conv3d(128, 1, 1,bias=False)
        self.dropout4 = nn.Dropout3d(p=dp_ratio)
        #add Drop out
        
    def forward(self, x_list):
        x1, x2, x3, x4 = x_list # 32*192, 32*96, 64*48, 64*24
        input_shape = x1.shape

        conv0 = self.conv0(x1)
        conv1 = self.conv1(x1)
        conv3 = self.conv3(conv1)
        conv5 = self.conv5(conv3)

        x = torch.cat([self.conv6(conv5), x4], 1)
        prob4 = self.dropout4(self.prob4(x))
        #prob4 = self.prob4(x)
        x = self.conv7(x) + self.conv4(conv3)
        x = torch.cat([x, x3, F.interpolate(prob4, scale_factor=2, mode='trilinear', align_corners=True)], 1)
        prob3 = self.dropout3(self.prob3(x))
        #prob3 = self.prob3(x)
        x = self.conv9(x) + self.conv2(conv1)
        x = torch.cat([x, x2, F.interpolate(prob3, scale_factor=2, mode='trilinear', align_corners=True)], 1)
        prob2 = self.dropout2(self.prob2(x))
        #prob2 = self.prob2(x)
        x = self.conv11(x) + conv0
        x = torch.cat([x, x1, F.interpolate(prob2, scale_factor=2, mode='trilinear', align_corners=True)], 1)

        if self.origin_size and self.image_scale == 0.50:
            x = F.interpolate(x, size=(input_shape[2], input_shape[3]*2, input_shape[4]*2), mode='trilinear', align_corners=True)
        prob1 = self.dropout1(self.prob1(x))
        #prob1 = self.prob1(x) # without dropout
        # if self.origin_size:
        #     x = F.interpolate(x, size=(input_shape[2], input_shape[3]*4, input_shape[4]*4), mode='trilinear', align_corners=True)
        return [prob1, prob2, prob3, prob4]


class RegNetUS0_Coarse2FineGN(nn.Module):
    def __init__(self, origin_size=False, dp_ratio=0.0, image_scale=0.25):
        super(RegNetUS0_Coarse2FineGN, self).__init__()
        self.origin_size = origin_size
        self.image_scale = image_scale

        self.conv0 = ConvGnReLU3D(32, 8)

        self.conv1 = ConvGnReLU3D(32, 16, stride=2)
        self.conv2 = ConvGnReLU3D(16, 16)

        self.conv3 = ConvGnReLU3D(16, 32, stride=2)
        self.conv4 = ConvGnReLU3D(32, 32)

        self.conv5 = ConvGnReLU3D(32, 64, stride=2)
        self.conv6 = ConvGnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(128, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            #nn.BatchNorm3d(32),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(97, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.GroupNorm(2, 16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(49, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.GroupNorm(1, 8),
            nn.ReLU(inplace=True))
        
        self.prob1 = nn.Conv3d(41, 1, 1,bias=False)
        self.dropout1 = nn.Dropout3d(p=dp_ratio)
        self.prob2 = nn.Conv3d(49, 1, 1,bias=False)
        self.dropout2 = nn.Dropout3d(p=dp_ratio)
        self.prob3 = nn.Conv3d(97, 1, 1,bias=False)
        self.dropout3 = nn.Dropout3d(p=dp_ratio)
        self.prob4 = nn.Conv3d(128, 1, 1,bias=False)
        self.dropout4 = nn.Dropout3d(p=dp_ratio)
        #add Drop out
        

    def forward(self, x_list):
        x1, x2, x3, x4 = x_list # 32*192, 32*96, 64*48, 64*24
        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        input_shape = x1.shape

        conv0 = self.conv0(x1)
        conv1 = self.conv1(x1)
        conv3 = self.conv3(conv1)
        conv5 = self.conv5(conv3)

        x = torch.cat([self.conv6(conv5), x4], 1)
        prob4 = self.dropout4(self.prob4(x))
        #prob4 = self.prob4(x)
        x = self.conv7(x) + self.conv4(conv3)
        x = torch.cat([x, x3, F.interpolate(prob4, scale_factor=2, mode='trilinear', align_corners=True)], 1)
        prob3 = self.dropout3(self.prob3(x))
        #prob3 = self.prob3(x)
        x = self.conv9(x) + self.conv2(conv1)
        x = torch.cat([x, x2, F.interpolate(prob3, scale_factor=2, mode='trilinear', align_corners=True)], 1)
        prob2 = self.dropout2(self.prob2(x))
        #prob2 = self.prob2(x)
        x = self.conv11(x) + conv0
        x = torch.cat([x, x1, F.interpolate(prob2, scale_factor=2, mode='trilinear', align_corners=True)], 1)

        if self.origin_size and self.image_scale == 0.50:
            x = F.interpolate(x, size=(input_shape[2], input_shape[3]*2, input_shape[4]*2), mode='trilinear', align_corners=True)
        prob1 = self.dropout1(self.prob1(x))
        #prob1 = self.prob1(x) # without dropout
        # if self.origin_size:
        #     x = F.interpolate(x, size=(input_shape[2], input_shape[3]*4, input_shape[4]*4), mode='trilinear', align_corners=True)
        return [prob1, prob2, prob3, prob4]

    
