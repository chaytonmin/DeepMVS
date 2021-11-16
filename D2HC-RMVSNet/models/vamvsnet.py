import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
import sys
from copy import deepcopy

from .submodule import volumegatelight, volumegatelightgn
# More scale feature map submodule
from .vamvsnet_high_submodule import *

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x

class UNetDS2GN(nn.Module): 
    def __init__(self):
        super(UNetDS2GN, self).__init__()
        base_filter = 8
        self.conv1_0 = ConvGnReLU(3              , base_filter * 2, stride=2)
        self.conv2_0 = ConvGnReLU(base_filter * 2, base_filter * 4, stride=2)
        self.conv3_0 = ConvGnReLU(base_filter * 4, base_filter * 8, stride=2)
        self.conv4_0 = ConvGnReLU(base_filter * 8, base_filter * 16, stride=2)

        self.conv0_1 = ConvGnReLU(3              , base_filter * 1)
        self.conv0_2 = ConvGnReLU(base_filter * 1, base_filter * 1)

        self.conv1_1 = ConvGnReLU(base_filter * 2, base_filter * 2)
        self.conv1_2 = ConvGnReLU(base_filter * 2, base_filter * 2)

        self.conv2_1 = ConvGnReLU(base_filter * 4, base_filter * 4)
        self.conv2_2 = ConvGnReLU(base_filter * 4, base_filter * 4)

        self.conv3_1 = ConvGnReLU(base_filter * 8, base_filter * 8)
        self.conv3_2 = ConvGnReLU(base_filter * 8, base_filter * 8)

        self.conv4_1 = ConvGnReLU(base_filter * 16, base_filter * 16)
        self.conv4_2 = ConvGnReLU(base_filter * 16, base_filter * 16)
        self.conv5_0 = deConvGnReLU(base_filter * 16, base_filter * 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)

        self.conv5_1 = ConvGnReLU(base_filter * 16, base_filter * 8)
        self.conv5_2 = ConvGnReLU(base_filter * 8, base_filter * 8)
        self.conv6_0 = deConvGnReLU(base_filter * 8, base_filter * 4, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)

        self.conv6_1 = ConvGnReLU(base_filter * 8, base_filter * 4)
        self.conv6_2 = ConvGnReLU(base_filter * 4, base_filter * 4)
        self.conv7_0 = deConvGnReLU(base_filter * 4, base_filter * 2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)

        self.conv7_1 = ConvGnReLU(base_filter * 4, base_filter * 2)
        self.conv7_2 = ConvGnReLU(base_filter * 2, base_filter * 2)
        self.conv8_0 = deConvGnReLU(base_filter * 2, base_filter * 1, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)

        self.conv8_1 = ConvGnReLU(base_filter * 2, base_filter * 1)
        self.conv8_2 = ConvGnReLU(base_filter * 1, base_filter * 1) # end of UNet
        self.conv9_0 = ConvGnReLU(base_filter * 1, base_filter * 2, 5, 2, 2)
        self.conv9_1 = ConvGnReLU(base_filter * 2, base_filter * 2)
        self.conv9_2 = ConvGnReLU(base_filter * 2, base_filter * 2)
        self.conv10_0 = ConvGnReLU(base_filter * 2, base_filter * 4, 5, 2, 2)
        self.conv10_1 = ConvGnReLU(base_filter * 4, base_filter * 4)
        self.conv10_2 = nn.Conv2d(base_filter * 4, base_filter * 4, 1, bias=False)

    def forward(self, x):
        conv1_0 = self.conv1_0(x)
        conv2_0 = self.conv2_0(conv1_0)
        conv3_0 = self.conv3_0(conv2_0)
        conv4_0 = self.conv4_0(conv3_0)

        conv0_2 = self.conv0_2(self.conv0_1(x))

        conv1_2 = self.conv1_2(self.conv1_1(conv1_0))

        conv2_2 = self.conv2_2(self.conv2_1(conv2_0))

        conv3_2 = self.conv3_2(self.conv3_1(conv3_0))

        conv4_2 = self.conv4_2(self.conv4_1(conv4_0))
        conv5_0 = self.conv5_0(conv4_2)

        x = torch.cat((conv5_0, conv3_2), dim=1)
        conv5_2 = self.conv5_2(self.conv5_1(x))
        conv6_0 = self.conv6_0(conv5_2)

        x = torch.cat((conv6_0, conv2_2), dim=1)
        conv6_2 = self.conv6_2(self.conv6_1(x))
        conv7_0 = self.conv7_0(conv6_2)
        
        x = torch.cat((conv7_0, conv1_2), dim=1)
        conv7_2 = self.conv7_2(self.conv7_1(x))
        conv8_0 = self.conv8_0(conv7_2)

        x = torch.cat((conv8_0, conv0_2), dim=1)
        conv8_2 = self.conv8_2(self.conv8_1(x))
        conv9_2 = self.conv9_2(self.conv9_1(self.conv9_0(conv8_2)))
        conv10_2 = self.conv10_2(self.conv10_1(self.conv10_0(conv9_2)))

        return conv10_2

class UNetDS2BN(nn.Module):
    def __init__(self):
        super(UNetDS2BN, self).__init__()
        base_filter = 8
        # in_channels, out_channels, kernel_size=3, stride=1, pad=1
        self.conv1_0 = ConvBnReLU(3              , base_filter * 2, stride=2)
        self.conv2_0 = ConvBnReLU(base_filter * 2, base_filter * 4, stride=2)
        self.conv3_0 = ConvBnReLU(base_filter * 4, base_filter * 8, stride=2)
        self.conv4_0 = ConvBnReLU(base_filter * 8, base_filter * 16, stride=2)

        self.conv0_1 = ConvBnReLU(3              , base_filter * 1)
        self.conv0_2 = ConvBnReLU(base_filter * 1, base_filter * 1)

        self.conv1_1 = ConvBnReLU(base_filter * 2, base_filter * 2)
        self.conv1_2 = ConvBnReLU(base_filter * 2, base_filter * 2)

        self.conv2_1 = ConvBnReLU(base_filter * 4, base_filter * 4)
        self.conv2_2 = ConvBnReLU(base_filter * 4, base_filter * 4)

        self.conv3_1 = ConvBnReLU(base_filter * 8, base_filter * 8)
        self.conv3_2 = ConvBnReLU(base_filter * 8, base_filter * 8)

        self.conv4_1 = ConvBnReLU(base_filter * 16, base_filter * 16)
        self.conv4_2 = ConvBnReLU(base_filter * 16, base_filter * 16)
        self.conv5_0 = deConvBnReLU(base_filter * 16, base_filter * 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)

        self.conv5_1 = ConvBnReLU(base_filter * 16, base_filter * 8)
        self.conv5_2 = ConvBnReLU(base_filter * 8, base_filter * 8)
        self.conv6_0 = deConvBnReLU(base_filter * 8, base_filter * 4, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)

        self.conv6_1 = ConvBnReLU(base_filter * 8, base_filter * 4)
        self.conv6_2 = ConvBnReLU(base_filter * 4, base_filter * 4)
        self.conv7_0 = deConvBnReLU(base_filter * 4, base_filter * 2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)

        self.conv7_1 = ConvBnReLU(base_filter * 4, base_filter * 2)
        self.conv7_2 = ConvBnReLU(base_filter * 2, base_filter * 2)
        self.conv8_0 = deConvBnReLU(base_filter * 2, base_filter * 1, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)

        self.conv8_1 = ConvBnReLU(base_filter * 2, base_filter * 1)
        self.conv8_2 = ConvBnReLU(base_filter * 1, base_filter * 1) # end of UNet
        self.conv9_0 = ConvBnReLU(base_filter * 1, base_filter * 2, 5, 2, 2)
        self.conv9_1 = ConvBnReLU(base_filter * 2, base_filter * 2)
        self.conv9_2 = ConvBnReLU(base_filter * 2, base_filter * 2)
        self.conv10_0 = ConvBnReLU(base_filter * 2, base_filter * 4, 5, 2, 2)
        self.conv10_1 = ConvBnReLU(base_filter * 4, base_filter * 4)
        self.conv10_2 = nn.Conv2d(base_filter * 4, base_filter * 4, 1, bias=False)

    def forward(self, x):
        conv1_0 = self.conv1_0(x)
        conv2_0 = self.conv2_0(conv1_0)
        conv3_0 = self.conv3_0(conv2_0)
        conv4_0 = self.conv4_0(conv3_0)

        conv0_2 = self.conv0_2(self.conv0_1(x))

        conv1_2 = self.conv1_2(self.conv1_1(conv1_0))

        conv2_2 = self.conv2_2(self.conv2_1(conv2_0))

        conv3_2 = self.conv3_2(self.conv3_1(conv3_0))

        conv4_2 = self.conv4_2(self.conv4_1(conv4_0))
        conv5_0 = self.conv5_0(conv4_2)

        x = torch.cat((conv5_0, conv3_2), dim=1)
        conv5_2 = self.conv5_2(self.conv5_1(x))
        conv6_0 = self.conv6_0(conv5_2)

        x = torch.cat((conv6_0, conv2_2), dim=1)
        conv6_2 = self.conv6_2(self.conv6_1(x))
        conv7_0 = self.conv7_0(conv6_2)

        x = torch.cat((conv7_0, conv1_2), dim=1)
        conv7_2 = self.conv7_2(self.conv7_1(x))
        conv8_0 = self.conv8_0(conv7_2)

        x = torch.cat((conv8_0, conv0_2), dim=1)
        conv8_2 = self.conv8_2(self.conv8_1(x))
        conv9_2 = self.conv9_2(self.conv9_1(self.conv9_0(conv8_2)))
        conv10_2 = self.conv10_2(self.conv10_1(self.conv10_0(conv9_2)))

        return conv10_2


class RegNetUS0(nn.Module):
    def __init__(self, origin_size=False):
        super(RegNetUS0, self).__init__()
        self.origin_size = origin_size

        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(32, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))
        
        self.prob = nn.Conv3d(8, 1, 1,bias=False)

    def forward(self, x):
        input_shape = x.shape

        conv0 = self.conv0(x)
        conv1 = self.conv1(x)
        conv3 = self.conv3(conv1)
        conv5 = self.conv5(conv3)

        x = self.conv7(self.conv6(conv5)) + self.conv4(conv3)
        x = self.conv9(x) + self.conv2(conv1)
        x = self.conv11(x) + conv0
        
        if self.origin_size:
            x = F.interpolate(x, size=(input_shape[2], input_shape[3]*4, input_shape[4]*4), mode='trilinear', align_corners=True)
        x = self.prob(x)
        return x

class RegNetUS0GN(nn.Module):
    def __init__(self, origin_size=False):
        super(RegNetUS0GN, self).__init__()
        self.origin_size = origin_size

        self.conv0 = ConvGnReLU3D(32, 8)

        self.conv1 = ConvGnReLU3D(32, 16, stride=2)
        self.conv2 = ConvGnReLU3D(16, 16)

        self.conv3 = ConvGnReLU3D(16, 32, stride=2)
        self.conv4 = ConvGnReLU3D(32, 32)

        self.conv5 = ConvGnReLU3D(32, 64, stride=2)
        self.conv6 = ConvGnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            #nn.BatchNorm3d(32),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            #nn.BatchNorm3d(16),
            nn.GroupNorm(2, 16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            # nn.BatchNorm3d(8),
            nn.GroupNorm(1, 8),
            nn.ReLU(inplace=True))
        
        self.prob = nn.Conv3d(8, 1, 1,bias=False)

    def forward(self, x):
        input_shape = x.shape

        conv0 = self.conv0(x)
        conv1 = self.conv1(x)
        conv3 = self.conv3(conv1)
        conv5 = self.conv5(conv3)

        x = self.conv7(self.conv6(conv5)) + self.conv4(conv3)
        x = self.conv9(x) + self.conv2(conv1)
        x = self.conv11(x) + conv0
        
        if self.origin_size:
            #x = F.interpolate(x, size=(input_shape[2], input_shape[3]*4, input_shape[4]*4), mode='trilinear', align_corners=True)
            x = F.interpolate(x, size=(input_shape[2], input_shape[3]*2, input_shape[4]*2), mode='trilinear', align_corners=True)

        x = self.prob(x)
        return x

class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x

class MVSNet(nn.Module):
    def __init__(self, refine=True, fea_net='FeatureNet', cost_net='CostRegNet', refine_net='RefineNet',
                 origin_size=False, cost_aggregation=0, dp_ratio=0.0, image_scale=0.25):
        super(MVSNet, self).__init__()
        self.refine = refine
        
        self.origin_size = origin_size
        self.cost_aggregation = cost_aggregation
        self.fea_net = fea_net
        self.cost_net = cost_net
        self.refine_net = refine_net
        self.dp_ratio = dp_ratio
        self.image_scale = image_scale
        print('MVSNet model , refine: {}, refine_net: {},  fea_net: {}, cost_net: {}, origin_size: {}, image_scale: {}'.format(self.refine, 
                                    refine_net, fea_net, cost_net, self.origin_size, self.image_scale))

        print('cost aggregation: ', self.cost_aggregation)

        if fea_net == 'FeatureNet':
            self.feature = FeatureNet()
        elif fea_net == 'UNetDS2GN':
            self.feature = UNetDS2GN()
        elif fea_net == 'FeatureNetHigh': 
            self.feature = FeatureNetHigh()
        elif fea_net == 'FeatureNetHighGN':
            self.feature = FeatureNetHighGN()
            
        if cost_net == 'CostRegNet':
            self.cost_regularization = CostRegNet()
        elif cost_net == 'RegNetUS0GN':
            self.cost_regularization = RegNetUS0GN(self.origin_size)
        elif cost_net == 'RegNetUS0_Coarse2Fine':
            self.cost_regularization = RegNetUS0_Coarse2Fine(self.origin_size, self.dp_ratio, self.image_scale)
        elif cost_net == 'RegNetUS0_Coarse2FineGN':
            self.cost_regularization = RegNetUS0_Coarse2FineGN(self.origin_size, self.dp_ratio, self.image_scale)

        
        if self.cost_aggregation == 91: #input 3D cost volume -> 3D Reweight map
            self.volumegate = volumegatelight(32, kernel_size=3, dilation=[1,3,5,7], bias=True)
            #self.volumegate = volumegatelightgn(32, kernel_size=3, dilation=[1,3,5,7], bias=True)
        elif self.cost_aggregation == 95: #input 3D cost volume -> 3D Reweight map
            self.volumegate = nn.ModuleList([
                volumegatelight(32, kernel_size=3, dilation=[1,3,5,7], bias=True),
                volumegatelight(32, kernel_size=3, dilation=[1,3,5,7], bias=True),
                volumegatelight(64, kernel_size=3, dilation=[1,3,5,7], bias=True),
                volumegatelight(64, kernel_size=3, dilation=[1,3,5,7], bias=True)]) 
        
    def forward(self, imgs, proj_matrices, depth_values):
        
        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)
        if ('High' in self.fea_net) and ('Coarse2Fine' in self.cost_net) :
            # step 1. feature extraction
            # in: images; out: 32-channel feature maps
            features = [self.feature(img) for img in imgs] #ref_num * 3
            ref_features, src_features_o = features[0], features[1:]
            ref_proj_o, src_projs_o = proj_matrices[0], proj_matrices[1:]
            # proj_mat[:3, :4] = proj_mat[:3, :4]  * sample_scale
            
            sample_scale = [1, 0.5, 0.25, 0.125]

            volume_variances = []
            new_depth_values_list = []
            src_features_o_transpose = list(map(list, zip(*(list(src_features_o))))) # N-1 * 4 -> 4 * N-1
            ii = 0
            for ref_feature, src_features, one_scale in zip(ref_features, src_features_o_transpose, sample_scale):
                ref_proj = ref_proj_o.clone()
                ref_proj[:, :3, :4] = ref_proj[:, :3, :4] * one_scale
                src_projs = deepcopy(src_projs_o) # deep copy
                for ti in range(len(src_projs)):
                    src_projs[ti][:, :3, :4] = src_projs[ti][:, :3, :4] * one_scale
                # step 2. differentiable homograph, build cost volume
                new_num_depth = int(num_depth * one_scale)
                new_index = torch.arange(0, 192, int(1/one_scale)).cuda()
                new_depth_values=depth_values.index_select(1, new_index)
                new_depth_values_list.append(new_depth_values)
                if self.cost_aggregation == 0:
                    ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, new_num_depth, 1, 1)
                    volume_sum = ref_volume
                    volume_sq_sum = ref_volume ** 2
                    del ref_volume
                    for src_fea, src_proj in zip(src_features, src_projs):
                        # warpped features
                        warped_volume = homo_warping(src_fea, src_proj, ref_proj, new_depth_values)
                        if self.training:
                            volume_sum = volume_sum + warped_volume
                            volume_sq_sum = volume_sq_sum + warped_volume ** 2
                        else:
                            # TODO: this is only a temporal solution to save memory, better way?
                            volume_sum += warped_volume
                            volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
                        del warped_volume
                    # aggregate multiple feature volumes by variance
                    volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
                elif self.cost_aggregation == 95: # More efficient, do not need to save warp_volumes
                    ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, new_num_depth, 1, 1)
                    #warp_volumes = []
                    warp_volumes = None
                    for src_fea, src_proj in zip(src_features, src_projs):
                        # warpped features
                        warped_volume = homo_warping(src_fea, src_proj, ref_proj, new_depth_values)
                        warped_volume = (warped_volume - ref_volume).pow_(2) #B,C,D,H,W
                        B,C,D,H,W = warped_volume.shape

                        reweight = self.volumegate[ii](warped_volume) #B, 1, D, H, W
                        if warp_volumes is None:
                            warp_volumes = (reweight + 1) * warped_volume
                        else:
                            warp_volumes += (reweight + 1) * warped_volume
                    aggregate_volume = warp_volumes / len(src_features)
                    volume_variance = aggregate_volume

                volume_variances.append(volume_variance)
                ii += 1

            # step 3. cost volume regularization
            cost_reg_list = self.cost_regularization(volume_variances)
            # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
            depth_list = []
            photometric_confidence_list = []
            i = 0
            for cost_reg in cost_reg_list:
                cost_reg = cost_reg.squeeze(1) # B, C, D, H, W
                # cost volume need to mul by -1
                prob_volume = F.softmax(-1*cost_reg, dim=1) # get prob volume
                depth = depth_regression(prob_volume, depth_values=new_depth_values_list[i])
                
                with torch.no_grad():
                    # photometric confidence
                    prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
                    depth_index = depth_regression(prob_volume, depth_values=torch.arange(int(num_depth*sample_scale[i]), device=prob_volume.device, dtype=torch.float)).long()
                    photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
                depth_list.append(depth)
                photometric_confidence_list.append(photometric_confidence)
                i += 1

            return {"depth": depth_list, "photometric_confidence": photometric_confidence_list}
            
        else:
            # step 1. feature extraction
            # in: images; out: 32-channel feature maps
            features = [self.feature(img) for img in imgs]
            ref_feature, src_features = features[0], features[1:]
            ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

            # step 2. differentiable homograph, build cost volume
            if self.cost_aggregation == 0:
                ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
                volume_sum = ref_volume
                volume_sq_sum = ref_volume ** 2
                del ref_volume
                for src_fea, src_proj in zip(src_features, src_projs):
                    # warpped features
                    warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)
                    if self.training:
                        volume_sum = volume_sum + warped_volume
                        volume_sq_sum = volume_sq_sum + warped_volume ** 2
                    else:
                        # TODO: this is only a temporal solution to save memory, better way?
                        volume_sum += warped_volume
                        volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
                    del warped_volume
                # aggregate multiple feature volumes by variance
                volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
            
            elif self.cost_aggregation == 91: # 1x1 element-wise reweight
                ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
                warp_volumes = None
                for src_fea, src_proj in zip(src_features, src_projs):
                    # warpped features
                    warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)
                    warped_volume = (warped_volume - ref_volume).pow_(2) #B,C,D,H,W
                    B,C,D,H,W = warped_volume.shape
                    reweight = self.volumegate(warped_volume) #B, 1, D, H, W
                    if warp_volumes is None:
                        warp_volumes = (reweight + 1) * warped_volume
                    else:
                        warp_volumes += (reweight + 1) * warped_volume
                    #del warped_volume
                aggregate_volume = warp_volumes / len(src_features)
                volume_variance = aggregate_volume

            # step 3. cost volume regularization
            cost_reg = self.cost_regularization(volume_variance)
            cost_reg = cost_reg.squeeze(1) # B, C, D, H, W
            # cost volume need to mul by -1
            prob_volume = F.softmax(-1*cost_reg, dim=1) # get prob volume, this is as same as Original MVSNet
            depth = depth_regression(prob_volume, depth_values=depth_values)

            with torch.no_grad():
                # photometric confidence
                prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
                depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
                photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
            
            return {"depth": depth, "photometric_confidence": photometric_confidence}
            
def get_propability_map(prob_volume, depth, depth_values):
    # depth_values: B,D
    shape = prob_volume.shape
    batch_size = shape[0]
    depth_num = shape[1]
    height = shape[2]
    width = shape[3]

    depth_delta = torch.abs(depth.unsqueeze(1).repeat(1, depth_num, 1, 1) - depth_values.repeat(height, width, 1, 1).permute(2, 3, 0, 1))
    _, index = torch.min(depth_delta, dim=1) # index: B, H, W
    index = index.unsqueeze(1).repeat(1, depth_num, 1, 1)
    index_left0 = index 
    index_left1 = torch.clamp(index - 1, 0, depth_num - 1)
    index_right0 = torch.clamp(index + 1, 0, depth_num - 1)
    index_right1 = torch.clamp(index + 2, 0, depth_num - 1)

    prob_map_left0 = torch.mean(torch.gather(prob_volume, 1, index_left0), dim=1)
    prob_map_left1 = torch.mean(torch.gather(prob_volume, 1, index_left1), dim=1)
    prob_map_right0 = torch.mean(torch.gather(prob_volume, 1, index_right0), dim=1)
    prob_map_right1 = torch.mean(torch.gather(prob_volume, 1, index_right1), dim=1)
    prob_map = torch.clamp(prob_map_left0 + prob_map_left1 + prob_map_right0 + prob_map_right1, 0, 0.9999)
    return prob_map

def mvsnet_loss(depth_est, depth_gt, mask):
    mask = mask > 0.5
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True) # TODO: origin using l1 loss, why use smooth_l1_loss?

def mvsnet_loss_l1norm(depth_est, depth_gt, mask):
    mask = mask > 0.5
    return F.l1_loss(depth_est[mask], depth_gt[mask], size_average=True)

def mvsnet_loss_divby_interval(depth_est, depth_gt, mask, depth_interval):
    mask = mask > 0.5
    #import pdb
    #pdb.set_trace()
    denom = torch.sum(mask, dim=(1, 2))
    loss_fn = nn.SmoothL1Loss(size_average=False, reduce=False)
    tmp = loss_fn(depth_est, depth_gt)
    tmp = mask.type(tmp.type()) * tmp
    #tmp = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=False, reduce=False)
    b_tmp = torch.sum(tmp, dim=(1,2))
    dtype = b_tmp.type()
    return torch.mean(b_tmp.div_(depth_interval.type(dtype)).div_(denom.type(dtype)))

def mvsnet_cls_loss(prob_volume, depth_gt, mask, depth_value, return_prob_map=False): #depth_num, depth_start, depth_interval):
    # depth_value: B * NUM
    # get depth mask
    # caculate depth_start, depth_num, depth_interval
    depth_start = depth_value[:, 0].view(-1, 1)
    depth_num = depth_value.shape[-1]
    depth_interval = (depth_value[:, 1] - depth_value[:, 0]).view(-1, 1)

    mask_true = mask 
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6#tf.reduce_sum(mask_true, axis=[1, 2, 3]) + 1e-7
    shape = depth_gt.shape #tf.shape(gt_depth_image) B, H, W;
    #depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)
    start_mat = depth_start.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)
    interval_mat = depth_interval.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)
    #gt_index_image = torch.argmin(torch.abs(depth_value_mat-depth_gt.unsqueeze(1)), dim=1) # round; B, H, W
    gt_index_image = (depth_gt.unsqueeze(1) - start_mat) / interval_mat
    # print('gt_index_img: ', gt_index_image.shape)
    # print('mask :', mask_true.shape)
    gt_index_image = torch.mul(mask_true.unsqueeze(1), gt_index_image)# - mask_true # remove mask=0 pixel
    gt_index_image = torch.round(gt_index_image).type(torch.long) # B, 1, H, W
    #print('gt_idx: max {}, min {}'.format(torch.max(gt_index_image), torch.min(gt_index_image)))
    # gt index map -> gt one hot volume (B x 1 x H x W )
    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)
    # print('shape:', gt_index_volume.shape, )
    # cross entropy image (B x D X H x W)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume), dim=1).squeeze(1) # B, 1, H, W
    
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image) # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])
    #print('masked_cross_entropy', masked_cross_entropy)
    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num) # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.float32)
    wta_depth_map = wta_index_map * interval_mat + start_mat
    #wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(prob_volume, dim=1)[0] # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map.squeeze(1), photometric_confidence
    return masked_cross_entropy, wta_depth_map.squeeze(1)

#def mvsnet_cls_winner_take_all(prob_volume, depth_value): #depth_num, depth_start, depth_interval):

def mvsnet_cls_loss_ori(prob_volume, depth_gt, mask, depth_value, return_prob_map=False): #depth_num, depth_start, depth_interval):
    # depth_value: B * NUM
    # get depth mask
    mask_true = mask #tf.cast(tf.not_equal(gt_depth_image, 0.0), dtype='float32')
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6#tf.reduce_sum(mask_true, axis=[1, 2, 3]) + 1e-7
    #print('valid pixel:', valid_pixel_num)
    # gt depth map -> gt index map
    shape = depth_gt.shape #tf.shape(gt_depth_image) B, H, W;
    #depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    #start_mat = tf.tile(tf.reshape(depth_start, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])

    #interval_mat = tf.tile(tf.reshape(depth_interval, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])
    #gt_index_image = tf.div(gt_depth_image - start_mat, interval_mat)
    depth_num = depth_value.shape[-1]
    depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)
    # print('shape: ', depth_value_mat.shape)
    #gt_index_image = torch.argmin(torch.abs(depth_value_mat-depth_gt.unsqueeze(1)), dim=1, keepdim=True) # B, 1, H, W
    gt_index_image = torch.argmin(torch.abs(depth_value_mat-depth_gt.unsqueeze(1)), dim=1) # round; B, H, W
    # print('gt idx:', gt_index_image.shape)
    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))# - mask_true # remove mask=0 pixel
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1) # B, 1, H, W
    #print('gt_idx: max {}, min {}'.format(torch.max(gt_index_image), torch.min(gt_index_image)))
    # gt index map -> gt one hot volume (B x 1 x H x W )
    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)
    # print('shape:', gt_index_volume.shape, )
    # cross entropy image (B x D X H x W)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume), dim=1).squeeze(1) # B, 1, H, W
    #print('cross_entropy_image', cross_entropy_image)
    # masked cross entropy loss
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image) # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])
    #print('masked_cross_entropy', masked_cross_entropy)
    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num) # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(prob_volume, dim=1)[0] # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map, photometric_confidence
    return masked_cross_entropy, wta_depth_map

    
