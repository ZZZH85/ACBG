# ------------------------------------------------------------------------------
# Author: Xiao Guo (guoxia11@msu.edu)
# CVPR2023: Hierarchical Fine-Grained Image Forgery Detection and Localization
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
from .GaussianSmoothing import GaussianSmoothing

class LaPlacianMs(nn.Module):
    def __init__(self,in_c,gauss_ker_size=3,scale=[2],drop_rate=0.2):
        super(LaPlacianMs, self).__init__()
        self.scale = scale
        self.gauss_ker_size = gauss_ker_size
        ## apply gaussian smoothing to input feature maps with 3 planes
        ## with kernel size K and sigma s
        self.smoothing = nn.ModuleDict()
        for s in self.scale:
            self.smoothing['scale-'+str(s)] = GaussianSmoothing(in_c, self.gauss_ker_size, s)

        self.conv_1x1 = nn.Sequential(nn.Conv2d(in_c*len(scale), in_c,
                                                kernel_size=1, stride=1,
                                                bias=False,groups=1),
                                                nn.BatchNorm2d(in_c),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(p=drop_rate)
        )
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def down(self,x,s):
        return F.interpolate(x,scale_factor=s,
                             mode='bilinear',
                             align_corners=False)
    def up (self,x, size):
        return F.interpolate(x,size=size,mode='bilinear',align_corners=False)

    def forward(self, x):
        # print(self.scale)
        # import sys;sys.exit(0)
        for i, s in enumerate(self.scale):
            # print()
            # print(x.size())
            sm = self.smoothing['scale-'+str(s)](x)
            # print(sm.size())
            sm = self.down(sm,1/s)
            # print(sm.size())
            sm = self.up(sm,(x.shape[2],x.shape[3]))
            # print(sm.size())
            # print("========================")
            if i == 0:
                diff = x - sm
            else:
                diff = torch.cat((diff, x - sm), dim=1)
        return self.conv_1x1(diff)