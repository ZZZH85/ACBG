#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Implementation of PointFlow ResNet series.
# Author: Xiangtai(lxt@pku.edu.cn)
import argparse

import torch.nn as nn
import torch

from config import assert_and_infer_cfg
from .nn.operators import PSPModule
from .nn.point_flow import PointFlowModuleWithMaxAvgpool
import resnet_d as Resnet_Deep
from .nn.mynn import Norm2d, Upsample


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        normal_layer(out_planes),
        nn.ReLU(inplace=True),
    )

class UperNetAlignHeadMaxAvgpool(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[64,128,256,512], fpn_dim=512,
                 fpn_dsn=True, reduce_dim=64, ignore_background=False, max_pool_size=8,
                 avgpool_size=8, edge_points=32):
        super(UperNetAlignHeadMaxAvgpool, self).__init__()

        self.ppm = PSPModule(inplane, norm_layer=norm_layer, out_features=fpn_dim)
        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)
        self.fpn_out = []
        self.fpn_out_align = []
        self.dsn = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))

            if ignore_background:
                self.fpn_out_align.append(
                    PointFlowModuleWithMaxAvgpool(fpn_dim, dim=reduce_dim, maxpool_size=max_pool_size,
                                                  avgpool_size=avgpool_size, edge_points=edge_points))
            else:
                self.fpn_out_align.append(
                    PointFlowModuleWithMaxAvgpool(fpn_dim, dim=reduce_dim, maxpool_size=max_pool_size,
                                                  avgpool_size=avgpool_size, edge_points=edge_points))

            if self.fpn_dsn:
                self.dsn.append(
                    # nn.Sequential(
                    #     nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
                    #     norm_layer(fpn_dim),
                    #     nn.ReLU(),
                    #     nn.Dropout2d(0.1),
                    #     nn.Conv2d(fpn_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True)
                    # )
                    nn.Conv2d(fpn_dim, 1, 7, 1, 3)
                )

        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        if self.fpn_dsn:
            self.dsn = nn.ModuleList(self.dsn)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )



    def forward(self, conv_out):
        psp_out = self.ppm(conv_out[-1])     #将最后一个特征层做四次averagepooling

        f = psp_out

        fpn_feature_list = [f]
        edge_preds = []
        out = []

        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)   #做一次CBR   所有特征层通道变成成和最后一层一样
            f, edge_pred= self.fpn_out_align[i]([f, conv_x])
            f = conv_x + f
            edge_preds.append(edge_pred)
            fpn_feature_list.append(self.fpn_out[i](f))
            if self.fpn_dsn:
                out.append(self.dsn[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:] #输入图像的4倍
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        return x, edge_preds,out




class AlignNetResNetMaxAvgpool(nn.Module):
    def __init__(self, num_classes, trunk='resnet-50-deep', criterion=None, variant='D', skip='m1', skip_num=48,
                 fpn_dsn=False, inplanes=128, reduce_dim=64, ignore_background=False,
                 max_pool_size=8, avgpool_size=8, edge_points=32):
        super(AlignNetResNetMaxAvgpool, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num
        self.fpn_dsn = fpn_dsn

        if trunk == trunk == 'resnet-50-deep':
            resnet = Resnet_Deep.resnet50()
        elif trunk == 'resnet-101-deep':
            resnet = Resnet_Deep.resnet101()
        else:
            raise ValueError("Not a valid network arch")

        resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        del resnet

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (2, 2), (2, 2)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (4, 4), (4, 4)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)
        else:
            print("Not using Dilation ")

        inplane_head = 2048
        self.head = UperNetAlignHeadMaxAvgpool(inplane_head, num_class=num_classes, norm_layer=Norm2d,
                                               fpn_dsn=fpn_dsn, reduce_dim=reduce_dim,
                                               ignore_background=ignore_background, max_pool_size=max_pool_size,
                                               avgpool_size=avgpool_size, edge_points=edge_points)

    def forward(self, x, gts=None):
        x_size = x.size()   #3,w,h
        x0 = self.layer0(x) #128 w/4 h/4
        x1 = self.layer1(x0) #256 w/4 h/4
        x2 = self.layer2(x1)#512 w/8 h/8
        x3 = self.layer3(x2)#1024 w/16 h/16
        x4 = self.layer4(x3)#2048 w/32 h/32
        x = self.head([x1, x2, x3, x4]) #4个stage最后输出
        main_out = Upsample(x[0], x_size[2:])#4倍上采样
        edge_preds = [Upsample(edge_pred, x_size[2:]) for edge_pred in x[1]]
        if self.training:
            if not self.fpn_dsn:
                return self.criterion([main_out, edge_preds], gts)
            return self.criterion(x, gts)
        return main_out


def DeepR101_PF_maxavg_deeply(num_classes, criterion, reduce_dim=64, max_pool_size=8, avgpool_size=8, edge_points=32):
    """
    ResNet-50 Based Network
    """
    return AlignNetResNetMaxAvgpool(num_classes, trunk='resnet-101-deep', criterion=criterion, variant='D', skip='m1',
                                    reduce_dim=reduce_dim, max_pool_size=max_pool_size, avgpool_size=avgpool_size,
                                    edge_points=edge_points)

def DeepR50_PF_maxavg_deeply(num_classes, criterion, reduce_dim=64, max_pool_size=8, avgpool_size=8, edge_points=32):
    """
    ResNet-50 Based Network
    """
    return AlignNetResNetMaxAvgpool(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D', skip='m1',
                                    reduce_dim=reduce_dim, max_pool_size=max_pool_size, avgpool_size=avgpool_size,
                                    edge_points=edge_points)


# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--arch', type=str, default='network.pointflow_resnet_with_max_avg_pool.DeepR50_PF_maxavg_deeply',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='cityscapes, mapillary, camvid, kitti')

parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')

parser.add_argument('--class_uniform_pct', type=float, default=0.0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')
parser.add_argument('--dice_loss', default=False, action='store_true', help="whether use dice loss in edge")
parser.add_argument("--ohem", action="store_true", default=False, help="start OHEM loss")
parser.add_argument("--aux", action="store_true", default=False, help="whether use Aux loss")
parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--joint_edge_loss_pfnet', action='store_true')
parser.add_argument('--edge_weight', type=float, default=1.0,
                    help='Edge loss weight for joint loss')
parser.add_argument('--seg_weight', type=float, default=1.0,
                    help='Segmentation loss weight for joint loss')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_epoch', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')
parser.add_argument('--apex', action='store_true', default=False,
                    help='Use Nvidia Apex Distributed Data Parallel')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')

parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--color_aug', type=float,
                    default=0.0, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=True,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=4,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=1.0,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=1.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--fix_bn', action='store_true', default=False,
                    help=" whether to fix bn for improving the performance")
parser.add_argument('--evaluateF', action='store_true', default=False,
                    help="whether to evaluate the F score")
parser.add_argument('--eval_thresholds', type=str, default='0.0005,0.001875,0.00375,0.005',
                    help='Thresholds for boundary evaluation')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--print_freq', type=int, default=5, help='frequency of print')
parser.add_argument('--eval_freq', type=int, default=1, help='frequency of evaluation during training')
parser.add_argument('--with_aug', action='store_true')
parser.add_argument('--thicky', default=8, type=int)
parser.add_argument('--draw_train', default=False, action='store_true')
parser.add_argument('--match_dim', default=64, type=int, help='dim when match in pfnet')
parser.add_argument('--ignore_background', action='store_true', help='whether to ignore background class when '
                                                                     'generating coarse mask in pfnet')
parser.add_argument('--maxpool_size', type=int, default=9)
parser.add_argument('--avgpool_size', type=int, default=9)
parser.add_argument('--edge_points', type=int, default=32)
parser.add_argument('--start_epoch', type=int, default=0)
args = parser.parse_args()
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                    'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
if __name__ == '__main__':
    x1=torch.rand(2,64,128,128)
    x2 = torch.rand(2,128, 64, 64)
    x3 = torch.rand(2, 256, 32, 32)
    x4 = torch.rand(2, 512,16, 16)
    x=[x1,x2,x3,x4]
    inplane_head = 512
    head = UperNetAlignHeadMaxAvgpool(inplane_head, num_class=2, norm_layer=Norm2d,
                                           fpn_dsn=True, reduce_dim=64,
                                           ignore_background=False, max_pool_size=8,
                                           avgpool_size=8, edge_points=32)
    x, edge_preds,out=head(x)
    print(1)