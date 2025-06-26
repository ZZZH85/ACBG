import torch
import torch.nn as nn
import torch.nn.functional as F
from network.nn.point_flow_learned_pe import PointFlowModuleWithMaxAvgpool
from network.nn.operators import PSPModule
###################################################################
# ################## Channel Attention Block ######################
###################################################################
class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : channel attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


###################################################################
# ################## Spatial Attention Block ######################
###################################################################
class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : spatial attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


###################################################################
# ################## Context Exploration Block ####################
###################################################################

class PEfusion(nn.Module):
    def __init__(self, channel1):
        super(PEfusion, self).__init__()
        self.channel = channel1

        self.mlp = nn.Conv2d(2, self.channel , 1, bias=False)
    def forward(self, x):
        _, _, h, w = x.shape
        # yy, xx = torch.meshgrid(torch.arange(h // 8), torch.arange(w // 8))
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w))
        yy = yy.float() / (h / 16) - 1.0
        xx = xx.float() / (w / 16) - 1.0
        cord = torch.stack((yy, xx), dim=0).unsqueeze(0).repeat(x.shape[0], 1, 1, 1).cuda()  # [2,4,4]
        cord = self.mlp(cord.float())
        furture_fosion = torch.cat([x, cord],dim=1)  # x[320,32,32]  cord[320,4,4]
        # furture_fosion = self.cr(furture_fosion)
        return furture_fosion

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()
        self.mlp = nn.Conv2d(2 * num_pos_feats, num_pos_feats, 1)

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self,x):
        # x = tensor_list.tensors
        # h, w = x.shape[-2:]
        _, _, h, w = x.shape  #h,w:32
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)  #32,320
        y_emb = self.row_embed(j)
        # test1 = x_emb.unsqueeze(0) #.repeat(h, 1, 1)  #16,16,256
        # test2 = y_emb.unsqueeze(1) #.repeat(1, w, 1)  #16,16,256
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        pos = self.mlp(pos)

        furture_fosion = torch.cat([x,pos], dim=1)

        return furture_fosion

###################################################################
# ##################### Positioning Module ########################
###################################################################
class Positioning(nn.Module):
    def __init__(self, channel):
        super(Positioning, self).__init__()
        self.channel = channel
        self.channels = 3 * channel
        self.cab = CA_Block(self.channel)
        self.sab = SA_Block(self.channel)
        self.cr =  nn.Conv2d(self.channels, self.channel,kernel_size=1, stride=1, padding=0)
        self.map = nn.Conv2d(self.channel, 1, 7, 1, 3)

    def forward(self, x):
        cab = self.cab(x)
        sab = self.sab(cab)
        mix = torch.cat((x,cab,sab),1)
        mix = self.cr(mix)
        map = self.map(mix)

        return mix, map,cab,sab



###################################################################
# ######################## Focus Module ###########################
###################################################################
class Focus(nn.Module):
    def __init__(self, channel1, channel2):
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))

        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())


        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)

        self.conv1= nn.Conv2d(in_channels=channel1*2, out_channels=channel1, kernel_size=1)

    def forward(self, x, y, in_map):

        up = self.up(y)

        refine2=self.conv1(torch.cat([x,up],dim=1))
        output_map = self.output_map(refine2)

        return refine2, output_map

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))


###################################################################
# ########################## NETWORK ##############################
###################################################################
class ACBG(nn.Module):
    def __init__(self, backbone_path=None):
        super(ACBG, self).__init__()
        # params

        # backbone
        import backbone.resnet.resnet as resnet
        resnet50 = resnet.resnet50(backbone_path)
        self.layer0 = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu)
        self.layer1 = nn.Sequential(resnet50.maxpool, resnet50.layer1)
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4

        # channel reduction
        self.cr4 = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.cr3 = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.cr2 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.cr1 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())

        # positioning
        self.positioning = Positioning(512)

        # focus
        self.focus3 = Focus(256, 512)
        self.focus2 = Focus(128, 256)
        self.focus1 = Focus(64, 128)


# 深层特征 平均池化
        self.ppm = PSPModule(512, norm_layer=nn.BatchNorm2d, out_features=512)



#F34边缘模块-------------------------------------------
        self.point_34 = PointFlowModuleWithMaxAvgpool(512, dim=256, maxpool_size=8,avgpool_size=8, edge_points=32)

        self.point_cbr_f3= nn.Sequential(
                    nn.Conv2d(256, 512, 1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=False))
            #恢复浅层特征通道数
        self.conv1_f3=nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
#--------------------------------------------------------------------------------------
# F23边缘模块


        self.point_23 = PointFlowModuleWithMaxAvgpool(256, dim=128, maxpool_size=8, avgpool_size=8, edge_points=32)

        self.point_cbr_f2= nn.Sequential(
            nn.Conv2d(128, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False))
        # 恢复浅层特征通道数
        self.conv1_f2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)


        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        # x: [batch_size, channel=3, h, w]
        layer0 = self.layer0(x)  # [-1, 64, h/2, w/2]
        layer1 = self.layer1(layer0)  # [-1, 256, h/4, w/4]
        layer2 = self.layer2(layer1)  # [-1, 512, h/8, w/8]
        layer3 = self.layer3(layer2)  # [-1, 1024, h/16, w/16]
        layer4 = self.layer4(layer3)  # [-1, 2048, h/32, w/32]

        # channel reduction
        cr4 = self.cr4(layer4)
        cr3 = self.cr3(layer3)
        cr2 = self.cr2(layer2)
        cr1 = self.cr1(layer1)

        # positioning
        positioning, predict4,ch,sp = self.positioning(cr4)

  #F4做四次平均池化

        F4_1=self.ppm(positioning)

        cr3_1=self.point_cbr_f3(cr3) #将F3卷积成和F4一样的维度  cr3_1 [bs,512,32,32]
        f_3, edge_predict4 ,sample_x4,sample_y4 = self.point_34([F4_1,cr3_1])   #16*16
        # f_3, edge_predict4 = self.point_34([F4_1, cr3_1])  # 16*16
        f_3=self.conv1_f3(f_3+cr3_1)

        # focus--输出mask图以及concat特征
        focus3, predict3 = self.focus3(f_3, positioning, predict4)
#边缘模块f23
        cr2_1 = self.point_cbr_f2(cr2)  # 将F3卷积成和F4一样的维度
        f_2, edge_predict3 ,sample_x3,sample_y3 = self.point_23([focus3, cr2_1])  #32*32
        # f_2, edge_predict3 = self.point_23([focus3, cr2_1])  # 32*32
        f_2 = self.conv1_f2(f_2 + cr2_1)

        focus2, predict2 = self.focus2(f_2, focus3, predict3)
        focus1, predict1 = self.focus1(cr1, focus2, predict2)

        # rescale
        edge_predict4=F.interpolate( edge_predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        edge_predict3 = F.interpolate(edge_predict3, size=x.size()[2:], mode='bilinear', align_corners=True)

        predict4 = F.interpolate(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)


        if self.training:
            return predict4, predict3, predict2, predict1,edge_predict3,edge_predict4,focus1,focus2,focus3



        #return torch.sigmoid(predict4), torch.sigmoid(predict3), torch.sigmoid(predict2), torch.sigmoid(predict1), torch.sigmoid(ch),torch.sigmoid(sp),torch.sigmoid(positioning)
        return torch.sigmoid(predict4), torch.sigmoid(predict3), torch.sigmoid(predict2), torch.sigmoid(predict1),torch.sigmoid(edge_predict3),torch.sigmoid(edge_predict4),focus1,focus2,focus3