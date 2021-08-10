"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import torch
import torch.nn as nn
from subpixel import shuffle_down, shuffle_up###################
import torch.nn.functional as F
from pdb import set_trace as stx
from Blanced_attention import BlancedAttention, BlancedAttention_CAM_SAM_ADD

##########################################################################
class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

def st_conv(in_channels, out_channels, kernel_size, bias=False, stride = 2):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)		
##########################################################################
class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x
##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.PReLU(),#nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
		
# contrast-aware channel attention module
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
	
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.PReLU(),#nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res
		
## Enhanced Channel Attention Block (ECAB)
class ECAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(ECAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
        self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)
    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = self.S2FB2(res, x)
        #res += x
        return res
		
## Channel Attention Block (CAB)
class CAB_dsc(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB_dsc, self).__init__()
        modules_body = []
        modules_body.append(depthwise_separable_conv(n_feat, n_feat))
        modules_body.append(act)
        modules_body.append(depthwise_separable_conv(n_feat, n_feat))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
        #self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)
    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        #res = self.S2FB2(res, x)
        res += x
        return res

##########################################################################
##---------- Resizing Modules ----------    
# class DownSample1(nn.Module):
    # def __init__(self):
        # super(DownSample, self).__init__()
        # self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)

    # def forward(self, x):
        # x = self.down(x)
        # return x

class DownSample(nn.Module):
    def __init__(self, in_channels):
    #def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x
		
class DownSample4(nn.Module):
    def __init__(self, in_channels):
    #def __init__(self, in_channels,s_factor):
        super(DownSample4, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x
	
class DownSample8(nn.Module):
    def __init__(self, in_channels):
    #def __init__(self, in_channels,s_factor):
        super(DownSample4, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x
# class UpSample1(nn.Module):
    # def __init__(self, in_channels):
    # #def __init__(self, in_channels,s_factor):
        # super(UpSample, self).__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    # def forward(self, x):
        # x = self.up(x)
        # return x
	
class UpSample(nn.Module):
    def __init__(self, in_channels):
    #def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class UpSample4(nn.Module):
    def __init__(self, in_channels):
    #def __init__(self, in_channels,s_factor):
        super(UpSample4, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x
		
class SkipUpSample(nn.Module):
    #def __init__(self, in_channels,s_factor):
    def __init__(self, in_channels):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

##########################################################################
## Spatial Attention Layer
class SALayer(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.PReLU(),#nn.PReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_du(x)
        return x * y
##########################################################################
## Long Feature Selection and Fusion Block (LFSFB)
class LFSFB(nn.Module):
    def __init__(self, n_feat, kernel_size, act, bias):
        super(LFSFB, self).__init__()
        #self.FS = nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1, padding=0, bias=False)
        #self.act1 =act
        self.FFU = nn.ConvTranspose2d(n_feat, n_feat,  kernel_size=3, stride=2, padding=1,output_padding=1, bias= False)
        self.act2 = act

    def forward(self, x1, x2):
        #res = self.act1(self.FS(x1))
        #res = self.act1(res)
        #print(res.shape)
        res_out = self.act2(self.FFU(x2))
        #res = self.act2(res)
        #print(res.shape)
        return res_out
##########################################################################
## Reconstruction and Reproduction Block (RRB)
class RRB(nn.Module):
    def __init__(self, n_feat, kernel_size, act, bias):
        super(RRB, self).__init__()
        self.recon_B =  conv(n_feat, 3, kernel_size, bias=bias)
        self.recon_R = conv(n_feat, 3, kernel_size, bias=bias)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(3, n_feat, 1, padding=0, bias=bias),
                nn.PReLU(),#nn.PReLU(inplace=True),
                #act,
                nn.Conv2d(n_feat, 3, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )
    def forward(self, x):
        xB = x[0]
        xR = x[1]
        recon_B = self.recon_B(xB)
        recon_R = self.recon_R(xR)
        res = self.avg_pool(recon_B + recon_R)
        res_att = self.conv_du(res)
        re_rain = recon_B*res_att + recon_R*(1-res_att)
        return [recon_B, re_rain]
##########################################################################
## Coupled Representation Block (CRB)
class CRB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_rcab):
        super(CRB, self).__init__()
        self.down_R = st_conv(n_feat, n_feat, kernel_size, bias=bias)
        self.act1=act
        modules_body = []
        modules_body = [CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_rcab)]
        self.body = nn.Sequential(*modules_body)

        self.lfsfb = LFSFB(n_feat, kernel_size, act, bias)
        self.CA_B = SALayer(n_feat, reduction, bias=bias)
        self.CA_R = SALayer(n_feat, reduction, bias=bias)
		
    def forward(self, x):
        xB = x[0]
        xR = x[1]
        res_down_R = self.act1(self.down_R(xR))
        #res_down_R = self.act1(res_down_R)
        res_R = self.body(res_down_R)
        #print(res_R.shape)
        #print(res_down_R.shape)
        xR_res = xR + self.lfsfb(res_down_R, res_R)
		
        res_BTOR = self.CA_B(xB)
        res_RTOB = self.CA_R(xR_res)
        x[0] = xB - res_BTOR + res_RTOB
        x[1] = xR_res - res_RTOB + res_BTOR
        return x
##########################################################################
## Coupled Representation Module (CRM)
class CRM(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_crb, num_rcab):
        super(CRM, self).__init__()
        modules_body = []
        modules_body = [CRB(n_feat, kernel_size, reduction, bias=bias, act=act, num_rcab=num_rcab) for _ in range(num_crb)]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x_B, x_R):
        res = self.body([x_B, x_R])
        #res += x
        return res
##########################################################################
class MODEL(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_crb, num_rcab):
        super(MODEL, self).__init__()
		
        self.image_fea = conv(3, n_feat, kernel_size, bias=bias)
		#### embedding
        self.shallow_fea_B = depthwise_separable_conv(n_feat, n_feat)#conv(n_feat*2, n_feat, kernel_size=1, bias=bias)
        self.shallow_fea_R = depthwise_separable_conv(n_feat, n_feat)#conv(n_feat*2, n_feat, kernel_size=1, bias=bias)
		
        self.down_B = st_conv(n_feat, n_feat, kernel_size, bias=bias)
        self.down_R = st_conv(n_feat, n_feat, kernel_size, bias=bias)
		
        self.crm = CRM(n_feat, kernel_size, reduction, act, bias, num_crb, num_rcab)
		
        #self.lfsfb_B = LFSFB(n_feat, kernel_size, act, bias)
        self.UP_B = nn.Sequential(nn.ConvTranspose2d(n_feat, n_feat,  kernel_size=3, stride=2, padding=1,output_padding=1, bias= False), act)#, conv(n_feat, n_feat, 1, bias=bias))
        #self.lfsfb_R = LFSFB(n_feat, kernel_size, act, bias)
        self.UP_R = nn.Sequential(nn.ConvTranspose2d(n_feat, n_feat,  kernel_size=3, stride=2, padding=1,output_padding=1, bias= False), act)#, conv(n_feat, n_feat, 1, bias=bias))
        self.rrb = RRB(n_feat, kernel_size, act, bias=bias)

    def forward(self, x):
		
        x_fea = self.image_fea(x)
        B_fea = self.shallow_fea_B(x_fea)
        R_fea = self.shallow_fea_R(x_fea)
		
        B_down_fea = self.down_B(B_fea)
        R_down_fea = self.down_R(R_fea)
        #cat_fea = [B_down_fea, R_down_fea]
        [fea_B, fea_R] = self.crm(B_down_fea, R_down_fea)
		
        #fea_B_fuse = self.lfsfb_B(B_down_fea, fea_B)
        #fea_R_fuse = self.lfsfb_R(R_down_fea, fea_R)
        fea_B_fuse = self.UP_B(fea_B)
        fea_R_fuse = self.UP_R(fea_R)
		
        [img_B, img_R] = self.rrb([fea_B_fuse, fea_R_fuse])
        return img_B, img_R
		
##########################################################################
class PCNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=32, kernel_size=3, reduction=4, num_crb=20, num_rcab=3, bias=False):
        super(PCNet, self).__init__()

        act=nn.PReLU()
        self.model = MODEL(n_feat, kernel_size, reduction, act, bias, num_crb, num_rcab)

    def forward(self, x_img): #####b,c,h,w
        #print(x_img.shape)
        imitation, rain_out = self.model(x_img)
        #print(imitation.shape)
        #print(rain_out.shape)
        return [imitation, rain_out]