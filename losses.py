import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

##########################################################################
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return [x_LL, x_HL, x_LH, x_HH]#torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


# 使用哈尔 haar 小波变换来实现二维逆向离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width]) #[1, 12, 56, 56]
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r**2)), r * in_height, r * in_width
    # print(out_batch, out_channel, out_height, out_width) #1 3 112 112
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    # print(x1.shape) #torch.Size([1, 3, 56, 56])
    # print(x2.shape) #torch.Size([1, 3, 56, 56])
    # print(x3.shape) #torch.Size([1, 3, 56, 56])
    # print(x4.shape) #torch.Size([1, 3, 56, 56])
    # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


# 二维离散小波
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)
		
		
class CharbonnierLoss_dwt(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss_dwt, self).__init__()
        self.eps = eps
        self.target_down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.x_dwt = DWT()
        self.y_dwt = DWT()
		
    def forward(self, x, y):
        x_fea = self.x_dwt(x)
        y_fea = self.y_dwt(y)
		
        #_, _, x_kw, x_kh = x_fea[0].shape
        #_, _, y_kw, y_kh = y_fea[0].shape
        #if x_kw == y_kw:
            #diff = x_fea - y_fea
        #else:
            #diff = x_fea - self.target_down(y_fea)
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = np.sum([torch.mean(torch.sqrt(((x_fea[j]-y_fea[j]) * (x_fea[j]-y_fea[j])) + (self.eps*self.eps))) for j in range(len(x_fea))])
        #loss = torch.mean(torch.sqrt(((x_fea[j]-y_fea[j]) * (x_fea[j]-y_fea[j])) + (self.eps*self.eps)))
        return loss
		
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.target_down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
    def forward(self, x, y):
        _, _, x_kw, x_kh = x.shape
        _, _, y_kw, y_kh = y.shape
        if x_kw == y_kw:
            diff = x - y
        else:
            diff = x - self.target_down(y)
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

		
class L1smooth(nn.Module):
    """L1smooth (L1)"""

    def __init__(self):
        super(L1smooth, self).__init__()
        self.target_down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.L1_smooth = torch.nn.SmoothL1Loss()
    def forward(self, x, y):
        _, _, x_kw, x_kh = x.shape
        _, _, y_kw, y_kh = y.shape
        if x_kw == y_kw:
            loss = self.L1_smooth(x,y)
        else:
            loss = self.L1_smooth(x,self.target_down(y))
        return loss
		
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()
        self.target_down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
		
    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        _, _, x_kw, x_kh = x.shape
        _, _, y_kw, y_kh = y.shape
        if x_kw == y_kw:
            loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        else:
            loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(self.target_down(y)))
        #loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss
