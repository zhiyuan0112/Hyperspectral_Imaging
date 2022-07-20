import math

import numpy as np
import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class ConvADMM(nn.Sequential):
    def __init__(
        self, n_convs, in_channels, out_channels, kernel_size, bias=True,
        bn=False, act=nn.ReLU(True)):

        m = []
        for i in range(n_convs):
            if (i > 0 and in_channels != out_channels):
                in_channels = out_channels
            m.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, 
                            padding=(kernel_size//2), bias=bias)
            )
            if bn:
                m.append(nn.BatchNorm2d(out_channels))
            if act is not None:
                m.append(act)

        super(ConvADMM, self).__init__(*m)
        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)



class MosaicAlpha2D(nn.Module):
    def __init__(self, pattern_size, channel):
        super(MosaicAlpha2D, self).__init__()
        torch.manual_seed(2021)
        
        self.pattern_size = pattern_size
        self.channel = channel
        self.weight = nn.Parameter(torch.ones([1, self.channel, self.pattern_size, self.pattern_size]), requires_grad=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, alpha):
        w = self.softmax(alpha * self.weight)
        w = w.repeat(1, 1, x.shape[-2]//w.shape[-2], x.shape[-1]//w.shape[-1])
        out = torch.sum(x * w, dim=1)
        return out, w


class MosaicSpectrum2D(nn.Module):
    def __init__(self, pattern_size, channel):
        super(MosaicSpectrum2D, self).__init__()
        torch.manual_seed(2021)
        
        self.pattern_size = pattern_size
        self.channel = channel
        self.weight = nn.Parameter(torch.full([1, self.channel, self.pattern_size, self.pattern_size], 1/31), requires_grad=True)  # bs,c,h,w
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        eta3 = 1e-3
        loss1 = eta3 * torch.norm(self.weight, p=2) ** 2
        eta4 = 1e-2
        loss2 = eta4 * torch.norm(self.weight[:, 1:, :, :] - self.weight[:, :-1, :, :], p=2) ** 2

        # if torch.min(self.weight) < 0:
        #     w = self.softmax(self.weight)
        # else:
        #     w = self.weight
        # if torch.min(self.weight) < 0:
        #     w = self.weight - torch.min(self.weight)
        # else:
        #     w = self.weight

        zero = torch.zeros_like(self.weight)
        w = torch.where(self.weight < 0, zero, self.weight)
        # w = self.weight
        w = w.repeat(1, 1, x.shape[-2]//self.weight.shape[-2], x.shape[-1]//self.weight.shape[-1])

        # non_zero_weight = torch.where(self.weight < 0, torch.zeros_like(self.weight), self.weight)
        # w = non_zero_weight.repeat(1, 1, x.shape[-2]//self.weight.shape[-2], x.shape[-1]//self.weight.shape[-1])

        out = torch.sum(x * w, dim=1)
        
        return out, w, loss1 + loss2


class Mosaic2DBase(nn.Module):
    def __init__(self, pattern_size=4):
        super(Mosaic2DBase, self).__init__()
        torch.manual_seed(2021)
        
        self.pattern_size = pattern_size
        MSFA = np.zeros((16, self.pattern_size, self.pattern_size))
        MSFA[0, :, :]  = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[1, :, :]  = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[2, :, :]  = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[3, :, :]  = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[4, :, :]  = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[5, :, :]  = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[6, :, :]  = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[7, :, :]  = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[8, :, :]  = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[9, :, :]  = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
        MSFA[10, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        MSFA[11, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        MSFA[12, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
        MSFA[13, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]])
        MSFA[14, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])
        MSFA[15, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        self.weight = torch.Tensor(MSFA).unsqueeze(0).cuda()

    def forward(self, x):
        w = self.weight.repeat(1, 1, x.shape[-2]//self.weight.shape[-2], x.shape[-1]//self.weight.shape[-1])
        out = torch.sum(x * w, dim=1)
        return out, w


class Mosaic3DBase(nn.Module):
    def __init__(self, pattern_size=4):
        super(Mosaic3DBase, self).__init__()
        torch.manual_seed(2021)
        
        self.pattern_size = pattern_size
        MSFA = np.zeros((16, self.pattern_size, self.pattern_size))
        MSFA[0, :, :] = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[1, :, :] = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[2, :, :] = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[3, :, :] = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[4, :, :] = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[5, :, :] = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[6, :, :] = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[7, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[8, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        MSFA[9, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
        MSFA[10, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        MSFA[11, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        MSFA[12, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
        MSFA[13, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]])
        MSFA[14, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])
        MSFA[15, :, :] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        self.weight = torch.Tensor(MSFA).unsqueeze(0).unsqueeze(0).cuda()

    def forward(self, x):
        w = self.weight.repeat(1, 1, 1, x.shape[-2]//self.weight.shape[-2], x.shape[-1]//self.weight.shape[-1])
        out = torch.sum(x * w, dim=2)
        return out, w


class SSRMosaicAlpha2D(nn.Module):
    def __init__(self, pattern_size, channel):
        super(SSRMosaicAlpha2D, self).__init__()
        torch.manual_seed(2021)
        
        self.pattern_size = pattern_size
        self.channel = channel
        self.weight = nn.Parameter(torch.ones([1, self.channel, self.pattern_size, self.pattern_size]), requires_grad=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, alpha):
        w = self.softmax(alpha * self.weight)
        w = w.repeat(1, 1, x.shape[-2]//w.shape[-2], x.shape[-1]//w.shape[-1])
        out = torch.sum(x * w, dim=1)
        return out, w
