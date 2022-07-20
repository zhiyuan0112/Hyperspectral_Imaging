import torch
import torch.nn as nn

from models.admmn import common
from models.admmn.edsr import EDSR
from models.admmn.common import MosaicAlpha2D, Mosaic2DBase, MosaicSpectrum2D


class ADMM_Block(nn.Module):
    def __init__(self, n_resblocks, in_channels, n_feats, n_convs, kernel_size):
        super(ADMM_Block, self).__init__()
        self.rho = nn.Parameter(torch.Tensor([1e-2]), requires_grad=True)
        
        self.denoiser = EDSR(n_resblocks=n_resblocks, n_colors=in_channels, n_feats=n_feats)

        self.net1 = common.ConvADMM(n_convs=n_convs, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size)
        self.net2 = common.ConvADMM(n_convs=n_convs, in_channels=1, out_channels=in_channels, kernel_size=kernel_size)

    def forward(self, v_pre, u_pre, y):
        x = self.net1(self.net2(y) + self.rho * (v_pre + u_pre))
        v = self.denoiser(x - u_pre)
        u = u_pre - x + v
        return v, u


class ADMMN_BASE(nn.Module):
    def __init__(self, n_resblocks, n_admmblocks, in_channels, n_feats, n_convs):
        super(ADMMN_BASE, self).__init__()

        kernel_size = 3
        
        self.rho = nn.Parameter(torch.Tensor([1e-2]), requires_grad=True)
        
        self.weight = Mosaic2DBase(4)

        self.net1 = common.ConvADMM(n_convs=n_convs, in_channels=1, out_channels=in_channels, kernel_size=kernel_size)
        self.net2 = common.ConvADMM(n_convs=n_convs, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size)
        self.net3 = common.ConvADMM(n_convs=n_convs, in_channels=1, out_channels=in_channels, kernel_size=kernel_size)

        self.admm_blocks = nn.ModuleList(
            [ADMM_Block(n_resblocks, in_channels, n_feats, n_convs, kernel_size) for _ in range(n_admmblocks)]
        )

        self.denoiser = EDSR(n_resblocks=n_resblocks, n_colors=in_channels, n_feats=n_feats)

    def forward(self, y):
        y, w = self.weight(y)
        y = y.unsqueeze(1)
        mosaic = y  # bs,1,h,w

        v = self.net1(y)
        x = self.net2(self.net3(y) + self.rho * v)
        v = self.denoiser(x)
        u = v - x

        v_list = [v]
        v_list.append(v)
        for block in self.admm_blocks:
            v, u = block(v, u, y)
            v_list.append(v)
        
        return v_list[-1], w, mosaic


class ADMMN_ALPHA(nn.Module):
    def __init__(self, n_resblocks, n_admmblocks, in_channels, n_feats, n_convs):
        super(ADMMN_ALPHA, self).__init__()

        kernel_size = 3
        
        self.rho = nn.Parameter(torch.Tensor([1e-2]), requires_grad=True)

        self.weight = MosaicAlpha2D(4, 16)

        self.net1 = common.ConvADMM(n_convs=n_convs, in_channels=1, out_channels=in_channels, kernel_size=kernel_size)
        self.net2 = common.ConvADMM(n_convs=n_convs, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size)
        self.net3 = common.ConvADMM(n_convs=n_convs, in_channels=1, out_channels=in_channels, kernel_size=kernel_size)

        self.admm_blocks = nn.ModuleList(
            [ADMM_Block(n_resblocks, in_channels, n_feats, n_convs, kernel_size) for _ in range(n_admmblocks)]
        )

        self.denoiser = EDSR(n_resblocks=n_resblocks, n_colors=in_channels, n_feats=n_feats)

    def forward(self, y, alpha):
        y, w = self.weight(y, alpha)
        y = y.unsqueeze(1)
        mosaic = y
        
        v = self.net1(y)
        x = self.net2(self.net3(y) + self.rho * v)
        v = self.denoiser(x)
        u = v - x

        v_list = [v]
        v_list.append(v)
        for block in self.admm_blocks:
            v, u = block(v, u, y)
            v_list.append(v)
        
        return v_list[-1], w, mosaic
    

class ADMMN_SPECTRUM(nn.Module):
    def __init__(self, n_resblocks, n_admmblocks, in_channels, n_feats, n_convs):
        super(ADMMN_SPECTRUM, self).__init__()

        kernel_size = 3
        
        self.rho = nn.Parameter(torch.Tensor([1e-2]), requires_grad=True)

        self.weight = MosaicSpectrum2D(4, 31)

        self.net1 = common.ConvADMM(n_convs=n_convs, in_channels=1, out_channels=in_channels, kernel_size=kernel_size)
        self.net2 = common.ConvADMM(n_convs=n_convs, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size)
        self.net3 = common.ConvADMM(n_convs=n_convs, in_channels=1, out_channels=in_channels, kernel_size=kernel_size)

        self.admm_blocks = nn.ModuleList(
            [ADMM_Block(n_resblocks, in_channels, n_feats, n_convs, kernel_size) for _ in range(n_admmblocks)]
        )

        self.denoiser = EDSR(n_resblocks=n_resblocks, n_colors=in_channels, n_feats=n_feats)

    def forward(self, y):
        y, w, spectrum_loss = self.weight(y)
        y = y.unsqueeze(1)
        mosaic = y

        v = self.net1(y)
        x = self.net2(self.net3(y) + self.rho * v)
        v = self.denoiser(x)
        u = v - x

        v_list = [v]
        v_list.append(v)
        for block in self.admm_blocks:
            v, u = block(v, u, y)
            v_list.append(v)
        
        return v_list[-1], w, mosaic, spectrum_loss

