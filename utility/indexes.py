import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from functools import partial


class Bandwise(object): 
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-3]
        # C = X.shape[-1]
        bwindex = []
        for ch in range(C):
            x = torch.squeeze(X[...,ch,:,:].data).cpu().numpy()
            y = torch.squeeze(Y[...,ch,:,:].data).cpu().numpy()
            # x = np.squeeze(X[...,ch])
            # y = np.squeeze(Y[...,ch])
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex


cal_bwpsnr = Bandwise(partial(peak_signal_noise_ratio, data_range=1))
cal_bwssim = Bandwise(structural_similarity)


def cal_sam(X, Y, eps=1e-8):
    X = torch.squeeze(X.data).cpu().numpy()
    Y = torch.squeeze(Y.data).cpu().numpy()
    # X = np.squeeze(X)
    # Y = np.squeeze(Y)
    tmp = (np.sum(X*Y, axis=0) + eps) / (np.sqrt(np.sum(X**2, axis=0)) + eps) / (np.sqrt(np.sum(Y**2, axis=0)) + eps)
    return np.mean(np.real(np.arccos(tmp)))


def MSIQA(X, Y):
    psnr = np.mean(cal_bwpsnr(X, Y))
    ssim = np.mean(cal_bwssim(X, Y))
    sam = cal_sam(X, Y)
    return psnr, ssim, sam


"""Depreciated"""
def cal_psnr(mse):
    return 10 * np.log10(1 / mse)


def mpsnr(bwmse, verbose=False):
    psnrs = []
    for mse in bwmse:
        cur_psnr = cal_psnr(mse)
        psnrs.append(cur_psnr)
    
    if not verbose:
        return np.mean(psnrs)
    else:
        return np.mean(psnrs), psnrs


# def cal_samloss(X, Y):
#     X = X.cpu()
#     Y = Y.cpu()
#     print(X.shape)
#     sam = 0
#     for i in range(X.shape[0]):
#         sam += 1 - 0.5 * torch.norm(X[i,...]-Y[i,...], p=2)
#     sam = sam / X.shape[0]
#     print(sam)
#     print(0.0001 * torch.arccos(sam))
#     # tmp = (np.sum(X*Y, axis=0) + eps) / (np.sqrt(np.sum(X**2, axis=0)) + eps) / (np.sqrt(np.sum(Y**2, axis=0)) + eps)
#     return 0.0001


class SAMLoss(nn.Module):
    def __init__(self, size_average = False):
        super(SAMLoss, self).__init__()

    def forward(self, img_base, img_out):
        sum1 = torch.sum(img_base * img_out, 1)
        sum2 = torch.sum(img_base * img_base, 1)
        sum3 = torch.sum(img_out * img_out, 1)
        t = (sum2 * sum3) ** 0.5
        numlocal = torch.gt(t, 0)
        num = torch.sum(numlocal)
        t = sum1 / t
        angle = torch.acos(t)
        sumangle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle).sum()
        if num == 0:
            averangle = sumangle
        else:
            averangle = sumangle / num
        SAM = averangle * 180 / math.pi
        return SAM


class FFTLoss(nn.Module):
    def forward(self, outputs, targets):
        o_fft = fft.fftn(outputs, dim=(-1, -2))
        t_fft = fft.fftn(targets, dim=(-1, -2))
        return torch.mean(torch.pow(torch.abs(o_fft-t_fft), 2))



class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1/len(self.losses)] * len(self.losses)
    
    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            l = loss(predict, target)
            total_loss += l * weight
        return total_loss

    def extra_repr(self):
        return 'weight={}'.format(self.weight)