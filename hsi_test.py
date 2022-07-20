import argparse
import os
from os.path import exists, join

import torch
import torch.nn as nn
from scipy.io import savemat

import models
from utility import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

prefix = 'DM'

def _parse_str_args(args):
    str_args = args.split(',')
    parsed_args = []
    for str_arg in str_args:
        arg = int(str_arg)
        if arg >= 0:
            parsed_args.append(arg)
    return parsed_args

if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(description='Hyperspectral Image Demosaicking.')
    parser.add_argument('--arch', '-a', metavar='ARCH', required=True,
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names))
    parser.add_argument('--wd', type=float, default=0, help='Weight Decay. Default=0')
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda?')
    parser.add_argument('--no-log', action='store_true', help='disable logger?')
    parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=2021, help='random seed to use. Default=2020')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resumePath', '-rp', type=str, default=None, help='checkpoint to use.')
    parser.add_argument('--test', action='store_true', help='test mode?')
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids')
    parser.add_argument('--use-2dconv', action="store_true", help='whether the network uses 2d convolution?')
    parser.add_argument('--bandwise', action="store_true", help='whether the network handles the input in a band-wise manner?')
    parser.add_argument('--fac', type=str, default='DM0', help='determine the value of fac in the Softmax layer.')
    parser.add_argument('--dataset', type=str, default='icvl', help='determine the testing dataset.')
    parser.add_argument('--no-save', action='store_true', help='saver mode?')

    opt = parser.parse_args()
    opt.gpu_ids = _parse_str_args(opt.gpu_ids)
    print(opt)

    cuda = not opt.no_cuda

    HSI2Tensor = partial(HSI2Tensor, use_2dconv=opt.use_2dconv)
    ImageTransformDataset = partial(ImageTransformDataset, target_transform=HSI2Tensor())

    spectral_downsample_transform = partial(sample16)
    common_transform = lambda x: x
    curves_dir = 'curves20'
    curves = np.loadtxt(curves_dir)

    # Baseline.
    index = []
    if 'base' in opt.arch:
        print("====== into baseline ======")
        index_dir = 'index'
        if os.path.exists(index_dir):
            index = np.loadtxt(index_dir).astype(int)
        else:
            print("!!!error!!!")

    select_transform = partial(channel_selection, curves=curves, index=index)
    # common_transform3 = partial(crop_center, cropx=64, cropy=64)
    if '16channel' in opt.arch:
        print('===== Task 1 ======')
        select_transform = spectral_downsample_transform
        ImageTransformDataset = partial(ImageTransformDataset,
                                    target_transform=Compose([spectral_downsample_transform, HSI2Tensor()]))
    if 'spectrum' in opt.arch:
        print('===== Task 3 ======')
        select_transform = common_transform

    print('==> Preparing data..')

    mat_transforms = Compose([
            select_transform,
            HSI2Tensor()
        ])

    if opt.dataset == 'icvl':
        datadir = '/media/exthdd/datasets/hsi/lzy_data/icvl512_101_gt'
    elif opt.dataset == 'harvard':
        datadir = '/media/exthdd/datasets/hsi/lzy_data/Harvard/test'
    else:
        print('===== REEOR =====')
    mat_dataset = MatDataFromFolder(datadir, size=None)
    # fns = ['urban_norm.mat']
    fns = os.listdir(datadir)
    mat_dataset.filenames = [os.path.join(datadir, fn) for fn in fns]

    mat_dataset = TransformDataset(mat_dataset, LoadMatKey(key='gt'))
    mat_dataset = TransformDataset(mat_dataset, lambda x: x.transpose(2,0,1))

    mat_datasets = ImageTransformDataset(mat_dataset, mat_transforms)
    mat_loaders = DataLoader(
                    mat_datasets,
                    batch_size=1, shuffle=False,
                    num_workers=opt.threads, pin_memory=cuda)

    """Model"""
    print("=> creating model '{}'".format(opt.arch))
    net = models.__dict__[opt.arch]()
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    if len(opt.gpu_ids) > 1:
        from models.sync_batchnorm import DataParallelWithCallback
        net = DataParallelWithCallback(net, device_ids=opt.gpu_ids)

    if cuda:
        net.cuda()
        criterion.cuda()

    """Resume previous model"""
    # Load checkpoint.
    print('==> Resuming from checkpoint %s..' %opt.resumePath)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(opt.resumePath or './checkpoint/%s/%s/model_best.pth'%(opt.arch, prefix))
    # net = nn.DataParallel(net)
    net.load_state_dict(checkpoint['net'])

    fac_prefix = {
        'DM0': np.sqrt(255.0)/8e5,
        'DM' : np.sqrt(399.0)/8e5,
        'DM2': np.sqrt(499.0)/8e5,
        'DM4': np.sqrt(127.0)/8e5,
        'DM5': np.sqrt(199.0)/8e5,
        # for L1 Loss.
        'DM_l1': np.sqrt(899.0)/8e5,
        'DM1_l1': np.sqrt(1047.0)/8e5,
        'DM2_l1': np.sqrt(579.0)/8e5,
    }
    fac = fac_prefix[opt.fac]
    iteration = checkpoint['iteration']
    softmax_alpha = np.float32(1. + (fac * iteration) ** 2)
    

    def torch2numpy(hsi):
        if opt.use_2dconv:
            R_hsi = hsi.data[0].cpu().numpy()
        else:
            R_hsi = hsi.data[0].cpu().numpy()[0,...]
        return R_hsi

    """Testing"""
    def test(test_loader):
        net.eval()
        total_psnr = 0
        total_ssim = 0
        total_sam = 0
        cnt = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if 'hsup' in opt.arch:
                t = torch.zeros([inputs.shape[0], inputs.shape[1], inputs.shape[2]//4, inputs.shape[3]//4])
                for i in range(inputs.shape[1]):
                    t[:,i,:,:] = inputs[:, i, i%4:inputs.shape[-2]:4, i//4:inputs.shape[-1]:4]
                inputs = t
            
            if not opt.no_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                if 'spectrum' in opt.arch:
                    outputs, weight, mosaic, spectrum_loss = net(inputs)
                elif 'alpha' in opt.arch:
                    outputs, weight, mosaic = net(inputs, softmax_alpha)
                else:
                    outputs, weight, mosaic = net(inputs)

            outputs = outputs.cpu()
            targets = targets.cpu()

            psnr,ssim,sam = MSIQA(outputs, targets)
            total_psnr += psnr
            avg_psnr = total_psnr / (batch_idx+1)
            total_ssim += ssim
            avg_ssim = total_ssim/(batch_idx+1)
            total_sam += sam
            avg_sam = total_sam/(batch_idx+1)

            progress_bar(batch_idx, len(test_loader), 'PSNR: %.4f | SSIM: %.4f | SAM: %.4f '
                % (avg_psnr, avg_ssim, avg_sam))


            if '16channel' in opt.arch:
                task = '1'
            elif 'spectrum' in opt.arch:
                task = '3'
            else:
                task = '2'

            if opt.no_save:
                filedir = False
            else:
                filedir = '/home/liangzhiyuan/Code/HSI_DM/log/lzy/result/'+opt.arch+'/'+opt.dataset+'/task'+task+'/'
            if filedir:
                outpath = join(filedir, fns[cnt])
                cnt += 1

                if not exists(filedir):
                    os.makedirs(filedir)

                if not exists(outpath):
                    savemat(outpath, {'pred': torch2numpy(outputs), 'pnsr': psnr, 'ssim': ssim, 'sam': sam})


    test(mat_loaders)


