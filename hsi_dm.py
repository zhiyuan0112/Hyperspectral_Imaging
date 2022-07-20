import argparse
import os

import models
from hsi_setup import Engine
from utility import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


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
    parser.add_argument('--batchSize', '-b', type=int, default=16, help='training batch size. Default=16')
    parser.add_argument('--nEpochs', '-n', type=int, default=100, help='number of epochs to train for. Default=50')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning Rate. Default=1e-4.')
    parser.add_argument('--lr2', type=float, default=1e-1, help='Learning Rate. Default=1e-1.')
    parser.add_argument('--min-lr', '-mlr', type=float, default=5e-6, help='Minimal Learning Rate. Default=1e-5.')
    parser.add_argument('--ri', type=int, default=1 , help='Record interval. Default=1')
    parser.add_argument('--wd', type=float, default=0, help='Weight Decay. Default=0')
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda?')
    parser.add_argument('--no-log', action='store_true', help='disable logger?')
    parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=2021, help='random seed to use. Default=2021')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resumePath', '-rp', type=str, default=None, help='checkpoint to use.')
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids')
    parser.add_argument('--prefix', '-p', type=str, default='icvl', help='distinguish checkpoint')
    parser.add_argument('--datadir', '-d', type=str, default='/media/exthdd/datasets/hsi/lzy_data/ICVL64_31_100_20210621.db', help='path to training set')
    parser.add_argument('--fac', type=str, default='DM0', help='determine the value of fac in the Softmax layer.')
    parser.add_argument('--loss', type=str, default='L1-SAM', help='determine the type of Loss Function.')

    opt = parser.parse_args()
    opt.gpu_ids = _parse_str_args(opt.gpu_ids)
    print(opt)

    cuda = not opt.no_cuda

    """Setup Engine"""
    engine = Engine(opt.prefix, opt)

    use_2dconv = engine.net.module.use_2dconv if len(opt.gpu_ids) > 1 else engine.net.use_2dconv
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=use_2dconv)
    ImageTransformDataset = partial(ImageTransformDataset, target_transform=HSI2Tensor())
    
    spectral_downsample_transform = partial(sample16)
    common_transform = lambda x: x

    curves_dir = 'curves20'
    curves = np.loadtxt(curves_dir)
    # curves = np.loadtxt(curves_dir) if os.path.exists(curves_dir) else generate_spectral_curve(ori_ndim=31, new_ndim=20)
    # np.savetxt(curves_dir, curves)

    # Baseline.
    index = []
    if 'base' in opt.prefix or 'base' in opt.arch:
        print("====== into baseline ======")
        index_dir = 'index'
        if os.path.exists(index_dir):
            index = np.loadtxt(index_dir).astype(int)
        else:
            index = np.random.randint(0, 20, size=16)
            index = np.sort(index)                            # base
            np.savetxt('index', index)

    print("====== length of index:", len(index))
    select_transform = partial(channel_selection, curves=curves, index=index)
    if '16channel' in opt.arch:
        print('===== Task 1 ======')
        select_transform = spectral_downsample_transform
        ImageTransformDataset = partial(ImageTransformDataset,
                                    target_transform=Compose([spectral_downsample_transform, HSI2Tensor()]))
    if 'spectrum' in opt.arch:
        print('===== Task 3 ======')
        select_transform = common_transform     

    train_transform = Compose([
        select_transform,
        HSI2Tensor()
    ])
    valid_transform = Compose([
        select_transform,
        HSI2Tensor()
    ])

    print('==> Preparing data..')

    icvl_64_31 = LMDBDataset(opt.datadir)
    """Split patches dataset into training, validation parts"""
    icvl_64_31 = TransformDataset(icvl_64_31, common_transform)

    icvl_64_31_T, icvl_64_31_V = get_train_valid_dataset(icvl_64_31, 1000)  # 1000 for icvl and 500 for harvard

    train_dataset = ImageTransformDataset(icvl_64_31_T, train_transform)
    valid_dataset = ImageTransformDataset(icvl_64_31_V, valid_transform)

    icvl_64_31_TL = DataLoader(train_dataset,
                    batch_size=opt.batchSize, shuffle=True,
                    num_workers=opt.threads, pin_memory=cuda)

    icvl_64_31_VL = DataLoader(valid_dataset,
                    batch_size=1, shuffle=False,
                    num_workers=opt.threads, pin_memory=cuda)

    
    adjust_learning_rate(engine.optimizer, opt.lr, opt.lr2)
    while engine.epoch < opt.nEpochs:
        engine.train(icvl_64_31_TL)
        psnr, loss = engine.validate(icvl_64_31_VL)
            
        engine.scheduler.step(loss)
        lrs = display_learning_rate(engine.optimizer)
        if engine.epoch % opt.ri == 0:
            engine.save_checkpoint(psnr, loss)

