import os
from os.path import join

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid

import models
from utility import *
from utility.indexes import FFTLoss, MultipleLoss, SAMLoss
from utility.metric import PSNR, SAM, SSIM, MetricTracker

loss_type = {
            'MSE': nn.MSELoss(),
            'L1': nn.L1Loss(),
            # 'L1-SAM': SAMLoss(),
            'L1-SAM':MultipleLoss([nn.L1Loss(), SAMLoss()], weight=[1,1e-3]),
            'FFT': FFTLoss()
        }

class Engine(object):
    def __init__(self, prefix, opt):
        self.prefix = prefix
        self.opt = opt
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.basedir = None
        self.iteration = None
        self.epoch = None
        self.best_psnr = None
        self.best_loss = None
        self.writer = None
        self.gpu_ids = self.opt.gpu_ids
        self.arch = self.opt.arch
        self.scheduler = None
        # self.softmax_alpha = 1.

        self.__setup()


    def __setup(self):
        self.basedir = join('checkpoint', self.opt.arch, self.opt.prefix, self.opt.fac)
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

        self.best_psnr = 0
        self.best_loss = 1e6
        self.epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.iteration = 0

        metrics = {'psnr': PSNR, 'ssim': SSIM, 'sam': SAM}
        self.metric_tracker = MetricTracker(metrics=metrics)

        cuda = not self.opt.no_cuda
        print('Cuda Acess: %d' %cuda)
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        torch.manual_seed(self.opt.seed)
        if cuda:
            torch.cuda.manual_seed(self.opt.seed)

        """Model"""
        print("==> creating model '{}'".format(self.opt.arch))
        self.net = models.__dict__[self.opt.arch]()
        # print(self.net)

        if len(self.opt.gpu_ids) > 1:
            from models.sync_batchnorm import DataParallelWithCallback
            self.net = DataParallelWithCallback(self.net, device_ids=self.opt.gpu_ids)

        self.criterion = loss_type[self.opt.loss]

        if cuda:
            self.net.cuda()

        """Logger Setup"""
        log = not self.opt.no_log
        if log:
            self.writer = get_summary_writer(self.opt.arch, self.opt.prefix)

        """Optimization Setup"""
        if 'base' in self.arch:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.opt.lr, weight_decay=self.opt.wd)
        else:
            weight_params = list(map(id, self.net.weight.parameters()))
            base_params = filter(lambda p: id(p) not in weight_params, self.net.parameters())
            self.optimizer = optim.Adam([
                                        {'params': base_params}, 
                                        {'params': self.net.weight.parameters(), 'lr': self.opt.lr2},
                                        ], lr=self.opt.lr, weight_decay=self.opt.wd)
        
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=5, min_lr=self.opt.min_lr, verbose=True)

        """Resume previous model"""
        if self.opt.resume:
            # Load checkpoint.
            self.load(self.opt.resumePath)
        else:
            print('==> Building model..')


    def __step(self, train, inputs, targets):
        start_time = time.time()

        if train:
            self.optimizer.zero_grad()
        loss_data = 0

        if 'spectrum' in self.arch:
            outputs, weight, mosaic, spectrum_loss = self.net(inputs)
        elif 'alpha' in self.arch:
            outputs, weight, mosaic = self.net(inputs, self.softmax_alpha)
            spectrum_loss = 0
        else:
            outputs, weight, mosaic = self.net(inputs)
            spectrum_loss = 0
        
        weight = weight[0, ...].squeeze()
        mosaic = mosaic[0, ...].squeeze()
        loss = self.criterion(outputs, targets) + spectrum_loss

        
        if train:
            loss.backward()
        loss_data += loss.item()
        
        if train:
            self.optimizer.step()
        
        end_time = time.time()

        return outputs, loss_data, weight, mosaic, end_time-start_time


    def load(self, resumePath=None):
        model_best_path = join(self.basedir, self.prefix, 'model_best.pth')
        if os.path.exists(model_best_path):
            best_model = torch.load(model_best_path)
            self.best_psnr = best_model['psnr']
            self.best_loss = best_model['loss']

        print('==> Resuming from checkpoint %s..' % resumePath)
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resumePath or model_best_path)
        self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.last_epoch = self.epoch


    """Training"""
    def train(self, train_loader):
        print('\nEpoch: %d' % self.epoch)

        self.net.train()

        self.metric_tracker.refresh()
        train_loss = 0      

        fac_prefix = {
            'DM0': np.sqrt(255.0)/8e5,
            'DM' : np.sqrt(399.0)/8e5,
            'DM2': np.sqrt(499.0)/8e5,
            'DM4': np.sqrt(127.0)/8e5,
            'DM5': np.sqrt(199.0)/8e5,
            # for L1 Loss
            'DM_l1': np.sqrt(899.0)/8e5,
            'DM1_l1': np.sqrt(1047.0)/8e5,
            'DM2_l1': np.sqrt(579.0)/8e5,
        }
        fac = fac_prefix[self.opt.fac]
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            self.softmax_alpha = np.float32(1. + (fac * self.iteration) ** 2)
            
            if not self.opt.no_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs, loss_data, weight, mosaic, _ = self.__step(True, inputs, targets)

            self.metric_tracker.update(outputs, targets)
            train_loss += loss_data
            avg_loss = train_loss / (batch_idx + 1)
            
            log_input = inputs[0, ...].squeeze()
            log_output = outputs[0, ...].squeeze()
            if not self.opt.no_log:
                self.writer.add_scalar(join(self.prefix,'train_loss'), loss_data, self.iteration)
                self.writer.add_scalar(join(self.prefix,'train_avg_loss'), avg_loss, self.iteration)
                if self.iteration % 100 == 0:
                    self.writer.add_image(join(self.prefix,'input_image'), make_grid(list(log_input[15:25:2,:,:].split(1,0))), self.iteration, dataformats='CHW')
                    self.writer.add_image(join(self.prefix,'output_image'), make_grid(list(log_output[15:25:2,:,:].split(1,0))), self.iteration, dataformats='CHW')
                if 'alpha' in self.opt.arch:
                    self.writer.add_image(join(self.prefix,'weight'), make_grid(list(weight[:,0:4,0:4].split(1,0))), self.iteration, dataformats='CHW')
                    self.writer.add_image(join(self.prefix,'mosaic_imgae'), mosaic, self.iteration, dataformats='HW')

            self.iteration += 1

            progress_bar(batch_idx, len(train_loader), 'AvgLoss: %.4e | PSNR: %.4f | SSIM: %.4f | SAM: %.4f'
                % (avg_loss, self.metric_tracker.get_all()['psnr'], self.metric_tracker.get_all()['ssim'], self.metric_tracker.get_all()['sam']))

        self.epoch += 1

        if not self.opt.no_log:
            self.writer.add_scalar(join(self.prefix,'train_loss_epoch'), avg_loss, self.epoch)


    """Validation"""
    def validate(self, valid_loader):     
        self.net.eval()

        self.metric_tracker.refresh()
        validate_loss = 0

        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            if not self.opt.no_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs, loss_data, weight, mosaic, _ = self.__step(False, inputs, targets)

            self.metric_tracker.update(outputs, targets)
            validate_loss += loss_data
            avg_loss = validate_loss / (batch_idx + 1)                  
            
            progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f | SSIM: %.4f | SAM: %.4f'
                % (avg_loss, self.metric_tracker.get_all()['psnr'], self.metric_tracker.get_all()['ssim'], self.metric_tracker.get_all()['sam']))
        
        avg_psnr = self.metric_tracker.get('psnr')
        log_input = inputs[0, ...].squeeze()
        log_output = outputs[0, ...].squeeze()
        if not self.opt.no_log:
            self.writer.add_scalar(join(self.prefix,'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(join(self.prefix,'val_psnr_epoch'), avg_psnr, self.epoch)
            if self.iteration % 100 == 0:
                    self.writer.add_image(join(self.prefix,'input_image'), make_grid(list(log_input[15:25:2,:,:].split(1,0))), self.iteration, dataformats='CHW')
                    self.writer.add_image(join(self.prefix,'output_image'), make_grid(list(log_output[15:25:2,:,:].split(1,0))), self.iteration, dataformats='CHW')


        """Save checkpoint"""
        if avg_loss < self.best_loss:
            print('Best Result Saving...')
            model_best_path = join(self.basedir, self.prefix, 'model_best.pth')
            self.save_checkpoint(psnr=avg_psnr, loss=avg_loss, model_out_path=model_best_path)
            self.best_psnr = avg_psnr
            self.best_loss = avg_loss
        
        return avg_psnr, avg_loss


    def save_checkpoint(self, psnr, loss, model_out_path=None):
        if not model_out_path:
            model_out_path = join(self.basedir, self.prefix, "model_epoch_%d_%d.pth" %(self.epoch, self.iteration))
        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'psnr': psnr,
            'loss': loss,
            'epoch': self.epoch,
            'iteration': self.iteration,
        }

        if not os.path.isdir('checkpoint'):
            os.makedirs('checkpoint')
        if not os.path.isdir(join(self.basedir, self.prefix)):
            os.makedirs(join(self.basedir, self.prefix))
        
        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
