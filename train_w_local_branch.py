# -*- coding: utf-8 -*-
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import torch
from torch import optim
import torch.nn.functional as F
from torch.backends import cudnn
import numpy as np
import os
import time
import datetime
import argparse
import random

from data.brain import get_loaders
from utils.util import check_dirs, parse_image_name
from py_ssim import ssim
from PIL import Image
from net.unet_concat import UNet_middle_concat, Discriminator, init_weights


class Solver:
    def __init__(self, data_files, opt):
        self.opt = opt
        self.best_ssim = 0
        self.best_epoch = 0

        self.start_epoch = self.opt.start_epoch

        # Data Loader.
        self.phase = self.opt.phase
        self.selected_modality = self.opt.selected_modality
        self.image_size = self.opt.image_size
        self.batch_size = self.opt.batch_size
        self.num_workers = self.opt.num_workers
        loaders = get_loaders(data_files, self.selected_modality, self.batch_size,
                              self.num_workers, self.image_size)
        self.loaders = {x: loaders[x] for x in ('train', 'val', 'test')}

        # Model Configurations.
        self.c_dim = len(self.selected_modality)
        self.d_conv_dim = self.opt.d_conv_dim
        self.d_repeat_num = self.opt.d_repeat_num

        self.lambda_cls = self.opt.lambda_cls
        self.lambda_rec = self.opt.lambda_rec
        self.lambda_gp = self.opt.lambda_gp
        self.lambda_real = self.opt.lambda_real
        self.lambda_fake = self.opt.lambda_fake
        self.lambda_local = self.opt.lambda_local

        # Train Configurations.
        self.max_epoch = self.opt.max_epoch
        self.decay_epoch = self.opt.decay_epoch
        self.g_lr = self.opt.g_lr
        self.min_g_lr = self.opt.min_g_lr
        self.d_lr = self.opt.d_lr
        self.min_d_lr = self.opt.min_d_lr
        self.beta1 = self.opt.beta1
        self.beta2 = self.opt.beta2
        self.n_critic = self.opt.n_critic

        self.test_epoch = self.opt.test_epoch
        self.use_tensorboard = self.opt.use_tensorboard
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.checkpoint_dir = self.opt.checkpoint_dir
        self.log_dir = os.path.join(self.checkpoint_dir, 'logs')
        self.sample_dir = os.path.join(self.checkpoint_dir, 'sample_dir')
        self.model_save_dir = os.path.join(
            self.checkpoint_dir, 'model_save_dir')
        self.result_dir = os.path.join(self.checkpoint_dir, 'result_dir')
        check_dirs([self.log_dir, self.sample_dir,
                   self.model_save_dir, self.result_dir])

        self.log_step = self.opt.log_step
        self.val_epoch = self.opt.val_epoch
        self.lr_update_epoch = self.opt.lr_update_epoch

        self.G = None
        self.D = None
        self.localD = None
        
        #  WT:0  TC:1    ET:2
        self.tumor_type = 0     # WT
        self.build_model()
        if self.start_epoch != 0 and self.phase != 'test':
            self.restore_model(self.start_epoch, onlyG=False)
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        self.G = UNet_middle_concat(used_modality_num=1)

        self.G.to(self.device)
        init_weights(self.G, init_type='kaiming', init_gain=0.02)

        self.D = Discriminator(
            self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
        self.D.to(self.device)
        init_weights(self.D, init_type='kaiming', init_gain=0.02)

        self.localD = Discriminator(
            self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num, input_dim=1)
        self.localD.to(self.device)
        init_weights(self.localD, init_type='kaiming', init_gain=0.02)
        
        self.g_optimizer = optim.Adam(self.G.parameters(), self.g_lr, [
                                      self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(self.D.parameters(), self.d_lr, [
                                      self.beta1, self.beta2])
        self.local_d_optimizer = optim.Adam(self.localD.parameters(), self.d_lr, [
                                      self.beta1, self.beta2])

    def build_tensorboard(self):
        self.writer = SummaryWriter(self.log_dir)

    def restore_model(self, start_epoch, onlyG=True):
        if onlyG == False:
            print('Resume the trained models from step {}...'.format(start_epoch), flush=True)
            G_path = os.path.join(self.model_save_dir,
                                  '{}-G.ckpt'.format(start_epoch))
            D_path = os.path.join(self.model_save_dir,
                                  '{}-D.ckpt'.format(start_epoch))
            local_D_path = os.path.join(self.model_save_dir,
                                  '{}-lD.ckpt'.format(start_epoch))
         
            GO_path = os.path.join(self.model_save_dir,
                                  '{}-GO.ckpt'.format(start_epoch))
            DO_path = os.path.join(self.model_save_dir,
                                  '{}-DO.ckpt'.format(start_epoch))
            local_DO_path = os.path.join(self.model_save_dir,
                                  '{}-lDO.ckpt'.format(start_epoch))
            self.G.load_state_dict(torch.load(
                G_path, map_location=lambda storage, loc: storage))
            self.D.load_state_dict(torch.load(
                D_path, map_location=lambda storage, loc: storage))
            self.localD.load_state_dict(torch.load(
                local_D_path, map_location=lambda storage, loc: storage))
            
            self.g_optimizer.load_state_dict(torch.load(
                GO_path, map_location=lambda storage, loc: storage))
            self.d_optimizer.load_state_dict(torch.load(
                DO_path, map_location=lambda storage, loc: storage))
            self.local_d_optimizer.load_state_dict(torch.load(
                local_DO_path, map_location=lambda storage, loc: storage))
        else:
            print('Loading the trained models from step {}...'.format(
                start_epoch), flush=True)
            G_path = os.path.join(self.model_save_dir,
                                  '{}-G.ckpt'.format(start_epoch))
            self.G.load_state_dict(torch.load(
                G_path, map_location=lambda storage, loc: storage))

    def save_model(self, save_iters):
        G_path = os.path.join(self.model_save_dir,
                              '{}-G.ckpt'.format(save_iters))
        D_path = os.path.join(self.model_save_dir,
                              '{}-D.ckpt'.format(save_iters))
        local_D_path = os.path.join(self.model_save_dir,
                                  '{}-lD.ckpt'.format(save_iters))
        GO_path = os.path.join(self.model_save_dir,
                              '{}-GO.ckpt'.format(save_iters))
        DO_path = os.path.join(self.model_save_dir,
                              '{}-DO.ckpt'.format(save_iters))  
        local_DO_path = os.path.join(self.model_save_dir,
                                  '{}-lDO.ckpt'.format(save_iters))                  
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        torch.save(self.localD.state_dict(), local_D_path)
        torch.save(self.g_optimizer.state_dict(), GO_path)
        torch.save(self.d_optimizer.state_dict(), DO_path)
        torch.save(self.local_d_optimizer.state_dict(), local_DO_path)
        print('Saved model checkpoints into {}...'.format(
            self.model_save_dir), flush=True)

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.local_d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.local_d_optimizer.zero_grad()

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def norm(self, x):
        out = x * 2 - 1
        return out.clamp_(-1, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onenot(self, labels, dim):
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    @staticmethod
    def classification_loss(logit, target):
        return F.cross_entropy(logit, target)
        
    def train(self):

        g_lr = self.g_lr
        d_lr = self.d_lr
        loader = self.loaders['train']

        start_epoch = self.start_epoch

        print('\nStart training...', flush=True)
        start_time = time.time()
        for epoch in range(start_epoch, self.max_epoch):
            self.G.train()
            self.D.train()
            self.localD.train()
            for i, (image, labels, vec_org, label_org, name, GT_img) in enumerate(loader):

                cur_step = epoch * len(loader) + i
                
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]  

                vec_trg = self.label2onenot(label_trg, self.c_dim)
                image = image.to(self.device)

                for kk in range(self.c_dim):
                    GT_img[kk] = GT_img[kk].to(self.device)
                
                labels[self.tumor_type] = labels[self.tumor_type].unsqueeze(1).float().to(self.device)
                
                x_local_gather = self.norm(labels[self.tumor_type] * self.denorm(image))


                label_org = label_org.to(self.device)
                label_trg = label_trg.to(self.device)
                vec_org = vec_org.to(self.device)
                vec_trg = vec_trg.to(self.device)
               
                out_src_real_global, out_cls_real_global = self.D(image)
                d_loss_real = - torch.mean(out_src_real_global)
                d_loss_cls = self.classification_loss(out_cls_real_global, label_org)

                out_src_real_local, out_cls_real_local = self.localD(x_local_gather)
                d_loss_real_local = - torch.mean(out_src_real_local)
                d_loss_cls_local = self.classification_loss(out_cls_real_local, label_org)

                x_fake, x_local_target_gather = self.G(image, x_local_gather, vec_trg)

                out_src_fake_global, _ = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src_fake_global)

                out_src_fake_local, _ = self.localD(x_local_target_gather.detach())
                d_loss_fake_local = torch.mean(out_src_fake_local)

                alpha = torch.rand(image.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * image.data + (1 - alpha)
                         * x_fake.data).requires_grad_(True)
                out_src_gp_global, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src_gp_global, x_hat)

                beta = torch.rand(x_local_gather.size(0), 1, 1, 1).to(self.device)
                t_hat = (beta * x_local_gather.data + (1 - beta)
                         * x_local_target_gather.data).requires_grad_(True)
                out_src_gp_local, _ = self.localD(t_hat)
                d_loss_gp_local = self.gradient_penalty(out_src_gp_local, t_hat)
                
                d_loss = self.lambda_real * d_loss_real + self.lambda_fake * d_loss_fake + \
                            self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
                
                locald_loss = self.lambda_real *  d_loss_real_local + self.lambda_fake * \
                    d_loss_fake_local + self.lambda_cls * d_loss_cls_local + self.lambda_gp * d_loss_gp_local

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                self.reset_grad()
                locald_loss.backward()
                self.local_d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/d_loss'] = d_loss.item()
                loss['D/d_loss_real'] = d_loss_real.item()
                loss['D/d_loss_fake'] = d_loss_fake.item()
                loss['D/d_loss_cls'] = d_loss_cls.item()
                loss['D/d_loss_gp'] = d_loss_gp.item()
                loss['D/d_loss_real_local'] = d_loss_real_local.item()
                loss['D/d_loss_fake_local'] = d_loss_fake_local.item()
                loss['D/d_loss_cls_local'] = d_loss_cls_local.item()
                loss['D/d_loss_gp_local'] = d_loss_gp_local.item()

                if (i + 1) % self.n_critic == 0:
                    x_fake, x_local_target_gather = self.G(image, x_local_gather, vec_trg)
                    out_src_fake_global2, out_cls_fake_global = self.D(x_fake)
                    g_loss_fake = - torch.mean(out_src_fake_global2)
                    g_loss_cls = self.classification_loss(out_cls_fake_global, label_trg)

                    out_src_fake_local2, out_cls_fake_local = self.localD(x_local_target_gather)
                    g_loss_fake_local = - torch.mean(out_src_fake_local2)
                    g_loss_cls_local = self.classification_loss(out_cls_fake_local, label_trg)
                    
                    x_target_local_gather = self.norm(labels[self.tumor_type] * self.denorm(x_fake))                
                    x_rec, x_local_rec_gather = self.G(x_fake, x_target_local_gather, vec_org)
                    
                    g_loss_rec = torch.mean(torch.abs(image - x_rec))

                    g_loss_local = torch.mean(
                        torch.abs(x_local_target_gather - x_target_local_gather)) + torch.mean(
                        torch.abs(x_local_gather - x_local_rec_gather))
                    
                    g_loss = self.lambda_fake * (g_loss_fake + g_loss_fake_local) + self.lambda_cls * \
                        (g_loss_cls + g_loss_cls_local) + self.lambda_rec * g_loss_rec +  \
                            self.lambda_local * g_loss_local

                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    loss['G/g_loss'] = g_loss.item()
                    loss['G/g_loss_fake'] = g_loss_fake.item()
                    loss['G/g_loss_cls'] = g_loss_cls.item()
                    loss['G/g_loss_fake_local'] = g_loss_fake_local.item()
                    loss['G/g_loss_cls_local'] = g_loss_cls_local.item()
                    loss['G/g_loss_rec'] = g_loss_rec.item()
                    loss['G/g_loss_local'] = g_loss_local.item()

                if (cur_step + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    line = "Elapsed [{}], Epoch [{}/{}], Iterations [{}]".format(et, epoch + 1, self.max_epoch,
                                                                                 cur_step)
                    for k, v in loss.items():
                        line += ", {}: {:.4f}".format(k, v)
                        if self.use_tensorboard:
                            self.writer.add_scalar(k, v, (cur_step + 1))
                    print(line, flush=True)

            if (epoch + 1) % self.val_epoch == 0:
                print()
                self.val(epoch + 1)
                self.save_model(epoch + 1)
                print()

            # Decay learning rates.
            if (epoch + 1) % self.lr_update_epoch == 0 and (epoch + 1) > (self.max_epoch - self.decay_epoch):
                g_dlr = (self.g_lr - self.min_g_lr) / \
                    (self.decay_epoch / self.lr_update_epoch)
                g_lr = self.g_lr - g_dlr * \
                    (epoch + 1 - (self.max_epoch - self.decay_epoch)) / \
                    self.lr_update_epoch
                d_dlr = (self.d_lr - self.min_d_lr) / \
                    (self.decay_epoch / self.lr_update_epoch)
                d_lr = self.d_lr - d_dlr * \
                    (epoch + 1 - (self.max_epoch - self.decay_epoch)) / \
                    self.lr_update_epoch
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(
                    g_lr, d_lr), flush=True)
            if self.use_tensorboard:
                self.writer.add_scalar('G/g_lr', g_lr, epoch + 1)
                self.writer.add_scalar('D/d_lr', d_lr, epoch + 1)
        return self.best_epoch

    def val(self, epoch):
        self.G.eval()
        loader = self.loaders['val']

        print('Start Valuing at iter {}...'.format(epoch))
        vis_index = []
        for k in range(10):
            vis_index.append(random.randint(0, len(loader) - 1))
        ssim_val = 0
        num = 0
        with torch.no_grad():
            for i, (x, labels, vec, cls, name, GT_img) in enumerate(loader):
                vis_list = [self.denorm(x).cpu()] 
                x = x.to(self.device)
                vec = vec.to(self.device)

                labels[self.tumor_type] = labels[self.tumor_type].unsqueeze(1).float().to(self.device)
                x_local_gather = self.norm(labels[self.tumor_type] * self.denorm(x))
                vis_list.append(self.denorm(x_local_gather).cpu())

                for kk in range(self.c_dim):
                    GT_img[kk] = GT_img[kk].to(self.device)
                for c in range(self.c_dim):
                    vis_list.append(self.denorm(GT_img[c]).cpu())

                for c in range(self.c_dim):
                    fake_cls = torch.zeros(cls.size())
                    fake_cls.fill_(c)
                    fake_vec = self.label2onenot(
                        fake_cls, self.c_dim).to(self.device)
                    x_fake, x_local_target_gather = self.G(x, x_local_gather, fake_vec)
                    vis_list.append(self.denorm(x_fake).cpu())
                    vis_list.append(self.denorm(x_local_target_gather).cpu())
                    ssim_val += ssim(self.denorm(GT_img[c]),
                                     self.denorm(x_fake), size_average=True).item()               
                num += 1
                if i in vis_index:
                    vis_list = torch.cat(vis_list, dim=3)
                    mod, _, _, _, _ = parse_image_name(name[0])
                    sample_path = os.path.join(
                        self.sample_dir, '{}-images-{}-{}.jpg'.format(epoch, mod, i))
                    save_image(vis_list.data.cpu(),
                               sample_path, nrow=1, padding=0)

        ssim_val = ssim_val / (self.c_dim * num)
        if ssim_val > self.best_ssim:
            self.best_epoch = epoch
            self.best_ssim = ssim_val
        # Log.
        print_str = 'val {} done, cur ssim:{}, best ssim{}, best_epoch{}'.format(epoch, ssim_val, self.best_ssim,
                                                                                 self.best_epoch)
        print(print_str, flush=True)
        if self.use_tensorboard:
            self.writer.add_scalar('val/ssim', ssim_val, epoch)

    def infer(self, epoch):
        if self.phase == 'test':
            save_dir = os.path.join(self.result_dir, str(epoch))
        else:
            save_dir = os.path.join(self.result_dir, str(epoch) + '_trans')
        check_dirs(save_dir)
        self.restore_model(epoch)
        self.G.eval()

        print('Start Testing at iter {}...'.format(epoch), flush=True)
        with torch.no_grad():
            for i, (x, labels, vec, cls, names, _) in enumerate(self.loaders['test']):
                x = x.to(self.device)
                vec = vec.to(self.device)
          
                labels[self.tumor_type] = labels[self.tumor_type].unsqueeze(1).float().to(self.device)
                x_local_gather = self.norm(labels[self.tumor_type] * self.denorm(x))

                for c in range(self.c_dim):
                    fake_cls = torch.zeros(cls.size())
                    fake_cls.fill_(c)
                    fake_vec = self.label2onenot(
                        fake_cls, self.c_dim).to(self.device)
                    x_fake, _ = self.G(x, x_local_gather, fake_vec)
                    x_fake = self.denorm(x_fake).cpu().numpy()
                    for b in range(x_fake.shape[0]):
                        mod, pid, index, _, _ = parse_image_name(names[b])
                        fake_img = x_fake[b][0]
                        img = Image.fromarray(
                            np.uint8(fake_img * 255).astype('uint8'))
                        result_path = os.path.join(save_dir,
                                                   '{}_{}_{}_{}.png'.format(mod, self.selected_modality[c], pid, index))
                        img.save(result_path)
        return
if __name__ == '__main__':
    cudnn.benchmark = True

    args = argparse.ArgumentParser()
    args.add_argument('--train_list', type=str, default='train.txt')
    args.add_argument('--val_list', type=str, default='val.txt')
    args.add_argument('--test_list', type=str, default='test.txt')

    # Data Loader.
    args.add_argument('--phase', type=str)
    args.add_argument('--selected_modality', nargs='+',
                      default=['flair', 't1', 't1ce', 't2'])
    args.add_argument('--image_size', type=int, default=128)
    args.add_argument('--batch_size', type=int, default=8)
    args.add_argument('--num_workers', type=int, default=6)

    # Model configurations.
    args.add_argument('--d_conv_dim', type=int, default=64)
    args.add_argument('--d_repeat_num', type=int, default=6)

    # Lambda.
    args.add_argument('--lambda_cls', type=float, default=1)
    args.add_argument('--lambda_rec', type=float, default=10)
    args.add_argument('--lambda_gp', type=float, default=10)
    args.add_argument('--lambda_real', type=float, default=1)
    args.add_argument('--lambda_fake', type=float, default=1)
    args.add_argument('--lambda_local', type=float, default=10)

    # Train configurations.
    args.add_argument('--max_epoch', type=int, default=100)
    args.add_argument('--decay_epoch', type=int, default=50)
    args.add_argument('--start_epoch', type=int, default=0)
    args.add_argument('--g_lr', type=float, default=1e-4)
    args.add_argument('--min_g_lr', type=float, default=1e-6)
    args.add_argument('--d_lr', type=float, default=1e-4)
    args.add_argument('--min_d_lr', type=float, default=1e-6)
    args.add_argument('--beta1', type=float, default=0.9)
    args.add_argument('--beta2', type=float, default=0.999)
    args.add_argument('--seed', type=int, default=1234)
    args.add_argument('--n_critic', type=int, default=5)   

    # Test configurations.
    args.add_argument('--test_epoch', nargs='+', default=[100])

    # Miscellaneous.
    args.add_argument('--use_tensorboard', type=bool, default=True)
    args.add_argument('--device', type=bool, default=True)
    args.add_argument('--gpu_id', type=str, default='0')

    # Directories.
    args.add_argument('--checkpoint_dir', type=str, default='checkpoint')

    # Step size.
    args.add_argument('--log_step', type=int, default=10)
    args.add_argument('--val_epoch', type=int, default=5)
    args.add_argument('--lr_update_epoch', type=int, default=1)

    args = args.parse_args()
    print('-----Config-----')
    for k, v in sorted(vars(args).items()):
        print('%s:\t%s' % (str(k), str(v)), flush=True)
    print('-------End------\n')

    # Set Random Seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    data_files = dict(train=args.train_list,
                      val=args.val_list, test=args.test_list)
    solver = Solver(data_files, args)

    if args.phase == 'train':
        best_epoch = solver.train()
        if best_epoch not in args.test_epoch:
            args.test_epoch.append(best_epoch)
    elif args.phase == 'test':
        for test_iter in args.test_epoch:
            test_iter = int(test_iter)
            solver.infer(test_iter)
            print()
    print('Done!')
