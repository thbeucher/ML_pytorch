import os
import json
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from copy import deepcopy
from tabulate import tabulate
from torch.autograd import grad
from collections import defaultdict
from itertools import product, chain
from torch.utils.data import DataLoader
from pytorch_msssim import SSIM, MS_SSIM
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from vision_transformer.cvt import CvT
from vision_transformer.vit import ViT
from vision_transformer.mae import MaskedAutoencoderViT


class NoiseLayer(nn.Module):
  """
  A layer that conditionally adds Gaussian noise to the input tensor.
  """
  def __init__(self, p=0.5, std=0.1):
    super().__init__()
    self.p = p
    self.std = std

  def forward(self, x, p=None, std=None):
    p = self.p if p is None else p
    std = self.std if std is None else std
    if self.training and p > 0 and std > 0:
      # Check if we should apply noise in this forward pass
      if random.random() < p:
        # Add Gaussian noise
        noise = torch.randn_like(x) * std
        # Add noise and clip the values to the valid [0, 1] range
        noisy_x = torch.clamp(x + noise, 0., 1.)
        return noisy_x
    return x


class Critic(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
      nn.Flatten(),
      nn.Linear(256*4*4, 1),
    )

  def forward(self, x):
    return self.net(x).view(-1)


class CNNEncoder(nn.Module):
  def __init__(self, add_noise=False, noise_p=0.5, noise_std=0.1, add_attn=False):
    super().__init__()
    self.add_noise = add_noise
    self.add_attn = add_attn

    self.down1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(True))
    self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True))
    self.down3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True))

    if add_noise:
      self.noise_layer = NoiseLayer(p=noise_p, std=noise_std)
    
    if add_attn:
      self.attn1 = ViT(image_size=16, patch_size=4, dim=64, depth=2, heads=4, mlp_dim=128, dim_head=32, channels=64)
      self.attn2 = ViT(image_size=8, patch_size=4, dim=64, depth=2, heads=4, mlp_dim=128, dim_head=32, channels=128)

  def forward(self, x):
    d1 = self.down1(x)  # [B, 3, 32, 32] -> [B, 64, 16, 16]
    if self.add_noise:
      d1 = self.noise_layer(d1)
    
    if self.add_attn:
      d1 = self.attn1(d1)
    
    d2 = self.down2(d1)  #               -> [B, 128, 8, 8]
    if self.add_noise:
      d2 = self.noise_layer(d2)
    
    if self.add_attn:
      d2 = self.attn2(d2)
    
    d3 = self.down3(d2)  #               -> [B, 256, 4, 4]
    if self.add_noise:
      d3 = self.noise_layer(d3)
    
    return d1, d2, d3


class CNNDecoder(nn.Module):
  def __init__(self, add_attn=True):
    super().__init__()
    self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True))
    self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True))
    self.up3 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid())

    self.add_attn = add_attn
    if add_attn:
      self.attn1 = ViT(image_size=8, patch_size=4, dim=64, depth=2, heads=4, mlp_dim=128, dim_head=32, channels=128)
      self.attn2 = ViT(image_size=16, patch_size=4, dim=64, depth=2, heads=4, mlp_dim=128, dim_head=32, channels=64)
  
  def forward(self, d3, d2=None, d1=None):
    u1 = self.up1(d3)   #                -> [B, 128, 8, 8]
    if self.add_attn:
      u1 = self.attn1(u1)
    if d2 is not None:  # skip connection
      u1 = u1 + d2

    u2 = self.up2(u1)   #                -> [B, 64, 16, 16]
    if self.add_attn:
      u2 = self.attn2(u2)
    if d1 is not None:  # skip connection
      u2 = u2 + d1

    u3 = self.up3(u2)  #                -> [B, 3, 32, 32]
    return u3


class CNNAE(nn.Module):
  CONFIG = {'skip_connection': True,
            'linear_bottleneck': False,
            'add_noise_bottleneck': False,
            'add_noise_encoder': False,
            'add_enc_attn': False,
            'add_dec_attn': True,
            'noise_prob': 0.5,
            'noise_std': 0.1,
            'latent_dim': 128}
  def __init__(self, config={}):
    super().__init__()
    self.config = {**CNNAE.CONFIG, **config}

    self.down = CNNEncoder(add_noise=self.config['add_noise_encoder'],
                           noise_p=self.config['noise_prob'],
                           noise_std=self.config['noise_std'],
                           add_attn=self.config['add_enc_attn'])

    if self.config['linear_bottleneck']:
      self.embedder = nn.Linear(256*4*4, self.config['latent_dim'])      # H=32,W=32 for cifar10
      self.fc_dec = nn.Linear(self.config['latent_dim'], 256*4*4)

    self.up = CNNDecoder(add_attn=self.config['add_dec_attn'])

    if self.config['add_noise_encoder'] or self.config['add_noise_bottleneck']:
      self.noise_layer = NoiseLayer(p=self.config['noise_prob'], std=self.config['noise_std'])
  
  def forward(self, x, return_enc=False):
    d1, d2, d3 = self.down(x)

    if self.config['linear_bottleneck']:
      d3 = self.fc_dec(self.embedder(d3.flatten(1))).view(d3.shape)
    
    if self.config['add_noise_bottleneck']:
      d3 = self.noise_layer(d3)

    u3 = self.up(d3,
                 d2 if self.config['skip_connection'] else None,
                 d1 if self.config['skip_connection'] else None)

    if return_enc:
      u3, d3
    return u3


class MixedAE(nn.Module):
  CONFIG = {'add_dec_attn': True, 'skip_connection': True}
  def __init__(self, config={}):
    super().__init__()
    self.config = {**MixedAE.CONFIG, **config}
    self.encoder = CvT()
    self.decoder = CNNDecoder(add_attn=self.config['add_dec_attn'])
  
  def forward(self, x):
    z1, z2, z3 = self.encoder(x)
    return self.decoder(z3,
                        z2 if self.config['skip_connection'] else None,
                        z1 if self.config['skip_connection'] else None)


class CNNAETrainer:
  '''ConvNeuralNet_AutoEncoder_Trainer'''
  CONFIG = {'lr':                1e-4,
            'n_epochs':          30,
            'batch_size':        64,
            'data_dir':          '../../../gpt_tests/data/',
            'save_dir':          'cifar10_exps/',
            'exp_name':          'cnn_ae_best',
            'log_dir':           'runs/',
            'save_rec_every':    5,
            'eval_every':        5,
            'use_tf_logger':     True,
            'seed':              42,
            'save_model_train':  True,
            'lambda_mse': 0.8,
            'lambda_ssim': 0.2,
            'model_config': {'skip_connection': True,
                             'linear_bottleneck': False,
                             'add_noise_bottleneck': False,
                             'add_noise_encoder': False,
                             'add_enc_attn': False,
                             'add_dec_attn': True}}
  def __init__(self, config={}):
    self.config = {**CNNAETrainer.CONFIG, **config}
    self.device = torch.device('cuda' if torch.cuda.is_available() else
                               'mps' if torch.backends.mps.is_available() else
                               'cpu')
    self.set_seed()
    self.instanciate_model()
    self.set_dataloader()
    self.set_optimizers_n_criterions()
    self.set_logger()  # tensorboard logger create the folders used by dump_config and save_model
    self.dump_config()
  
  def set_logger(self, exp_name=None):
    exp_name = self.config['exp_name'] if exp_name is None else exp_name
    save_dir_run = os.path.join(self.config['save_dir'], exp_name, self.config['log_dir'])
    self.tf_logger = SummaryWriter(save_dir_run) if self.config['use_tf_logger'] else None
  
  def dump_config(self):
    with open(os.path.join(self.config['save_dir'],
                           self.config['exp_name'],
                           f"{self.config['exp_name']}_CONFIG.json"), 'w') as f:
      json.dump(self.config, f)
  
  def set_seed(self):
    # Set seeds for reproducibility
    torch.manual_seed(self.config['seed'])
    np.random.seed(self.config['seed'])
    random.seed(self.config['seed'])
    if self.device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

  def instanciate_model(self):
    self.auto_encoder = CNNAE(self.config['model_config']).to(self.device)
    self.n_trainable_params = sum(p.numel() for p in self.auto_encoder.parameters() if p.requires_grad)

  def set_dataloader(self):
    os.makedirs(self.config['data_dir'], exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(root=self.config['data_dir'], train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=self.config['data_dir'], train=False, download=True, transform=transform)

    self.train_dataloader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True,
                                       num_workers=6, pin_memory=True if torch.cuda.is_available() else False)
    self.test_dataloader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=True,
                                      num_workers=6, pin_memory=True if torch.cuda.is_available() else False)

  def set_optimizers_n_criterions(self):
    self.ae_optim = torch.optim.AdamW(self.auto_encoder.parameters(), lr=self.config['lr'], betas=(0.9, 0.95))
    self.mse_criterion = nn.MSELoss()
    self.ssim_criterion = SSIM(data_range=1, size_average=True, channel=3)
    # Image size should be larger than 160 due to the 4 downsamplings in ms-ssim
    # self.ms_ssim_criterion = MS_SSIM(data_range=1, size_average=True, channel=3)

  def save_model(self):
    os.makedirs(self.config['save_dir'], exist_ok=True)
    torch.save({'auto_encoder': self.auto_encoder.state_dict()},
               os.path.join(self.config['save_dir'], self.config['exp_name'], f"{self.config['exp_name']}.pt"))

  def load_model(self):
    path = os.path.join(self.config['save_dir'], self.config['exp_name'], f"{self.config['exp_name']}.pt")
    if os.path.isfile(path):
      model = torch.load(path, map_location=self.device)
      self.auto_encoder.load_state_dict(model['auto_encoder'])
      print(f'Model loaded successfully from {path}...')
    else:
      print(f'File {path} not found... No loaded model.')

  def train(self):
    self.auto_encoder.train()
    fixed_img = None
    best_loss = torch.inf
    pbar = tqdm(range(self.config['n_epochs']))
    for epoch in pbar:
      running_rec_loss, running_ssim_loss, running_mse_loss = 0., 0., 0.
      for img, _ in self.train_dataloader:
        img = img.to(self.device)

        if fixed_img is None:
          fixed_img = img[:8]

        rec = self.auto_encoder(img)
        mse_loss = self.mse_criterion(rec, img)
        ssim_loss = 1 - self.ssim_criterion(rec, img)
        # ms_ssim_loss = 1 - self.ms_ssim_criterion(rec, img)
        rec_loss = self.config['lambda_mse'] * mse_loss + self.config['lambda_ssim'] * ssim_loss

        self.ae_optim.zero_grad()
        rec_loss.backward()
        self.ae_optim.step()

        running_mse_loss += mse_loss.item()
        running_ssim_loss += ssim_loss.item()
        running_rec_loss += rec_loss.item()

      if self.tf_logger is not None:
        self.tf_logger.add_scalar('reconstruction_loss', running_rec_loss/len(pbar), epoch)
        self.tf_logger.add_scalar('mse_loss', running_mse_loss/len(pbar), epoch)
        self.tf_logger.add_scalar('ssim_loss', running_ssim_loss/len(pbar), epoch)

        if epoch % self.config['save_rec_every'] == 0 or epoch == (self.config['n_epochs'] - 1):
          with torch.no_grad():
            rec = self.auto_encoder(fixed_img)
          ori_rec = torch.cat([fixed_img, rec], dim=0)
          self.tf_logger.add_images(f'generated_epoch_{epoch}', ori_rec)
      
      if self.config['save_model_train'] and (running_rec_loss/len(pbar)) < best_loss:
        best_loss = running_rec_loss/len(pbar)
        self.save_model()

      pbar.set_postfix(loss=f'{running_rec_loss/len(pbar):.4f}')
    return running_rec_loss/len(pbar)

  @torch.no_grad()
  def evaluate(self):
    self.auto_encoder.eval()
    running_loss = 0.
    for img, _ in tqdm(self.test_dataloader):
      img = img.to(self.device)
      rec = self.auto_encoder(img)
      mse_loss = self.mse_criterion(rec, img)
      ssim_loss = 1 - self.ssim_criterion(rec, img)
      loss = self.config['lambda_mse'] * mse_loss + self.config['lambda_ssim'] * ssim_loss
      running_loss += loss.item()
    rec_loss = running_loss/len(self.test_dataloader)
    print(f'Reconstruction loss on test data: {rec_loss:.4f}')
    return rec_loss


class WGANGPTrainer(CNNAETrainer):
  '''Wasserstein_GAN_GradientPenalty_Trainer'''
  CONFIG = {'exp_name': 'wgan_gp_best',
            'n_critic_train_steps_per_epoch': 5,
            'gradient_penalty_lambda': 10.0,
            'rec_loss_lambda': 10.0,
            'model_config': {'skip_connection': True,
                             'linear_bottleneck': False,
                             'add_noise_bottleneck': True,
                             'add_noise_encoder': False,
                             'noise_prob': 1.0,
                             'add_enc_attn': False,
                             'add_dec_attn': True}}
  def __init__(self, config={}):
    super().__init__(WGANGPTrainer.CONFIG)
    self.config = {**self.config, **config}
  
  @staticmethod
  def gradient_penalty(critic, real, fake, device, gp_lambda=10.0):
    batch_size = real.size(0)
    # Random weight for interpolation between real and fake
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    critic_interpolates = critic(interpolates)
    # For autograd.grad to work we need a scalar for each sample — use ones
    ones = torch.ones_like(critic_interpolates, device=device)

    gradients = grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    gp = gp_lambda * ((grad_norm - 1) ** 2).mean()
    return gp
  
  def instanciate_model(self):
    self.auto_encoder = CNNAE(self.config['model_config']).to(self.device)
    self.critic = Critic().to(self.device)
    self.n_trainable_params = sum(p.numel() for p in chain(self.auto_encoder.parameters(), self.critic.parameters())
                                  if p.requires_grad)

  def set_optimizers_n_criterions(self):
    self.ae_optim = torch.optim.AdamW(self.auto_encoder.parameters(), lr=self.config['lr'], betas=(0.9, 0.95))
    self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=self.config['lr'], betas=(0.9, 0.95))
    self.mse_criterion = nn.MSELoss()
    self.ssim_criterion = SSIM(data_range=1, size_average=True, channel=3)
  
  def train(self):
    self.auto_encoder.train()
    self.critic.train()

    fixed_img = None
    best_loss = torch.inf

    pbar = tqdm(range(self.config['n_epochs']))
    for epoch in pbar:
      running_gp = 0.0
      running_critic_loss = 0.0
      running_gen_loss = 0.0
      running_mse_loss = 0.0
      running_ssim_loss = 0.0
      running_rec_loss = 0.0
      for real_imgs, _ in self.train_dataloader:
        real_imgs = real_imgs.to(self.device)

        if fixed_img is None:
          fixed_img = real_imgs[:8]
        
        # -----------------------------
        # Train Critic (multiple steps)
        # -----------------------------
        fake_imgs = self.auto_encoder(real_imgs).detach()
        for _ in range(self.config['n_critic_train_steps_per_epoch']):
          real_scores = self.critic(real_imgs)
          fake_scores = self.critic(fake_imgs)

          # WGAN-GP critic loss: E[fake] - E[real] + GP
          gp = WGANGPTrainer.gradient_penalty(self.critic, real_imgs, fake_imgs, self.device,
                                              gp_lambda=self.config['gradient_penalty_lambda'])
          loss_critic = fake_scores.mean() - real_scores.mean() + gp

          self.critic_optim.zero_grad()
          loss_critic.backward()
          self.critic_optim.step()

          running_gp += gp.item()
          running_critic_loss += loss_critic.item()
        
        # -----------------------------------
        # Train Generator (Encoder + Decoder)
        # -----------------------------------
        fake_imgs = self.auto_encoder(real_imgs)
        # Generator tries to minimize -E[critic(fake)] (i.e. maximize critic score)
        gen_adv = -self.critic(fake_imgs).mean()

        mse_loss = self.mse_criterion(fake_imgs, real_imgs)
        ssim_loss = 1 - self.ssim_criterion(fake_imgs, real_imgs)
        rec_loss = self.config['lambda_mse'] * mse_loss + self.config['lambda_ssim'] * ssim_loss

        gen_loss = gen_adv + self.config['rec_loss_lambda'] * rec_loss

        self.ae_optim.zero_grad()
        gen_loss.backward()
        self.ae_optim.step()

        running_gen_loss += gen_adv.item()
        running_rec_loss += rec_loss.item()
        running_mse_loss += mse_loss.item()
        running_ssim_loss += ssim_loss.item()

      if self.tf_logger is not None:
        self.tf_logger.add_scalar('reconstruction_loss', running_rec_loss/len(pbar), epoch)
        self.tf_logger.add_scalar('mse_loss', running_mse_loss/len(pbar), epoch)
        self.tf_logger.add_scalar('ssim_loss', running_ssim_loss/len(pbar), epoch)

        if epoch % self.config['save_rec_every'] == 0 or epoch == (self.config['n_epochs'] - 1):
          with torch.no_grad():
            rec = self.auto_encoder(fixed_img)
          ori_rec = torch.cat([fixed_img, rec], dim=0)
          self.tf_logger.add_images(f'generated_epoch_{epoch}', ori_rec)
      
      if self.config['save_model_train'] and (running_rec_loss/len(pbar)) < best_loss:
        best_loss = running_rec_loss/len(pbar)
        self.save_model()

      pbar.set_postfix(loss=f'{running_rec_loss/len(pbar):.4f}')
    return running_rec_loss/len(pbar)

  @torch.no_grad()
  def evaluate(self):
    self.auto_encoder.eval()
    running_loss = 0.
    for img, _ in tqdm(self.test_dataloader):
      img = img.to(self.device)
      rec = self.auto_encoder(img)
      mse_loss = self.mse_criterion(rec, img)
      ssim_loss = 1 - self.ssim_criterion(rec, img)
      loss = self.config['lambda_mse'] * mse_loss + self.config['lambda_ssim'] * ssim_loss
      running_loss += loss.item()
    rec_loss = running_loss/len(self.test_dataloader)
    print(f'Reconstruction loss on test data: {rec_loss:.4f}')
    return rec_loss


class MAETrainer(CNNAETrainer):
  CONFIG = {'lr':             1e-4,
            'n_epochs':       200,
            'save_rec_every': 20,
            'exp_name':       'mae_base_longMRschedule_recwarmup',
            'warmup_epochs':  2,
            'max_mask_ratio': 0.75,
            'min_mask_ratio': 0.1,
            'n_step_mask_ratio_schedule': 50,
            'n_repeat_step_max': 100,
            'n_step_repeat_schedule': 20,
            'n_epochs_repeat_stop': 20,
            'model_config': {'img_size':32, 'patch_size':8, 'in_chans':3,
                             'embed_dim':128, 'depth':6, 'num_heads':8,
                             'decoder_embed_dim':64, 'decoder_depth':2, 'decoder_num_heads':8,
                             'mlp_ratio':4.}}
  def __init__(self, config={}):
    super().__init__(MAETrainer.CONFIG)
    self.config = {**self.config, **config}
  
  def instanciate_model(self):
    self.auto_encoder = MaskedAutoencoderViT(**self.config['model_config']).to(self.device)
    self.n_trainable_params = sum(p.numel() for p in self.auto_encoder.parameters() if p.requires_grad)
  
  @staticmethod
  def exponential_schedule(epoch, n_epochs_decay, start=0.1, end=0.75):
    op_check = min if end > start else max
    return round(op_check(end, start * (end / start) ** (epoch / (n_epochs_decay - 1))), 2)
  
  @torch.no_grad()
  def get_rec_to_plot(self, fixed_img, mask_ratio=0.75):
    loss, pred, mask = self.auto_encoder(fixed_img, mask_ratio=mask_ratio)
    pred = self.auto_encoder.unpatchify(pred)
    if torch.isnan(pred).any():
      # if mask_ratio = 0. the loss will be nan
      print(f'Prediction contains Nan values...')
      pred = torch.nan_to_num(pred, nan=0.0)
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, self.auto_encoder.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = self.auto_encoder.unpatchify(mask)  # 1 is removing, 0 is keeping
    im_masked = fixed_img * (1 - mask)
    ori_masked_rec = torch.cat([fixed_img, im_masked, pred], dim=0)
    return ori_masked_rec
  
  def train(self):
    self.auto_encoder.train()
    fixed_img = None
    best_loss = torch.inf
    pbar = tqdm(range(self.config['n_epochs']))
    for epoch in pbar:
      running_rec_loss = 0.
      mask_ratio = MAETrainer.exponential_schedule(epoch, self.config['n_step_mask_ratio_schedule'])
      for i, (img, _) in enumerate(self.train_dataloader):
        img = img.to(self.device)

        if fixed_img is None:
          fixed_img = img[:8]
        
        if epoch < self.config['n_epochs_repeat_stop']:
          n_repeat = int(MAETrainer.exponential_schedule(i, self.config['n_step_repeat_schedule'],
                                                         start=self.config['n_repeat_step_max'], end=1))
        for j in range(n_repeat):
          loss, pred, mask = self.auto_encoder(img, mask_ratio=mask_ratio)  # pred = [B, n_patch, c*h*w]
          self.tf_logger.add_scalar('repeat_batch_loss', loss.item(), j)

          if epoch < self.config['warmup_epochs']:
            pred = self.auto_encoder.unpatchify(pred)
            loss = loss + self.mse_criterion(pred, img)

          self.ae_optim.zero_grad()
          loss.backward()
          self.ae_optim.step()

        running_rec_loss += loss.item()

      if self.tf_logger is not None:
        self.tf_logger.add_scalar('masked_reconstruction_loss', running_rec_loss/len(pbar), epoch)
        if epoch % self.config['save_rec_every'] == 0 or epoch == (self.config['n_epochs'] - 1):
          ori_masked_rec = self.get_rec_to_plot(fixed_img, mask_ratio=mask_ratio)
          self.tf_logger.add_images(f'generated_epoch_{epoch}', ori_masked_rec)
      
      if self.config['save_model_train'] and (running_rec_loss/len(pbar)) < best_loss:
        best_loss = running_rec_loss/len(pbar)
        self.save_model()
          
      pbar.set_postfix(loss=f'{running_rec_loss/len(pbar):.4f}')
    return running_rec_loss/len(pbar)

  @torch.no_grad()
  def evaluate(self):
    self.auto_encoder.eval()
    running_loss, running_rec_loss = 0., 0.
    for img, _ in tqdm(self.test_dataloader):
      img = img.to(self.device)
      loss, pred, mask = self.auto_encoder(img)
      pred = self.auto_encoder.unpatchify(pred)
      rec_loss = self.mse_criterion(pred, img)
      running_loss += loss.item()
      running_rec_loss += rec_loss.item()
    loss = running_loss / len(self.test_dataloader)
    rec_loss = running_rec_loss / len(self.test_dataloader)
    print(f'Masked reconstruction loss on test data: {loss:.4f}')
    print(f'Reconstruction loss on test data: {rec_loss:.4f}')
    return rec_loss


class SnakeAETrainer(CNNAETrainer):
  CONFIG = {'lr':             1e-4,
            'n_epochs':       30,
            'save_rec_every': 5,
            'exp_name':       'snakeAE_best',
            'model_config': {'encoder_config': {'add_noise': False, 'add_attn': False},
                             'decoder_config': {'add_attn': True}}}  # the .train() logic made skip_connection to True
  def __init__(self, config={}):
    super().__init__(SnakeAETrainer.CONFIG)
    self.config = {**self.config, **config}
  
  def instanciate_model(self):
    self.encoder = CNNEncoder(**self.config['model_config']['encoder_config']).to(self.device)
    self.decoder = CNNDecoder(**self.config['model_config']['decoder_config']).to(self.device)
    self.n_trainable_params = sum(p.numel() for p in chain(self.encoder.parameters(), self.decoder.parameters())
                                  if p.requires_grad)
  
  def set_optimizers_n_criterions(self):
    self.encoder_optim = torch.optim.AdamW(self.encoder.parameters(), lr=self.config['lr'], betas=(0.9, 0.95))
    self.decoder_optim = torch.optim.AdamW(self.decoder.parameters(), lr=self.config['lr'], betas=(0.9, 0.95))
    self.mse_criterion = nn.MSELoss()
    self.ssim_criterion = SSIM(data_range=1, size_average=True, channel=3)
  
  def save_model(self):
    os.makedirs(self.config['save_dir'], exist_ok=True)
    torch.save({'encoder': self.encoder.state_dict(), 'decoder': self.decoder.state_dict()},
               os.path.join(self.config['save_dir'], self.config['exp_name'], f"{self.config['exp_name']}.pt"))

  def load_model(self):
    path = os.path.join(self.config['save_dir'], self.config['exp_name'], f"{self.config['exp_name']}.pt")
    if os.path.isfile(path):
      model = torch.load(path, map_location=self.device)
      self.encoder.load_state_dict(model['encoder'])
      self.decoder.load_state_dict(model['decoder'])
      print(f'Model loaded successfully from {path}...')
    else:
      print(f'File {path} not found... No loaded model.')

  def train(self):
    self.encoder.train()
    self.decoder.train()

    fixed_img = None
    best_loss = torch.inf

    pbar = tqdm(range(self.config['n_epochs']))
    for epoch in pbar:
      running_rec_loss, running_mse_loss, running_ssim_loss = 0., 0., 0.
      for img, _ in self.train_dataloader:
        img = img.to(self.device)

        if fixed_img is None:
          fixed_img = img[:8]
        
        # ============================================
        # Phase 1: Autoencoder (train E + D)
        # ============================================
        z1, z2, z3 = self.encoder(img)
        rec = self.decoder(z3, z2, z1)

        mse_loss = self.mse_criterion(rec, img)
        ssim_loss = 1 - self.ssim_criterion(rec, img)
        rec_loss = self.config['lambda_mse'] * mse_loss + self.config['lambda_ssim'] * ssim_loss

        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()
        rec_loss.backward()
        self.encoder_optim.step()
        self.decoder_optim.step()

        running_rec_loss += rec_loss.item()
        running_mse_loss += mse_loss.item()
        running_ssim_loss += ssim_loss.item()

        # ============================================
        # Phase 2: Latent consistency (train D only)
        # ============================================
        with torch.no_grad():
          z1, z2, z3 = self.encoder(img)
        
        rec_img = self.decoder(z3, z2, z1)
        rec_z1, rec_z2, rec_z3 = self.encoder(rec_img)

        loss = self.mse_criterion(rec_z3, z3) + self.mse_criterion(rec_z2, z2) + self.mse_criterion(rec_z1, z1)
        self.decoder_optim.zero_grad()
        loss.backward()
        self.decoder_optim.step()

      if self.tf_logger is not None:
        self.tf_logger.add_scalar('reconstruction_loss', running_rec_loss/len(pbar), epoch)
        self.tf_logger.add_scalar('mse_loss', running_mse_loss/len(pbar), epoch)
        self.tf_logger.add_scalar('ssim_loss', running_ssim_loss/len(pbar), epoch)

        if epoch % self.config['save_rec_every'] == 0 or epoch == (self.config['n_epochs'] - 1):
          with torch.no_grad():
            z1, z2, z3 = self.encoder(fixed_img)
            rec = self.decoder(z3, z2, z1)
          ori_rec = torch.cat([fixed_img, rec], dim=0)
          self.tf_logger.add_images(f'generated_epoch_{epoch}', ori_rec)
      
      if self.config['save_model_train'] and (running_rec_loss/len(pbar)) < best_loss:
        best_loss = running_rec_loss/len(pbar)
        self.save_model()

      pbar.set_postfix(loss=f'{running_rec_loss/len(pbar):.4f}')
    return running_rec_loss/len(pbar)

  @torch.no_grad()
  def evaluate(self):
    self.encoder.eval()
    self.decoder.eval()

    running_loss = 0.
    for img, _ in tqdm(self.test_dataloader):
      img = img.to(self.device)

      z1, z2, z3 = self.encoder(img)
      rec = self.decoder(z3, z2, z1)

      mse_loss = self.mse_criterion(rec, img)
      ssim_loss = 1 - self.ssim_criterion(rec, img)
      loss = self.config['lambda_mse'] * mse_loss + self.config['lambda_ssim'] * ssim_loss
      running_loss += loss.item()

    rec_loss = running_loss/len(self.test_dataloader)
    print(f'Reconstruction loss on test data: {rec_loss:.4f}')
    return rec_loss


class CvTTrainer(CNNAETrainer):
  CONFIG = {'lr':             1e-4,
            'n_epochs':       30,
            'save_rec_every': 5,
            'exp_name':       'cvt_best',
            'model_config': {'add_dec_attn': True, 'skip_connection': True}}
  def __init__(self, config={}):
    super().__init__(CvTTrainer.CONFIG)
    self.config = {**self.config, **config}
  
  def instanciate_model(self):
    self.auto_encoder = MixedAE(self.config['model_config']).to(self.device)
    self.n_trainable_params = sum(p.numel() for p in self.auto_encoder.parameters() if p.requires_grad)

def flatten_dict(d, parent_key="", sep="."):
  """Recursively flattens a nested dictionary."""
  items = {}
  for k, v in d.items():
    new_key = f"{parent_key}{sep}{k}" if parent_key else k
    if isinstance(v, dict):
      items.update(flatten_dict(v, new_key, sep=sep))
    else:
      items[new_key] = v
  return items


def unflatten_dict(d, sep="."):
  result = {}
  for key, value in d.items():
    keys = key.split(sep)
    current = result
    for k in keys[:-1]:
      current = current.setdefault(k, {})
    current[keys[-1]] = value
  return result


def generate_boolean_configs(base_config):
  """
  Takes a full trainer CONFIG and returns a list of configs
  with all boolean combinations in model_config.
  """
  base_config = deepcopy(base_config)
  model_cfg = base_config.get("model_config", {})

  flat_cfg = flatten_dict(model_cfg)

  bool_keys = [k for k, v in flat_cfg.items() if isinstance(v, bool)]

  configs = []
  for values in product([False, True], repeat=len(bool_keys)):
    new_flat_cfg = flat_cfg.copy()
    for k, v in zip(bool_keys, values):
      new_flat_cfg[k] = v

    new_cfg = deepcopy(base_config)
    new_cfg["model_config"] = unflatten_dict(new_flat_cfg)
    configs.append(new_cfg)

  return configs


def build_exp_name(trainer_name, config):
  """
  Builds experiment name by appending boolean flags that are True
  from model_config.
  """
  exp_name = trainer_name

  model_cfg = config.get("model_config", {})
  flat_cfg = flatten_dict(model_cfg)

  for key, value in flat_cfg.items():
    if isinstance(value, bool) and value:
      # flag_name = '_'.join(key.split(".")[-2:])
      exp_name += f"_{key.replace('.', '_')}"
      # exp_name += f"_{flag_name}"

  return exp_name


def run_all_experiments(trainers, only_best=True):
  print(f'Run all experiments ({only_best=})')
  del trainers['mae']  # Currently not converging
  results = defaultdict(list)
  for trainer_name, trainer_cls in trainers.items():
    if only_best:
      all_configs = [trainer_cls.CONFIG]
    else:
      all_configs = generate_boolean_configs(trainer_cls.CONFIG)

    for config in all_configs:
      if not only_best:
        config['exp_name'] = build_exp_name(trainer_name, config)
      trainer = trainer_cls(config)

      print(f"Start experiment {trainer.config['exp_name']}")
      start = time.perf_counter()
      train_loss = trainer.train()
      train_time = time.perf_counter() - start
      train_epoch_time = train_time / trainer.config['n_epochs']

      start = time.perf_counter()
      test_loss = trainer.evaluate()
      test_time = time.perf_counter() - start
      test_batch_time = test_time / len(trainer.test_dataloader)

      results['trainer_name'].append(trainer_name)
      results['exp_name'].append(trainer.config['exp_name'])
      results['train_loss'].append(round(train_loss, 5))
      results['test_loss'].append(round(test_loss, 5))
      results['train_time'].append(round(train_time, 3))
      results['train_epoch_time'].append(round(train_epoch_time, 3))
      results['test_time'].append(round(test_time, 3))
      results['test_batch_time'].append(round(test_batch_time, 3))
      results['n_trainable_parameters'].append(f'{trainer.n_trainable_params:,}')

  df = pd.DataFrame.from_dict(results)
  df = df.sort_values('test_loss')
  print(tabulate(df, headers='keys', tablefmt='psql'))
  df.to_csv('report_encoding_exps_all_experiments.csv')


def get_args():
  parser = argparse.ArgumentParser(description='Image encoding experiments')
  parser.add_argument('--trainer', '-t', type=str, default='cnn_ae')
  parser.add_argument('--run_all_exps', '-rae', action='store_true')
  parser.add_argument('--load_model', '-lm', action='store_true')
  parser.add_argument('--train_model', '-tm', action='store_true')
  parser.add_argument('--eval_model', '-em', action='store_true')
  parser.add_argument('--save_model', '-sm', action='store_true')
  parser.add_argument('--rae_only_best', '-rob', action='store_false')
  return parser.parse_args()


if __name__ == '__main__':
  trainers = {'cnn_ae': CNNAETrainer, 'mae': MAETrainer, 'wgan_gp': WGANGPTrainer,
              'snake_ae': SnakeAETrainer, 'cvt_ae': CvTTrainer}
  args = get_args()

  if args.run_all_exps:
    run_all_experiments(trainers, only_best=args.rae_only_best)
  else:
    trainer = trainers[args.trainer]()

    if args.load_model:
      print(f'Loading model... ({trainers[args.trainer].__name__})')
      trainer.load_model()

    if args.train_model:
      print(f'Training model... ({trainers[args.trainer].__name__})')
      trainer.train()
    
    if args.eval_model:
      print(f'Evaluating model... ({trainers[args.trainer].__name__})')
      trainer.evaluate()
    
    if args.save_model:
      print(f'Saving model... ({trainers[args.trainer].__name__})')
      trainer.save_model()
'''
SSIM — Structural Similarity Index
What it measures
  SSIM compares images based on:
    1) Luminance (brightness)
    2) Contrast
    3) Structure (local patterns / edges)
  Instead of treating each pixel independently (like MSE), SSIM looks at local windows and compares statistics.

Output range
  -1 to 1 (practically 0 to 1)
  1.0 = identical images

Interpretation (rule of thumb)
  |       SSIM | Perceptual quality            |
  | ---------: | ----------------------------- |
  |      < 0.7 | Poor                          |
  |        0.8 | Acceptable                    |
  |        0.9 | High quality                  |
  |     ≥ 0.95 | Visually almost identical.    |

+---+--------------+--------------+-------------+--------------+--------------------+-------------+-------------------+
|   | exp_name     |   train_loss |   test_loss |   train_time |   train_epoch_time |   test_time |   test_batch_time |
|---+--------------+--------------+-------------+--------------+--------------------+-------------+-------------------|
| 1 | wgan_gp_best |       0.1382 |      0.0049 |      940.316 |             31.344 |       0.868 |             0.006 |
| 2 | snakeAE_best |       0.0083 |      0.0003 |      555.021 |             18.501 |       0.869 |             0.006 |
| 3 | cvt_best     |       0.0023 |      0.0003 |      1265.78 |             42.193 |       3.048 |             0.019 |
| 0 | cnn_ae_best  |       0.0042 |      0.0002 |      277.828 |              9.261 |       0.852 |             0.005 |
+---+--------------+--------------+-------------+--------------+--------------------+-------------+-------------------+
'''