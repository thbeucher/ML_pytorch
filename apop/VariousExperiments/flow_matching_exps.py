import os
import json
import math
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter


class TimeEmbedding(nn.Module):
  """
  Classical sinusoidal time embedding (like in diffusion/transformers).
  """
  def __init__(self, dim, max_positions=10000):
    super().__init__()
    self.dim = dim
    self.max_positions = max_positions

  def forward(self, t):
    """
    t: (batch,) in [0,1]
    returns: (batch, dim)
    """
    t = t * self.max_positions
    half = self.dim // 2

    freqs = torch.exp(
        torch.arange(half, device=t.device) * -(math.log(self.max_positions) / (half - 1))
    )
    emb = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if self.dim % 2 == 1:
        emb = F.pad(emb, (0,1))

    return emb


class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.double_conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.double_conv(x)


class Unet(nn.Module):
  CONFIG = {'time_dim': 128}
  def __init__(self, config={}):
    super().__init__()
    self.config = {**Unet.CONFIG, **config}
    self.time_embed = TimeEmbedding(self.config['time_dim'])
    self.time_proj = nn.Linear(self.config['time_dim'], 16)

    self.inp = DoubleConv(3, 16)                                                 # [B, 3, 32, 32] -> [B, 16, 32, 32]
    self.down1 = nn.Sequential(nn.Conv2d(16, 16, 4, 2, 1), DoubleConv(16, 32))   # -> [B, 16, 16, 16] -> [B, 32, 16, 16]
    self.down2 = nn.Sequential(nn.Conv2d(32, 32, 4, 2, 1), DoubleConv(32, 64))   # -> [B, 32, 8, 8]   -> [B, 64, 8, 8]
    self.down3 = nn.Sequential(nn.Conv2d(64, 64, 4, 2, 1), DoubleConv(64, 128))  # -> [B, 64, 4, 4]   -> [B, 128, 4, 4]

    self.up1 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 2, 1), DoubleConv(128, 64))
    self.up2 = nn.Sequential(nn.ConvTranspose2d(64, 64, 4, 2, 1), DoubleConv(64, 32))
    self.up3 = nn.Sequential(nn.ConvTranspose2d(32, 32, 4, 2, 1), DoubleConv(32, 16))
    self.out = DoubleConv(16, 3)
    self.out = nn.Sequential(DoubleConv(16, 3), nn.Sigmoid())
  
  def forward(self, x, t):
    t = self.time_embed(t)                 # -> [B, 128]
    t = self.time_proj(t)                  # -> [B, 16]
    
    x = self.inp(x) + t[:, :, None, None]  # -> [B, 16, 32, 32]
    
    d1 = self.down1(x)                     # -> [B, 32, 16, 16]
    d2 = self.down2(d1)                    # -> [B, 64, 8, 8]
    d3 = self.down3(d2)                    # -> [B, 128, 4, 4]

    u1 = self.up1(d3)                      # -> [B, 64, 8, 8]
    u2 = self.up2(u1 + d2)                 # -> [B, 32, 16, 16]
    u3 = self.up3(u2 + d1)                 # -> [B, 16, 32, 32]

    out = self.out(u3 + x)                 # -> [B, 3, 32, 32]
    return out


class FlowMatchingTrainer:
  CONFIG = {'batch_size':        64,
            'lr':                1e-4,
            'n_training_steps':  80_000,
            'sample_step':       10_000,
            'data_dir':          '../../../gpt_tests/data/',
            'save_dir':          'cifar10_exps/',
            'exp_name':          'fm_base',
            'log_dir':           'runs/',
            'save_model_train':  True,
            'use_tf_logger':     True,
            'seed':              42,
            'model_config':      {'time_dim': 128}}
  def __init__(self, config={}):
    self.config = {**FlowMatchingTrainer.CONFIG, **config}

    self.device = torch.device('cuda' if torch.cuda.is_available() else
                               'mps' if torch.backends.mps.is_available() else
                               'cpu')

    save_dir_run = os.path.join(self.config['save_dir'], self.config['exp_name'], self.config['log_dir'])
    self.tf_logger = SummaryWriter(save_dir_run) if self.config['use_tf_logger'] else None

    self.set_seed()
    self.instanciate_model()
    self.set_dataloader()
    self.set_optimizers_n_criterions()
    self.dump_config()
  
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
    self.unet = Unet(self.config['model_config']).to(self.device)

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
    self.unet_optim = torch.optim.AdamW(self.unet.parameters(), lr=self.config['lr'], betas=(0.9, 0.95))
    self.recon_criterion = nn.MSELoss()

  def save_model(self):
    os.makedirs(self.config['save_dir'], exist_ok=True)
    torch.save({'unet': self.unet.state_dict()},
                os.path.join(self.config['save_dir'], self.config['exp_name'], f"{self.config['exp_name']}.pt"))

  def load_model(self):
    path = os.path.join(self.config['save_dir'], self.config['exp_name'], f"{self.config['exp_name']}.pt")
    if os.path.isfile(path):
      model = torch.load(path)
      self.unet.load_state_dict(model['unet'])
      print(f'Model loaded successfully from {path}...')
    else:
      print(f'File {path} not found... No loaded model.')

  @torch.no_grad()
  def sample(self, n_samples, n_steps=100):
    self.unet.eval()

    # Start from pure Gaussian noise
    xt = torch.randn(n_samples, 3, 32, 32).to(self.device)

    # Euler integration from t=0 → t=1
    for i, t in enumerate(torch.linspace(0, 1, n_steps).to(self.device)):
      tt = torch.full((n_samples,), t, device=self.device)
      v = self.unet(xt, tt)  # vector field
      xt = xt + (1 / n_steps) * v  # Euler update

      if i % (n_steps//2) == 0:
        mid_samples = xt.clamp(0, 1)

    # xt now contains generated samples in [0,1]
    samples = xt.clamp(0, 1)
    return mid_samples, samples

  def train(self):
    self.unet.train()

    best_loss = torch.inf

    pbar = tqdm(range(self.config['n_training_steps']))
    for step in pbar:
      # ---- Sample a batch of images ----
      x1, _ = next(iter(self.train_dataloader))
      x1 = x1.to(self.device)

      # ---- Sample Gaussian noise as x0 ----
      x0 = torch.randn_like(x1)

      # ---- The target vector field (x1 - x0) ----
      target = x1 - x0

      # ---- Sample t uniformly ----
      t = torch.rand(x1.size(0), device=self.device)
      t_expand = t[:, None, None, None]

      # ---- Interpolate x_t = (1-t)x0 + t x1 ----
      xt = (1 - t_expand) * x0 + t_expand * x1

      pred = self.unet(xt, t)
      loss = self.recon_criterion(pred, target)

      self.unet_optim.zero_grad()
      loss.backward()
      self.unet_optim.step()

      if self.tf_logger is not None:
        self.tf_logger.add_scalar('fm_loss', loss.item(), step)

        if step % self.config['sample_step'] == 0 or step == (self.config['n_training_steps'] - 1):
          real_imgs = x1[:8]
          mid_samples, samples = self.sample(8)
          ori_sample = torch.cat([real_imgs, mid_samples, samples], dim=0)
          self.tf_logger.add_images(f'generated_epoch_{step}', ori_sample)
      
          if self.config['save_model_train'] and loss < best_loss:
            best_loss = loss.item()
            self.save_model()

      pbar.set_postfix(loss=f'{loss.item():.4f}')

  @torch.no_grad()
  def evaluate(self):
    pass


def run_all_experiments(trainers, args):
  pass


def get_args():
  parser = argparse.ArgumentParser(description='Flow matching experiments')
  parser.add_argument('--trainer', '-t', type=str, default='fm_base')
  parser.add_argument('--run_all_exps', '-rae', action='store_true')
  parser.add_argument('--load_model', '-lm', action='store_true')
  parser.add_argument('--train_model', '-tm', action='store_true')
  parser.add_argument('--eval_model', '-em', action='store_true')
  parser.add_argument('--save_model', '-sm', action='store_true')
  return parser.parse_args()


if __name__ == '__main__':
  trainers = {'fm_base': FlowMatchingTrainer}
  args = get_args()

  if args.run_all_exps:
    run_all_experiments(trainers, args)
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
  
# =============================================================================================== #
# Papers to read on Flow Matching                                                                 #
# Guided Flows for Generative Modeling and Decision Making (https://arxiv.org/pdf/2311.13443v2)   #
# FLOW GENERATOR MATCHING (https://arxiv.org/pdf/2410.19310)                                      #
# Contrastive Flow Matching (https://arxiv.org/pdf/2506.05350)                                    #
# Efficient Flow Matching using Latent Variables (https://arxiv.org/pdf/2505.04486)               #
# https://rfangit.github.io/blog/2025/optimal_flow_matching/                                      #
# =============================================================================================== #