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
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils

import cnn_layers as cl


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


class ClassConditionalFlowUnet(nn.Module):
  def __init__(self, img_chan=3, time_emb_dim=64, n_classes=10, class_emb_dim=32):
    super().__init__()
    # self.time_emb = nn.Sequential(nn.Linear(1, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))
    self.time_emb = nn.Sequential(TimeEmbedding(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU())
    self.class_emb = nn.Sequential(nn.Embedding(n_classes, class_emb_dim), nn.Linear(class_emb_dim, class_emb_dim))

    self.init_conv = nn.Conv2d(img_chan + class_emb_dim + time_emb_dim, 64, 3, 1, 1)

    self.down1 = cl.EnhancedResidualFullBlock(64, 128, batch_norm_first=True, pooling=True)
    self.down2 = cl.EnhancedResidualFullBlock(128, 256, pooling=True)
    self.down3 = cl.EnhancedResidualFullBlock(256, 512, pooling=True)

    self.up1 = cl.EnhancedResidualFullBlock(512, 256, upscaling=True)
    self.up2 = cl.EnhancedResidualFullBlock(256, 128, upscaling=True)
    self.up3 = cl.EnhancedResidualFullBlock(128, 64, upscaling=True)
    
    self.final_conv = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.SiLU(),
                                    nn.Conv2d(32, img_chan, 3, 1, 1))
  
  def forward(self, x, t, labels):
    B, C, H, W = x.shape
    c_embed = self.class_emb(labels)[:, :, None, None].expand(-1, -1, H, W)  # -> [B, c_dim, H, W]
    # t_embed = self.time_emb(t)[:, :, None, None].expand(-1, -1, H, W)  # -> [B, t_dim, H, W]
    t_embed = self.time_emb(t.squeeze(-1))[:, :, None, None].expand(-1, -1, H, W)  # when using sinusoidal embedding

    x = torch.cat([x, c_embed, t_embed], dim=1)   # -> [B, C + c_dim + (t_dim or 0), H, W]
    x = self.init_conv(x)                         # -> [B, 64, H, W]

    d1 = self.down1(x)                            # -> [B, 128, H/2, W/2]
    d2 = self.down2(d1)                           # -> [B, 256, H/4, W/4]
    d3 = self.down3(d2)                           # -> [B, 512, H/8, W/8]

    u1 = self.up1(d3)                             # -> [B, 256, H/4, W/4]
    u2 = self.up2(u1 + d2)                        # -> [B, 128, H/2, W/2]
    u3 = self.up3(u2 + d1)                        # -> [B, 64, H, W]

    velocity = self.final_conv(u3)                # -> [B, img_chan, H, W]
    return velocity


class ClassConditionalFlowMatchingTrainer:
  CONFIG = {'n_epochs':          200,
            'sample_every':      10,
            'batch_size':        128,
            'lr':                2e-4,
            'data_dir':          'data/',
            'save_dir':          'cifar10_exps/',
            'exp_name':          'cc_fm_sinemb',
            'log_dir':           'runs/',
            'save_model_train':  True,
            'use_tf_logger':     True,
            'seed':              42}
  def __init__(self, config={}):
    self.config = {**ClassConditionalFlowMatchingTrainer.CONFIG, **config}
  
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

    self.cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
  
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
    self.unet = ClassConditionalFlowUnet().to(self.device)
    self.n_trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
  
  def set_dataloader(self):
    os.makedirs(self.config['data_dir'], exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Maps [0,1] to [-1,1]
                                    ])

    train_dataset = datasets.CIFAR10(root=self.config['data_dir'], train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=self.config['data_dir'], train=False, download=True, transform=transform)

    self.train_dataloader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True,
                                       num_workers=min(6, os.cpu_count()),
                                       pin_memory=True if torch.cuda.is_available() else False)
    self.test_dataloader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=True,
                                      num_workers=min(6, os.cpu_count()),
                                      pin_memory=True if torch.cuda.is_available() else False)

  def set_optimizers_n_criterions(self):
    self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.config['lr'],
                                        weight_decay=1e-4, betas=(0.9, 0.999))
    self.criterion = nn.MSELoss()
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                T_max=self.config['n_epochs'],
                                                                eta_min=1e-6)

  def save_model(self):
    os.makedirs(self.config['save_dir'], exist_ok=True)
    torch.save({'unet': self.unet.state_dict()},
                os.path.join(self.config['save_dir'], self.config['exp_name'], f"{self.config['exp_name']}.pt"))

  def load_model(self):
    path = os.path.join(self.config['save_dir'], self.config['exp_name'], f"{self.config['exp_name']}.pt")
    if os.path.isfile(path):
      model = torch.load(path, map_location=self.device)
      self.unet.load_state_dict(model['unet'])
      print(f'Model loaded successfully from {path}...')
    else:
      print(f'File {path} not found... No loaded model.')
  
  @torch.no_grad()
  def show_sample(self, n=8, n_steps=10, get_step=5, solver='euler'):
    print('Generating Images...')
    real_imgs, class_labels = next(iter(self.train_dataloader))
    real_imgs = (real_imgs.to(self.device)+1)/2
    class_labels = class_labels.to(self.device)

    if solver == 'euler':
      samples = self.euler_sampling(n, n_steps=n_steps, get_step=get_step, class_labels=class_labels[:n])
    else:
      samples = self.rk45_sampling(n, n_steps=n_steps, get_step=get_step, class_labels=class_labels[:n])

    samples = [(s.clamp(-1, 1)+1)/2 for s in samples]  # denormalize [-1, 1] -> [0, 1]
    ori_sample = torch.cat([real_imgs[:n]] + samples, dim=0)

    show_sample_tag = f"show_sample_{solver}_{self.config['exp_name']}"
    self.tf_logger.add_images(show_sample_tag, ori_sample)

    path = os.path.join(self.config['save_dir'], self.config['exp_name'], 'samples/')
    os.makedirs(path, exist_ok=True)
    grid = utils.make_grid(ori_sample, nrow=8, normalize=True, value_range=(0, 1))
    utils.save_image(grid, os.path.join(path, f'{show_sample_tag}.png'))
    print(f'Images generated upload in tensorboard {show_sample_tag}')

  @torch.no_grad()
  def euler_sampling(self, n_samples, n_steps=10, get_step=5, class_labels=None, clamp=True):
    self.unet.eval()
    samples = []

    # Start from pure Gaussian noise
    x = torch.randn(n_samples, 3, 32, 32).to(self.device)

    # Euler integration from t=0 → t=1
    dt = 1.0 / n_steps
    for i, step in enumerate(range(n_steps)):
      t = torch.full((n_samples, 1), step * dt, device=self.device)
      v = self.unet(x, t, labels=class_labels)  # vector field
      x = x + v * dt  # Euler update

      if clamp:
        x = torch.clamp(x, -3.0, 3.0)

      if i % get_step == 0:
        samples.append(x.clone())

    return samples + [x]
  
  @torch.no_grad()
  def rk45_step(self, f, t, x, dt):
    """
    One Dormand–Prince RK45 step.
    Args:
        f : function f(t, x) -> dx/dt
        t : scalar float tensor
        x : tensor (B, C, H, W)
        dt : step size (float)
    """
    k1 = f(t, x)
    k2 = f(t + dt*1/5, x + dt*(1/5)*k1)
    k3 = f(t + dt*3/10, x + dt*(3/40*k1 + 9/40*k2))
    k4 = f(t + dt*4/5, x + dt*(44/45*k1 - 56/15*k2 + 32/9*k3))
    k5 = f(t + dt*8/9, x + dt*(19372/6561*k1 - 25360/2187*k2 + 64448/6561*k3 - 212/729*k4))
    k6 = f(t + dt, x + dt*(9017/3168*k1 - 355/33*k2 + 46732/5247*k3 + 49/176*k4 - 5103/18656*k5))

    # 5th-order solution
    x_next = x + dt * (
        35/384*k1 + 500/1113*k3 + 125/192*k4
        - 2187/6784*k5 + 11/84*k6
    )

    return x_next
  
  @torch.no_grad()
  def rk45_sampling(self, n_samples, n_steps=10, get_step=5, class_labels=None, clamp=True):
    self.unet.eval()

    def ode_fn(t, x):
      t_tensor = torch.full((x.shape[0], 1), t, device=self.device)
      return self.unet(x, t_tensor, labels=class_labels)

    # Initial noise
    x = torch.randn(n_samples, 3, 32, 32, device=self.device)

    t = 0.0
    dt = 1.0 / n_steps

    samples = []
    for i in range(n_steps):
      x = self.rk45_step(ode_fn, t, x, dt)
      t += dt

      if clamp:
        x = torch.clamp(x, -3.0, 3.0)
      
      if i % get_step == 0:
        samples.append(x.clone())

    return samples + [x]
  
  def flow_matching_loss(self, x1, class_labels=None):
    '''
    Training logic of flow model:
    1) Sample random points from our source and target distributions, and pair the points (x1 and x0)
    2) Sample random times between 0 and 1
    3) Calculate the locations where these points would be at those times
       if they were moving at constant velocity from source to target locations (interpolation, xt)
    4) Calculate what velocity they would have at those locations if they were moving at constant velocity (x1-x0)
    5) Train the network to predict these velocities – which will end up “seeking the mean”
       when the network has to do the same for many, many points.
    '''
    # ---- Sample Gaussian noise as x0 ----
    x0 = torch.randn_like(x1)

    # ---- Sample t uniformly ----
    t = torch.rand(x1.size(0), 1, device=self.device)
    t_expand = t[:, :, None, None]

    # ---- The target vector field (x1 - x0) ----
    v_target = x1 - x0

    # ---- Interpolate x_t = (1-t)x0 + t x1 ----
    xt = (1 - t_expand) * x0 + t_expand * x1

    # ---- Predict velocity and compute loss ----
    v_pred = self.unet(xt, t, labels=class_labels)
    return self.criterion(v_pred, v_target)
  
  def get_real_samples_by_class(self):
    samples = list(range(10))
    class_ok = [False] * 10
    for images, labels in self.test_dataloader:
      for img , label in zip(images, labels):
        samples[label.item()] = img
        class_ok[label.item()] = True
      
      if all(class_ok):
        break
    return samples
  
  def train(self):
    self.unet.train()

    best_loss = torch.inf

    fixed_imgs = (torch.stack(self.get_real_samples_by_class()[:8]).to(self.device)+1)/2

    pbar = tqdm(range(self.config['n_epochs']))
    for epoch in pbar:
      running_loss = 0.0
      for real_imgs, class_labels in tqdm(self.train_dataloader, leave=False):
        x1 = real_imgs.to(self.device)
        class_labels = class_labels.to(self.device)
        
        loss = self.flow_matching_loss(x1, class_labels=class_labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        running_loss += loss.item()
      
      self.scheduler.step()

      if self.tf_logger is not None:
        self.tf_logger.add_scalar('fm_loss', running_loss / len(self.train_dataloader), epoch)

        if epoch % self.config['sample_every'] == 0 or epoch == (self.config['n_epochs'] - 1):
          mid_samples, samples = self.euler_sampling(8, class_labels=torch.tensor(list(range(8)), dtype=torch.long, device=self.device))
          mid_samples, samples = mid_samples.clamp(-1, 1), samples.clamp(-1, 1)
          mid_samples, samples = (mid_samples+1)/2, (samples+1)/2  # denormalize [-1, 1] -> [0, 1]
          ori_sample = torch.cat([fixed_imgs, mid_samples, samples], dim=0)
          self.tf_logger.add_images(f'generated_epoch_{epoch}', ori_sample)
      
          if self.config['save_model_train'] and loss < best_loss:
            best_loss = loss.item()
            self.save_model()

      pbar.set_postfix(loss=f'{running_loss/len(self.train_dataloader):.4f}')
  
  @torch.no_grad()
  def evaluate(self):
    pass


def run_all_experiments(trainers, args):
  pass


def get_args():
  parser = argparse.ArgumentParser(description='Flow matching experiments')
  parser.add_argument('--trainer', '-t', type=str, default='ccfm')
  parser.add_argument('--run_all_exps', '-rae', action='store_true')
  parser.add_argument('--load_model', '-lm', action='store_true')
  parser.add_argument('--train_model', '-tm', action='store_true')
  parser.add_argument('--eval_model', '-em', action='store_true')
  parser.add_argument('--save_model', '-sm', action='store_true')
  parser.add_argument('--show_sample', '-ss', action='store_true')
  parser.add_argument('--n_steps', '-ns', type=int, default=10, help='Number of denoising steps')
  parser.add_argument('--get_step', '-gs', type=int, default=5, help='Get intermediate states during sampling')
  parser.add_argument('--ode_solver', '-os', type=str, default='euler', choices=['euler', 'RK45'])
  return parser.parse_args()


if __name__ == '__main__':
  trainers = {'ccfm': ClassConditionalFlowMatchingTrainer}
  args = get_args()

  if args.run_all_exps:
    run_all_experiments(trainers, args)
  else:
    trainer = trainers[args.trainer]()

    if args.load_model:
      print(f'Loading model... ({trainers[args.trainer].__name__})')
      trainer.load_model()

    if args.train_model:
      print(f'Training model... ({trainers[args.trainer].__name__})({trainer.n_trainable_params=:,})')
      trainer.train()
    
    if args.eval_model:
      print(f'Evaluating model... ({trainers[args.trainer].__name__})')
      trainer.evaluate()
    
    if args.show_sample:
      print(f'Generating samples... ({trainers[args.trainer].__name__})')
      trainer.load_model()
      trainer.show_sample(n_steps=args.n_steps, get_step=args.get_step, solver=args.ode_solver)
    
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