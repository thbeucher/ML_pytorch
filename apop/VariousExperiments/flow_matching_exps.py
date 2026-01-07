import os
import copy
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


def sample_time(batch_size, device):
  u = torch.rand(batch_size, device=device)
  t = 1 - u**2   # bias toward small t
  return t[:, None]


def flow_matching_loss(model, x1, x0=None, condition=None, weighted_time_sampling=False, noise_scale=1.0):
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
  if x0 is None:
    # ---- Sample Gaussian noise as x0 ----
    x0 = torch.randn_like(x1) * noise_scale

  # ---- Sample t uniformly or weighted ----
  if weighted_time_sampling:
    t = sample_time(x1.size(0), x1.device)
  else:
    t = torch.rand(x1.size(0), 1, device=x1.device)
  t_expand = t[:, :, None, None]

  # ---- The target vector field (x1 - x0) ----
  v_target = x1 - x0

  # ---- Interpolate x_t = (1-t)x0 + t x1 ----
  xt = (1 - t_expand) * x0 + t_expand * x1

  # ---- Predict velocity and compute loss ----
  v_pred = model(xt, t, condition)
  return F.mse_loss(v_pred, v_target)


@torch.no_grad()
def euler_sampling(model, device, x=None, n_samples=8, n_steps=10, get_step=5, condition=None, clamp=True):
  model.eval()
  samples = []

  if x is None:
    # Start from pure Gaussian noise
    x = torch.randn(n_samples, 3, 32, 32).to(device)

  # Euler integration from t=0 → t=1
  dt = 1.0 / n_steps
  for i, step in enumerate(range(n_steps)):
    t = torch.full((x.shape[0], 1), step * dt, device=device)
    v = model(x, t, condition)  # vector field
    x = x + v * dt  # Euler update

    if clamp:
      x = torch.clamp(x, -3.0, 3.0)

    if i % get_step == 0 and n_steps > 1:
      samples.append(x.clone())

  return samples + [x]


@torch.no_grad()
def rk45_step(f, t, x, dt):
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
  x_next = x + dt * (35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)
  return x_next


@torch.no_grad()
def rk45_sampling(model, device, x=None, n_samples=8, n_steps=10, get_step=5, condition=None, clamp=True):
  model.eval()

  def ode_fn(t, x):
    t_tensor = torch.full((x.shape[0], 1), t, device=device)
    return model(x, t_tensor, condition)

  if x is None:
    # Initial noise
    x = torch.randn(n_samples, 3, 32, 32, device=device)

  t = 0.0
  dt = 1.0 / n_steps

  samples = []
  for i in range(n_steps):
    x = rk45_step(ode_fn, t, x, dt)
    t += dt

    if clamp:
      x = torch.clamp(x, -3.0, 3.0)
    
    if i % get_step == 0 and n_steps > 1:
      samples.append(x.clone())

  return samples + [x]


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


class ConditionalFlowUnet(nn.Module):
  def __init__(self, img_chan=3, time_dim=64, n_condition_values=10, condition_dim=32, condition_all_layers=False):
    super().__init__()
    # self.time_emb = nn.Sequential(nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
    self.time_emb = nn.Sequential(TimeEmbedding(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU())
    self.condition_emb = nn.Sequential(nn.Embedding(n_condition_values, condition_dim),
                                       nn.Linear(condition_dim, condition_dim))

    self.condition_all_layers = condition_all_layers
    cond_dim = condition_dim + time_dim
    self.init_conv = nn.Conv2d(img_chan + cond_dim, 64, 3, 1, 1)

    if not condition_all_layers:
      cond_dim = None

    self.down1 = cl.EnhancedResidualFullBlock(64, 128, pooling=True, cond_emb=cond_dim)
    self.down2 = cl.EnhancedResidualFullBlock(128, 256, pooling=True, cond_emb=cond_dim)
    self.down3 = cl.EnhancedResidualFullBlock(256, 512, pooling=True, cond_emb=cond_dim, groups=1)

    self.up1 = cl.EnhancedResidualFullBlock(512, 256, upscaling=True, cond_emb=cond_dim, groups=1)
    self.up2 = cl.EnhancedResidualFullBlock(256, 128, upscaling=True, cond_emb=cond_dim)
    self.up3 = cl.EnhancedResidualFullBlock(128, 64, upscaling=True, cond_emb=cond_dim)
    
    self.final_conv = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.SiLU(),
                                    nn.Conv2d(32, img_chan, 3, 1, 1))
  
  def forward(self, x, t, condition):
    B, C, H, W = x.shape
    c_embed = self.condition_emb(condition)      # -> [B, c_dim]
    # t_embed = self.time_emb(t)                   # -> [B, t_dim]
    t_embed = self.time_emb(t.squeeze(-1))       # when using sinusoidal embedding
    all_embed = torch.cat([c_embed, t_embed], dim=1)

    x = torch.cat([x, all_embed[:, :, None, None].expand(-1, -1, H, W)], dim=1)
    x = self.init_conv(x)                        # -> [B, 64, H, W]

    if not self.condition_all_layers:
      all_embed = None

    d1 = self.down1(x, c=all_embed)              # -> [B, 128, H/2, W/2]
    d2 = self.down2(d1, c=all_embed)             # -> [B, 256, H/4, W/4]
    d3 = self.down3(d2, c=all_embed)             # -> [B, 512, H/8, W/8]

    u1 = self.up1(d3, c=all_embed)               # -> [B, 256, H/4, W/4]
    u2 = self.up2(u1 + d2, c=all_embed)          # -> [B, 128, H/2, W/2]
    u3 = self.up3(u2 + d1, c=all_embed)          # -> [B, 64, H, W]

    velocity = self.final_conv(u3)                # -> [B, img_chan, H, W]
    return velocity


class ClassConditionalFlowMatchingTrainer:
  CONFIG = {'n_epochs':          500,
            'sample_every':      10,
            'batch_size':        128,
            'lr':                2e-4,
            'n_epochs_reflow':   30,
            'n_steps_reflow':    100,
            'data_dir':          'data/',
            'save_dir':          'cifar10_exps/',
            'exp_name':          'cc_fm_sinemb_base',
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
    self.set_training_utils()
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
    self.unet = ConditionalFlowUnet().to(self.device)
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

  def set_training_utils(self, reflow=False):
    model = self.reflow if reflow else self.unet
    self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['lr'],
                                       weight_decay=1e-4, betas=(0.9, 0.999))
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                T_max=self.config['n_epochs'],
                                                                eta_min=1e-6)

  def save_model(self, reflow=False):
    save_dir = os.path.join(self.config['save_dir'], self.config['exp_name'])
    os.makedirs(save_dir, exist_ok=True)
    reflow_suffix = '_reflow' if reflow else ''
    path = os.path.join(save_dir, f"{self.config['exp_name']}{reflow_suffix}.pt")
    model = self.reflow if reflow else self.unet
    torch.save({'unet': model.state_dict()}, path)

  def load_model(self, load_reflow=False):
    reflow_suffix = '_reflow' if load_reflow else ''
    path = os.path.join(self.config['save_dir'],
                        self.config['exp_name'],
                        f"{self.config['exp_name']}{reflow_suffix}.pt")
    if os.path.isfile(path):
      model = torch.load(path, map_location=self.device)
      self.unet.load_state_dict(model['unet'])
      print(f'Model loaded successfully from {path}...')
    else:
      print(f'File {path} not found... No loaded model.')
  
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
  
  @torch.no_grad()
  def show_sample(self, n=8, n_steps=10, get_step=5, solver='euler', compare_n_steps=False, reflow=False):
    print(f'Generating Images using {solver} sampler...')
    samplers = {'euler': euler_sampling, 'RK45': rk45_sampling}
    sampler = samplers[solver]
    
    real_imgs, class_labels = next(iter(self.train_dataloader))
    real_imgs = (real_imgs.to(self.device)+1)/2
    class_labels = class_labels.to(self.device)

    if compare_n_steps:
      samples = []
      print('Comparing generation using those number of steps: [1, 2, 4, 6, 8, 10, 20]')
      for n_steps in [1, 2, 4, 6, 8, 10, 20]:
        samples.append(sampler(self.unet, self.device, n_samples=n, n_steps=n_steps, get_step=30,
                               condition=class_labels[:n])[-1])
    else:
      samples = sampler(self.unet, self.device, n_samples=n, n_steps=n_steps, get_step=get_step,
                        condition=class_labels[:n])

    samples = [(s.clamp(-1, 1)+1)/2 for s in samples]  # denormalize [-1, 1] -> [0, 1]
    ori_sample = torch.cat([real_imgs[:n]] + samples, dim=0)

    reflow_suffix = '_reflow' if reflow else ''
    show_sample_tag = f"show_sample_{solver}_{self.config['exp_name']}{reflow_suffix}"
    self.tf_logger.add_images(show_sample_tag, ori_sample)

    path = os.path.join(self.config['save_dir'], self.config['exp_name'], 'samples/')
    os.makedirs(path, exist_ok=True)
    grid = utils.make_grid(ori_sample, nrow=8, normalize=True, value_range=(0, 1))
    utils.save_image(grid, os.path.join(path, f'{show_sample_tag}.png'))
    print(f'Images generated upload in tensorboard {show_sample_tag}')

  def train_log_n_save(self, fixed_imgs, epoch, best_loss, loss, reflow=False):
    if self.tf_logger is not None:
      reflow_suffix = '_reflow' if reflow else ''
      self.tf_logger.add_scalar(f'fm_loss{reflow_suffix}', loss, epoch)

      if epoch % self.config['sample_every'] == 0 or epoch == (self.config['n_epochs'] - 1):
        samples = euler_sampling(self.unet, device=self.device,
                                 condition=torch.tensor(list(range(8)), dtype=torch.long, device=self.device))
        samples = (samples[-1].clamp(-1, 1) + 1) / 2  # clamp and denormalize [-1, 1] -> [0, 1]
        ori_sample = torch.cat([fixed_imgs, samples], dim=0)
        self.tf_logger.add_images(f'generated_epoch_{epoch}{reflow_suffix}', ori_sample)
    
        if self.config['save_model_train'] and loss < best_loss:
          best_loss = loss
          self.save_model(reflow=reflow)
    return best_loss
  
  def train(self):
    self.unet.train()

    best_loss = torch.inf

    fixed_imgs = (torch.stack(self.get_real_samples_by_class()[:8]).to(self.device) + 1) / 2

    pbar = tqdm(range(self.config['n_epochs']))
    for epoch in pbar:
      running_loss = 0.0
      for real_imgs, class_labels in tqdm(self.train_dataloader, leave=False):
        x1 = real_imgs.to(self.device)
        class_labels = class_labels.to(self.device)
        
        loss = flow_matching_loss(self.unet, x1, condition=class_labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        running_loss += loss.item()
      
      self.scheduler.step()

      epoch_loss = running_loss / len(self.train_dataloader)

      best_loss = self.train_log_n_save(fixed_imgs, epoch, best_loss, epoch_loss)

      pbar.set_postfix(loss=f'{epoch_loss:.4f}')
  
  def train_reflow(self):
    '''
    Reflow (Rectified Flow) does self-distillation:
    1) Start with a pretrained flow model
    2) Sample data using this frozen model to obtain x1
    3) Train a new model to match the same straight-line velocity field: v(xt, t) = x1 - x0
    '''
    self.reflow = copy.deepcopy(self.unet)
    self.set_training_utils(reflow=True)
    self.reflow.train()

    best_loss = torch.inf

    fixed_imgs = (torch.stack(self.get_real_samples_by_class()[:8]).to(self.device) + 1) / 2

    n_classes = len(self.cifar_classes)
    batch_size = self.config['batch_size']

    pbar = tqdm(range(self.config['n_epochs_reflow']))
    for epoch in pbar:
      running_loss = 0.0
      for _ in tqdm(range(self.config['n_steps_reflow']), leave=False):
        class_labels = torch.arange(n_classes, device=self.device).repeat(batch_size // 10 + 1)[:batch_size].long()

        samples = rk45_sampling(self.unet, self.device, n_samples=batch_size, n_steps=4, condition=class_labels)
        x1 = samples[-1].detach()
        
        loss = flow_matching_loss(self.reflow, x1, condition=class_labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        running_loss += loss.item()

      epoch_loss = running_loss / len(self.train_dataloader)

      best_loss = self.train_log_n_save(fixed_imgs, epoch, best_loss, epoch_loss, reflow=True)

      pbar.set_postfix(loss=f'{epoch_loss:.4f}')
    
    self.unet = self.reflow  # Replace the teacher with the student

  @torch.no_grad()
  def evaluate(self):
    pass


def run_all_experiments(trainers, args):
  pass


def get_args():
  parser = argparse.ArgumentParser(description='Flow matching experiments')
  parser.add_argument('--trainer', '-t', type=str, default='ccfm')
  parser.add_argument('--run_all_exps', '-rae', action='store_true')
  parser.add_argument('--exp_name', '-en', type=str, default=None)
  parser.add_argument('--load_model', '-lm', action='store_true')
  parser.add_argument('--train_model', '-tm', action='store_true')
  parser.add_argument('--eval_model', '-em', action='store_true')
  parser.add_argument('--save_model', '-sm', action='store_true')
  parser.add_argument('--show_sample', '-ss', action='store_true')
  parser.add_argument('--compare_nsteps', '-cn', action='store_true')
  parser.add_argument('--train_reflow', '-tr', action='store_true')
  parser.add_argument('--load_reflow', '-lr', action='store_true')
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
    trainer = trainers[args.trainer]({} if args.exp_name is None else {'exp_name': args.exp_name})
    n_params = trainer.n_trainable_params
    trainer_name = trainers[args.trainer].__name__

    if args.load_model:
      print(f'Loading model... ({trainer_name})')
      trainer.load_model(load_reflow=args.load_reflow)

    if args.train_model:
      print(f'Training model... ({trainer_name})({n_params=:,})')
      trainer.train()
    
    if args.train_reflow:
      print(f'Training ReFlow model (Flow model should have been trained first) ({trainer_name}) ({n_params=:,})')
      trainer.load_model()  # Load the teacher
      trainer.train_reflow()
    
    if args.eval_model:
      print(f'Evaluating model... ({trainer_name})')
      trainer.evaluate()
    
    if args.show_sample:
      print(f'Generating samples... ({trainer_name})')
      trainer.load_model(load_reflow=args.load_reflow)
      trainer.show_sample(n_steps=args.n_steps, get_step=args.get_step, solver=args.ode_solver,
                          compare_n_steps=args.compare_nsteps, reflow=args.load_reflow)
    
    if args.save_model:
      print(f'Saving model... ({trainer_name})')
      trainer.save_model()
  
# =============================================================================================== #
# Papers to read on Flow Matching:                                                                #
# Guided Flows for Generative Modeling and Decision Making (https://arxiv.org/pdf/2311.13443v2)   #
# FLOW GENERATOR MATCHING (https://arxiv.org/pdf/2410.19310)                                      #
# Contrastive Flow Matching (https://arxiv.org/pdf/2506.05350)                                    #
# Efficient Flow Matching using Latent Variables (https://arxiv.org/pdf/2505.04486)               #
# https://rfangit.github.io/blog/2025/optimal_flow_matching/                                      #
# https://drscotthawley.github.io/blog/posts/FlowModels.html                                      #
# =============================================================================================== #