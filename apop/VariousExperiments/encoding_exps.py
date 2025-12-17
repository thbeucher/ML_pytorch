import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils

from masked_autoencoder import MaskedAutoencoderViT


class CNNAE(nn.Module):
  CONFIG = {'skip_connection': True,
            'linear_bottleneck': False,
            'add_noise_bottleneck': False,
            'noise_std': 0.1,
            'latent_dim': 128}
  def __init__(self, config={}):
    super().__init__()
    self.config = {**self.CONFIG, **config}

    self.down1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(True))
    self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True))
    self.down3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True))

    if self.config['linear_bottleneck']:
      self.embedder = nn.Linear(256*4*4, self.config['latent_dim'])      # H=32,W=32 for cifar10
      self.fc_dec = nn.Linear(self.config['latent_dim'], 256*4*4)

    self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True))
    self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True))
    self.up3 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid())
  
  def forward(self, x):
    d1 = self.down1(x)   # [B, 3, 32, 32] -> [B, 64, 16, 16]
    d2 = self.down2(d1)  #                -> [B, 128, 8, 8]
    d3 = self.down3(d2)  #                -> [B, 256, 4, 4]

    if self.config['linear_bottleneck']:
      d3 = self.fc_dec(self.embedder(d3.flatten(1))).view(d3.shape)
    
    if self.config['add_noise_bottleneck']:
      d3 = d3 + torch.randn_like(d3) * self.config['noise_std']

    u1 = self.up1(d3)   #                -> [B, 128, 8, 8]
    if self.config['skip_connection']:
      u1 = u1 + d2

    u2 = self.up2(u1)   #                -> [B, 64, 16, 16]
    if self.config['skip_connection']:
      u2 = u2 + d1

    up3 = self.up3(u2)  #                -> [B, 3, 32, 32]

    return up3


class CNNAETrainer:
  '''ConvNeuralNet_AutoEncoder_Trainer'''
  CONFIG = {'lr':                1e-4,
            'n_epochs':          30,
            'batch_size':        64,
            'data_dir':          '../../../gpt_tests/data/',
            'save_dir':          'cifar10_exps/',
            'exp_name':          'cnn_ae_base',
            'log_dir':           'runs/',
            'save_rec_every':    5,
            'eval_every':        5,
            'use_tf_logger':     True,
            'seed':              42,
            'save_model_train':  True,
            'model_config': {'skip_connection': False,
                             'linear_bottleneck': True,
                             'add_noise_bottleneck': False}}
  def __init__(self, config={}, set_specific_exp_name=False):
    self.config = {**CNNAETrainer.CONFIG, **config}

    self.device = torch.device('cuda' if torch.cuda.is_available() else
                               'mps' if torch.backends.mps.is_available() else
                               'cpu')

    if set_specific_exp_name:
      self.config['exp_name'] = self.config['exp_name'] +\
        ''.join(s for s, cond in [(f'_{k}', v) for k, v in trainer.config['model_config'].items()] if cond)
    save_dir_run = os.path.join(self.config['save_dir'], self.config['exp_name'], self.config['log_dir'])
    self.tf_logger = SummaryWriter(save_dir_run) if self.config['use_tf_logger'] else None

    self.set_seed()
    self.instanciate_model()
    self.set_dataloader()
    self.set_optimizers_n_criterions()
  
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
    self.opt_critic = torch.optim.AdamW(self.auto_encoder.parameters(), lr=self.config['lr'], betas=(0.9, 0.95))
    self.recon_criterion = nn.MSELoss()

  def save_model(self):
    os.makedirs(self.config['save_dir'], exist_ok=True)
    torch.save({'auto_encoder': self.auto_encoder.state_dict()},
               os.path.join(self.config['save_dir'], f"{self.config['exp_name']}.pt"))

  def load_model(self):
    path = os.path.join(self.config['save_dir'], f"{self.config['exp_name']}.pt")
    if os.path.isfile(path):
      model = torch.load(path)
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
      running_rec_loss = 0.
      for img, _ in self.train_dataloader:
        img = img.to(self.device)

        if fixed_img is None:
          fixed_img = img[:16]
          self.tf_logger.add_images('Training Fixed Image', fixed_img)

        rec = self.auto_encoder(img)
        loss = self.recon_criterion(rec, img)

        self.opt_critic.zero_grad()
        loss.backward()
        self.opt_critic.step()

        running_rec_loss += loss.item()

      if self.tf_logger is not None:
        self.tf_logger.add_scalar('reconstruction_loss', running_rec_loss/len(pbar), epoch)
        if epoch % self.config['save_rec_every'] == 0 or epoch == (self.config['n_epochs'] - 1):
          rec = self.auto_encoder(fixed_img)
          self.tf_logger.add_images(f'generated_epoch_{epoch+1}', rec)
      
      if self.config['save_model_train'] and (running_rec_loss/len(pbar)) < best_loss:
        best_loss = running_rec_loss/len(pbar)
        self.save_model()

      pbar.set_postfix(loss=f'{running_rec_loss/len(pbar):.4f}')

  @torch.no_grad()
  def evaluate(self):
    self.auto_encoder.eval()
    running_loss = 0.
    for img, _ in tqdm(self.test_dataloader):
      img = img.to(self.device)
      rec = self.auto_encoder(img)
      loss = self.recon_criterion(rec, img)
      running_loss += loss.item()
    rec_loss = running_loss/len(self.test_dataloader)
    print(f'Reconstruction loss on test data: {rec_loss:.4f}')
    return rec_loss


class MAETrainer(CNNAETrainer):
  CONFIG = {'lr':             1e-4,
            'n_epochs':       200,
            'save_rec_every': 20,
            'exp_name':       'mae_base',
            'warmup_epochs':  2,
            'max_mask_ratio': 0.75,
            'model_config': {'img_size':32, 'patch_size':8, 'in_chans':3,
                             'embed_dim':128, 'depth':6, 'num_heads':8,
                             'decoder_embed_dim':64, 'decoder_depth':2, 'decoder_num_heads':8,
                             'mlp_ratio':4.}}
  def __init__(self, config={}):
    super().__init__(MAETrainer.CONFIG)
    self.config = {**self.config, **config}
  
  def instanciate_model(self):
    self.auto_encoder = MaskedAutoencoderViT(**self.config['model_config']).to(self.device)
  
  def train(self):
    self.auto_encoder.train()
    fixed_img = None
    best_loss = torch.inf
    pbar = tqdm(range(self.config['n_epochs']))
    for epoch in pbar:
      running_rec_loss = 0.
      mask_ratio = 0.1 if epoch < self.config['warmup_epochs'] else 0.75
      for img, _ in self.train_dataloader:
        img = img.to(self.device)

        if fixed_img is None:
          fixed_img = img[:8]

        loss, pred, mask = self.auto_encoder(img, mask_ratio=mask_ratio)  # pred = [B, n_patch, c*h*w]

        if epoch < self.config['warmup_epochs']:
          pred = self.auto_encoder.unpatchify(pred)
          loss = loss + self.recon_criterion(pred, img)

        self.opt_critic.zero_grad()
        loss.backward()
        self.opt_critic.step()

        running_rec_loss += loss.item()

      if self.tf_logger is not None:
        self.tf_logger.add_scalar('masked_reconstruction_loss', running_rec_loss/len(pbar), epoch)
        if epoch % self.config['save_rec_every'] == 0 or epoch == (self.config['n_epochs'] - 1):
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
          self.tf_logger.add_images(f'generated_epoch_{epoch+1}', ori_masked_rec)
      
      if self.config['save_model_train'] and (running_rec_loss/len(pbar)) < best_loss:
        best_loss = running_rec_loss/len(pbar)
        self.save_model()
          
      pbar.set_postfix(loss=f'{running_rec_loss/len(pbar):.4f}')

  @torch.no_grad()
  def evaluate(self):
    self.auto_encoder.eval()
    running_loss, running_rec_loss = 0., 0.
    for img, _ in tqdm(self.test_dataloader):
      img = img.to(self.device)
      loss, pred, mask = self.auto_encoder(img)
      pred = self.auto_encoder.unpatchify(pred)
      rec_loss = self.recon_criterion(pred, img)
      running_loss += loss.item()
      running_rec_loss += rec_loss.item()
    loss = running_loss / len(self.test_dataloader)
    rec_loss = running_rec_loss / len(self.test_dataloader)
    print(f'Masked reconstruction loss on test data: {loss:.4f}')
    print(f'Reconstruction loss on test data: {rec_loss:.4f}')
    return rec_loss


def get_args():
  parser = argparse.ArgumentParser(description='Image encoding experiment')
  parser.add_argument('--trainer', '-t', type=str, default='cnn_ae')
  parser.add_argument('--run_all_exps', '-rae', action='store_true')
  parser.add_argument('--load_model', '-lm', action='store_true')
  parser.add_argument('--train_model', '-tm', action='store_true')
  parser.add_argument('--eval_model', '-em', action='store_true')
  parser.add_argument('--save_model', '-sm', action='store_true')
  return parser.parse_args()


def run_all_experiments(trainers, args):
  results = {'exp_name': [], 'test_loss': []}
  for config in [{},
                 {'model_config': {'skip_connection': False, 'linear_bottleneck': False, 'add_noise_bottleneck': False}},
                 {'model_config': {'skip_connection': True, 'linear_bottleneck': False, 'add_noise_bottleneck': False}},
                 {'model_config': {'skip_connection': True, 'linear_bottleneck': False, 'add_noise_bottleneck': True}}]:
    trainer = trainers[args.trainer](config, set_specific_exp_name=True)
    print(f'Start experiment {trainer.exp_name}')
    trainer.train()
    loss = trainer.evaluate()
    results['exp_name'].append(trainer.exp_name)
    results['test_loss'].append(round(loss, 4))
  df = pd.DataFrame.from_dict(results)
  print(tabulate(df, headers='keys', tablefmt='psql'))


if __name__ == '__main__':
  trainers = {'cnn_ae': CNNAETrainer, 'mae': MAETrainer}
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