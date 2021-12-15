import os
import torch
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import f1_score
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

import models as m
from plotter import plot_generated


class MNISTGANTrainer(object):
  BASE_CONFIG = {'dataset_path': 'data/', 'batch_size': 128, 'n_workers': 8, 'save_name': 'model/mnist_gan_model.pt',
                 'lr_generator': 2e-4, 'lr_discriminator': 2e-4, 'n_epochs': 200, 'noise_dim': 100, 'eval_step': 20,
                 'betas': (0.9, 0.999), 'save_img_folder': 'generated_gan_imgs/'}
  def __init__(self, config):
    self.config = {**MNISTGANTrainer.BASE_CONFIG, **config}

    if not os.path.isdir(self.config['save_img_folder']):
      os.makedirs(self.config['save_img_folder'])

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.set_dataloaders()
    self.instanciate_model()

    self.criterion = torch.nn.BCELoss()
    self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.config['lr_generator'],
                                                betas=self.config['betas'])
    self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.config['lr_discriminator'],
                                                    betas=self.config['betas'])

    # self.lowest_f1 = 1.
    self.best_d_x, self.best_d_g_z1, self.best_d_g_z2 = 1., 0., 0.
  
  def instanciate_model(self):
    self.discriminator = m.MLPDiscriminator({}).to(self.device)
    self.generator = m.MLPGenerator({}).to(self.device)

  def set_dataloaders(self):
    self.transform = Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])
    self.train_data_loader = DataLoader(MNIST(self.config['dataset_path'], train=True,
                                              download=True, transform=self.transform),
                                        batch_size=self.config['batch_size'], num_workers=self.config['n_workers'],
                                        shuffle=True, pin_memory=True)
    self.test_data_loader = DataLoader(MNIST(self.config['dataset_path'], train=False,
                                             download=True, transform=self.transform),
                                       batch_size=self.config['batch_size'], num_workers=self.config['n_workers'],
                                       pin_memory=True)

  def discriminator_train_pass(self, img, target):
    self.discriminator.zero_grad()

    # Train on real MNIST data
    out = self.discriminator(img.view(img.shape[0], -1))
    real_loss = self.criterion(out, torch.ones(img.shape[0], 1, device=self.device))
    D_x = out.mean().item()

    # Train on fake MNIST data
    fake_img = self.generator(torch.randn(img.shape[0], self.config['noise_dim'], device=self.device))
    out = self.discriminator(fake_img)
    fake_loss = self.criterion(out, torch.zeros(img.shape[0], 1, device=self.device))
    D_G_z = out.mean().item()

    # Optimize only discriminator's parameters
    loss = real_loss + fake_loss
    loss.backward()
    self.discriminator_optimizer.step()

    return loss.item(), D_x, D_G_z

  def generator_train_pass(self):
    self.generator.zero_grad()

    fake_img = self.generator(torch.randn(self.config['batch_size'], self.config['noise_dim'], device=self.device))
    out = self.discriminator(fake_img)

    loss = self.criterion(out, torch.ones(self.config['batch_size'], 1, device=self.device))
    loss.backward()
    self.generator_optimizer.step()

    return loss.item(), out.mean().item()
  
  def train(self):
    for epoch in tqdm(range(self.config['n_epochs'])):
      d_losses, g_losses = [], []
      d_xs, d_g_z1s, d_g_z2s = [], [], []
      for img, target in tqdm(self.train_data_loader, leave=False):
        d_loss, d_x, d_g_z1 = self.discriminator_train_pass(img.to(self.device), target.to(self.device))
        g_loss, d_g_z2 = self.generator_train_pass()

        d_losses.append(d_loss)
        g_losses.append(g_loss)
        d_xs.append(d_x)
        d_g_z1s.append(d_g_z1)
        d_g_z2s.append(d_g_z2)

      logging.info(f'Epoch {epoch} | discriminator_loss={np.mean(d_losses):.3f} | generator_loss={np.mean(g_losses):.3f}')
      logging.info(f'Epoch {epoch} | d_x={np.mean(d_xs):.3f} | d_g_z1={np.mean(d_g_z1s):.3f} | d_g_z2={np.mean(d_g_z2s):.3f}')

      if epoch % self.config['eval_step'] == 0:
        self.save_model()
        scores = self.get_metrics()
        logging.info(f"Epoch {epoch} | {[(m, round(s, 4)) for m, s in scores.items()]}")
        self.evaluation(save_name=f"{self.config['save_img_folder']}imgs_generated_epoch{epoch}.png")

        # if scores['fake_data_f1'] < self.lowest_f1:
        #   logging.info(f"Saving model with f1 score on fake data = {scores['fake_data_f1']:.3f}")
        #   self.save_model(save_name=self.config['save_name'].replace('.pt', '_lowest_fake_f1.pt'))
        #   self.lowest_f1 = scores['fake_data_f1']
        if np.mean(d_xs) < self.best_d_x and np.mean(d_g_z1s) > self.best_d_g_z1 and np.mean(d_g_z2s) > self.best_d_g_z2:
          logging.info(f'Saving model! d_x={np.mean(d_xs):.3f} | d_g_z1={np.mean(d_g_z1s):.3f} | d_g_z2={np.mean(d_g_z2s):.3f}')
          self.save_model(save_name=self.config['save_name'].replace('.pt', '_best_theoretically_dxdgz.pt'))
          self.best_d_x = np.mean(d_xs)
          self.best_d_g_z1 = np.mean(d_g_z1s)
          self.best_d_g_z2 = np.mean(d_g_z1s)
  
  @torch.no_grad()
  def get_metrics(self, n_examples=10000):
    scores = {}

    # Discriminator f1 score on real train and test data
    real_data_preds = {'train': [], 'test': []}
    for train in [True, False]:
      for img, _ in tqdm(self.train_data_loader if train else self.test_data_loader, leave=False):
        out = self.discriminator(img.view(img.shape[0], -1).to(self.device))
        real_data_preds['train' if train else 'test'] += (out > 0.5).int().view(-1).cpu().tolist()

        if len(real_data_preds['train' if train else 'test']) > n_examples:
          break
    
    scores['real_train_data_f1'] = f1_score([1] * len(real_data_preds['train']), real_data_preds['train'], average='weighted')
    scores['real_test_data_f1'] = f1_score([1] * len(real_data_preds['test']), real_data_preds['test'], average='weighted')
    
    # Discriminator f1 score on fake data from the Generator
    fake_data_preds = []
    for _ in tqdm(range(n_examples // self.config['batch_size'] + 1), leave=False):
      fake_img = self.generator(torch.randn(self.config['batch_size'], self.config['noise_dim'], device=self.device))
      out = self.discriminator(fake_img)
      fake_data_preds += (out > 0.5).int().view(-1).cpu().tolist()
    
    scores['fake_data_f1'] = f1_score([0] * len(fake_data_preds), fake_data_preds, average='weighted')

    return scores

  @torch.no_grad()
  def evaluation(self, seed=42, save_name=None):
    torch.manual_seed(seed)  # to generate images with always the same random inputs
    generated_imgs = self.generator(torch.randn(10, self.config['noise_dim'], device=self.device))
    plot_generated(generated_imgs.view(10, 28, 28).cpu(), save_name=save_name)

  def save_model(self, save_name=None):
    save_name = self.config['save_name'] if save_name is None else save_name
    if not os.path.isdir(os.path.dirname(save_name)):
      os.makedirs(os.path.dirname(save_name))
    torch.save({'generator': self.generator.state_dict(), 'discriminator': self.discriminator.state_dict()}, save_name)

  def load_model(self, save_name=None, map_location=None):
    save_name = self.config['save_name'] if save_name is None else save_name
    if os.path.isfile(save_name):
      data = torch.load(save_name, map_location=map_location)
      self.generator.load_state_dict(data['generator'])
      self.discriminator.load_state_dict(data['discriminator'])
    else:
      print(f"File {save_name} doesn't exist")


class MNISTDCGANTrainer(MNISTGANTrainer):
  BASE_CONFIG = {'save_name': 'model/mnist_dcgan_model.pt', 'save_img_folder': 'generated_dcgan_imgs/',
                 'betas': (0.5, 0.999)}
  def __init__(self, config):
    super().__init__({**MNISTDCGANTrainer.BASE_CONFIG, **config})
  
  def instanciate_model(self):
    self.discriminator = m.CNNDiscriminator({}).to(self.device)
    self.generator = m.CNNGenerator({}).to(self.device)
  
  def discriminator_train_pass(self, img, target):  # img = [batch_size, 1, 28, 28]
    self.discriminator.zero_grad()

    # Train on real MNIST data
    out = self.discriminator(img)
    real_loss = self.criterion(out.view(out.shape[0], -1), torch.ones(img.shape[0], 1, device=self.device))
    real_loss.backward()
    D_x = out.mean().item()  # theoretically, this quantity should start close to 1 then converge to 0.5

    # Train on fake MNIST data
    fake_img = self.generator(torch.randn(img.shape[0], self.config['noise_dim'], 1, 1, device=self.device))
    out = self.discriminator(fake_img)
    fake_loss = self.criterion(out.view(out.shape[0], -1), torch.zeros(img.shape[0], 1, device=self.device))
    fake_loss.backward()
    D_G_z = out.mean().item()  # theoretically, this quantity should start close to 0 then converge to 0.5

    # Optimize only discriminator's parameters
    loss = real_loss + fake_loss
    self.discriminator_optimizer.step()

    return loss.item(), D_x, D_G_z

  def generator_train_pass(self):
    self.generator.zero_grad()

    fake_img = self.generator(torch.randn(self.config['batch_size'], self.config['noise_dim'], 1, 1, device=self.device))
    out = self.discriminator(fake_img)

    loss = self.criterion(out.view(out.shape[0], -1), torch.ones(self.config['batch_size'], 1, device=self.device))
    loss.backward()
    self.generator_optimizer.step()

    return loss.item(), out.mean().item()
  
  @torch.no_grad()
  def get_metrics(self, n_examples=10000):
    scores = {}

    # Discriminator f1 score on real train and test data
    real_data_preds = {'train': [], 'test': []}
    for train in [True, False]:
      for img, _ in tqdm(self.train_data_loader if train else self.test_data_loader, leave=False):
        out = self.discriminator(img.to(self.device))
        real_data_preds['train' if train else 'test'] += (out > 0.5).int().view(-1).cpu().tolist()

        if len(real_data_preds['train' if train else 'test']) > n_examples:
          break
    
    scores['real_train_data_f1'] = f1_score([1] * len(real_data_preds['train']), real_data_preds['train'], average='weighted')
    scores['real_test_data_f1'] = f1_score([1] * len(real_data_preds['test']), real_data_preds['test'], average='weighted')
    
    # Discriminator f1 score on fake data from the Generator
    fake_data_preds = []
    for _ in tqdm(range(n_examples // self.config['batch_size'] + 1), leave=False):
      fake_img = self.generator(torch.randn(self.config['batch_size'], self.config['noise_dim'], 1, 1, device=self.device))
      out = self.discriminator(fake_img)
      fake_data_preds += (out > 0.5).int().view(-1).cpu().tolist()
    
    scores['fake_data_f1'] = f1_score([0] * len(fake_data_preds), fake_data_preds, average='weighted')

    return scores

  @torch.no_grad()
  def evaluation(self, seed=42, save_name=None):
    torch.manual_seed(seed)  # to generate images with always the same random inputs
    generated_imgs = self.generator(torch.randn(10, self.config['noise_dim'], 1, 1, device=self.device))
    plot_generated(generated_imgs.squeeze(1).cpu(), save_name=save_name)


if __name__ == "__main__":
  # Tips & tricks to train GAN -> https://github.com/soumith/ganhacks
  # TODO
  # Feature matching: Develop a GAN using semi-supervised learning.
  # Minibatch discrimination: Develop features across multiple samples in a minibatch.
  # Historical averaging: Update the loss function to incorporate history.
  # One-sided label smoothing: Scaling target values for the discriminator away from 1.0.
  # Virtual batch normalization: Calculation of batch norm statistics using a reference batch of real images.
  argparser = argparse.ArgumentParser(prog='mnist_exps.py', description='')
  argparser.add_argument('--log_file', default='_tmp_mnist_gan_logs.txt', type=str)
  argparser.add_argument('--dataset_path', default='data/', type=str)
  argparser.add_argument('--n_workers', default=8, type=int)
  argparser.add_argument('--random_seed', default=42, type=int)
  argparser.add_argument('--save_model', default='model/mnist_gan_model.pt', type=str)
  argparser.add_argument('--batch_size', default=128, type=int)
  argparser.add_argument('--trainer', default='gan', type=str)
  args = argparser.parse_args()

  logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

  torch.manual_seed(args.random_seed)

  map_trainer = {'gan': MNISTGANTrainer, 'dcgan': MNISTDCGANTrainer}

  mnist_trainer = map_trainer[args.trainer]({'dataset_path': args.dataset_path, 'n_workers': args.n_workers,
                                             'save_name': args.save_model, 'batch_size': args.batch_size})

  rep = input(f'Load MNIST {args.trainer.upper()} model? (y or n): ')
  if rep == 'y':
    print(f'Model {args.save_model} loaded.')
    mnist_trainer.load_model(map_location=mnist_trainer.device)

  rep = input(f'Train MNIST {args.trainer.upper()}? (y or n): ')
  if rep == 'y':
    mnist_trainer.train()
  
  rep = input(f'Eval MNIST {args.trainer.upper()}? (y or n): ')
  if rep == 'y':
    mnist_trainer.evaluation()