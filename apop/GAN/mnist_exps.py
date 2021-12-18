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
from sklearn.model_selection import train_test_split
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

  def generator_train_pass(self, target):
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
        g_loss, d_g_z2 = self.generator_train_pass(target.to(self.device))

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
          self.best_d_x, self.best_d_g_z1, self.best_d_g_z2 = np.mean(d_xs), np.mean(d_g_z1s), np.mean(d_g_z2s)
          logging.info(f'Saving model! d_x={self.best_d_x:.3f} | d_g_z1={self.best_d_g_z1:.3f} | d_g_z2={self.best_d_g_z2:.3f}')
          self.save_model(save_name=self.config['save_name'].replace('.pt', '_best_theoretically_dxdgz.pt'))
  
  @torch.no_grad()
  def get_metrics(self, n_examples=10000):
    scores = {}
    self.generator.eval()
    self.discriminator.eval()

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

    self.generator.train()
    self.discriminator.train()
    return scores

  @torch.no_grad()
  def evaluation(self, seed=42, save_name=None):
    self.generator.eval()
    torch.manual_seed(seed)  # to generate images with always the same random inputs
    generated_imgs = self.generator(torch.randn(10, self.config['noise_dim'], device=self.device))
    plot_generated(generated_imgs.view(10, 28, 28).cpu(), save_name=save_name)
    self.generator.train()

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

  def generator_train_pass(self, target):
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


class MNISTSSDCGANTrainer(MNISTGANTrainer):
  # With around 12 examples per class, a classifier CNN give f1 = 0.82 where SSDCGan give f1 = 0.9
  # With this current implementation, the training is slow and the more you add labeled example
  #                                   the more you loose advantage against a classic CNN classifier
  # With a CNN classifier (classif_mnist_exps.py) we get:
  # n_examples_per_class = 12  | f1 = 0.8177
  # n_examples_per_class = 24  | f1 = 0.9285
  # n_examples_per_class = 53  | f1 = 0.9696
  # n_examples_per_class = 102 | f1 = 0.98
  # With this Semi-Supervised DC GAN we get:
  # n_examples_per_class = 12  | f1 = 0.905
  # n_examples_per_class = 24  | f1 = 0.946
  # n_examples_per_class = 53  | f1 = 0.949
  # n_examples_per_class = 102 | f1 = 0.954
  BASE_CONFIG = {'save_name': 'model/mnist_ssdcgan_model.pt', 'save_img_folder': 'generated_ssdcgan_imgs/',
                 'betas': (0.5, 0.999), 'percent': 0.002}
  def __init__(self, config):
    super().__init__({**MNISTSSDCGANTrainer.BASE_CONFIG, **config})
    self.best_f1 = 0.

    self.activation = {}
    def get_activation(name):
      def hook(model, input, output):
        self.activation[name] = output
      return hook
    self.discriminator.network[7].register_forward_hook(get_activation('7'))
  
  def instanciate_model(self):
    self.discriminator = m.CNNDiscriminator({'n_classes': 10}).to(self.device)
    self.generator = m.CNNGenerator({}).to(self.device)
  
  def set_dataloaders(self):
    self.transform = Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])

    dataset = MNIST(self.config['dataset_path'], train=True, download=True, transform=self.transform)
    if self.config['percent'] < 1.:
      train_idx, test_idx = train_test_split(list(range(len(dataset))), train_size=self.config['percent'], stratify=dataset.targets)
      dataset_labeled = torch.utils.data.Subset(dataset, train_idx)
      dataset_unlabeled = torch.utils.data.Subset(dataset, test_idx)

    self.train_data_loader_labeled = DataLoader(dataset_labeled, batch_size=self.config['batch_size'],
                                                num_workers=self.config['n_workers'], shuffle=True, pin_memory=True)
    self.train_data_loader_unlabeled = DataLoader(dataset_unlabeled, batch_size=self.config['batch_size'],
                                                  num_workers=self.config['n_workers'], shuffle=True, pin_memory=True)
    self.test_data_loader = DataLoader(MNIST(self.config['dataset_path'], train=False,
                                             download=True, transform=self.transform),
                                       batch_size=self.config['batch_size'], num_workers=self.config['n_workers'],
                                       pin_memory=True)
  
  def discriminator_train_pass(self, img_unlbl, target_unlbl, img_lbl, target_lbl):  # img = [batch_size, 1, 28, 28]
    self.discriminator.zero_grad()

    # Train on real MNIST data considered as unlabeled
    out = self.discriminator(img_unlbl).squeeze(-1).squeeze(-1)
    D_x = out.mean().item()  # theoretically, this quantity should start close to 1 then converge to 0.5
    logz_unlbl = torch.logsumexp(out, dim=1)
    loss_unlbl = 0.5 * (-logz_unlbl.mean() + torch.nn.functional.softplus(logz_unlbl).mean())
    loss_unlbl.backward()

    # Train on fake MNIST data
    fake_img = self.generator(torch.randn(img_unlbl.shape[0], self.config['noise_dim'], 1, 1, device=self.device))
    out = self.discriminator(fake_img).squeeze(-1).squeeze(-1)
    D_G_z = out.mean().item()  # theoretically, this quantity should start close to 0 then converge to 0.5
    loss_fake = 0.5 * torch.nn.functional.softplus(torch.logsumexp(out, dim=1)).mean()
    loss_fake.backward()

    # Train on real annotated MNIST data
    out = self.discriminator(img_lbl).squeeze(-1).squeeze(-1)
    logz_lbl = torch.logsumexp(out, dim=1)
    p_lbl = torch.gather(out, 1, target_lbl.unsqueeze(1))
    loss_lbl = -p_lbl.mean() + logz_lbl.mean()
    loss_lbl.backward()

    # Optimize only discriminator's parameters
    loss = loss_unlbl + loss_fake + loss_lbl
    self.discriminator_optimizer.step()

    return loss.item(), D_x, D_G_z

  def generator_train_pass(self, img_unlbl, feature_matching=False):
    self.generator.zero_grad()

    fake_img = self.generator(torch.randn(self.config['batch_size'], self.config['noise_dim'], 1, 1, device=self.device))
    out = self.discriminator(fake_img).squeeze(-1).squeeze(-1)

    if feature_matching:
      m1 = self.activation['7'].mean(0)
      self.discriminator(img_unlbl)
      m2 = self.activation['7'].mean(0)
      loss = torch.mean((m1-m2)**2)
    else:
      logz = torch.logsumexp(out, dim=1)
      loss = 0.5 * (-logz.mean() + torch.nn.functional.softplus(logz).mean())

    loss.backward()
    self.generator_optimizer.step()

    return loss.item(), out.mean().item()
  
  def train(self):
    for epoch in tqdm(range(self.config['n_epochs'])):
      d_losses, g_losses = [], []
      d_xs, d_g_z1s, d_g_z2s = [], [], []
      for img_unlbl, target_unlbl in tqdm(self.train_data_loader_unlabeled, leave=False):
        img_lbl, target_lbl = next(iter(self.train_data_loader_labeled))
        d_loss, d_x, d_g_z1 = self.discriminator_train_pass(img_unlbl.to(self.device), target_unlbl.to(self.device),
                                                            img_lbl.to(self.device), target_lbl.to(self.device))
        g_loss, d_g_z2 = self.generator_train_pass(img_unlbl.to(self.device))

        d_losses.append(d_loss)
        g_losses.append(g_loss)
        d_xs.append(d_x)
        d_g_z1s.append(d_g_z1)
        d_g_z2s.append(d_g_z2)

      logging.info(f'Epoch {epoch} | discriminator_loss={np.mean(d_losses):.3f} | generator_loss={np.mean(g_losses):.3f}')
      logging.info(f'Epoch {epoch} | d_x={np.mean(d_xs):.3f} | d_g_z1={np.mean(d_g_z1s):.3f} | d_g_z2={np.mean(d_g_z2s):.3f}')

      if epoch % self.config['eval_step'] == 0:
        self.save_model()
        f1 = self.get_metrics()
        logging.info(f"Epoch {epoch} | f1 = {f1:.3f}")
        self.evaluation(save_name=f"{self.config['save_img_folder']}imgs_generated_epoch{epoch}.png")

        if f1 > self.best_f1:
          self.best_f1 = f1
          logging.info(f'Saving model with classification f1 = {f1:.3f}')
          self.save_model(save_name=self.config['save_name'].replace('.pt', f'_best_f1_{f1:.3f}.pt'))
  
  @torch.no_grad()
  def get_metrics(self):
    self.generator.eval()
    self.discriminator.eval()

    preds, targets = [], []
    for img, target in tqdm(self.test_data_loader, leave=False):
      targets += target.tolist()
      out = self.discriminator(img.to(self.device)).squeeze(-1).squeeze(-1)  # [batch_size, n_classes]
      preds += out.argmax(-1).tolist()
    
    self.generator.train()
    self.discriminator.train()
    return f1_score(targets, preds, average='weighted')
  
  @torch.no_grad()
  def evaluation(self, seed=42, save_name=None):
    torch.manual_seed(seed)  # to generate images with always the same random inputs
    generated_imgs = self.generator(torch.randn(10, self.config['noise_dim'], 1, 1, device=self.device))
    plot_generated(generated_imgs.squeeze(1).cpu(), save_name=save_name)


class MNISTConditionalDCGANTrainer(MNISTGANTrainer):
  BASE_CONFIG = {'save_name': 'model/mnist_cdcgan_model.pt', 'save_img_folder': 'generated_cdcgan_imgs/',
                 'betas': (0.5, 0.999), 'percent': 0.002}
  def __init__(self, config):
    super().__init__({**MNISTSSDCGANTrainer.BASE_CONFIG, **config})
  
  def instanciate_model(self):
    self.discriminator = m.ConditionalCNNDiscriminator({}).to(self.device)
    self.generator = m.ConditionalCNNGenerator({}).to(self.device)
  
  def discriminator_train_pass(self, img, target):  # img = [batch_size, 1, 28, 28]
    self.discriminator.zero_grad()

    # Train on real MNIST data
    out = self.discriminator(img, target)
    real_loss = self.criterion(out.view(out.shape[0], -1), torch.ones(img.shape[0], 1, device=self.device))
    real_loss.backward()
    D_x = out.mean().item()  # theoretically, this quantity should start close to 1 then converge to 0.5

    # Train on fake MNIST data
    fake_img = self.generator(torch.randn(img.shape[0], self.config['noise_dim'], 1, 1, device=self.device), target)
    out = self.discriminator(fake_img, target)
    fake_loss = self.criterion(out.view(out.shape[0], -1), torch.zeros(img.shape[0], 1, device=self.device))
    fake_loss.backward()
    D_G_z = out.mean().item()  # theoretically, this quantity should start close to 0 then converge to 0.5

    # Optimize only discriminator's parameters
    loss = real_loss + fake_loss
    self.discriminator_optimizer.step()

    return loss.item(), D_x, D_G_z

  def generator_train_pass(self, target):
    self.generator.zero_grad()

    fake_img = self.generator(torch.randn(len(target), self.config['noise_dim'], 1, 1, device=self.device), target)
    out = self.discriminator(fake_img, target)

    loss = self.criterion(out.view(out.shape[0], -1), torch.ones(out.shape[0], 1, device=self.device))
    loss.backward()
    self.generator_optimizer.step()

    return loss.item(), out.mean().item()
  
  @torch.no_grad()
  def get_metrics(self, n_examples=10000):
    scores = {}
    self.generator.eval()
    self.discriminator.eval()

    # Discriminator f1 score on real train and test data
    real_data_preds = {'train': [], 'test': []}
    for train in [True, False]:
      for img, target in tqdm(self.train_data_loader if train else self.test_data_loader, leave=False):
        out = self.discriminator(img.to(self.device), target.to(self.device))
        real_data_preds['train' if train else 'test'] += (out > 0.5).int().view(-1).cpu().tolist()

        if len(real_data_preds['train' if train else 'test']) > n_examples:
          break
    
    scores['real_train_data_f1'] = f1_score([1] * len(real_data_preds['train']), real_data_preds['train'], average='weighted')
    scores['real_test_data_f1'] = f1_score([1] * len(real_data_preds['test']), real_data_preds['test'], average='weighted')
    
    # Discriminator f1 score on fake data from the Generator
    fake_data_preds = []
    for _ in tqdm(range(n_examples // self.config['batch_size'] + 1), leave=False):
      targets = torch.randint(0, 10, (self.config['batch_size'],)).to(self.device)
      fake_img = self.generator(torch.randn(self.config['batch_size'], self.config['noise_dim'], 1, 1, device=self.device), targets)
      out = self.discriminator(fake_img, targets)
      fake_data_preds += (out > 0.5).int().view(-1).cpu().tolist()
    
    scores['fake_data_f1'] = f1_score([0] * len(fake_data_preds), fake_data_preds, average='weighted')

    self.generator.train()
    self.discriminator.train()
    return scores

  @torch.no_grad()
  def evaluation(self, seed=42, save_name=None):
    torch.manual_seed(seed)  # to generate images with always the same random inputs
    targets = torch.arange(0, 10).to(self.device)
    generated_imgs = self.generator(torch.randn(10, self.config['noise_dim'], 1, 1, device=self.device), targets)
    plot_generated(generated_imgs.squeeze(1).cpu(), save_name=save_name)
  


if __name__ == "__main__":
  # Tips & tricks to train GAN -> https://github.com/soumith/ganhacks
  # Comparison of different GAN -> https://sci-hub.mksa.top/https://link.springer.com/article/10.1007/s11042-019-08600-2
  # TODO
  # Feature matching | Minibatch discrimination | Historical averaging | One-sided label smoothing | Virtual batch normalization
  argparser = argparse.ArgumentParser(prog='mnist_exps.py', description='')
  argparser.add_argument('--log_file', default='_tmp_mnist_gan_logs.txt', type=str)
  argparser.add_argument('--dataset_path', default='data/', type=str)
  argparser.add_argument('--n_workers', default=8, type=int)
  argparser.add_argument('--random_seed', default=42, type=int)
  argparser.add_argument('--save_model', default='model/mnist_gan_model.pt', type=str)
  argparser.add_argument('--batch_size', default=128, type=int)
  argparser.add_argument('--trainer', default='gan', type=str)
  argparser.add_argument('--percent', default=0.002, type=float)
  argparser.add_argument('--save_img_folder', default='generated_gan_imgs/', type=str)
  args = argparser.parse_args()

  logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

  torch.manual_seed(args.random_seed)

  map_trainer = {'gan': MNISTGANTrainer, 'dcgan': MNISTDCGANTrainer, 'ssdcgan': MNISTSSDCGANTrainer,
                 'cdcgan': MNISTConditionalDCGANTrainer}

  mnist_trainer = map_trainer[args.trainer]({'dataset_path': args.dataset_path, 'n_workers': args.n_workers,
                                             'save_name': args.save_model, 'batch_size': args.batch_size,
                                             'percent': args.percent, 'save_img_folder': args.save_img_folder})

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