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


class Discriminator(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    base_config = {'layers_config': [{'type': torch.nn.Linear, 'params': {'in_features': 784, 'out_features': 1024}},
                                     {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2}},
                                     {'type': torch.nn.Dropout, 'params': {'p': 0.3}},
                                     {'type': torch.nn.Linear, 'params': {'in_features': 1024, 'out_features': 512}},
                                     {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2}},
                                     {'type': torch.nn.Dropout, 'params': {'p': 0.3}},
                                     {'type': torch.nn.Linear, 'params': {'in_features': 512, 'out_features': 256}},
                                     {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2}},
                                     {'type': torch.nn.Dropout, 'params': {'p': 0.3}},
                                     {'type': torch.nn.Linear, 'params': {'in_features': 256, 'out_features': 1}},
                                     {'type': torch.nn.Sigmoid, 'params': {}}]}
    self.config = {**base_config, **config}

    network = []
    for layer_config in self.config['layers_config']:
      network.append(layer_config['type'](**layer_config['params']))
    self.network = torch.nn.Sequential(*network)
  
  def forward(self, x):
    return self.network(x)


class Generator(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    base_config = {'layers_config': [{'type': torch.nn.Linear, 'params': {'in_features': 100, 'out_features': 256}},
                                     {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2}},
                                     {'type': torch.nn.Linear, 'params': {'in_features': 256, 'out_features': 512}},
                                     {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2}},
                                     {'type': torch.nn.Linear, 'params': {'in_features': 512, 'out_features': 1024}},
                                     {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2}},
                                     {'type': torch.nn.Linear, 'params': {'in_features': 1024, 'out_features': 784}},
                                     {'type': torch.nn.Tanh, 'params': {}}]}
    self.config = {**base_config, **config}

    network = []
    for layer_config in self.config['layers_config']:
      network.append(layer_config['type'](**layer_config['params']))
    self.network = torch.nn.Sequential(*network)
  
  def forward(self, x):
    return self.network(x)


class MNISTGANTrainer(object):
  def __init__(self, config):
    base_config = {'dataset_path': 'data/', 'batch_size': 128, 'n_workers': 8, 'save_name': 'model/mnist_gan_model.pt',
                   'lr_generator': 2e-4, 'lr_discriminator': 2e-4, 'n_epochs': 200, 'noise_dim': 100, 'eval_step': 20}
    self.config = {**base_config, **config}

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.set_dataloaders()
    self.instanciate_model()

    self.criterion = torch.nn.BCELoss()
    self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.config['lr_generator'])
    self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.config['lr_discriminator'])

    self.lowest_f1 = 1.
  
  def instanciate_model(self):
    self.discriminator = Discriminator({}).to(self.device)
    self.generator = Generator({}).to(self.device)

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

    # Train on fake MNIST data
    fake_img = self.generator(torch.randn(img.shape[0], self.config['noise_dim'], device=self.device))
    out = self.discriminator(fake_img)
    fake_loss = self.criterion(out, torch.zeros(img.shape[0], 1, device=self.device))

    # Optimize only discriminator's parameters
    loss = real_loss + fake_loss
    loss.backward()
    self.discriminator_optimizer.step()

    return loss.item()

  def generator_train_pass(self):
    self.generator.zero_grad()

    fake_img = self.generator(torch.randn(self.config['batch_size'], self.config['noise_dim'], device=self.device))
    out = self.discriminator(fake_img)

    loss = self.criterion(out, torch.ones(self.config['batch_size'], 1, device=self.device))
    loss.backward()
    self.generator_optimizer.step()

    return loss.item()
  
  def train(self):
    for epoch in tqdm(range(self.config['n_epochs'])):
      d_losses, g_losses = [], []
      for img, target in tqdm(self.train_data_loader, leave=False):
        d_losses.append(self.discriminator_train_pass(img.to(self.device), target.to(self.device)))
        g_losses.append(self.generator_train_pass())

      logging.info(f'Epoch {epoch} | discriminator_loss={np.mean(d_losses):.3f} | generator_loss={np.mean(g_losses):.3f}')

      if epoch % self.config['eval_step'] == 0:
        self.save_model()
        scores = self.get_metrics()
        logging.info(f"Epoch {epoch} | {[(m, round(s, 4)) for m, s in scores.items()]}")

        if scores['fake_data_f1'] < self.lowest_f1:
          logging.info(f"Saving model with f1 score on fake data = {scores['fake_data_f1']:.3f}")
          self.save_model(save_name=self.config['save_name'].replace('.pt', '_lowest_fake_f1.pt'))
          self.lowest_f1 = scores['fake_data_f1']
  
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
  def evaluation(self):
    generated_imgs = self.generator(torch.randn(10, self.config['noise_dim'], device=self.device))
    MNISTGANTrainer.plot_generated(generated_imgs.view(10, 28, 28))
  
  @staticmethod
  def plot_generated(generated_imgs, dim=(1, 10), figsize=(12, 2)):
    plt.figure(figsize=figsize)
    for i in range(generated_imgs.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_imgs[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    # plt.draw()
    # plt.pause(0.01)
    plt.show()

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


if __name__ == "__main__":
  # https://github.com/Sleepychord/ImprovedGAN-pytorch/blob/master/ImprovedGAN.py
  argparser = argparse.ArgumentParser(prog='mnist_exps.py', description='')
  argparser.add_argument('--log_file', default='_tmp_mnist_gan_logs.txt', type=str)
  argparser.add_argument('--dataset_path', default='data/', type=str)
  argparser.add_argument('--n_workers', default=8, type=int)
  argparser.add_argument('--random_seed', default=42, type=int)
  argparser.add_argument('--save_model', default='model/mnist_gan_model.pt', type=str)
  argparser.add_argument('--batch_size', default=128, type=int)
  args = argparser.parse_args()

  logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

  torch.manual_seed(args.random_seed)


  # generator = Generator({})
  # out = generator(torch.randn(32, 100))
  # print(f'Generator output = {out.shape}')
  # print(generator)

  # discriminator = Discriminator({})
  # out = discriminator(torch.randn(32, 784))
  # print(f'Generator output = {out.shape}')
  # print(discriminator)

  mnist_gan_trainer = MNISTGANTrainer({'dataset_path': args.dataset_path, 'n_workers': args.n_workers,
                                       'save_name': args.save_model, 'batch_size': args.batch_size})

  rep = input(f'Load MNIST GAN model? (y or n): ')
  if rep == 'y':
    mnist_gan_trainer.load_model(map_location=mnist_gan_trainer.device)

  rep = input(f'Train MNIST GAN? (y or n): ')
  if rep == 'y':
    mnist_gan_trainer.train()
  
  rep = input(f'Eval MNIST GAN? (y or n): ')
  if rep == 'y':
    mnist_gan_trainer.evaluation()