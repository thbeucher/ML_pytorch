import os
import torch
import logging
import argparse
import numpy as np

from tqdm import tqdm
from sklearn.metrics import f1_score
from torchvision.datasets import MNIST
from torchvision.utils import make_grid, save_image
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Normalize

import models as m


class VQVAETrainer(object):
  BASE_CONFIG = {'dataset_path': 'data/', 'batch_size': 128, 'n_workers': 8, 'save_name': 'model/vqvae_mnist_model.pt',
                 'n_training_update': 5001, 'eval_step': 100, 'percent': 1., 'save_img_folder': 'generated_vqvae_imgs/',
                 'lr': 1e-3, 'n_examples': 10}
  def __init__(self, config):
    super().__init__()
    self.config = {**VQVAETrainer.BASE_CONFIG, **config}

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.instanciate_model()
    self.set_dataloader()

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
    self.criterion = torch.nn.MSELoss()

    self.train_imgs_to_rec = torch.stack([self.train_data_loader.dataset.__getitem__(i)[0] for i in range(self.config['n_examples'])])
    self.test_imgs_to_rec = torch.stack([self.test_data_loader.dataset.__getitem__(i)[0] for i in range(self.config['n_examples'])])

    if not os.path.isdir(self.config['save_img_folder']):
      os.makedirs(self.config['save_img_folder'])
    
    save_image(make_grid(self.test_imgs_to_rec, nrow=10), os.path.join(self.config['save_img_folder'], 'imgs_original.png'))
    # plt.imshow(grid_img.permute(1, 2, 0), cmap='gray_r')
  
  def instanciate_model(self):
    self.model = m.VQVAEModel({}).to(self.device)

  def set_dataloader(self):
    self.transform = Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])

    dataset = MNIST(self.config['dataset_path'], train=True, download=True, transform=self.transform)
    if self.config['percent'] < 1.:
      train_idx, _ = train_test_split(list(range(len(dataset))), train_size=self.config['percent'], stratify=dataset.targets)
      dataset = torch.utils.data.Subset(dataset, train_idx)

    self.train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.config['batch_size'],
                                                         num_workers=self.config['n_workers'], shuffle=True, pin_memory=True)
    self.test_data_loader = torch.utils.data.DataLoader(MNIST(self.config['dataset_path'], train=False, download=True,
                                                              transform=self.transform),
                                                        batch_size=self.config['batch_size'], num_workers=self.config['n_workers'],
                                                        shuffle=False, pin_memory=True)

  def train(self):
    rec_losses, perplexities = [], []
    i = 0
    for _ in tqdm(range(self.config['n_training_update'])):
      for img, _ in tqdm(self.train_data_loader, leave=False):
        img = img.to(self.device)
        self.optimizer.zero_grad()
        vq_loss, img_rec, perplexity, encodings, quantized = self.model(img)
        rec_loss = self.criterion(img_rec, img)
        loss = vq_loss + rec_loss
        loss.backward()
        self.optimizer.step()

        rec_losses.append(rec_loss.item())
        perplexities.append(perplexity.item())

        if i % self.config['eval_step'] == 0:
          logging.info(f'Step {i} | rec_loss={np.mean(rec_losses)} | perplexity={np.mean(perplexities)}')
          rec_losses, perplexities = [], []
          self.generate_img(save_name=os.path.join(self.config['save_img_folder'], f'imgs_reconstructed_step{i}.png'))
          self.save_model()
        
        i += 1

        if i > self.config['n_training_update']:
          return

  @torch.no_grad()
  def evaluation(self):
    self.model.eval()
    self.model.train()

  @torch.no_grad()
  def generate_img(self, save_name=None):
    self.model.eval()
    _, img_rec, *_ = self.model(self.test_imgs_to_rec.to(self.device))
    save_image(make_grid(img_rec.cpu(), nrow=10), save_name)
    self.model.train()

  def save_model(self, save_name=None):
    save_name = self.config['save_name'] if save_name is None else save_name
    if not os.path.isdir(os.path.dirname(save_name)):
      os.makedirs(os.path.dirname(save_name))
    torch.save({'model': self.model.state_dict()}, save_name)

  def load_model(self, save_name=None, map_location=None):
    save_name = self.config['save_name'] if save_name is None else save_name
    if os.path.isfile(save_name):
      data = torch.load(save_name, map_location=map_location)
      self.model.load_state_dict(data['model'])
    else:
      print(f"File {save_name} doesn't exist")


class VQVAEClassifierTrainer(VQVAETrainer):
  BASE_CONFIG = {'dataset_path': 'data/', 'batch_size': 128, 'n_workers': 8, 'save_name': 'model/vqvae_classif_mnist_model.pt',
                 'n_training_update': 15001, 'eval_step': 100, 'percent': 1., 'save_img_folder': 'generated_vqvae_classif_imgs/',
                 'lr': 1e-3, 'n_examples': 10}
  def __init__(self, config):
    config = {**VQVAEClassifierTrainer.BASE_CONFIG, **config}
    super().__init__(config)
    self.classifier_criterion = torch.nn.CrossEntropyLoss()
  
  def instanciate_model(self):
    self.model = m.VQVAEClassifierModel({}).to(self.device)
  
  def train(self):
    rec_losses, perplexities, classif_losses = [], [], []
    i = 0
    f1_memory = 0.
    for _ in tqdm(range(self.config['n_training_update'])):
      for img, target in tqdm(self.train_data_loader, leave=False):
        img = img.to(self.device)
        self.optimizer.zero_grad()
        vq_loss, img_rec, perplexity, encodings, quantized, out_classif = self.model(img)
        rec_loss = self.criterion(img_rec, img)
        classif_loss = self.classifier_criterion(out_classif, target.to(self.device))
        loss = vq_loss + rec_loss + classif_loss
        loss.backward()
        self.optimizer.step()

        rec_losses.append(rec_loss.item())
        perplexities.append(perplexity.item())
        classif_losses.append(classif_loss.item())

        if i % self.config['eval_step'] == 0:
          logging.info(f'Step {i} | rec_loss={np.mean(rec_losses)} | perplexity={np.mean(perplexities)}')
          rec_losses, perplexities = [], []
          self.generate_img(save_name=os.path.join(self.config['save_img_folder'], f'imgs_reconstructed_step{i}.png'))
          self.save_model()

          f1 = self.evaluation()
          logging.info(f"Step {i} | ({'new_best' if f1 > f1_memory else ''}) f1 = {f1:.3f}")
          f1_memory = f1 if f1 > f1_memory else f1_memory
        
        i += 1

        if i > self.config['n_training_update']:
          return
  
  @torch.no_grad()
  def evaluation(self):
    self.model.eval()
    preds, targets = [], []
    for img, target in tqdm(self.test_data_loader, leave=False):
      targets += target.tolist()
      _, _, _, _, _, out_classif = self.model(img.to(self.device))
      preds += out_classif.argmax(-1).cpu().tolist()
    self.model.train()
    return f1_score(targets, preds, average='weighted')


class VQVAETClassifierTrainer(VQVAEClassifierTrainer):
  BASE_CONFIG = {'dataset_path': 'data/', 'batch_size': 128, 'n_workers': 8, 'save_name': 'model/vqvae_Tclassif_mnist_model.pt',
                 'n_training_update': 15001, 'eval_step': 100, 'percent': 1., 'save_img_folder': 'generated_vqvae_Tclassif_imgs/',
                 'lr': 1e-3, 'n_examples': 10}
  def __init__(self, config):
    config = {**VQVAETClassifierTrainer.BASE_CONFIG, **config}
    super().__init__(config)
    self.classifier_criterion = torch.nn.CrossEntropyLoss()
  
  def instanciate_model(self):
    self.model = m.VQVAEClassifierModel({'classifier_type': m.TransformerHead}).to(self.device)


if __name__ == "__main__":
  # https://nbviewer.org/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
  argparser = argparse.ArgumentParser(prog='vqvae_mnist_exps.py', description='')
  argparser.add_argument('--log_file', default='_tmp_mnist_vqvae_logs.txt', type=str)
  argparser.add_argument('--dataset_path', default='data/', type=str)
  argparser.add_argument('--n_workers', default=8, type=int)
  argparser.add_argument('--random_seed', default=42, type=int)
  argparser.add_argument('--save_model', default='', type=str)
  argparser.add_argument('--batch_size', default=128, type=int)
  argparser.add_argument('--trainer', default='vqvae', type=str)
  argparser.add_argument('--percent', default=1., type=float)
  argparser.add_argument('--save_img_folder', default='', type=str)
  args = argparser.parse_args()

  logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

  torch.manual_seed(args.random_seed)

  map_trainer = {'vqvae': VQVAETrainer, 'vqvae_classif': VQVAEClassifierTrainer, 'vqvae_Tclassif': VQVAETClassifierTrainer}

  mnist_trainer = map_trainer[args.trainer]({'dataset_path': args.dataset_path, 'n_workers': args.n_workers,
                                             'batch_size': args.batch_size, 'percent': args.percent})

  if args.save_model != '':
    mnist_trainer.config['save_name'] = args.save_model
  
  if args.save_img_folder != '':
    mnist_trainer.config['save_img_folder'] = args.save_img_folder

  rep = input(f'Load MNIST {args.trainer.upper()} model? (y or n): ')
  if rep == 'y':
    print(f"Model {mnist_trainer.config['save_name']} loaded.")
    mnist_trainer.load_model(map_location=mnist_trainer.device)

  rep = input(f'Train MNIST {args.trainer.upper()}? (y or n): ')
  if rep == 'y':
    mnist_trainer.train()
  
  # rep = input(f'Eval MNIST {args.trainer.upper()}? (y or n): ')
  # if rep == 'y':
  #   mnist_trainer.evaluation()

  rep = input(f'Generate Images with current model? (y or n): ')
  if rep == 'y':
    mnist_trainer.generate_img(save_name=f"{mnist_trainer.config['save_img_folder']}current_model_generated_imgs.png")