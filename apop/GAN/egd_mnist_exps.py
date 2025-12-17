import os
import sys
import torch
import logging
import argparse
import numpy as np

from tqdm import tqdm
from itertools import chain
from torchvision.datasets import MNIST
from sklearn.metrics import classification_report
from torchvision.utils import make_grid, save_image
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Normalize

sys.path.append(os.path.abspath(__file__).replace('GAN/egd_mnist_exps.py', ''))

import models.gan_vae_divers as m


class EGDTrainer(object):
  BASE_CONFIG = {'dataset_path': 'data/', 'batch_size': 32, 'n_workers': 8, 'save_name': 'model/egd_mnist_model.pt',
                 'n_epochs': 1001, 'eval_step': 10, 'percent': 1., 'save_img_folder': 'generated_egd_imgs/',
                 'save_preds_tmp': '_tmp_egd_mnist_test_preds.pt'}
  def __init__(self, config):
    super().__init__()
    self.config = {**EGDTrainer.BASE_CONFIG, **config}

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.instanciate_model()
    self.set_dataloader()

    self.rec_criterion = torch.nn.MSELoss()
    self.classif_criterion = torch.nn.CrossEntropyLoss()

    self.optimizer = torch.optim.AdamW(chain(self.vqvae.parameters(), self.classifier.parameters()))

    self.test_imgs = torch.stack([self.test_data_loader.dataset.__getitem__(i)[0] for i in range(10)]).to(self.device)
    self.test_targets = torch.tensor([self.test_data_loader.dataset.__getitem__(i)[1] for i in range(10)]).to(self.device)

    self.best_f1 = 0.

    if not os.path.isdir(self.config['save_img_folder']):
      os.makedirs(self.config['save_img_folder'])
    save_image(make_grid(self.test_imgs, nrow=10), os.path.join(self.config['save_img_folder'], 'imgs_original.png'))
  
  def instanciate_model(self):
    self.vqvae = m.VQVAEModel({}).to(self.device)
    self.classifier = m.MNISTClassifier({}).to(self.device)  # B*1*28*28 -> B*10
  
  def set_dataloader(self):
    self.transform = Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])

    dataset = MNIST(self.config['dataset_path'], train=True, download=True, transform=self.transform)
    if self.config['percent'] < 1.:
      train_idx, _ = train_test_split(list(range(len(dataset))), train_size=self.config['percent'], stratify=dataset.targets)
      dataset = torch.utils.data.Subset(dataset, train_idx)

    self.train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.config['batch_size'],
                                                         num_workers=self.config['n_workers'], shuffle=True, pin_memory=True)
    self.test_data_loader = torch.utils.data.DataLoader(MNIST(self.config['dataset_path'], train=False,
                                                              download=True, transform=self.transform),
                                                        batch_size=self.config['batch_size'], num_workers=self.config['n_workers'],
                                                        shuffle=False, pin_memory=True)
  
  def train(self):
    for epoch in tqdm(range(self.config['n_epochs'])):
      losses = []
      for img, target in tqdm(self.train_data_loader, leave=False):
        img, target = img.to(self.device), target.to(self.device)
        vq_loss, img_rec, perplexity, encodings, quantized = self.vqvae(img)
        out = self.classifier(img_rec)

        self.optimizer.zero_grad()
        rec_loss = self.rec_criterion(img_rec, img)
        classif_loss = self.classif_criterion(out, target)
        loss = vq_loss + rec_loss + classif_loss
        loss.backward()
        self.optimizer.step()

        losses.append(loss.item())

      self.generate_img(save_name=os.path.join(self.config['save_img_folder'], f'imgs_reconstructed_epoch{epoch}.png'))
      logging.info(f"Epoch {epoch} | loss = {np.mean(losses)}")
      f1 = self.evaluation()
      logging.info(f'Epoch {epoch} | f1 = {f1:.3f}')
      
      if f1 > self.best_f1:
        logging.info(f'Epoch {epoch} | save model with f1={f1:.4f}')
        self.save_model()
        self.best_f1 = f1

  @torch.no_grad()
  def evaluation(self, print_res=False, digits=2):
    self._mode(mode='eval')
    predictions, targets, confidences = [], [], []
    for img, target in tqdm(self.test_data_loader, leave=False):
      targets += target.tolist()
      vq_loss, img_rec, perplexity, encodings, quantized = self.vqvae(img.to(self.device))
      out = self.classifier(img_rec)
      confidences += out.softmax(-1).cpu().tolist()
      predictions += out.argmax(-1).cpu().tolist()
    self._mode()

    f1 = classification_report(targets, predictions, zero_division=0, output_dict=True)['weighted avg']['f1-score']
    if f1 > self.best_f1:
      torch.save({'targets': targets, 'predictions': predictions, 'confidences': confidences}, self.config['save_preds_tmp'])

    if print_res:
      print(f'TEST results:\n{classification_report(targets, predictions, zero_division=0, digits=digits)}')

    return f1
  
  @torch.no_grad()
  def generate_img(self, save_name=None):
    self._mode(mode='eval')
    _, img_rec, *_ = self.vqvae(self.test_imgs)
    save_image(make_grid(img_rec.cpu(), nrow=10), save_name)
    self._mode()

  def _mode(self, mode='train'):
    if mode == 'eval':
      self.vqvae.eval()
      self.classifier.eval()
    else:
      self.vqvae.train()
      self.classifier.train()
  
  def save_model(self, save_name=None):
    save_name = self.config['save_name'] if save_name is None else save_name
    if not os.path.isdir(os.path.dirname(save_name)):
      os.makedirs(os.path.dirname(save_name))
    torch.save({'vqvae': self.vqvae.state_dict(),
                'classifier': self.classifier.state_dict()}, save_name)

  def load_model(self, save_name=None, map_location=None):
    save_name = self.config['save_name'] if save_name is None else save_name
    if os.path.isfile(save_name):
      data = torch.load(save_name, map_location=map_location)
      self.vqvae.load_state_dict(data['vqvae'])
      self.classifier.load_state_dict(data['classifier'])
    else:
      print(f"File {save_name} doesn't exist")


class EGDrelaxTrainer(EGDTrainer):
  BASE_CONFIG = {'dataset_path': 'data/', 'batch_size': 32, 'n_workers': 8, 'save_name': 'model/egd_relax_mnist_model.pt',
                 'n_epochs': 1001, 'eval_step': 10, 'percent': 1., 'save_img_folder': 'generated_egd_relax_imgs/',
                 'save_preds_tmp': '_tmp_egd_relax_mnist_test_preds.pt'}
  def __init__(self, config):
    config = {**EGDrelaxTrainer.BASE_CONFIG, **config}
    super().__init__(config)
  
  def train(self):
    start_relaxing_factor, n_timestep_relaxing = 1., 200
    relaxing_decay = 1 / n_timestep_relaxing
    relaxing_factor = start_relaxing_factor
    for epoch in tqdm(range(self.config['n_epochs'])):
      losses = []
      for img, target in tqdm(self.train_data_loader, leave=False):
        img, target = img.to(self.device), target.to(self.device)

        vq_loss, img_rec, perplexity, encodings, quantized = self.vqvae(img)
        out = self.classifier(img_rec)

        self.optimizer.zero_grad()
        rec_loss = self.rec_criterion(img_rec, img)
        classif_loss = self.classif_criterion(out, target)
        loss = vq_loss + relaxing_factor * rec_loss + classif_loss
        loss.backward()
        self.optimizer.step()

        losses.append(loss.item())

      relaxing_factor = max(0, start_relaxing_factor - relaxing_decay * epoch)

      self.generate_img(save_name=os.path.join(self.config['save_img_folder'], f'imgs_reconstructed_epoch{epoch}.png'))
      logging.info(f"Epoch {epoch} | loss = {np.mean(losses)} | relaxing_factor = {relaxing_factor:.3f}")
      f1 = self.evaluation()
      logging.info(f'Epoch {epoch} | f1 = {f1:.3f}')
      
      if f1 > self.best_f1:
        logging.info(f'Epoch {epoch} | save model with f1={f1:.4f}')
        self.save_model()
        self.best_f1 = f1
  
  @torch.no_grad()
  def evaluation(self, print_res=False, digits=2, save_preds=True):
    self._mode(mode='eval')
    predictions, targets = [], []
    for img, target in tqdm(self.test_data_loader, leave=False):
      targets += target.tolist()
      target = torch.ones(len(target)).long().to(self.device)
      vq_loss, img_rec, perplexity, encodings, quantized = self.vqvae(img.to(self.device))
      out = self.classifier(img_rec)
      predictions += out.argmax(-1).cpu().tolist()
    self._mode()

    if save_preds:
      torch.save({'targets': targets, 'predictions': predictions}, self.config['save_preds_tmp'])

    if print_res:
      print(f'TEST results:\n{classification_report(targets, predictions, zero_division=0, digits=digits)}')
    else:
      return classification_report(targets, predictions, zero_division=0, output_dict=True)['weighted avg']['f1-score']


if __name__ == "__main__":
  # https://github.com/bshall/VectorQuantizedVAE/
  argparser = argparse.ArgumentParser(prog='egd_mnist_exps.py', description='')
  argparser.add_argument('--log_file', default='_tmp_egd_mnist_exps_logs.txt', type=str)
  argparser.add_argument('--dataset_path', default='data/', type=str)
  argparser.add_argument('--n_workers', default=8, type=int)
  argparser.add_argument('--random_seed', default=42, type=int)
  argparser.add_argument('--save_model', default='', type=str)
  argparser.add_argument('--batch_size', default=128, type=int)
  argparser.add_argument('--trainer', default='egd', type=str)
  argparser.add_argument('--digits', default=2, type=int)
  args = argparser.parse_args()

  logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

  torch.manual_seed(args.random_seed)

  trainers = {'egd': EGDTrainer, 'egd_relax': EGDrelaxTrainer}

  mnist_trainer = trainers[args.trainer]({'dataset_path': args.dataset_path, 'n_workers': args.n_workers,
                                          'batch_size': args.batch_size})
  
  if args.save_model != '':
    mnist_trainer.config['save_name'] = args.save_model
  
  rep = input(f'Load {args.trainer} model? (y or n): ')
  if rep == 'y':
    print(f"Load {mnist_trainer.config['save_name']} model...")
    mnist_trainer.load_model(map_location=mnist_trainer.device, save_name=mnist_trainer.config['save_name'])

  rep = input(f'Start {args.trainer} training? (y or n): ')
  if rep == 'y':
    mnist_trainer.train()
  
  rep = input(f'Start {args.trainer} evaluation? (y or n): ')
  if rep == 'y':
    mnist_trainer.evaluation(print_res=True, digits=args.digits)