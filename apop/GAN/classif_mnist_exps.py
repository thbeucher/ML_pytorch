import os
import sys
import torch
import random
import logging
import argparse
import torchvision
import numpy as np

from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torchvision.datasets import MNIST
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Normalize

sys.path.append(os.path.abspath(__file__).replace('GAN/classif_mnist_exps.py', ''))

import utils as u
import models as m


class MNISTSiamese(MNIST):
  def __init__(self, root, train=True, transform=None, target_transform=None, download=False, classif_mode=False):
    super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
    self.classif_mode = classif_mode

    imgs_folder = os.path.abspath(__file__).replace('classif_mnist_exps.py', 'imgs/')
    self.ref_imgs = [Image.open(os.path.join(imgs_folder, img_fname)) for img_fname in sorted(os.listdir(imgs_folder))]
    self.ref_imgs_target = list(range(10))
  
  def __getitem__(self, index):
    img, target = self.data[index], int(self.targets[index])

    if not self.classif_mode:
      if torch.rand(1) < 0.5:
        idx = target
      else:
        idx = random.sample([i for i in self.ref_imgs_target if i != target], 1)[0]

      img_ref = self.ref_imgs[idx]
      target_ref = idx

      # doing this so that it is consistent with all other datasets
      # to return a PIL Image
      img = Image.fromarray(img.numpy(), mode="L")

      if self.transform is not None:
          img = self.transform(img)
          img_ref = self.transform(img_ref)

      if self.target_transform is not None:
          target = self.target_transform(target)
          target_ref = self.target_transform(target_ref)

      return (img, img_ref), (0 if target == target_ref else 1, target)
    else:
      img = Image.fromarray(img.numpy(), mode="L")

      ref_imgs = self.ref_imgs
      ref_targets = self.ref_imgs_target

      if self.transform is not None:
          img = self.transform(img)
          ref_imgs = [self.transform(im) for im in ref_imgs]

      if self.target_transform is not None:
          target = self.target_transform(target)
          ref_targets = [self.target_transform(t) for t in ref_targets]
      
      return (img, ref_imgs), (target, ref_targets)


class CompTrainer(object):
  def __init__(self, config):
    base_config = {'dataset_path': 'data/', 'batch_size': 128, 'n_workers': 8, 'save_name': 'model/comp_mnist_model.pt',
                   'n_epochs': 101, 'eval_step': 20, 'percent': 1.}
    self.config = {**base_config, **config}
    u.dump_dict(self.config, 'MNIST Experiment configuration')

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.instanciate_model()
    self.set_dataloader()

    self.criterion = torch.nn.BCELoss()
    self.optimizer = torch.optim.Adam(self.model.parameters())

    self.best_f1 = 0.
  
  def instanciate_model(self):  # 128 if 28x28, 1024 if 28x56
    self.model = m.MNISTClassifier({'n_classes': 1,
                                    'heads_config': [[{'type': torch.nn.Linear,
                                                       'params': {'in_features': 1024, 'out_features': 1}}]]})
    self.model.to(self.device)

  def set_dataloader(self):
    self.transform = Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])

    dataset = MNISTSiamese(self.config['dataset_path'], train=True, download=True, transform=self.transform)
    if self.config['percent'] < 1.:
      train_idx, _ = train_test_split(list(range(len(dataset))), train_size=self.config['percent'], stratify=dataset.targets)
      dataset = torch.utils.data.Subset(dataset, train_idx)

    self.train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.config['batch_size'],
                                                         num_workers=self.config['n_workers'], shuffle=True, pin_memory=True)
    self.test_data_loader = torch.utils.data.DataLoader(MNISTSiamese(self.config['dataset_path'], train=False,
                                                                     download=True, transform=self.transform),
                                                        batch_size=self.config['batch_size'], num_workers=self.config['n_workers'],
                                                        shuffle=False, pin_memory=True)
    self.test_data_loader2 = torch.utils.data.DataLoader(MNISTSiamese(self.config['dataset_path'], train=False, download=True,
                                                                      transform=self.transform, classif_mode=True),
                                                        batch_size=self.config['batch_size'], num_workers=self.config['n_workers'],
                                                        shuffle=False, pin_memory=True)
  
  def train_siamese(self):
    pass

  def train(self):
    for epoch in tqdm(range(self.config['n_epochs'])):
      losses = []
      for (img, img_ref), (target, _) in tqdm(self.train_data_loader, leave=False):
        out = self.model(torch.cat([img, img_ref], dim=-1).to(self.device))
        loss = self.criterion(out.sigmoid(), target.float().view(-1, 1).to(self.device))
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        losses.append(loss.item())

      logging.info(f'CompTrainer - Epoch {epoch} | loss={np.mean(losses):.4f}')
      f1 = self.evaluation()
      logging.info(f'CompTrainer - Epoch {epoch} | f1 = {f1:.4f}')

      if f1 > self.best_f1:
        f1_classif = self.classif_evaluation()
        logging.info(f'CompTrainer - Epoch {epoch} | save model with f1={f1:.4f} | f1_classif={f1_classif:.4f}')
        self.save_model()
        self.best_f1 = f1

  @torch.no_grad()
  def evaluation(self, print_res=False, digits=2):
    self.model.eval()
    predictions, targets = [], []
    for (img, img_ref), (target, _) in tqdm(self.test_data_loader, leave=False):
      targets += target.tolist()
      out = self.model(torch.cat([img, img_ref], dim=-1).to(self.device))
      predictions += (out.sigmoid() > 0.5).int().cpu().tolist()
    self.model.train()
    if print_res:
      print(f'TEST results:\n{classification_report(targets, predictions, zero_division=0, digits=digits)}')
    else:
      return classification_report(targets, predictions, zero_division=0, output_dict=True)['weighted avg']['f1-score']
  
  @torch.no_grad()
  def classif_evaluation(self, print_res=False, digits=2):
    self.model.eval()
    predictions, targets = [], []
    for (img, imgs_ref), (target, _) in tqdm(self.test_data_loader2, leave=False):
      targets += target.tolist()
      outs = [self.model(torch.cat([img, img_ref], dim=-1).to(self.device)) for img_ref in imgs_ref]
      predictions += torch.stack(outs).squeeze(-1).T.argmin(-1).cpu().tolist()
    self.model.train()
    if print_res:
      print(f'TEST results:\n{classification_report(targets, predictions, zero_division=0, digits=digits)}')
    else:
      return classification_report(targets, predictions, zero_division=0, output_dict=True)['weighted avg']['f1-score']
  
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


class ClassifTrainer(CompTrainer):
  def __init__(self, config):
    config['save_name'] = 'model/classif_mnist_model.pt'
    super().__init__(config)

    self.criterion = torch.nn.CrossEntropyLoss()
  
  def instanciate_model(self):
    self.model = m.MNISTClassifier({})
    self.model.to(self.device)
  
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
        out = self.model(img.to(self.device))
        loss = self.criterion(out, target.to(self.device))
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        losses.append(loss.item())

      logging.info(f'ClassifTrainer - Epoch {epoch} | loss={np.mean(losses):.4f}')
      f1 = self.evaluation()
      logging.info(f'ClassifTrainer - Epoch {epoch} | f1 = {f1:.4f}')

      if f1 > self.best_f1:
        logging.info(f'ClassifTrainer - Epoch {epoch} | save model with f1={f1:.4f}')
        self.save_model()
        self.best_f1 = f1

  @torch.no_grad()
  def evaluation(self, print_res=False, digits=2):
    self.model.eval()
    predictions, targets = [], []
    for img, target in tqdm(self.test_data_loader, leave=False):
      targets += target.tolist()
      out = self.model(img.to(self.device))
      predictions += out.argmax(-1).cpu().tolist()
    self.model.train()
    if print_res:
      print(f'TEST results:\n{classification_report(targets, predictions, zero_division=0, digits=digits)}')
    else:
      return classification_report(targets, predictions, zero_division=0, output_dict=True)['weighted avg']['f1-score']


class CompClassifTrainer(CompTrainer):
  def __init__(self, config):
    config['save_name'] = 'model/comp_classif_mnist_model.pt'
    super().__init__(config)

    self.criterion_classif = torch.nn.CrossEntropyLoss()
  
  def instanciate_model(self):
    self.model = m.MNISTClassifier({'n_classes': 1,
                                    'heads_config': [[{'type': torch.nn.Linear,
                                                       'params': {'in_features': 1024, 'out_features': 1}},
                                                       {'type': torch.nn.Sigmoid, 'params': {}}],
                                                      [{'type': torch.nn.Linear,
                                                        'params': {'in_features': 1025, 'out_features': 10}}]]})
    self.model.to(self.device)
  
  def train(self):
    for epoch in tqdm(range(self.config['n_epochs'])):
      losses = []
      for (img, img_ref), (target, target_classif) in tqdm(self.train_data_loader, leave=False):
        out_comp, out_classif = self.model(torch.cat([img, img_ref], dim=-1).to(self.device))
        loss_comp = self.criterion(out_comp.sigmoid(), target.float().view(-1, 1).to(self.device))
        loss_classif = self.criterion_classif(out_classif, target_classif.to(self.device))
        loss = loss_comp + loss_classif
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        losses.append(loss.item())

      logging.info(f'CompClassifTrainer - Epoch {epoch} | loss={np.mean(losses):.4f}')
      f1_comp, f1_classif = self.evaluation()
      logging.info(f'CompClassifTrainer - Epoch {epoch} | f1_comp = {f1_comp:.4f} | f1_classif = {f1_classif:.4f}')

      if f1_classif > self.best_f1:
        logging.info(f'CompClassifTrainer - Epoch {epoch} | save model with f1={f1_comp:.4f} | f1_classif={f1_classif:.4f}')
        self.save_model()
        self.best_f1 = f1_classif

  @torch.no_grad()
  def evaluation(self, print_res=False, digits=2):
    self.model.eval()
    predictions_comp, predictions_classif, targets_comp, targets_classif = [], [], [], []
    for (img, img_ref), (target, target_classif) in tqdm(self.test_data_loader, leave=False):
      targets_comp += target.tolist()
      targets_classif += target_classif.tolist()
      out_comp, out_classif = self.model(torch.cat([img, img_ref], dim=-1).to(self.device))
      predictions_comp += (out_comp.sigmoid() > 0.5).int().cpu().tolist()
      predictions_classif += out_classif.argmax(-1).cpu().tolist()
    self.model.train()
    if print_res:
      print(f'TEST results:\nCOMP:\n{classification_report(targets_comp, predictions_comp, zero_division=0, digits=digits)}')
      print(f'\nCLASSIF:\n{classification_report(targets_classif, predictions_classif, zero_division=0, digits=digits)}')
    else:
      f1_comp = classification_report(targets_comp, predictions_comp, zero_division=0,
                                      output_dict=True)['weighted avg']['f1-score']
      f1_classif = classification_report(targets_classif, predictions_classif, zero_division=0,
                                         output_dict=True)['weighted avg']['f1-score']
      return f1_comp, f1_classif


if __name__ == "__main__":
  argparser = argparse.ArgumentParser(prog='classif_mnist_exps.py', description='')
  argparser.add_argument('--log_file', default='_tmp_classif_mnist_exps_logs.txt', type=str)
  argparser.add_argument('--dataset_path', default='data/', type=str)
  argparser.add_argument('--n_workers', default=8, type=int)
  argparser.add_argument('--random_seed', default=42, type=int)
  argparser.add_argument('--save_model', default='model/comp_mnist_model.pt', type=str)
  argparser.add_argument('--batch_size', default=128, type=int)
  argparser.add_argument('--trainer', default='comp', type=str)
  argparser.add_argument('--digits', default=2, type=int)
  args = argparser.parse_args()

  logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

  torch.manual_seed(args.random_seed)

  trainers = {'classif': ClassifTrainer, 'comp': CompTrainer, 'comp_classif': CompClassifTrainer}

  mnist_trainer = trainers[args.trainer]({'dataset_path': args.dataset_path, 'n_workers': args.n_workers,
                                          'save_name': args.save_model, 'batch_size': args.batch_size})
  
  rep = input(f'Load {args.trainer} model? (y or n): ')
  if rep == 'y':
    print(f'Load {args.save_model} model...')
    mnist_trainer.load_model(map_location=mnist_trainer.device, save_name=args.save_model)

  rep = input(f'Start {args.trainer} training? (y or n): ')
  if rep == 'y':
    mnist_trainer.train()
  
  rep = input(f'Start {args.trainer} evaluation? (y or n): ')
  if rep == 'y':
    mnist_trainer.evaluation(print_res=True, digits=args.digits)
  
  rep = input(f'Train/evaluate {args.trainer} on subsampling? (y or n): ')
  if rep == 'y':
    # [12, 24, 54, 102, 504, 1020, 2040, 4020, 6000]
    for percent in [0.002, 0.004, 0.009, 0.017, 0.084, 0.17, 0.34, 0.67, 1.]:
      logging.info(f'START {args.trainer} training with train_size = {percent}')
      mnist_trainer.config['percent'] = percent
      mnist_trainer.set_dataloader()
      mnist_trainer.train()
      logging.info(f'END {args.trainer} training with train_size = {percent}')