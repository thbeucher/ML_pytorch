import os
import torch
import random
import logging
import argparse
import torchvision
import numpy as np

from PIL import Image
from tqdm import tqdm
from torchvision.datasets import MNIST
from sklearn.metrics import classification_report
from torchvision.transforms import Compose, ToTensor, Normalize


class MNISTSiamese(MNIST):
  def __init__(self, root, train=True, transform=None, target_transform=None, download=False, classif_mode=False):
    super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
    self.classif_mode = classif_mode

    self.ref_imgs = [Image.open(os.path.join('imgs/', img_fname)) for img_fname in sorted(os.listdir('imgs/'))]
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

      return (img, img_ref), 0 if target == target_ref else 1
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


class MNISTCompTrainer(object):
  def __init__(self, config):
    base_config = {'dataset_path': 'data/', 'batch_size': 128, 'n_workers': 8, 'save_name': 'model/comp_mnist_model.pt',
                   'n_epochs': 200, 'eval_step': 20}
    self.config = {**base_config, **config}

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.instanciate_model()
    self.set_dataloader()

    self.criterion = torch.nn.BCELoss()
    self.optimizer = torch.optim.Adam(self.model.parameters())

    self.best_f1 = 0.
  
  def instanciate_model(self):  # 128 if 28x28, 1024 if 28x56
    self.model = torch.nn.Sequential(torch.nn.Conv2d(1, 32, 3), torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(32),
                                     torch.nn.Conv2d(32, 32, 3), torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(32),
                                     torch.nn.Conv2d(32, 32, 5, 2, 2), torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(32),
                                     torch.nn.Dropout(0.4),
                                     torch.nn.Conv2d(32, 64, 3), torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(64),
                                     torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(64),
                                     torch.nn.Conv2d(64, 64, 5, 2, 2), torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(64),
                                     torch.nn.Dropout(0.4),
                                     torch.nn.Conv2d(64, 128, 4), torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(128),
                                     torch.nn.Flatten(), torch.nn.Dropout(0.4), torch.nn.Linear(1024, 1))
    self.model.to(self.device)

  def set_dataloader(self):
    self.transform = Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])
    self.train_data_loader = torch.utils.data.DataLoader(MNISTSiamese(self.config['dataset_path'], train=True,
                                                                      download=True, transform=self.transform),
                                                         batch_size=self.config['batch_size'], num_workers=self.config['n_workers'],
                                                         shuffle=True, pin_memory=True)
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
      for (img, img_ref), target in tqdm(self.train_data_loader, leave=False):
        out = self.model(torch.cat([img, img_ref], dim=-1).to(self.device))
        loss = self.criterion(out.sigmoid(), target.float().view(-1, 1).to(self.device))
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        losses.append(loss.item())

      logging.info(f'Epoch {epoch} | loss={np.mean(losses):.4f}')
      f1 = self.evaluation()
      logging.info(f'Epoch {epoch} | f1 = {f1:.3f}')

      if f1 > self.best_f1:
        self.save_model()
        self.best_f1 = f1

  @torch.no_grad()
  def evaluation(self, print_res=False):
    predictions, targets = [], []
    for (img, img_ref), target in tqdm(self.test_data_loader, leave=False):
      targets += target.tolist()
      out = self.model(torch.cat([img, img_ref], dim=-1).to(self.device))
      predictions += (out.sigmoid() > 0.5).int().cpu().tolist()
    if print_res:
      print(f'TEST results:\n{classification_report(targets, predictions, zero_division=0)}')
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


if __name__ == "__main__":
  argparser = argparse.ArgumentParser(prog='comp_exps_mnist.py', description='')
  argparser.add_argument('--log_file', default='_tmp_comp_exps_mnist_logs.txt', type=str)
  argparser.add_argument('--dataset_path', default='data/', type=str)
  argparser.add_argument('--n_workers', default=8, type=int)
  argparser.add_argument('--random_seed', default=42, type=int)
  argparser.add_argument('--save_model', default='model/comp_mnist_model.pt', type=str)
  argparser.add_argument('--batch_size', default=128, type=int)
  args = argparser.parse_args()

  logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

  torch.manual_seed(args.random_seed)

  mnist_comp_trainer = MNISTCompTrainer({'dataset_path': args.dataset_path, 'n_workers': args.n_workers,
                                         'save_name': args.save_model, 'batch_size': args.batch_size})
  
  rep = input('Load model? (y or n): ')
  if rep == 'y':
    mnist_comp_trainer.load_model(map_location=mnist_comp_trainer.device, save_name=args.save_model)

  rep = input('Start training? (y or n): ')
  if rep == 'y':
    mnist_comp_trainer.train()
  
  rep = input('Start evaluation? (y or n): ')
  if rep == 'y':
    mnist_comp_trainer.evaluation(print_res=True)