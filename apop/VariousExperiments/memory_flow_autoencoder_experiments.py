import os
import torch
import random
import argparse
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from trainers_zoo import CNNAETrainer, FlowImagePredictorTrainer


class Trainer:
  CONFIG = {
    'experiment_save_dir': 'cifar10_exps',
    'experiment_name':     'memory_flow_ae',
    'data_dir':            'data/',
    'seed':                42,
    'batch_size':          128,
    'use_tf_logger':       True,
    'ae_config': {
      'experiment_name': 'memory_flow_ae',
      'model_config': {
        'encoder_archi':     'BigCNNEncoder',
        'skip_connection':   False,
        'linear_bottleneck': True,
        'latent_dim':        512,
      }
    },
    'flow_config': {
      'experiment_name': 'memory_flow_ae',
      'model_config': {
        'img_chan':       3,
        'time_dim':       64,
        'add_action':     False,
        'add_other_cond': True,
        'other_cond_dim': 512,
      }
    }
  }
  def __init__(self, config={}):
    self.config = {**Trainer.CONFIG, **config}
    self.config['ae_config']['experiment_name'] = self.config['experiment_name']
    self.config['flow_config']['experiment_name'] = self.config['experiment_name']
    self.device = torch.device('cuda' if torch.cuda.is_available() else
                               'mps' if torch.backends.mps.is_available() else
                               'cpu')
    self.set_seed()
    self.instanciate_trainers()
    self.set_dataloader()

    save_dir_run = os.path.join(self.config['experiment_save_dir'], self.config['experiment_name'], 'runs/')
    self.tf_logger = SummaryWriter(save_dir_run) if self.config['use_tf_logger'] else None
  
  def set_seed(self):
    # Set seeds for reproducibility
    torch.manual_seed(self.config['seed'])
    np.random.seed(self.config['seed'])
    random.seed(self.config['seed'])
    if self.device.type == 'cuda':
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

  def instanciate_trainers(self):
    self.ae_trainer = CNNAETrainer(self.config['ae_config'])
    self.flow_trainer = FlowImagePredictorTrainer(self.config['flow_config'])

  def set_dataloader(self, train_dataset=None, test_dataset=None, transform=None, num_workers=None):
    os.makedirs(self.config['data_dir'], exist_ok=True)

    if transform is None:
      transform = transforms.Compose([transforms.ToTensor()])

    if train_dataset is None:
      train_dataset = datasets.CIFAR10(root=self.config['data_dir'], train=True, download=True, transform=transform)
    if test_dataset is None:
      test_dataset = datasets.CIFAR10(root=self.config['data_dir'], train=False, download=True, transform=transform)

    if num_workers is None:
      num_workers = min(6, os.cpu_count())

    self.train_dataloader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True,
                                       num_workers=num_workers,
                                       pin_memory=True if torch.cuda.is_available() else False)
    self.test_dataloader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=True if torch.cuda.is_available() else False)
  
  def train(self):
    def infinite_loader(dataloader):
      while True:
        for batch in dataloader:
          yield batch
    dataiter = infinite_loader(self.train_dataloader)

    def get_ae_data_fn(*args, **kwargs):
      img, _ = next(dataiter)
      img = img.to(self.device)
      return {'image': img, 'target_image': img}

    if not self.ae_trainer.load():
      losses = self.ae_trainer.train(get_ae_data_fn, n_max_steps=len(self.train_dataloader)*20,
                                     tf_logger=self.tf_logger)
      print(f'mean_mse_loss={losses[0]:.4f} | mean_ssim_loss={losses[1]:.4f} | mean_rec_loss={losses[2]:.4f}')
      self.ae_trainer.save()

    self.ae_trainer.model.eval()
    def get_flow_data_fn(*args, **kwargs):
      img, _ = next(dataiter)
      img = img.to(self.device)
      with torch.no_grad():
        _, features = self.ae_trainer.model(img, return_latent=True)
      return {'target_image': img, 'other': features}
    
    self.flow_trainer.load()
    loss = self.flow_trainer.train(get_flow_data_fn, n_max_steps=len(self.train_dataloader)*50,
                                   condition_key='other', generate_every=100, tf_logger=self.tf_logger,
                                   n_gen_steps=4)
    self.flow_trainer.save()
    print(f'Mean_flow_loss={loss:.4f}')


def get_args():
  parser = argparse.ArgumentParser(description='Memory Flow AutoEncoder experiments')
  parser.add_argument('--experiment_name', '-en', type=str, default=None)
  return parser.parse_args()



if __name__ == '__main__':
  args = get_args()

  config = {} if args.experiment_name is None else {'experiment_name': args.experiment_name}

  trainer = Trainer(config)
  trainer.train()