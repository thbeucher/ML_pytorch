import os
import sys
import torch
import logging
import argparse
import torchvision
import numpy as np

from tqdm import tqdm
from bayes_opt import BayesianOptimization
from collections import defaultdict, Counter
from sklearn.metrics import classification_report

sys.path.append(os.path.abspath(__file__).replace('SNN/simple_snn_exp.py', ''))

import models.snn.stdp as lf
import models.snn.functional as f
import models.snn.temporal_order_coding_image as toci

from models.snn.layer import Linear, Convolution


class ReceptiveField(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
    self.bias = bias
    
    weights = ReceptiveField._weights_manhattan(kernel_size=self.kernel_size)
    self.weights = torch.nn.Parameter(weights.unsqueeze(0).unsqueeze(0), requires_grad=False)
  
  def forward(self, x, padding=None):
    return torch.nn.functional.conv2d(x, self.weights, bias=self.bias, stride=self.stride,
                                      padding=self.padding if padding is None else padding,
                                      dilation=self.dilation, groups=self.groups)
  
  @staticmethod
  def _weights_manhattan(kernel_size=(5, 5)):
    ker_x, ker_y = kernel_size
    w = torch.zeros([ker_x, ker_y])
    x_ori, y_ori = ker_x // 2, ker_y // 2
    for i in range(ker_x):
        for j in range(ker_y):
            d = abs(x_ori - i) + abs(y_ori - j)
            w[i][j] = (-0.375) * d + 1
    return w


class DataTransformer(object):
  def __init__(self, configuration={}):
    base_config = {'n_time_steps': 20}
    self.config = {**base_config, **configuration}

    self.to_tensor = torchvision.transforms.ToTensor()
  
  def __call__(self, image):
    return toci.cumulative_intensity_to_latency(self.to_tensor(image) * 255, self.config['n_time_steps'])


class SimpleSNNExp(object):
  def __init__(self, config):
    base_config = {'dataset_path': 'data/', 'batch_size': 1, 'n_workers': 8, 'image_size': (28, 28),
                   'n_output_neurons': 16, 'threshold': 50, 'n_epoch': 10, 'strategy': 'wda',
                   'max_ap': torch.tensor([0.15]), 'an_update': -0.75, 'timestep_update_lr': 2000,
                   'save_path': 'model/simple_snn_exp.pt', 'n_winners': 1}
    self.config = {**base_config, **config}

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.lr = [torch.tensor([0.004]).to(self.device), torch.tensor([-0.003]).to(self.device)]

    self.set_dataloader()
    self.instanciate_model()
  
  def set_dataloader(self):
    self.train_data_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(self.config['dataset_path'], train=True,
                                                                                    download=True, transform=DataTransformer()),
                                                         batch_size=self.config['batch_size'], num_workers=self.config['n_workers'],
                                                         shuffle=True, pin_memory=True, collate_fn=lambda x: x[0])
    self.test_data_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(self.config['dataset_path'], train=False,
                                                                                   download=True, transform=DataTransformer()),
                                                        batch_size=self.config['batch_size'], num_workers=self.config['n_workers'],
                                                        shuffle=True, pin_memory=True, collate_fn=lambda x: x[0])
  
  def instanciate_model(self):
    self.layer = Linear(np.prod(self.config['image_size']), self.config['n_output_neurons'], self.lr, None,
                        strategy=self.config['strategy'])
  
  def forward(self, input_spikes, training=False):
    input_spikes = input_spikes.view(20, -1)
    potentials = self.layer(input_spikes)
    spikes, potentials = f.fire(potentials, self.config['threshold'], return_thresholded_potentials=True)
    if training:
      self.layer.stdp(potentials, input_spikes, spikes)
    return spikes, potentials
  
  def train(self):
    for epoch in tqdm(range(self.config['n_epoch'])):
        for j, (spikes_in, target) in enumerate(tqdm(self.train_data_loader, leave=False)):  # [20, 28, 28]
          self.forward(spikes_in.to(self.device), training=True)  # [20, 16]

          if j % self.config['timestep_update_lr'] == 0:
            new_ap = torch.min(self.layer.lr[0] * 2, self.config['max_ap'])
            new_an = self.layer.lr[0] * self.config['an_update']
            self.layer._update_all_lr(new_ap, new_an)
          
          # if j % 20000 == 0:
          #   network_f1 = self.evaluate(print_res=False)
          #   print(f'f1 = {network_f1:.3f}')

          if j % 5000 == 0:
            C = (self.layer.weights * (1 - self.layer.weights)).sum() / np.prod(self.layer.weights.shape)
            # print(f'C = {C}')
            if C < 1e-4:
              break
  
  def get_internal_mapping(self):
    counter = defaultdict(list)
    targets = []
    for spikes_in, target in tqdm(self.train_data_loader, leave=False):
      targets.append(target)
      spikes_out, potentials = self.forward(spikes_in.to(self.device))
      pred = self.layer.get_winner(potentials, spikes_out).item()
      counter[pred].append(target)
    
    # for i in range(16):
    #   print(i, Counter(counter[i]))

    mapping = {n: Counter(t).most_common(1)[0][0] for n, t in counter.items()}
    mapping[0] = 8
    return mapping
  
  def evaluate(self, print_res=True):
    mapping = self.get_internal_mapping()
    targets, predictions = [], []
    for spikes_in, target in tqdm(self.test_data_loader, leave=False):
      targets.append(target)
      spikes_out, potentials = self.forward(spikes_in.to(self.device))
      winner = self.layer.get_winner(potentials, spikes_out).item()
      predictions.append(mapping[winner])
    
    if print_res:
      print(f'TEST results:\n{classification_report(targets, predictions, zero_division=0)}')
    else:
      return classification_report(targets, predictions, zero_division=0, output_dict=True)['weighted avg']['f1-score']
  
  def load_model(self, map_location=None, save_name=None):
    save_name = self.config['save_path'] if save_name is None else save_name
    if os.path.isfile(save_name):
      data = torch.load(save_name, map_location=map_location)
      self.layer.load_state_dict(data['model'])
      self.layer.lr = data['lr']
    else:
      print(f"File {save_name} doesn't exist")
  
  def save_model(self, save_name=None):
    save_name = self.config['save_path'] if save_name is None else save_name
    if not os.path.isdir(os.path.dirname(save_name)):
      os.makedirs(os.path.dirname(save_name))
    torch.save({'model': self.layer.state_dict(), 'lr': self.layer.lr}, save_name)


def to_optimize(threshold=50, n_output_neurons=16, strategy=0, timestep_update_lr=2000):
  params = {'n_workers': 0, 'threshold': int(threshold), 'timestep_update_lr': int(timestep_update_lr),
            'strategy': ['wta', 'wda'][int(round(strategy))], 'n_output_neurons': int(n_output_neurons)}
  try:
    simple = SimpleSNNExp(params)
    simple.train()
    f1 = simple.evaluate(print_res=False)
    status = 'OK'
  except:
    f1 = 0.
    status = 'ERROR'
  logging.info(f'Optimization - params={params} - status={status} - f1={f1:.3f}')
  return f1


if __name__ == "__main__":
  # https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-015-0059-4#additional-information
  # linear -> conv
  # linear -> input = [20, 1, 28, 28] | weights = [784, 16]       -> input.view(20, -1).mm(weights) = [20, 16]
  # conv   ->                         | weights = [16, 784, 1, 1] -> 
  # torch.nn.functional.conv2d(input.view(20, 1, 784, 1).permute(0, 2, 1, 3), weights) = [20, 16, 1, 1]
  argparser = argparse.ArgumentParser(prog='simple_snn_exp.py', description='')
  argparser.add_argument('--log_file', default='_tmp_simple_snn_exp_logs.txt', type=str)
  argparser.add_argument('--dataset_path', default='data/', type=str)
  argparser.add_argument('--n_workers', default=8, type=int)
  args = argparser.parse_args()

  logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

  random_seed = 42
  torch.manual_seed(random_seed)

  simple = SimpleSNNExp({'n_workers': args.n_workers})

  rep = input('Load model? (y or n): ')
  if rep == 'y':
    simple.load_model()

  rep = input('Train Network? (y or n): ')
  if rep == 'y':
    simple.train()
    simple.save_model()
  
  rep = input('Eval network? (y or n): ')
  if rep == 'y':
    simple.load_model()
    simple.evaluate()
  
  rep = input('Optimize hyperparameters? (y or n): ')
  if rep == 'y':
    pbounds = {'threshold': (1, 500), 'n_output_neurons': (10, 100), 'strategy': (0, 1), 'timestep_update_lr': (500, 20000)}
    optimizer = BayesianOptimization(f=to_optimize, pbounds=pbounds, verbose=2, random_state=1)
    optimizer.maximize(init_points=15, n_iter=10)
    print(optimizer.max)

    for i, res in enumerate(optimizer.res):
      print(f"Iteration {i}: {res}")
  

  '''
  Params:
    ->threshold = 50    ->n_output_neurons=16    ->strategy=wda    ->timestep_update_lr=2000
  Results:
                  precision    recall  f1-score   support

              0       0.73      0.80      0.77       980
              1       0.96      0.23      0.37      1135
              2       0.80      0.58      0.68      1032
              3       0.60      0.72      0.65      1010
              4       0.52      0.51      0.51       982
              5       0.35      0.42      0.38       892
              6       0.83      0.66      0.73       958
              7       0.82      0.69      0.75      1028
              8       0.22      0.47      0.30       974
              9       0.47      0.46      0.46      1009

       accuracy                           0.55     10000
      macro avg       0.63      0.55      0.56     10000
   weighted avg       0.64      0.55      0.56     10000
  '''