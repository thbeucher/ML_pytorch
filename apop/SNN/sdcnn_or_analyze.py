# Taken (with modifications) from https://github.com/miladmozafari/SpykeTorch
# Take 1Go of memory on GPU for training, trained in 2h47, accuracy obtained = 98% (0.9829)
# using caching, training time = 2h35
# By using Convergence criteria to stop training, the training is done in 7min41 with acc=0.9798
# Convergence criteria + 16 winning neurons instead of 8 for 2nd layer -> 6min38, acc=0.9781
import os
import ast
import sys
import h5py
import torch
import logging
import argparse
import torchvision
import numpy as np
import pickle as pk
import torch.nn as nn

from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization

sys.path.append(os.path.abspath(__file__).replace('SNN/sdcnn_or_analyze.py', ''))

import utils as u
import models.snn.stdp as lf
import models.snn.functional as f
import models.snn.temporal_order_coding_image as toci

from models.snn.layer import Convolution, Linear


class DataTransformer(object):
  def __init__(self, configuration={}):
    base_config = {'kernel_type': toci.DoGKernel, 'kernels_conf': [[7, 1, 2], [7, 2, 1]], 'padding': 3,
                   'threshold': 50, 'use_threshold': True, 'normalization_radius': 8, 'n_time_steps': 15}
    self.config = u.populate_configuration(base_config, configuration)

    self.to_tensor = torchvision.transforms.ToTensor()
    self.filters = toci.Filter(self.config)
  
  def __call__(self, image):
    image = self.to_tensor(image) * 255
    image = self.filters(image.unsqueeze(0))
    image = toci.local_normalization(image, self.config['normalization_radius'])
    return toci.cumulative_intensity_to_latency(image, self.config['n_time_steps'])


class SDCNNExperiment(object):
  '''
    Spiking Deep Convolutional Neural Network for Object Recognition
    paper: https://www.sciencedirect.com/science/article/abs/pii/S0893608017302903
    Handle padding here and not leave it to conv2d as we need the input_spikes to be padded in computation of pre-post ordering
  '''
  def __init__(self, configuration={}):
    self.base_config = {'layers_config': [{'in_channels': 2, 'out_channels': 32, 'kernel_size': 5,
                                           'weight_mean': 0.8, 'weight_std': 0.05},
                                          {'in_channels': 32, 'out_channels': 150, 'kernel_size': 2,
                                           'weight_mean': 0.8, 'weight_std': 0.05}],
                        'learning_rates': (0.004, -0.003), 'max_ap': torch.tensor([0.15]), 'an_update': -0.75,
                        'thresholds': [10, 1], 'timestep_update_lr': 500,
                        'pooling_vars': [{'kernel_size': 2, 'stride': 2, 'padding': 1}] * 2,
                        'n_winners': [5, 8], 'inhibition_radius': [2, 1],
                        'batch_size': 1, 'dataset_path': 'data/', 'n_epochs': [2, 20],
                        'save_path': 'model/sdcnne_exp.pt', 'subset_percent': 1.0}
    self.config = u.populate_configuration(configuration, self.base_config)

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.config['max_ap'] = self.config['max_ap'].to(self.device)

    pad_fn = lambda x: (x - 1) // 2 + ((x-1) % 2 > 0)
    self.paddings_train = [[pad_fn(conf['kernel_size'])] * 4 for conf in self.config['layers_config']]
    self.paddings_forward = [pad_fn(conf['kernel_size']) for conf in self.config['layers_config']]

    self.set_metadata()
    self.set_dataloader()
    self.instanciate_model()

    # print('Caching dataset...')
    # self.train_dataset = [spike_target for spike_target in tqdm(self.train_data_loader)]

  def set_metadata(self):
    self.learning_rates = []
    for conf in self.config['layers_config']:
      ap, an = self.config['learning_rates']
      lr = [[torch.tensor([ap]).to(self.device), torch.tensor([an]).to(self.device)] for _ in range(conf['out_channels'])]
      self.learning_rates.append(lr)
  
  def set_dataloader(self):
    dataset = u.extract_subset(torchvision.datasets.MNIST(self.config['dataset_path'], train=True, download=True,
                                                          transform=DataTransformer()), self.config['subset_percent'])
    self.train_data_loader = torch.utils.data.DataLoader(dataset,
                                                         batch_size=self.config['batch_size'], num_workers=16,
                                                         shuffle=True, pin_memory=True, collate_fn=lambda x: x[0])
    self.test_data_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(self.config['dataset_path'], train=False,
                                                                                   download=True, transform=DataTransformer()),
                                                        batch_size=self.config['batch_size'], num_workers=8,
                                                        shuffle=True, pin_memory=True, collate_fn=lambda x: x[0])
  
  def instanciate_model(self):
    self.layers = nn.ModuleList([Convolution(**lc) for lc in self.config['layers_config']])
    self.layers.to(self.device)
  
  def update_all_lr(self, new_ap, new_an, layer_idx):
    for i in range(len(self.learning_rates[layer_idx])):
      self.learning_rates[layer_idx][i][0] = new_ap
      self.learning_rates[layer_idx][i][1] = new_an
  
  def competition_winner_stdp(self, input_spikes, potentials, layer_idx):
    potentials = f.pointwise_feature_competition_inhibition(potentials)  # [timestep, feat_out(eg32), height, width]
    spikes = potentials.sign()
    # logging.info(f'layer_idx={layer_idx} | sparsity_ratio={(spikes.max(dim=0)[0] == 0).sum()/np.prod(spikes.shape[1:])}')
    winners = f.get_k_winners(potentials, kwta=self.config['n_winners'][layer_idx],
                              inhibition_radius=self.config['inhibition_radius'][layer_idx], spikes=spikes)
    lf.functional_stdp(self.layers[layer_idx], self.learning_rates[layer_idx], input_spikes, spikes, winners)
    return spikes, potentials
  
  def forward(self, input_spikes, layer_idx, training=False):  # [timestep, feat_in(eg2), height, width]
    for i, layer in enumerate(self.layers):
      input_spikes = nn.functional.pad(input_spikes, self.paddings_train[i])  # [timestep, feat_in, height_pad, width_pad]

      if i > 0 and training:
        input_spikes = f.pointwise_feature_competition_inhibition(input_spikes)

      potentials = layer(input_spikes)  # [timestep, feat_out(eg32), heigth, width]
      spikes, potentials = f.fire(potentials, self.config['thresholds'][i], return_thresholded_potentials=True)

      if i == layer_idx and training:
        return self.competition_winner_stdp(input_spikes, potentials, layer_idx)
      
      input_spikes = nn.functional.max_pool2d(spikes, **self.config['pooling_vars'][i])
    
    return input_spikes
  
  def train(self):
    for i, n_epoch in enumerate(tqdm(self.config['n_epochs'])):
      for epoch in tqdm(range(n_epoch), leave=False):
        for j, (spikes_in, target) in enumerate(tqdm(self.train_data_loader, leave=False)):
        # for j, (spikes_in, target) in enumerate(tqdm(self.train_dataset, leave=False)):
          if i == 0 and j > 0 and j % self.config['timestep_update_lr'] == 0:
            new_ap = torch.min(self.learning_rates[0][0][0] * 2, self.config['max_ap'])
            new_an = self.learning_rates[0][0][0] * self.config['an_update']
            # logging.info(f'New learning rates = ({new_ap.cpu().item()},{new_an.cpu().item()})')
            self.update_all_lr(new_ap, new_an, 0)

          self.forward(spikes_in.to(self.device), i, training=True)

          if j % 5000 == 0:
            C = (self.layers[i].weights * (1 - self.layers[i].weights)).sum() / np.prod(self.layers[i].weights.shape)
            # logging.info(f'Layer={i} - epoch={epoch} - step={j} - Convergence_value={C.cpu().item()}')
            if C < 0.008:
              break
          
          if i == 0 and j % 50 == 0:
            with h5py.File('_tmp_kernel_conv1.h5', 'a') as hf:  # '_tmp_kernel_conv1_WLRScheduling.h5'
              hf.create_dataset(f'dump{j}', data=self.layers[i].weights.cpu().numpy())
  
  def evaluate(self, print_cr=True, save_res=True):
    from sklearn.svm import LinearSVC
    from sklearn.metrics import classification_report, f1_score

    classifier = LinearSVC(C=2.4)

    print('Retrieve train data...')
    x_train, y_train = [], []
    for spikes_in, target in tqdm(self.train_data_loader):
      out = self.forward(spikes_in.to(self.device), None)  # (15, 150, 9, 9)
      x_train.append(out.max(dim=0)[0].reshape(-1).cpu().numpy())
      y_train.append(target)
    
    print('Retrieve test data...')
    x_test, y_test = [], []
    for spikes_in, target in tqdm(self.test_data_loader):
      out = self.forward(spikes_in.to(self.device), None)  # (15, 150, 9, 9)
      x_test.append(out.max(dim=0)[0].reshape(-1).cpu().numpy())
      y_test.append(target)
    
    print('Launch classifier training...')
    classifier.fit(np.array(x_train), np.array(y_train))
    print('Start predictions...')
    preds_train = classifier.predict(np.array(x_train))
    preds_test = classifier.predict(np.array(x_test))

    if print_cr:
      print(f'TRAIN results:\n{classification_report(np.array(y_train), preds_train)}')
      print(f'TEST results:\n{classification_report(np.array(y_test), preds_test)}')

    if save_res:
      import pickle as pk
      with open('_tmp_xy_traintest_preds.pk', 'wb') as f:
        pk.dump({'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test,
                'preds_train': preds_train, 'preds_test': preds_test}, f)
    
    return f1_score(np.array(y_test), preds_test, average='weighted')
  
  def load_model(self, map_location='cpu'):
    if os.path.isfile(self.config['save_path']):
      data = torch.load(self.config['save_path'], map_location=map_location)
      self.layers.load_state_dict(data['model'])
      self.learning_rates = data['lr']
    else:
      print(f"File {self.config['save_path']} doesn't exist")
  
  def save_model(self):
    if not os.path.isdir(os.path.dirname(self.config['save_path'])):
      os.makedirs(os.path.dirname(self.config['save_path']))
    torch.save({'model': self.layers.state_dict(), 'lr': self.learning_rates}, self.config['save_path'])


class SDCNNExperiment2(SDCNNExperiment):
  def __init__(self, fire_threshold=450, n_output_neurons=200, convergence_value=0.008):
    super().__init__()
    self.convergence_value = convergence_value
    self.fire_threshold = int(fire_threshold)
    self.n_output_neurons = int(n_output_neurons)
    self.lr = [torch.tensor([0.004]).to(self.device), torch.tensor([0.003]).to(self.device)]

    self.decision_layer = Linear(12150, self.n_output_neurons, self.lr)
    self.decision_layer.to(self.device)

    self.save_path = 'model/decision_layer.pt'
  
  def forward2(self, input_spikes, training=False):
    potentials = self.decision_layer(input_spikes)
    spikes, potentials = f.fire(potentials, self.fire_threshold, return_thresholded_potentials=True)

    if training:
      self.decision_layer.stdp(potentials, input_spikes, spikes)
    
    return spikes, potentials
  
  def train2(self):
    for epoch in tqdm(range(20)):
      for j, (spikes_in, target) in enumerate(tqdm(self.train_data_loader, leave=False)):
        out = self.forward(spikes_in.to(self.device), 2)
        out = self.forward2(out.view(out.size(0), -1), training=True)

        if j > 0 and j % 1000 == 0:
          new_ap = torch.min(self.lr[0] * 2, self.config['max_ap'])
          new_an = self.lr[0] * self.config['an_update']
          # logging.info(f'New learning rates = ({new_ap.cpu().item()},{new_an.cpu().item()})')
          self.decision_layer._update_all_lr(new_ap, new_an)

        if j % 5000 == 0:
          C = (self.decision_layer.weights * (1 - self.decision_layer.weights)).sum() / np.prod(self.decision_layer.weights.shape)
          logging.info(f'EXP2 - epoch={epoch} - step={j} - C={C:.5f}')
          if C < self.convergence_value:
            break
  
  def evaluation2(self, train_data=True):
    targets, winners = [], []
    for spikes_in, target in tqdm(self.train_data_loader if train_data else self.test_data_loader):
      out = self.forward(spikes_in.to(self.device), 2)
      spikes, pot = self.forward2(out.view(out.size(0), -1))
      winner = self.decision_layer.get_winner(pot, spikes)

      targets.append(target)
      winners.append(winner.cpu().item())
    
    targets = np.array(targets)
    winners = np.array(winners)
    
    mc = [[el[0] for el in Counter(winners[targets==i].tolist()).most_common(100)] for i in range(10)]
    fmm = [(i, j, el2) for i, el1 in enumerate(mc) for j, el2 in enumerate(el1)]

    mymap = {}
    for i in range(len(set(winners))):
      candidats = [el for el in fmm if el[-1] == i]
      if len(candidats) > 0:
        mymap[i] = sorted(candidats, key=lambda x: x[1])[0][0]

    preds = np.array([mymap[el] for el in winners])

    f1 = f1_score(targets, preds, average='weighted')
    return f1
  
  def dump_evaluation2(self):
    print('Retrieves train predictions...')
    targets, winners, spikes_sum = [], [], []
    for spikes_in, target in tqdm(self.train_data_loader):
      out = self.forward(spikes_in.to(self.device), 2)
      spikes, pot = self.forward2(out.view(out.size(0), -1))
      winner = self.decision_layer.get_winner(pot, spikes)

      targets.append(target)
      winners.append(winner.cpu().item())
      spikes_sum.append(spikes.sum(dim=0).cpu().numpy())
    
    with open('_tmp_eval2_train.pk', 'wb') as f:
      pk.dump({'targets': targets, 'winners': winners, 'spikes_sum': spikes_sum}, f)
    
    print('Retrieves test predictions...')
    targets, winners, spikes_sum = [], [], []
    for spikes_in, target in tqdm(self.test_data_loader):
      out = self.forward(spikes_in.to(self.device), 2)
      spikes, pot = self.forward2(out.view(out.size(0), -1))
      winner = self.decision_layer.get_winner(pot, spikes)

      targets.append(target)
      winners.append(winner.cpu().item())
      spikes_sum.append(spikes.sum(dim=0).cpu().numpy())
    
    with open('_tmp_eval2_test.pk', 'wb') as f:
      pk.dump({'targets': targets, 'winners': winners, 'spikes_sum': spikes_sum}, f)
  
  def load_model2(self, map_location='cpu'):
    if os.path.isfile(self.save_path):
      data = torch.load(self.save_path, map_location=map_location)
      self.decision_layer.load_state_dict(data['model'])
      self.lr = data['lr']
    else:
      print(f"File {self.save_path} doesn't exist")
  
  def save_model2(self):
    if not os.path.isdir(os.path.dirname(self.save_path)):
      os.makedirs(os.path.dirname(self.save_path))
    torch.save({'model': self.decision_layer.state_dict(), 'lr': self.lr}, self.save_path)


def to_optimize(fire_threshold=450, n_output_neurons=200):
  try:
    sdcnn_exp2 = SDCNNExperiment2(fire_threshold=fire_threshold, n_output_neurons=n_output_neurons)
    sdcnn_exp2.load_model(map_location=None)
    sdcnn_exp2.train2()
    f1 = sdcnn_exp2.evaluation2(train_data=False)
  except:
    f1 = 0.
  logging.info(f'Optimization - fire_threshold={fire_threshold} - n_output_neurons={n_output_neurons} - f1={f1:.3f}')
  return f1


if __name__ == "__main__":
  argparser = argparse.ArgumentParser(prog='sdcnn_or.py', description='')
  argparser.add_argument('--log_file', default='_tmp_sdcnnOR_logs.txt', type=str)
  argparser.add_argument('--dataset_path', default='data/', type=str)
  argparser.add_argument('--subset_percent', default=1.0, type=float)
  argparser.add_argument('--save_res', default=False, type=ast.literal_eval)
  args = argparser.parse_args()

  logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

  random_seed = 42
  torch.manual_seed(random_seed)

  # try:
  #   new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
  #   torchvision.datasets.MNIST.resources = [('/'.join([new_mirror, url.split('/')[-1]]), md5)\
  #                                           for url, md5 in torchvision.datasets.MNIST.resources]
  # except:
  #   pass

  sdcnn_exp = SDCNNExperiment({'dataset_path': args.dataset_path, 'subset_percent': args.subset_percent})

  rep = input('Start training? (y or n): ')
  if rep == 'y':
    print('Launch training...')
    sdcnn_exp.train()
    print('Saving model...')
    sdcnn_exp.save_model()
  
  rep = input('Start evaluation? (y or n): ')
  if rep == 'y':
    print('Loading train model...')
    sdcnn_exp.load_model(map_location=None)
    f1 = sdcnn_exp.evaluate(save_res=args.save_res)
    print(f'Test f1 = {f1:.4f}')
  
  rep = input('Checking subset performance curve? (y or n): ')
  if rep == 'y':
    sps, f1s = [0.01, 0.02, 0.03, 0.04, 0.05], []
    for sp in sps:
      sdcnn_exp = SDCNNExperiment({'dataset_path': args.dataset_path, 'subset_percent': sp})
      sdcnn_exp.train()
      f1s.append(sdcnn_exp.evaluate(print_cr=False, save_res=False))
    u.board_printing({'subset_percent': sps, 'f1': f1s})
    # +----+------------------+----------+----------------+
    # |    |   subset_percent |       f1 |  Training time |
    # |----+------------------+----------|----------------+
    # |  0 |             0.01 | 0.669552 |      1min57s   |
    # |  1 |             0.02 | 0.847369 |      3min39s   |
    # |  2 |             0.03 | 0.848092 |      5min32s   |
    # |  3 |             0.04 | 0.931198 |      7min06s   |
    # |  4 |             0.05 | 0.958679 |      8min55s   |
    # +----+------------------+----------+----------------+
  
  rep = input('Train decision layer? (y or n): ')
  if rep == 'y':
    sdcnn_exp2 = SDCNNExperiment2()
    sdcnn_exp2.load_model(map_location=None)
    sdcnn_exp2.load_model2(map_location=None)
    sdcnn_exp2.train2()
    sdcnn_exp2.save_model2()
  
  rep = input('Evaluate decision layer? (y or n): ')
  if rep == 'y':
    sdcnn_exp2 = SDCNNExperiment2()
    sdcnn_exp2.load_model(map_location=None)
    sdcnn_exp2.load_model2(map_location=None)
    sdcnn_exp2.dump_evaluation2()
  
  # +----+------------------+-----------+--------+
  # |    |   output_neuron  | threshold |   f1   |
  # |----+------------------+-----------|--------+
  # |  0 |        100       |    150    |  0.75  |
  # |  1 |        100       |    300    |  0.79  |
  # |  2 |        100       |    450    |  0.80  |
  # |  3 |        200       |    450    |  0.83  |
  # |  4 |                  |           |        |
  # +----+------------------+-----------+--------+

  rep = input('Optimize hyperparameters? (y or n): ')
  if rep == 'y':
    pbounds = {'fire_threshold': (1, 2000), 'n_output_neurons': (10, 300)}
    optimizer = BayesianOptimization(f=to_optimize, pbounds=pbounds, verbose=2, random_state=1)
    optimizer.maximize(init_points=50, n_iter=10)
    print(optimizer.max)

    for i, res in enumerate(optimizer.res):
      print(f"Iteration {i}: {res}")


## visualize kernel weights distribution
# import numpy as np;import torch;import seaborn as sns;import matplotlib.pyplot as plt;import pandas as pd
# x = np.linspace(0, 1, 20)
# xx, yy = [], []
# for i, el in enumerate(x[:-1]):
#  xx.append(round((el + x[i+1])/2, 2))
#  yy.append(W[(W >= el) & (W < x[i+1])].count_nonzero().item())
# sns.barplot(x='x', y='y', data=pd.DataFrame.from_dict({'x': xx, 'y': yy}))
# plt.show()