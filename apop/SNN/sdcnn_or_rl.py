import os
import sys
import torch
import logging
import argparse
import torchvision
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import classification_report

sys.path.append(os.path.abspath(__file__).replace('SNN/sdcnn_or_rl.py', ''))

import utils as u
import models.snn.stdp as lf
import models.snn.functional as f
import models.snn.temporal_order_coding_image as toci

from models.snn.layer import Convolution


class DataTransformer(object):
  def __init__(self, configuration={}):
    base_config = {'kernel_type': toci.DoGKernel, 'kernels_conf': [[3, 3/9, 6/9], [3, 6/9, 3/9],
                                                                   [7, 7/9, 14/9], [7, 14/9, 7/9],
                                                                   [13, 13/9, 26/9], [13, 26/9, 13/9]],
                   'padding': 6, 'threshold': 50, 'use_threshold': True, 'normalization_radius': 8, 'n_time_steps': 15}
    self.config = u.populate_configuration(base_config, configuration)

    self.to_tensor = torchvision.transforms.ToTensor()
    self.filters = toci.Filter(self.config)
  
  def __call__(self, image):
    image = self.to_tensor(image) * 255
    image = self.filters(image.unsqueeze(0))
    image = toci.local_normalization(image, self.config['normalization_radius'])
    return toci.cumulative_intensity_to_latency(image, self.config['n_time_steps'])


class SDCNNExperiment(object):
  def __init__(self, configuration={}):
    self.base_config = {'layers_config': [{'in_channels': 6, 'out_channels': 30, 'kernel_size': 5,
                                           'weight_mean': 0.8, 'weight_std': 0.05, 'anti_stdp': False},
                                          {'in_channels': 30, 'out_channels': 250, 'kernel_size': 3,
                                           'weight_mean': 0.8, 'weight_std': 0.05, 'anti_stdp': False},
                                          {'in_channels': 250, 'out_channels': 200, 'kernel_size': 5,
                                          'weight_mean': 0.8, 'weight_std': 0.05, 'anti_stdp': True}],
                        'learning_rates': (0.004, -0.003), 'max_ap': torch.tensor([0.15]), 'an_update': -0.75,
                        'anti_learning_rates': (-0.004, 0.0005),
                        'thresholds': [15, 10, None], 'timestep_update_lr': 500,
                        'pooling_vars': [{'kernel_size': 2, 'stride': 2, 'padding': 0},
                                         {'kernel_size': 3, 'stride': 3, 'padding': 0}],
                        'n_winners': [5, 8, 1], 'inhibition_radius': [3, 1, 0],
                        'batch_size': 1, 'dataset_path': 'data/', 'n_epochs': [2, 4, 680],
                        'save_path': 'model/sdcnn_rl_exp.pt',
                        'check_convergence_step': 5000, 'convergence_threshold': 0.008,
                        'n_labels': 10, 'adaptive_lr': True, 'n_workers': 8}
    self.config = u.populate_configuration(configuration, self.base_config)

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.config['max_ap'] = self.config['max_ap'].to(self.device)

    pad_fn = lambda x: (x - 1) // 2 + ((x-1) % 2 > 0)
    self.paddings_train = [[pad_fn(conf['kernel_size'])] * 4 for conf in self.config['layers_config']]

    n_neuron_per_label = self.config['layers_config'][-1]['out_channels'] // self.config['n_labels']
    self.decision_map = np.array([[i] * n_neuron_per_label for i in range(self.config['n_labels'])]).flatten().tolist()

    self.predictions_history = np.array([0., 0., 0.])  # correct, wrong, silent
    self.prev_network_f1 = 0.

    self.set_metadata()
    self.set_dataloader()
    self.instanciate_model()
  
  def set_metadata(self):
    self.learning_rates, self.anti_learning_rates = [], []
    for conf in self.config['layers_config']:
      ap, an = self.config['learning_rates']
      lr = [[torch.tensor([ap]).to(self.device), torch.tensor([an]).to(self.device)] for _ in range(conf['out_channels'])]
      self.learning_rates.append(lr)

      if conf['anti_stdp']:
        ap, an = self.config['anti_learning_rates']
        lr = [[torch.tensor([ap]).to(self.device), torch.tensor([an]).to(self.device)] for _ in range(conf['out_channels'])]
        self.anti_learning_rates.append(lr)
      else:
        self.anti_learning_rates.append(None)
  
  def set_dataloader(self):
    self.train_data_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(self.config['dataset_path'], train=True,
                                                                                    download=True, transform=DataTransformer()),
                                                         batch_size=self.config['batch_size'], num_workers=self.config['n_workers'],
                                                         shuffle=True, pin_memory=True, collate_fn=lambda x: x[0])
    self.test_data_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(self.config['dataset_path'], train=False,
                                                                                   download=True, transform=DataTransformer()),
                                                        batch_size=self.config['batch_size'], num_workers=self.config['n_workers'],
                                                        shuffle=False, pin_memory=True, collate_fn=lambda x: x[0])
  
  def instanciate_model(self):
    self.layers = nn.ModuleList([Convolution(**lc) for lc in self.config['layers_config']])
    self.layers.to(self.device)
  
  def update_all_lr(self, new_ap, new_an, layer_idx):
    for i in range(len(self.learning_rates[layer_idx])):
      self.learning_rates[layer_idx][i][0] = new_ap
      self.learning_rates[layer_idx][i][1] = new_an
  
  def update_all_anti_lr(self, new_ap, new_an, layer_idx):
    for i in range(len(self.anti_learning_rates[layer_idx])):
      self.anti_learning_rates[layer_idx][i][0] = new_ap
      self.anti_learning_rates[layer_idx][i][1] = new_an
  
  def adaptive_update_lr(self, layer_idx):
    ap_ori, an_ori = self.config['learning_rates']
    anti_ap_ori, anti_an_ori = self.config['anti_learning_rates']
    self.predictions_history /= self.predictions_history.sum()
    new_ap = torch.tensor([ap_ori]).to(self.device) * self.predictions_history[1]  # N_miss / N
    new_an = torch.tensor([an_ori]).to(self.device) * self.predictions_history[1]
    new_anti_ap = torch.tensor([anti_ap_ori]).to(self.device) * self.predictions_history[0]  # N_hit / N
    new_anti_an = torch.tensor([anti_an_ori]).to(self.device) * self.predictions_history[0]
    self.update_all_lr(new_ap, new_an, layer_idx)
    self.update_all_anti_lr(new_anti_ap, new_anti_an, layer_idx)
    self.predictions_history = np.array([0., 0., 0.])
    # logging.info(f'new_ap = {new_ap.item():.5f} | new_an = {new_an.item():.5f}')
    # logging.info(f'new_anti_ap = {new_anti_ap.item():.5f} | new_anti_an = {new_anti_an.item():.5f}')
  
  def competition_winner(self, potentials, layer_idx):
    potentials = f.pointwise_feature_competition_inhibition(potentials)  # [timestep, feat_out(eg32), height, width]
    spikes = potentials.sign()
    # winners = f.get_k_winners(potentials, kwta=self.config['n_winners'][layer_idx],
    #                           inhibition_radius=self.config['inhibition_radius'][layer_idx], spikes=spikes)
    winners = f.get_k_winners_ori(potentials, kwta=self.config['n_winners'][layer_idx],
                                  inhibition_radius=self.config['inhibition_radius'][layer_idx], spikes=spikes)
    return potentials, spikes, winners
  
  def competition_winner_stdp(self, input_spikes, potentials, layer_idx, target, stdp=True):
    potentials, spikes, winners = self.competition_winner(potentials, layer_idx)

    use_stabilizer, lower_bound, upper_bound = True, 0, 1
    lrs = self.learning_rates[layer_idx]

    if target is not None:
      use_stabilizer, lower_bound, upper_bound = False, 0.2, 0.8

      if len(winners) != 0:
        pred = self.decision_map[winners[0][0]]
        if pred != target:
          lrs = self.anti_learning_rates[layer_idx]
          self.predictions_history[1] += 1
        else:
          self.predictions_history[0] += 1
      else:
        stdp = False
        self.predictions_history[2] += 1

    if stdp:
      lf.functional_stdp(self.layers[layer_idx], lrs, input_spikes, spikes, winners,
                        use_stabilizer=use_stabilizer, lower_bound=lower_bound, upper_bound=upper_bound)
    return spikes, potentials
  
  def forward(self, input_spikes, layer_idx, target=None, training=False):  # [timestep, feat_in(eg2), height, width]
    for i, layer in enumerate(self.layers):
      input_spikes = nn.functional.pad(input_spikes, self.paddings_train[i])  # [timestep, feat_in, height_pad, width_pad]

      potentials = layer(input_spikes)  # [timestep, feat_out(eg32), heigth, width]
      spikes, potentials = f.fire(potentials, self.config['thresholds'][i], return_thresholded_potentials=True)

      if i == layer_idx and training:
        return self.competition_winner_stdp(input_spikes, potentials, layer_idx, target)
      
      if target is not None and i == layer_idx:
        potentials, spikes, winners = self.competition_winner(potentials, layer_idx)
        pred = -1 if len(winners) == 0 else self.decision_map[winners[0][0]]
        return potentials, spikes, pred

      input_spikes = nn.functional.max_pool2d(spikes, **self.config['pooling_vars'][i])
    
    return input_spikes
  
  def forward_fullret(self, input_spikes, layer_idx, target=None, training=False):
    returns = {'input_spikes': [], 'potentials': []}
    for i, layer in enumerate(self.layers):
      returns['input_spikes'].append([i, input_spikes.cpu().short()])
      input_spikes = nn.functional.pad(input_spikes, self.paddings_train[i])

      potentials = layer(input_spikes)  # [timestep, feat_out(eg32), heigth, width]
      spikes, potentials = f.fire(potentials, self.config['thresholds'][i], return_thresholded_potentials=True)
      returns['potentials'].append([i, potentials.cpu().short()])

      if i == layer_idx and training:
        return self.competition_winner_stdp(input_spikes, potentials, layer_idx, target)
      
      if target is not None and i == layer_idx:
        potentials, spikes, winners = self.competition_winner(potentials, layer_idx)
        pred = -1 if len(winners) == 0 else self.decision_map[winners[0][0]]
        returns['pred'] = pred
        return returns

      input_spikes = nn.functional.max_pool2d(spikes, **self.config['pooling_vars'][i])
    
    return input_spikes
  
  def train(self):
    for i, n_epoch in enumerate(tqdm(self.config['n_epochs'])):
      for epoch in tqdm(range(n_epoch), leave=False):
        for j, (spikes_in, target) in enumerate(tqdm(self.train_data_loader, leave=False)):
          if i in [0, 1] and j > 0 and j % self.config['timestep_update_lr'] == 0:
            new_ap = torch.min(self.learning_rates[i][0][0] * 2, self.config['max_ap'])
            new_an = self.learning_rates[i][0][0] * self.config['an_update']
            self.update_all_lr(new_ap, new_an, i)

          self.forward(spikes_in.to(self.device), i, target=target if i == len(self.layers) - 1 else None, training=True)

          if i == len(self.layers) - 1 and self.config['adaptive_lr'] and j > 0 and j % 1000 == 0:
            self.adaptive_update_lr(i)
          
          if i == len(self.layers) - 1 and j % 20000 == 0:
            network_f1 = self.evaluate(print_res=False)
            logging.info(f'Network performance - f1 = {network_f1:.3f}')

            if network_f1 > self.prev_network_f1:
              if os.path.isfile(self.config['save_path'].replace('.pt', f'_{self.prev_network_f1:.2f}.pt')):
                os.remove(self.config['save_path'].replace('.pt', f'_{self.prev_network_f1:.2f}.pt'))

              self.save_model(save_name=self.config['save_path'].replace('.pt', f'_{network_f1:.2f}.pt'))
              self.prev_network_f1 = network_f1

          if j % self.config['check_convergence_step'] == 0:
            C = (self.layers[i].weights * (1 - self.layers[i].weights)).sum() / np.prod(self.layers[i].weights.shape)
            logging.info(f'Layer {i} - Epoch {epoch} - n_data {j} - C = {C:.4f}')
            if (C < self.config['convergence_threshold'] and epoch != 0) or round(C.item(), 5) == 0.:
              self.save_model()
              break
  
  def evaluate(self, print_res=True):
    targets, predictions = [], []
    for spikes_in, target in tqdm(self.test_data_loader, leave=False):
      targets.append(target)
      _, _, pred = self.forward(spikes_in.to(self.device), len(self.layers) - 1, target=target)
      predictions.append(pred)
    
    if print_res:
      print(f'TEST results:\n{classification_report(targets, predictions, zero_division=0)}')
    else:
      return classification_report(targets, predictions, zero_division=0, output_dict=True)['weighted avg']['f1-score']
  
  def out_for_analyze(self, train=False, save_out=True):
    outputs = []
    for spikes_in, target in tqdm(self.train_data_loader if train else self.test_data_loader, leave=False):
      returns = self.forward_fullret(spikes_in.to(self.device), len(self.layers) - 1, target=target)
      returns['target'] = target
      outputs.append(returns)
    
    if save_out:
      torch.save(outputs, '_tmp_out_for_analyze.pt')
    else:
      return outputs
  
  def evaluate_stdp_layers(self):
    from sklearn.svm import LinearSVC

    classifier = LinearSVC(C=2.4)

    print('Retrieve train data...')
    self.load_model()
    ori_layers = self.layers
    self.layers = self.layers[:-1]

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
    
    self.layers = ori_layers
    
    print('Launch classifier training...')
    classifier.fit(np.array(x_train), np.array(y_train))
    print('Start predictions...')
    preds_train = classifier.predict(np.array(x_train))
    preds_test = classifier.predict(np.array(x_test))

    print(f'TRAIN results:\n{classification_report(np.array(y_train), preds_train)}')
    print(f'TEST results:\n{classification_report(np.array(y_test), preds_test)}')
  
  def load_model(self, map_location=None, save_name=None):
    save_name = self.config['save_path'] if save_name is None else save_name
    if os.path.isfile(save_name):
      data = torch.load(save_name, map_location=map_location)
      self.layers.load_state_dict(data['model'])
      self.learning_rates = data['lr']
    else:
      print(f"File {save_name} doesn't exist")
  
  def save_model(self, save_name=None):
    save_name = self.config['save_path'] if save_name is None else save_name
    if not os.path.isdir(os.path.dirname(save_name)):
      os.makedirs(os.path.dirname(save_name))
    torch.save({'model': self.layers.state_dict(), 'lr': self.learning_rates}, save_name)


if __name__ == "__main__":
  # Layer 1 is trained in 2mn (STDP)
  # Layer 2 is trained in 8mn (STDP)
  # Layer 3 achieve performance of 0.9 in 22mn
  argparser = argparse.ArgumentParser(prog='sdcnn_or_rl.py', description='')
  argparser.add_argument('--log_file', default='_tmp_sdcnn_or_rl_logs.txt', type=str)
  argparser.add_argument('--dataset_path', default='data/', type=str)
  argparser.add_argument('--n_workers', default=8, type=int)
  args = argparser.parse_args()

  logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

  random_seed = 42
  torch.manual_seed(random_seed)

  sdcnn_rl_exp = SDCNNExperiment({'dataset_path': args.dataset_path, 'n_workers': args.n_workers})

  rep = input('Load saved model? (y or n): ')
  if rep == 'y':
    sdcnn_rl_exp.load_model()

  rep = input('Start training? (y or n): ')
  if rep == 'y':
    sdcnn_rl_exp.train()

  rep = input('Start evaluation? (y or n): ')
  if rep == 'y':
    sdcnn_rl_exp.evaluate()

  rep = input('Save model? (y or n): ')
  if rep == 'y':
    sdcnn_rl_exp.save_model()
  
  rep = input('Retrieves all outputs then save it? (y or n): ')
  if rep == 'y':
    sdcnn_rl_exp.out_for_analyze()