import os
import sys
import torch
import random
import logging
import argparse
import torchvision
import numpy as np
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report

sys.path.append(os.path.abspath(__file__).replace('SNN/sdcnn_or_rl_compTask.py', ''))

import utils as u
import models.snn.stdp as lf
import models.snn.functional as f
import models.snn.temporal_order_coding_image as toci

from models.snn.layer import Convolution, Linear


class DataTransformer(object):
  def __init__(self, configuration={}):
    base_config = {'kernel_type': toci.DoGKernel, 'kernels_conf': [[3, 3/9, 6/9], [3, 6/9, 3/9],
                                                                   [7, 7/9, 14/9], [7, 14/9, 7/9],
                                                                   [13, 13/9, 26/9], [13, 26/9, 13/9]],
                   'padding': 6, 'threshold': 50, 'use_threshold': True, 'normalization_radius': 8, 'n_time_steps': 20}
    self.config = u.populate_configuration(base_config, configuration)

    self.to_tensor = torchvision.transforms.ToTensor()
    self.filters = toci.Filter(self.config)
  
  def __call__(self, image):
    image = self.to_tensor(image) * 255
    image = self.filters(image.unsqueeze(0))
    image = toci.local_normalization(image, self.config['normalization_radius'])
    return toci.cumulative_intensity_to_latency(image, self.config['n_time_steps'])


class MNISTSiamese(torchvision.datasets.MNIST):
  def __init__(self, root, train=True, transform=None, target_transform=None, download=False, classif_mode=False):
    super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
    self.classif_mode = classif_mode

    self.index_same = {i: [j for j, t in enumerate(self.targets) if t == i] for i in range(10)}
    self.index_allothers = {i: [j for j, t in enumerate(self.targets) if t != i] for i in range(10)}

    self.ref_imgs_idx = [self.index_same[i][1 if i in [3, 5] else 0] for i in range(10)]
    self.ref_imgs = [Image.fromarray(self.data[i].numpy(), mode='L') for i in self.ref_imgs_idx]
    self.ref_imgs_target = list(range(10))
  
  def __getitem__(self, index):
    img, target = self.data[index], int(self.targets[index])

    if not self.classif_mode:
      if torch.rand(1) < 0.5:
        number_idx = self.index_same[target][torch.randint(0, len(self.index_same[target]), (1,))]
      else:
        number_idx = self.index_allothers[target][torch.randint(0, len(self.index_allothers[target]), (1,))]

      img2, target2 = self.data[number_idx], int(self.targets[number_idx])

      # doing this so that it is consistent with all other datasets
      # to return a PIL Image
      img = Image.fromarray(img.numpy(), mode="L")
      img2 = Image.fromarray(img2.numpy(), mode='L')

      if self.transform is not None:
          img = self.transform(img)
          img2 = self.transform(img2)

      if self.target_transform is not None:
          target = self.target_transform(target)
          target2 = self.target_transform(target2)

      return (img, img2), 0 if target == target2 else 1
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


class SDCNNExperiment(object):
  def __init__(self, configuration={}):
    self.base_config = {'layers_config': [{'in_channels': 6, 'out_channels': 30, 'kernel_size': 5,
                                           'weight_mean': 0.8, 'weight_std': 0.05, 'anti_stdp': False},
                                          {'in_channels': 30, 'out_channels': 250, 'kernel_size': 3,
                                           'weight_mean': 0.8, 'weight_std': 0.05, 'anti_stdp': False}],
                        'learning_rates': (0.004, -0.003), 'max_ap': torch.tensor([0.15]), 'an_update': -0.75,
                        'anti_learning_rates': (-0.004, 0.0005),
                        'thresholds': [15, 10, None], 'timestep_update_lr': 500,
                        'pooling_vars': [{'kernel_size': 2, 'stride': 2, 'padding': 0},
                                         {'kernel_size': 3, 'stride': 3, 'padding': 0}],
                        'n_winners': [5, 8, 1], 'inhibition_radius': [3, 1, 0],
                        'batch_size': 1, 'dataset_path': 'data/', 'n_epochs': [2, 4, 680],
                        'save_path': 'model/sdcnn_rl_compTask_exp.pt',
                        'check_convergence_step': 5000, 'convergence_threshold': 0.008,
                        'n_labels': 10, 'adaptive_lr': True, 'n_workers': 8}
    self.config = u.populate_configuration(configuration, self.base_config)

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.config['max_ap'] = self.config['max_ap'].to(self.device)

    pad_fn = lambda x: (x - 1) // 2 + ((x-1) % 2 > 0)
    self.paddings_train = [[pad_fn(conf['kernel_size'])] * 4 for conf in self.config['layers_config']]

    self.predictions_history = np.array([0., 0.])  # correct, wrong
    self.prev_network_f1 = 0.

    self.set_metadata()
    self.set_dataloader()
    self.instanciate_model()

    # one neuron say "YES! SAME NUMBER" and the other "NO"
    self.lr = [torch.tensor([0.004]).to(self.device), torch.tensor([-0.003]).to(self.device)]
    self.anti_lr = [torch.tensor([-0.004]).to(self.device), torch.tensor([0.0005]).to(self.device)]
    self.decision_layer = Linear(250*4*4*2, 2, self.lr, self.anti_lr)
    self.decision_layer.to(self.device)
  
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
    self.train_data_loader2 = torch.utils.data.DataLoader(MNISTSiamese(self.config['dataset_path'], train=True,
                                                                       download=True, transform=DataTransformer()),
                                                          batch_size=self.config['batch_size'], num_workers=self.config['n_workers'],
                                                          shuffle=True, pin_memory=True, collate_fn=lambda x: x[0])
    self.test_data_loader = torch.utils.data.DataLoader(MNISTSiamese(self.config['dataset_path'], train=False,
                                                                     download=True, transform=DataTransformer()),
                                                        batch_size=self.config['batch_size'], num_workers=self.config['n_workers'],
                                                        shuffle=True, pin_memory=True, collate_fn=lambda x: x[0])
    self.test_data_loader2 = torch.utils.data.DataLoader(MNISTSiamese(self.config['dataset_path'], train=False,
                                                                      download=True, transform=DataTransformer(),
                                                                      classif_mode=True),
                                                        batch_size=self.config['batch_size'], num_workers=self.config['n_workers'],
                                                        shuffle=True, pin_memory=True, collate_fn=lambda x: x[0])
  
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
    ap_ori, an_ori = self.lr
    anti_ap_ori, anti_an_ori = self.anti_lr
    self.predictions_history /= self.predictions_history.sum()
    new_ap = ap_ori * self.predictions_history[1]  # N_miss / N
    new_an = an_ori * self.predictions_history[1]
    new_anti_ap = anti_ap_ori * self.predictions_history[0]  # N_hit / N
    new_anti_an = anti_an_ori * self.predictions_history[0]
    self.decision_layer._update_all_lr(new_ap, new_an)
    self.decision_layer._update_all_anti_lr(new_anti_ap, new_anti_an)
    self.predictions_history = np.array([0., 0.])
    # logging.info(f'new_ap = {new_ap.item():.5f} | new_an = {new_an.item():.5f}')
    # logging.info(f'new_anti_ap = {new_anti_ap.item():.5f} | new_anti_an = {new_anti_an.item():.5f}')
  
  def competition_winner(self, potentials, layer_idx):
    potentials = f.pointwise_feature_competition_inhibition(potentials)  # [timestep, feat_out(eg32), height, width]
    spikes = potentials.sign()
    winners = f.get_k_winners_ori(potentials, kwta=self.config['n_winners'][layer_idx],
                                  inhibition_radius=self.config['inhibition_radius'][layer_idx], spikes=spikes)
    return potentials, spikes, winners
  
  def competition_winner_stdp(self, input_spikes, potentials, layer_idx, stdp=True):
    potentials, spikes, winners = self.competition_winner(potentials, layer_idx)

    use_stabilizer, lower_bound, upper_bound = True, 0, 1
    lrs = self.learning_rates[layer_idx]

    if stdp:
      lf.functional_stdp(self.layers[layer_idx], lrs, input_spikes, spikes, winners,
                        use_stabilizer=use_stabilizer, lower_bound=lower_bound, upper_bound=upper_bound)
    return spikes, potentials
  
  def forward(self, input_spikes, layer_idx, training=False):  # [timestep, feat_in(eg2), height, width]
    for i, layer in enumerate(self.layers):
      input_spikes = nn.functional.pad(input_spikes, self.paddings_train[i])  # [timestep, feat_in, height_pad, width_pad]

      potentials = layer(input_spikes)  # [timestep, feat_out(eg32), heigth, width]
      spikes, potentials = f.fire(potentials, self.config['thresholds'][i], return_thresholded_potentials=True)

      if i == layer_idx and training:
        return self.competition_winner_stdp(input_spikes, potentials, layer_idx)

      input_spikes = nn.functional.max_pool2d(spikes, **self.config['pooling_vars'][i])
    
    return input_spikes
  
  def forward_comp(self, input_spikes_img1, input_spikes_img2, target, training=False):  # currently = [15, 250, 4, 4]
    input_spikes = torch.cat([input_spikes_img1.view(20, -1), input_spikes_img2.view(20, -1)], dim=1)  # [15, 8.000]

    potentials = self.decision_layer(input_spikes)
    spikes, potentials = f.fire(potentials, self.config['thresholds'][-1], return_thresholded_potentials=True)

    if training:
      winner = self.decision_layer.get_winner(potentials, spikes)
      if winner != target:
        self.predictions_history[1] += 1
      else:
        self.predictions_history[0] += 1
      return self.decision_layer.rstdp(potentials, input_spikes, spikes, target, winner=winner)
    
    return self.decision_layer.get_winner(potentials, spikes)
  
  def train(self):
    for i, n_epoch in enumerate(tqdm(self.config['n_epochs'])):
      train_data_loader = self.train_data_loader if i < 2 else self.train_data_loader2
      for epoch in tqdm(range(n_epoch), leave=False):
        for j, (spikes_in, target) in enumerate(tqdm(train_data_loader, leave=False)):
          if i in [0, 1] and j > 0 and j % self.config['timestep_update_lr'] == 0:
            new_ap = torch.min(self.learning_rates[i][0][0] * 2, self.config['max_ap'])
            new_an = self.learning_rates[i][0][0] * self.config['an_update']
            self.update_all_lr(new_ap, new_an, i)

          if i < 2:
            self.forward(spikes_in.to(self.device), i, training=True)
          else:
            spikes_in1 = self.forward(spikes_in[0].to(self.device), None)
            spikes_in2 = self.forward(spikes_in[1].to(self.device), None)
            self.forward_comp(spikes_in1, spikes_in2, target, training=True)

          # if i == 2 and self.config['adaptive_lr'] and j > 0 and j % 1000 == 0:
          #   self.adaptive_update_lr(i)
          
          if i == 2 and j % 20000 == 0:
            network_f1 = self.evaluate(print_res=False)
            network_f1_classif = self.evaluate_classif(print_res=False)
            logging.info(f'Network performance - f1 = {network_f1:.3f} - f1_classif = {network_f1_classif:.3f}')

            if network_f1 > self.prev_network_f1:
              if os.path.isfile(self.config['save_path'].replace('.pt', f'_{self.prev_network_f1:.2f}.pt')):
                os.remove(self.config['save_path'].replace('.pt', f'_{self.prev_network_f1:.2f}.pt'))

              self.save_model(save_name=self.config['save_path'].replace('.pt', f'_{network_f1:.2f}.pt'))
              self.prev_network_f1 = network_f1

          if j % self.config['check_convergence_step'] == 0 and i != 2:
            C = (self.layers[i].weights * (1 - self.layers[i].weights)).sum() / np.prod(self.layers[i].weights.shape)
            logging.info(f'Layer {i} - Epoch {epoch} - n_data {j} - C = {C:.4f}')
            if (C < self.config['convergence_threshold'] and epoch != 0) or round(C.item(), 5) == 0.:
              self.save_model()
              break
  
  def evaluate(self, print_res=True):
    targets, predictions = [], []
    for (spikes_in1, spikes_in2), target in tqdm(self.test_data_loader, leave=False):
      targets.append(target)
      spikes_out1 = self.forward(spikes_in1.to(self.device), None)
      spikes_out2 = self.forward(spikes_in2.to(self.device), None)
      pred = self.forward_comp(spikes_out1, spikes_out2, target)
      predictions.append(pred.item())
    
    if print_res:
      print(f'TEST results:\n{classification_report(targets, predictions, zero_division=0)}')
    else:
      return classification_report(targets, predictions, zero_division=0, output_dict=True)['weighted avg']['f1-score']
  
  def evaluate_classif(self, print_res=True):
    targets, predictions = [], []
    for (img, ref_imgs), (target, ref_targets) in tqdm(self.test_data_loader2, leave=False):
      targets.append(target)
      spikes_out = self.forward(img.to(self.device), None)
      pred = None

      for ref_img, ref_target in zip(ref_imgs, ref_targets):
        spikes_out_ref = self.forward(ref_img.to(self.device), None)
        sim = self.forward_comp(spikes_out, spikes_out_ref, None)

        if sim == 0:
          pred = ref_target
        
      if pred is None:
        pred = random.sample(list(range(10)), 1)[0]

      predictions.append(pred)
    
    if print_res:
      print(f'TEST results:\n{classification_report(targets, predictions, zero_division=0)}')
    else:
      return classification_report(targets, predictions, zero_division=0, output_dict=True)['weighted avg']['f1-score']
  
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
  argparser = argparse.ArgumentParser(prog='sdcnn_or_rl_compTask.py', description='')
  argparser.add_argument('--log_file', default='_tmp_sdcnn_or_rl_compTask_logs.txt', type=str)
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