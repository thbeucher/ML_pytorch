# Taken (with modifications) from https://github.com/miladmozafari/SpykeTorch
# Take 1Go of memory on GPU for training, trained in 2h47, accuracy obtained = 98%
# Using Convergence criterion, training is done in 8mn with an accuracy of 98%
import os
import sys
import torch
import logging
import argparse
import torchvision
import numpy as np
import pickle as pk
import torch.nn as nn

from tqdm import tqdm

sys.path.append(os.path.abspath(__file__).replace('SNN/sdcnn_or.py', ''))

import utils as u
import models.snn.stdp as lf
import models.snn.functional as f
import models.snn.temporal_order_coding_image as toci

from models.snn.layer import Convolution


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
                        'save_path': 'model/sdcnne_exp.pt',
                        'check_convergence_step': 5000, 'convergence_threshold': 0.008}
    self.config = u.populate_configuration(configuration, self.base_config)

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.config['max_ap'] = self.config['max_ap'].to(self.device)

    pad_fn = lambda x: (x - 1) // 2 + ((x-1) % 2 > 0)
    self.paddings_train = [[pad_fn(conf['kernel_size'])] * 4 for conf in self.config['layers_config']]
    self.paddings_forward = [pad_fn(conf['kernel_size']) for conf in self.config['layers_config']]

    self.set_metadata()
    self.set_dataloader()
    self.instanciate_model()

  def set_metadata(self):
    self.learning_rates = []
    for conf in self.config['layers_config']:
      ap, an = self.config['learning_rates']
      lr = [[torch.tensor([ap]).to(self.device), torch.tensor([an]).to(self.device)] for _ in range(conf['out_channels'])]
      self.learning_rates.append(lr)
  
  def set_dataloader(self):
    self.train_data_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(self.config['dataset_path'], train=True,
                                                                                    download=True, transform=DataTransformer()),
                                                         batch_size=self.config['batch_size'], num_workers=8,
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
          if i == 0 and j > 0 and j % self.config['timestep_update_lr'] == 0:
            new_ap = torch.min(self.learning_rates[0][0][0] * 2, self.config['max_ap'])
            new_an = self.learning_rates[0][0][0] * self.config['an_update']
            self.update_all_lr(new_ap, new_an, 0)
          self.forward(spikes_in.to(self.device), i, training=True)

          if j % self.config['check_convergence_step'] == 0:
            C = (self.layers[i].weights * (1 - self.layers[i].weights)).sum() / np.prod(self.layers[i].weights.shape)
            logging.info(f'Layer {i} - Epoch {epoch} - n_data {j} - C = {C:.4f}')
            if C < self.config['convergence_threshold']:
              break
  
  def evaluate(self):
    from sklearn.svm import LinearSVC
    from sklearn.metrics import classification_report

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

    print(f'TRAIN results:\n{classification_report(np.array(y_train), preds_train)}')
    print(f'TEST results:\n{classification_report(np.array(y_test), preds_test)}')

    import pickle as pk
    with open('_tmp_xy_traintest_preds.pk', 'wb') as f:
      pk.dump({'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test,
               'preds_train': preds_train, 'preds_test': preds_test}, f)
  
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


if __name__ == "__main__":
  argparser = argparse.ArgumentParser(prog='sdcnn_or.py', description='')
  argparser.add_argument('--log_file', default='_tmp_sdcnnOR_logs.txt', type=str)
  argparser.add_argument('--dataset_path', default='data/', type=str)
  args = argparser.parse_args()

  logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

  random_seed = 42
  torch.manual_seed(random_seed)

  try:
    new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
    torchvision.datasets.MNIST.resources = [('/'.join([new_mirror, url.split('/')[-1]]), md5)\
                                            for url, md5 in torchvision.datasets.MNIST.resources]
  except:
    pass

  sdcnn_exp = SDCNNExperiment({'dataset_path': args.dataset_path})

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
    sdcnn_exp.evaluate()


###################################################################################################################################
## STORY                                                                                                                         ##
##                                                                                                                               ##
## Input:                                                                                                                        ##
##  image from mnist = grayscale -> [1, 28, 28]                                                                                  ##
##  Creation of different kernels that modelize the retina and ganglion cells. We use these kernels to transform                 ##
##  the grayscale image (pixel values between 0-255) into intencitie levels                                                      ##
##  (convolution between kernels and image -> [n_kernels=2, 28, 28]). We localy normalize obtained intencities.                  ##
##  Finally, we transform the intencities matrix (2D) into temporal spiking matrix (3D) (see cumulative_intensity_to_latency)    ##
## Model:                                                                                                                        ##
##  Convolution layer, kernel interpretation -> at each position, there is feature_out neurons that shared feature_in set of     ##
##                                              kernel_size*2 synapses                                                           ##
##                                             (eg at each position, there is 32 neurons that shared 2 set of 25 synapses)       ##
## Training:                                                                                                                     ##
##  spike_in = [n_timesteps=15, n_retina_kernels=2, 28, 28], we pad it in order to go through convolution layer [15, 2, 32, 32]  ##
##  -We go through convolutional layer, as we have cumulative temporal spiking input, we obtained at each timestep the potential ##
##  of each neuron so then with a defined threshold we obtain a cumulative spike train (functional.fire).                        ##
##  spike_out = [15, 32, 28, 28]                                                                                                 ##
##  -We perform a pointwise feature competition. At each position, through all features, we retrieve the first neuron to spike   ##
##  then inhibit (by putting to 0) all other neuron at that position on other features.                                          ##
##  -We find k=5 winners by using the criteria of first to spike then highest potential. When a winner is found, all neurons     ##
##  in the same feature are set to 0 to prevent the same feature to be the next winner then to increase the chance of learning   ##
##  diverse features we perform a columnar inhibition where with a specified radius=2 we set to 0 all neurons in that radius     ##
##  on all other features.                                                                                                       ##
##  -We finally allow winners to learn by updating their synapses using STDP process                                             ##
##                                                                                                                               ##
## HYPERPARAMETERS:                                                                                                              ##
##  - learning rates (0.004, -0.003) & learning rates update (max_ap=0.15, an_update=-0.75, timestep_update_lr=500)              ##
##  - firing threshold (layer1 = 10, layer2 = 1)                                                                                 ##
##  - number of winners (layer1 = 5, layer2 = 8)                                                                                 ##
##  - inhibition radius (layer1 = 2, layer2 = 1)                                                                                 ##
##  - number of epochs (layer1 = 2, layer2 = 20) -> can be removed by using Convergence criteria                                 ##
##  - layers configuration (n_channel, kernel_size)                                                                              ##
##  - input transformation hyperparameters: kernel_type, kernel_configuration, normalization_radius, n_time_steps                ##
###################################################################################################################################