# Functions & class taken (with modifications) from https://github.com/miladmozafari/SpykeTorch
import os
import sys
import math
import torch
import numpy as np

sys.path.append(os.path.abspath(__file__).replace('models/snn/temporal_order_coding_image.py', ''))

import utils as u


def construct_DoG_kernel(kernel_size, sigma1, sigma2, to_tensor=False):
  ks = kernel_size // 2
  ker_x, ker_y = np.mgrid[-ks:ks+1, -ks:ks+1]
  ker_prd = ker_x**2 + ker_y**2
  gauss1 = 1 / sigma1**2 * np.exp(-1 / (2 * sigma1**2) * ker_prd)
  gauss2 = 1 / sigma2**2 * np.exp(-1 / (2 * sigma2**2) * ker_prd)
  dog = 1 / (2 * math.pi) * (gauss1 - gauss2)
  dog = dog - np.mean(dog)
  dog = dog / np.max(dog)
  return torch.FloatTensor(dog) if to_tensor else dog


def local_normalization(x, normalization_radius, eps=1e-12):
  aera_size = normalization_radius * 2 + 1
  kernel = torch.ones(1, 1, aera_size, aera_size).float() / aera_size**2
  x = x.permute(1, 0, 2, 3)
  means = torch.nn.functional.conv2d(x, kernel, padding=normalization_radius) + eps
  x = x / means
  return x.permute(1, 0, 2, 3)


class DoGKernel(object):
  def __init__(self, kernel_size, sigma1, sigma2, to_tensor=True):
    self.kernel_size = kernel_size
    self.sigma1 = sigma1
    self.sigma2 = sigma2
    self.to_tensor = to_tensor
  
  def __call__(self):
    return construct_DoG_kernel(self.kernel_size, self.sigma1, self.sigma2, to_tensor=self.to_tensor)
    
    
class Filter(object):
  def __init__(self, configuration):
    base_configuration = {'kernel_type': DoGKernel,
                          'kernels_conf': [[3, 3/9, 6/9], [3, 6/9, 3/9],
                                          [7, 7/9, 14/9], [7, 14/9, 7/9],
                                          [13, 13/9, 26/9], [13, 26/9, 13/9]],
                          'padding': 6,
                          'threshold': 50, 'use_threshold': True}
    self.configuration = u.populate_configuration(configuration, base_configuration)
    # creates kernels and pad smaller kernels to fit biggest one
    self.max_kernel_size = max([conf[0] for conf in self.configuration['kernels_conf']])
    
    kernels = []
    for conf in self.configuration['kernels_conf']:
      kernel = self.configuration['kernel_type'](*conf)().unsqueeze(0)
      pad = (self.max_kernel_size - conf[0]) // 2
      kernels.append(torch.nn.functional.pad(kernel, (pad, pad, pad, pad)))
        
    self.kernels = torch.stack(kernels)
  
  def __call__(self, x):  # x = (1, 1, height, width)
    out = torch.nn.functional.conv2d(x, self.kernels, padding=self.configuration['padding'])
    
    if self.configuration['use_threshold']:
      out = torch.where(out < self.configuration['threshold'], torch.zeros(1), out)
    return out


def cumulative_intensity_to_latency(intencities, n_time_steps, to_spike=True):
  '''intencities = [1, n_feats, height, width]
  We compute bin_size in order to fill all non-zero intencities in the n_time_steps
  We then sort flattened intencities then split it in n bins
  Finally, the cumulative latency matrice is created where we find at each timestep
  the neurons (spike=1) that fire first (higher intencities received)
  Cumulative because when a neuron fire at time_step n, it "keep firing" at n+t timesteps
  '''
  # bin size to get all non-zero intencities into the n_time_steps defined
  bin_size = (intencities != 0).sum() // n_time_steps  # for pytorch < 1.6
  # bin_size = torch.count_nonzero(intencities) // n_time_steps
  
  intencities_flattened_sorted = torch.sort(intencities.view(-1), descending=True)
  
  sorted_bins_value = torch.split(intencities_flattened_sorted[0], bin_size.item())  # for pytorch < 1.6
  sorted_bins_idx = torch.split(intencities_flattened_sorted[1], bin_size.item())  # for pytorch < 1.6
  # sorted_bins_value = torch.split(intencities_flattened_sorted[0], bin_size)
  # sorted_bins_idx = torch.split(intencities_flattened_sorted[1], bin_size)
  
  spike_map = torch.zeros(intencities_flattened_sorted[0].shape)
  
  bins_intencities = []
  for i in range(n_time_steps):
    spike_map.scatter_(0, sorted_bins_idx[i], sorted_bins_value[i])  # cumulative line
    bins_intencities.append(spike_map.clone().reshape(intencities.shape[1:]))
  
  out = torch.stack(bins_intencities)
  return out.sign() if to_spike else out