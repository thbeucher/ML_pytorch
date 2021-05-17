# Functions taken (with possible modifications) from https://github.com/miladmozafari/SpykeTorch
import torch
import numpy as np


def pointwise_feature_competition_inhibition(potentials):
  '''In each position, the first spiking neuron will inhibit his concurrents'''
  pot_max = potentials.max(dim=1, keepdim=True)
  # use topk instead of max as since torch version 1.0.0, max doesn't return first index when equal max values
  earliest_spikes = pot_max[0].sign().topk(1, dim=0)
  winners = pot_max[1].gather(0, earliest_spikes[1])  # keep values of only winning neurons
  coefs = torch.zeros_like(potentials[0]).unsqueeze_(0).scatter_(1, winners, earliest_spikes[0])  # inhibition coefs
  return torch.mul(potentials, coefs)  # broadcast on each timesteps


def unravel_index(index, shape):
  out = []
  for dim in reversed(shape):
    out.append(index % dim)
    index = index // dim
  return tuple(reversed(out))


def get_k_winners(potentials, kwta=1, inhibition_radius=0, spikes=None):
  if spikes is None:
    spikes = potentials.sign()
  # finding earliest potentials for each position in each feature
  # use topk instead of max as since torch version 1.0, max doesn't return first index when equal max values
  maximum = torch.topk(spikes, 1, dim=0)
  values = potentials.gather(dim=0, index=maximum[1]) # gathering values
  # propagating the earliest potential through the whole timesteps
  truncated_pot = spikes * values
  # summation with a high enough value (maximum of potential summation over timesteps) at spike positions
  v = truncated_pot.max() * potentials.size(0)
  truncated_pot.addcmul_(spikes,v)
  # summation over all timesteps
  total = truncated_pot.sum(dim=0,keepdim=True)
  
  total.squeeze_(0)
  global_pooling_size = tuple(total.size())
  winners = []
  for k in range(kwta):
    max_val,max_idx = total.view(-1).max(0)
    if max_val.item() != 0:
      # finding the 3d position of the maximum value
      max_idx_unraveled = np.unravel_index(max_idx.item(),global_pooling_size)
      # adding to the winners list
      winners.append(max_idx_unraveled)
      # preventing the same feature to be the next winner
      total[max_idx_unraveled[0],:,:] = 0
      # columnar inhibition (increasing the chance of leanring diverse features)
      if inhibition_radius != 0:
        rowMin,rowMax = max(0,max_idx_unraveled[-2]-inhibition_radius),min(total.size(-2),max_idx_unraveled[-2]+inhibition_radius+1)
        colMin,colMax = max(0,max_idx_unraveled[-1]-inhibition_radius),min(total.size(-1),max_idx_unraveled[-1]+inhibition_radius+1)
        total[:,rowMin:rowMax,colMin:colMax] = 0
    else:
      break
  return winners


def fire(potentials, threshold=None, return_thresholded_potentials=False):
  thresholded = potentials.clone()
  if threshold is None:
    thresholded[:-1]=0
  else:
    torch.nn.functional.threshold_(thresholded, threshold, 0)
  return (thresholded.sign(), thresholded) if return_thresholded_potentials else thresholded.sign()


if __name__ == "__main__":
  potentials = torch.tensor([1, 2, 0, 0, 2, 1, 0, 0] + [1, 1, 2, 3, 1, 1, 3, 2] + [1] * 8).reshape(3, 2, 2, 2)
  print(f'potentials:\n{potentials}')
  potentials = pointwise_feature_competition_inhibition(potentials)
  print(f'potentials after inhibition:\n{potentials}')
  spikes = potentials.sign()
  print(get_k_winners(potentials, kwta=2, inhibition_radius=0, spikes=spikes))