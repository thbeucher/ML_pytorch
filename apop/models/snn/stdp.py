# Functional version (with modifications) of STDP class from https://github.com/miladmozafari/SpykeTorch
import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(__file__).replace('stdp.py', ''))

import functional as sf


def get_pre_post_ordering(input_spikes, output_spikes, winners, kernel_size, winners_shifted=None):
  """
  Computes the ordering of the input and output spikes with respect to the position of each winner and
  returns them as a list of boolean tensors.
  True for pre-then-post (or concurrency) and False for post-then-pre.
  Input and output tensors must be spike-waves.

  Args:
      input_spikes (Tensor): Input spike-wave (shape = [timestep, feat_in(eg2), height_pad, width_pad])
      output_spikes (Tensor): Output spike-wave (shape = [timestep, feat_out(eg32), height, width])
      winners (List of Tuples): List of winners.
                                Each tuple denotes a winner in a form of a triplet (feature, row, column).

  Returns:
      List: pre-post ordering of spikes
  """
  # accumulating input and output spikes to get latencies
  input_latencies = torch.sum(input_spikes, dim=0)  # [feat_in, height_pad, width_pad]
  output_latencies = torch.sum(output_spikes, dim=0)  # [feat_out, height, width]

  result = []
  for i, winner in enumerate(winners):
    # generating repeated output tensor with the same size of the receptive field
    out_tensor = torch.ones(*kernel_size, device=output_latencies.device) * output_latencies[winner]
    # slicing input tensor with the same size of the receptive field centered around winner
    # since input_latencies is padded and winners are computes on unpadded input we do not need to shift it to the center
    winner = winners_shifted[i] if winners_shifted is not None else winner
    in_tensor = input_latencies[:, winner[-2]:winner[-2] + kernel_size[-2], winner[-1]:winner[-1] + kernel_size[-1]]
    result.append(torch.ge(in_tensor, out_tensor))  # ge = in_tensor >= out_tensor

  return result  # results_1 shape = [feat_in, kernel_size, kernel_size]


def functional_stdp(conv_layer, learning_rate, input_spikes, output_spikes, winners,
                    use_stabilizer=True, lower_bound=0, upper_bound=1, strategy='wta', winners_shifted=None):
  pairings = get_pre_post_ordering(input_spikes, output_spikes, winners, conv_layer.kernel_size, winners_shifted=winners_shifted)
  
  lr = torch.zeros_like(conv_layer.weights) if strategy == 'wta' else torch.ones_like(conv_layer.weights) * learning_rate[0][1]
  for i in range(len(winners)):
    feat = winners[i][0]
    lr[feat] = torch.where(pairings[i], *(learning_rate[feat]))

  conv_layer.weights += lr * ((conv_layer.weights - lower_bound) * (upper_bound - conv_layer.weights) if use_stabilizer else 1)
  conv_layer.weights.clamp_(lower_bound, upper_bound)