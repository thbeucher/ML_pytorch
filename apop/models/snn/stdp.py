# Functional version (with possible modifications) of STDP class from https://github.com/miladmozafari/SpykeTorch
import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(__file__).replace('stdp.py', ''))

import functional as sf


def get_pre_post_ordering(input_spikes, output_spikes, winners, kernel_size):
  """
  Computes the ordering of the input and output spikes with respect to the position of each winner and
  returns them as a list of boolean tensors.
  True for pre-then-post (or concurrency) and False for post-then-pre.
  Input and output tensors must be spike-waves.

  Args:
      input_spikes (Tensor): Input spike-wave
      output_spikes (Tensor): Output spike-wave
      winners (List of Tuples): List of winners.
                                Each tuple denotes a winner in a form of a triplet (feature, row, column).

  Returns:
      List: pre-post ordering of spikes
  """
  # accumulating input and output spikes to get latencies
  input_latencies = torch.sum(input_spikes, dim=0)
  output_latencies = torch.sum(output_spikes, dim=0)

  result = []
  for winner in winners:
    # generating repeated output tensor with the same size of the receptive field
    out_tensor = torch.ones(*kernel_size, device=output_latencies.device) * output_latencies[winner]
    # slicing input tensor with the same size of the receptive field centered around winner
    # since input_latencies is padded and winners are computes on unpadded input we do not need to shift it to the center
    in_tensor = input_latencies[:, winner[-2]:winner[-2] + kernel_size[-2], winner[-1]:winner[-1] + kernel_size[-1]]
    result.append(torch.ge(in_tensor, out_tensor))
  return result


def functional_stdp(conv_layer, learning_rate, input_spikes, output_spikes, winners,
                    use_stabilizer=True, lower_bound=0, upper_bound=1):
  pairings = get_pre_post_ordering(input_spikes, output_spikes, winners, conv_layer.kernel_size)
  
  lr = torch.zeros_like(conv_layer.weights)
  for i in range(len(winners)):
    feat = winners[i][0]
    lr[feat] = torch.where(pairings[i], *(learning_rate[feat]))

  conv_layer.weights += lr * (conv_layer.weights * (1 - conv_layer.weights) if use_stabilizer else 1)
  conv_layer.weights.clamp_(lower_bound, upper_bound)