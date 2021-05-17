# Taken (with possible modifications) from https://github.com/miladmozafari/SpykeTorch
import torch
import torch.nn as nn


class Convolution(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None,
               weight_mean=0.8, weight_std=0.02):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
    self.bias = bias

    self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size), requires_grad=False)
    self._reset_weights(weight_mean=weight_mean, weight_std=weight_std)
  
  def _reset_weights(self, weight_mean=0.8, weight_std=0.02):
    self.weights.normal_(weight_mean, weight_std)
  
  def _load_weight(self, target):
    self.weights.copy_(target)
  
  def forward(self, x, padding=None):
    return nn.functional.conv2d(x, self.weights, bias=self.bias, stride=self.stride,
                                padding=self.padding if padding is None else padding,
                                dilation=self.dilation, groups=self.groups)