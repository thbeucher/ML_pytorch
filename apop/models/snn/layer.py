# Taken (with modifications) from https://github.com/miladmozafari/SpykeTorch
import torch
import torch.nn as nn


class Layer(nn.Module):
  def __init__(self):
    super().__init__()
  
  def _reset_weights(self, weight_mean=0.8, weight_std=0.02):
    self.weights.normal_(weight_mean, weight_std)
  
  def _load_weight(self, target):
    self.weights.copy_(target)


class Convolution(Layer):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None,
               weight_mean=0.8, weight_std=0.02, *args, **kwargs):
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
  
  def forward(self, x, padding=None):
    return nn.functional.conv2d(x, self.weights, bias=self.bias, stride=self.stride,
                                padding=self.padding if padding is None else padding,
                                dilation=self.dilation, groups=self.groups)


class Linear(Layer):
  def __init__(self, in_feat, out_feat, learning_rates, weight_mean=0.8, weight_std=0.02, *args, **kwargs):
    super().__init__()
    self.in_feat = in_feat
    self.out_feat = out_feat

    self.lr = learning_rates

    self.weights = nn.Parameter(torch.Tensor(in_feat, out_feat), requires_grad=False)
    self._reset_weights(weight_mean=weight_mean, weight_std=weight_std)
  
  def _update_all_lr(self, new_ap, new_an):
    self.lr = [new_ap, new_an]
  
  def forward(self, x):
    return x.mm(self.weights)
  
  def get_winner(self, potentials, spikes):
    truncated_pot = spikes * potentials.gather(dim=0, index=torch.topk(spikes, 1, dim=0)[1])
    truncated_pot.addcmul_(spikes, truncated_pot.max() * potentials.size(0))
    return truncated_pot.sum(dim=0).max(dim=0)[1]
  
  def stdp(self, potentials, input_spikes, output_spikes, use_stabilizer=True, lower_bound=0, upper_bound=1):
    '''
    Parameters:
      * potentials : torch.Tensor, shape = [timesteps, out_feat]
      * input_spikes : torch.Tensor, shape = [timesteps, in_feat]
      * output_spikes : torch.Tensor, shape = [timesteps, out_feat]
    '''
    winner = self.get_winner(potentials, output_spikes)

    in_latencies = input_spikes.sum(dim=0)
    out_latencies_winner = torch.ones(in_latencies.shape, device=potentials.device) * output_spikes.sum(dim=0)[winner]

    lr = torch.zeros_like(self.weights)
    lr.T[winner] = torch.where(torch.ge(in_latencies, out_latencies_winner), *self.lr)

    self.weights += lr * (self.weights * (1 - self.weights) if use_stabilizer else 1)
    self.weights.clamp_(lower_bound, upper_bound)



if __name__ == "__main__":
  linear = Linear(12127, 10, [0.004, -0.003])
  in_ = torch.randn(15, 12127)
  out = linear(in_)
  print(out.shape)
  print(linear.state_dict())