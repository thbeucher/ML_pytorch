import math
import torch
import torch.nn as nn
from collections import OrderedDict


#############################################################################
###  Implementation of https://arxiv.org/pdf/1608.06993.pdf               ###
###  I have removed all Batch-Norms                                       ###   
#############################################################################
class BasicBlock(nn.Module):
  def __init__(self, in_chan, out_chan, dropout=0.0):
    super().__init__()
    self.relu = nn.ReLU(inplace=True)
    self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    # x = (batch, num_chan, H, W)
    out = self.dropout(self.conv(self.relu(x)))
    # out = (batch, growth_rate, H, W)
    out = torch.cat([x, out], 1)
    # out = (batch, num_chan + growth_rate, H, W)
    return out


class BottleneckBlock(nn.Module):
  def __init__(self, in_chan, out_chan, dropout=0.0):
    super().__init__()
    # Ref paper: Bottleneck layers - In our experiments, we let each 1x1 convolution produce 4k feature-maps
    inter_chan = out_chan * 4
    self.relu = nn.ReLU(inplace=True)
    self.conv1 = nn.Conv2d(in_chan, inter_chan, kernel_size=1, stride=1, padding=0, bias=False)
    self.conv2 = nn.Conv2d(inter_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    # x = (batch, num_chan, H, W)
    out = self.dropout(self.conv1(self.relu(x)))
    # out = (batch, growth_rate * 4, H, W)
    out = self.dropout(self.conv2(self.relu(out)))
    # out = (batch, growth_rate, H, W)
    out = torch.cat([x, out], 1)
    # out = (batch, num_chan + growth_rate, H, W)
    return out


class TransitionBlock(nn.Module):
  def __init__(self, in_chan, out_chan, pool, dropout=0.0):
    super().__init__()
    self.relu = nn.ReLU(inplace=True)
    self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
    self.avg_pool2d = nn.AvgPool2d(pool)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    # x = (batch, num_chan, H, W)
    out = self.dropout(self.conv(self.relu(x)))
    # out = (batch, num_chan * compression, H, W)
    out = self.avg_pool2d(out)
    # out = (batch, num_chan * compression, H/2, W/2)
    return out


class DenseBlock(nn.Module):
  def __init__(self, num_layers, in_chan, growth_rate, block, dropout=0.0):
    super().__init__()
    self.layer = nn.Sequential(*[block(in_chan + i * growth_rate, growth_rate, dropout=dropout) for i in range(num_layers)])
  
  def forward(self, x):
    return self.layer(x)


class DenseNet(nn.Module):
  def __init__(self, configuration):
    base_configuration = {'in_chan': 1, 'num_classes': None, 'block_config': [3, 3], 'height': 103, 'width': 300,
                          'classifier_on': False, 'final_pool': 8, 'dropout': 0., 'growth_rate': 12, 'compression': 0.5,
                          'num_init_features': 24, 'bottleneck': True, 'transition_pool': 2}
    self.configuration = {k: v if k not in configuration else configuration[k] for k, v in base_configuration.items()}
    super().__init__()
    self.classifier_on = self.configuration['classifier_on']

    self.relu = nn.ReLU(inplace=True)
    self.avg_pool2d = nn.AvgPool2d(self.configuration['final_pool'])
    # First Convolution
    self.network = nn.Sequential(OrderedDict([
      ('conv0', nn.Conv2d(self.configuration['in_chan'], self.configuration['num_init_features'], kernel_size=3, stride=1,
                          padding=1, bias=False))
    ]))

    block = BottleneckBlock if self.configuration['bottleneck'] else BasicBlock

    num_feats = self.configuration['num_init_features']
    
    # DenseBlocks
    for i, num_layers in enumerate(self.configuration['block_config']):
      self.network.add_module(f'denseblock{i + 1}', DenseBlock(num_layers, num_feats, self.configuration['growth_rate'], block,
                                                               dropout=self.configuration['dropout']))
      num_feats = num_feats + num_layers * self.configuration['growth_rate']

      if i != len(self.configuration['block_config']) - 1:
        self.network.add_module(f'transition{i + 1}', TransitionBlock(num_feats, int(num_feats * self.configuration['compression']),
                                                                      self.configuration['transition_pool']))
        num_feats = int(num_feats * self.configuration['compression'])
      
    # Final Layer
    self.num_feats = self._compute_final_num_feats(self.configuration['block_config'], self.configuration['height'],
                                                   self.configuration['width'], self.configuration['growth_rate'],
                                                   self.configuration['compression'], self.configuration['num_init_features'],
                                                   self.configuration['transition_pool'], self.configuration['final_pool'])
    
    if self.classifier_on:
      self.classifier = nn.Linear(num_feats, self.configuration['num_classes'])
  
  def _compute_final_num_feats(self, block_config, height, width, growth_rate, compression, init_feats, transition_pool, final_pool):
    '''

    Params:
      * block_config : list of int
      * height : int
      * width : int
      * growth_rate : int
      * compression : float
      * init_feats : int
      * transition_pool : int
      * final_pool : int
    
    Returns:
      * int
    '''
    f_feats = init_feats

    for i, n_layers in enumerate(block_config):
      f_feats = f_feats + n_layers * growth_rate

      if i != len(block_config) - 1:
        f_feats = int(f_feats * compression)

    dimension_div = transition_pool ** (len(block_config) - 1) * final_pool

    self.h_mul = 1 if int(height / dimension_div) == 0 else int(height / dimension_div)
    self.w_mul = 1 if int(width / dimension_div) == 0 else int(width / dimension_div)

    self.f_feats = f_feats
      
    return self.f_feats * self.h_mul * self.w_mul
  
  def forward(self, x):
    # x = (batch, num_chan, H, W)
    out = self.relu(self.network(x))
    out = self.avg_pool2d(out)

    if self.classifier_on:
      out = out.view(out.shape[0], -1)
      out = self.classifier(out)
    
    return out


if __name__ == "__main__":
  t = torch.ones(2, 1, 120, 150)

  dn = DenseNet(1, 3, [4, 3, 3, 3], 120, 150)

  res = dn(t)
  print(f'Network output: {res.shape}')