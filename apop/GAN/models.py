import torch
from torch.nn.modules.activation import Tanh


def weights_init(m, mean=0.0, std=0.02):
  # from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
  # Initialization preconised by https://arxiv.org/pdf/1511.06434.pdf
  # Normal distribution with mean=0 and std=0.02
  # Apply this on the CNN with generator.apply(weights_init)
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    torch.nn.init.normal_(m.weight.data, mean, std)
  elif classname.find('BatchNorm') != -1:
    torch.nn.init.normal_(m.weight.data, 1.0, std)
    torch.nn.init.constant_(m.bias.data, 0)


def sequential_constructor(network_config):
  network = []
  for layer_config in network_config:
    network.append(layer_config['type'](**layer_config['params']))
  return torch.nn.Sequential(*network)


def modulelist_sequential_constructor(networks_config):
  networks = []
  for network in networks_config:
    networks.append(sequential_constructor(network))
  return torch.nn.ModuleList(networks)


class MLPDiscriminator(torch.nn.Module):
  BASE_CONFIG = {'layers_config': [{'type': torch.nn.Linear,    'params': {'in_features': 784, 'out_features': 1024}},
                                   {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2}},
                                   {'type': torch.nn.Dropout,   'params': {'p': 0.3}},
                                   {'type': torch.nn.Linear,    'params': {'in_features': 1024, 'out_features': 512}},
                                   {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2}},
                                   {'type': torch.nn.Dropout,   'params': {'p': 0.3}},
                                   {'type': torch.nn.Linear,    'params': {'in_features': 512, 'out_features': 256}},
                                   {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2}},
                                   {'type': torch.nn.Dropout,   'params': {'p': 0.3}},
                                   {'type': torch.nn.Linear,    'params': {'in_features': 256, 'out_features': 1}},
                                   {'type': torch.nn.Sigmoid,   'params': {}}]}
  def __init__(self, config):
    super().__init__()
    MLPDiscriminator.BASE_CONFIG['layers_config'][-2]['params']['out_features'] = config.get('n_classes', 1)
    self.config = {**MLPDiscriminator.BASE_CONFIG, **config}
    self.network = sequential_constructor(self.config['layers_config'])
  
  def forward(self, x):
    return self.network(x)


class MLPGenerator(torch.nn.Module):
  BASE_CONFIG = {'layers_config': [{'type': torch.nn.Linear,    'params': {'in_features': 100, 'out_features': 256}},
                                   {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2}},
                                   {'type': torch.nn.Linear,    'params': {'in_features': 256, 'out_features': 512}},
                                   {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2}},
                                   {'type': torch.nn.Linear,    'params': {'in_features': 512, 'out_features': 1024}},
                                   {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2}},
                                   {'type': torch.nn.Linear,    'params': {'in_features': 1024, 'out_features': 784}},
                                   {'type': torch.nn.Tanh,      'params': {}}]}
  def __init__(self, config):
    super().__init__()
    self.config = {**MLPGenerator.BASE_CONFIG, **config}
    self.network = sequential_constructor(self.config['layers_config'])
  
  def forward(self, x):
    return self.network(x)


class CNNGenerator(torch.nn.Module):
  BASE_CONFIG = {'layers_config': [
    {'type': torch.nn.ConvTranspose2d,  # 100*1*1 -> 256*4*4
     'params': {'in_channels': 100, 'out_channels': 256, 'kernel_size': 4, 'stride': 1, 'padding': 0, 'bias': False}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 256}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},

    {'type': torch.nn.ConvTranspose2d,  # 256*4*4 -> 128*7*7
     'params': {'in_channels': 256, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'bias': False}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 128}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},

    {'type': torch.nn.ConvTranspose2d,  # 128*7*7 -> 64*14*14
     'params': {'in_channels': 128, 'out_channels': 64, 'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 64}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},

    {'type': torch.nn.ConvTranspose2d,  # 64*14*14 -> 1*28*28
     'params': {'in_channels': 64, 'out_channels': 1, 'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False}},
    {'type': torch.nn.Tanh, 'params': {}}]}
  def __init__(self, config):
    super().__init__()
    # H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    CNNGenerator.BASE_CONFIG['layers_config'][0]['params']['in_channels'] = config.get('latent_vector_size', 100)
    CNNGenerator.BASE_CONFIG['layers_config'][-2]['params']['out_channels'] = config.get('n_channels', 1)
    self.config = {**CNNGenerator.BASE_CONFIG, **config}

    self.network = sequential_constructor(self.config['layers_config'])
    self._reset_weights()

  def forward(self, x):
    return self.network(x)
  
  def _reset_weights(self):
    self.network.apply(weights_init)


class CNNDiscriminator(torch.nn.Module):
  BASE_CONFIG = {'layers_config': [
    {'type': torch.nn.Conv2d,  # 1*28*28 -> 64*14*14
     'params': {'in_channels': 1, 'out_channels': 64, 'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False}},
    {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2, 'inplace': True}},

    {'type': torch.nn.Conv2d,  # 64*14*14 -> 128*7*7
     'params': {'in_channels': 64, 'out_channels': 128, 'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 128}},
    {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2, 'inplace': True}},

    {'type': torch.nn.Conv2d,  # 128*7*7 -> 256*4*4
     'params': {'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'bias': False}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 256}},
    {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2, 'inplace': True}},

    {'type': torch.nn.Conv2d,  # 256*4*4 -> 1*1*1
     'params': {'in_channels': 256, 'out_channels': 1, 'kernel_size': 4, 'stride': 1, 'padding': 0, 'bias': False}},
    {'type': torch.nn.Sigmoid, 'params': {}}]}
  def __init__(self, config):
    super().__init__()
    # H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    CNNDiscriminator.BASE_CONFIG['layers_config'][0]['params']['in_channels'] = config.get('n_channels', 1)
    CNNDiscriminator.BASE_CONFIG['layers_config'][-2]['params']['out_channels'] = config.get('n_classes', 1)
    self.config = {**CNNDiscriminator.BASE_CONFIG, **config}

    self.network = sequential_constructor(self.config['layers_config'])
    self._reset_weights()

  def forward(self, x):
    return self.network(x)
  
  def _reset_weights(self):
    self.network.apply(weights_init)


class ConditionalCNNDiscriminator(CNNDiscriminator):
  def __init__(self, config):
    config['n_channels'] = 2
    super().__init__(config)
    self.embed = torch.nn.Embedding(config.get('n_classes', 10), config.get('embedding_size', 28*28))
  
  def forward(self, x, labels):
    embedding = self.embed(labels).view(x.shape[0], 1, x.shape[2], x.shape[3])
    return self.network(torch.cat([x, embedding], dim=1))


class ConditionalCNNGenerator(CNNGenerator):
  def __init__(self, config):
    config['latent_vector_size'] = 100 + 100
    super().__init__(config)
    self.embed = torch.nn.Embedding(config.get('n_classes', 10), config.get('embedding_size', 100))
  
  def forward(self, x, labels):
    embedding = self.embed(labels).unsqueeze(-1).unsqueeze(-1)
    return self.network(torch.cat([x, embedding], dim=1))


class ACCNNDiscriminator(torch.nn.Module):
  BASE_CONFIG = {'body_config': [
    {'type': torch.nn.Conv2d,  # 1*28*28 -> 64*14*14
     'params': {'in_channels': 1, 'out_channels': 64, 'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False}},
    {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2, 'inplace': True}},

    {'type': torch.nn.Conv2d,  # 64*14*14 -> 128*7*7
     'params': {'in_channels': 64, 'out_channels': 128, 'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 128}},
    {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2, 'inplace': True}},

    {'type': torch.nn.Conv2d,  # 128*7*7 -> 256*4*4
     'params': {'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'bias': False}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 256}},
    {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2, 'inplace': True}}],

                  'heads_config': [
    [{'type': torch.nn.Conv2d, # 256*4*4 -> 1*1*1
      'params': {'in_channels': 256, 'out_channels': 1, 'kernel_size': 4, 'stride': 1, 'padding': 0, 'bias': False}},
     {'type': torch.nn.Sigmoid, 'params': {}}],

    [{'type': torch.nn.Conv2d, # 256*4*4 -> 10*1*1
      'params': {'in_channels': 256, 'out_channels': 10, 'kernel_size': 4, 'stride': 1, 'padding': 0, 'bias': False}}]]}
  def __init__(self, config):
    super().__init__()
    # H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    ACCNNDiscriminator.BASE_CONFIG['body_config'][0]['params']['in_channels'] = config.get('n_channels', 1)
    ACCNNDiscriminator.BASE_CONFIG['heads_config'][1][0]['params']['out_channels'] = config.get('n_classes', 10)
    self.config = {**ACCNNDiscriminator.BASE_CONFIG, **config}

    self.body = sequential_constructor(self.config['body_config'])
    self.heads = modulelist_sequential_constructor(self.config['heads_config'])
    self._reset_weights()

  def forward(self, x):
    out = self.body(x)
    return [head(out) for head in self.heads]
  
  def _reset_weights(self):
    self.body.apply(weights_init)
    for head in self.heads:
      head.apply(weights_init)


class MNISTClassifier(torch.nn.Module):
  BASE_CONFIG = {'body_config': [
    {'type': torch.nn.Conv2d, 'params': {'in_channels': 1, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 32}},

    {'type': torch.nn.Conv2d, 'params': {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 32}},

    {'type': torch.nn.Conv2d, 'params': {'in_channels': 32, 'out_channels': 32, 'kernel_size': 5, 'stride': 2, 'padding': 2}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 32}},

    {'type': torch.nn.Dropout, 'params': {'p': 0.4}},

    {'type': torch.nn.Conv2d, 'params': {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 64}},

    {'type': torch.nn.Conv2d, 'params': {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 64}},

    {'type': torch.nn.Conv2d, 'params': {'in_channels': 64, 'out_channels': 64, 'kernel_size': 5, 'stride': 2, 'padding': 2}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 64}},

    {'type': torch.nn.Dropout, 'params': {'p': 0.4}},

    {'type': torch.nn.Conv2d, 'params': {'in_channels': 64, 'out_channels': 128, 'kernel_size': 4, 'stride': 1, 'padding': 0}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 128}},

    {'type': torch.nn.Flatten, 'params': {}},
    {'type': torch.nn.Dropout, 'params': {'p': 0.4}}],

                  'heads_config': [[{'type': torch.nn.Linear, 'params': {'in_features': 128, 'out_features': 10}}]]}
  def __init__(self, config):
    super().__init__()
    MNISTClassifier.BASE_CONFIG['body_config'][0]['params']['in_channels'] = config.get('n_channels', 1)
    MNISTClassifier.BASE_CONFIG['heads_config'][0][0]['params']['out_features'] = config.get('n_classes', 10)
    self.config = {**MNISTClassifier.BASE_CONFIG, **config}

    self.body = sequential_constructor(self.config['body_config'])
    self.heads = modulelist_sequential_constructor(self.config['heads_config'])
  
  def forward(self, x):
    out = self.body(x)
    out_comp = self.heads[0](out)

    if len(self.heads) == 1:
      return out_comp
    else:
      return out_comp, self.heads[1](torch.cat([out, out_comp], dim=1))


if __name__ == '__main__':
  import pandas as pd
  from tabulate import tabulate

  def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

  models = [{'type': MLPGenerator,                'args': {}},
            {'type': MLPDiscriminator,            'args': {}},
            {'type': CNNGenerator,                'args': {}},
            {'type': CNNDiscriminator,            'args': {}},
            {'type': CNNDiscriminator,            'args': {'n_classes': 10}},
            {'type': ConditionalCNNDiscriminator, 'args': {}},
            {'type': ConditionalCNNGenerator,     'args': {}},
            {'type': ACCNNDiscriminator,          'args': {}},
            {'type': MNISTClassifier,             'args': {}}]
  
  inputs = [ {'type': torch.randn, 'args': (4, 100)},
             {'type': torch.randn, 'args': (4, 28*28)},
             {'type': torch.randn, 'args': (4, 100, 1, 1)},
             {'type': torch.randn, 'args': (4, 1, 28, 28)},
             {'type': torch.randn, 'args': (4, 1, 28, 28)},
            [{'type': torch.randn, 'args': (4, 1, 28, 28)}, {'type': torch.randint, 'args': [0, 10, (4,)]}],
            [{'type': torch.randn, 'args': (4, 100, 1, 1)}, {'type': torch.randint, 'args': [0, 10, (4,)]}],
             {'type': torch.randn, 'args': (4, 1, 28, 28)},
             {'type': torch.randn, 'args': (4, 1, 28, 28)}]
  
  tab = {'Classname': [], 'input_shape': [], 'output_shape': [], 'n_parameters': []}
  for model, inp in zip(models, inputs):
    mdl = model['type'](model['args'])

    if isinstance(inp, list):
      args = [i['type'](*i['args']) if isinstance(i['args'], list) else i['type'](i['args']) for i in inp]
      args_shape = ', '.join([f"{i['args'][-1]}" if isinstance(i['args'], list) else f"{i['args']}" for i in inp]) 
      out = mdl(*args)
    else:
      args_shape = inp['args']
      out = mdl(inp['type'](inp['args']))

    tab['Classname'].append(type(mdl).__name__)
    tab['input_shape'].append(args_shape)

    out_shape = ', '.join([f'{list(o.shape)}' for o in out]) if isinstance(out, list) else f'{list(out.shape)}'
    tab['output_shape'].append(out_shape)

    tab['n_parameters'].append(f'{count_parameters(mdl):,}')
  
  print(tabulate(pd.DataFrame.from_dict(tab), headers='keys', tablefmt='psql'))