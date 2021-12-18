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


class MLPDiscriminator(torch.nn.Module):
  BASE_CONFIG = {'layers_config': [{'type': torch.nn.Linear, 'params': {'in_features': 784, 'out_features': 1024}},
                                   {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2}},
                                   {'type': torch.nn.Dropout, 'params': {'p': 0.3}},
                                   {'type': torch.nn.Linear, 'params': {'in_features': 1024, 'out_features': 512}},
                                   {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2}},
                                   {'type': torch.nn.Dropout, 'params': {'p': 0.3}},
                                   {'type': torch.nn.Linear, 'params': {'in_features': 512, 'out_features': 256}},
                                   {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2}},
                                   {'type': torch.nn.Dropout, 'params': {'p': 0.3}},
                                   {'type': torch.nn.Linear, 'params': {'in_features': 256, 'out_features': 1}},
                                   {'type': torch.nn.Sigmoid, 'params': {}}]}
  def __init__(self, config):
    super().__init__()
    MLPDiscriminator.BASE_CONFIG['layers_config'][-2]['params']['out_features'] = config.get('n_classes', 1)
    self.config = {**MLPDiscriminator.BASE_CONFIG, **config}

    network = []
    for layer_config in self.config['layers_config']:
      network.append(layer_config['type'](**layer_config['params']))
    self.network = torch.nn.Sequential(*network)
  
  def forward(self, x):
    return self.network(x)


class MLPGenerator(torch.nn.Module):
  BASE_CONFIG = {'layers_config': [{'type': torch.nn.Linear, 'params': {'in_features': 100, 'out_features': 256}},
                                   {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2}},
                                   {'type': torch.nn.Linear, 'params': {'in_features': 256, 'out_features': 512}},
                                   {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2}},
                                   {'type': torch.nn.Linear, 'params': {'in_features': 512, 'out_features': 1024}},
                                   {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2}},
                                   {'type': torch.nn.Linear, 'params': {'in_features': 1024, 'out_features': 784}},
                                   {'type': torch.nn.Tanh, 'params': {}}]}
  def __init__(self, config):
    super().__init__()
    self.config = {**MLPGenerator.BASE_CONFIG, **config}

    network = []
    for layer_config in self.config['layers_config']:
      network.append(layer_config['type'](**layer_config['params']))
    self.network = torch.nn.Sequential(*network)
  
  def forward(self, x):
    return self.network(x)


class CNNGenerator(torch.nn.Module):
  BASE_CONFIG = {'layers_config': [{'type': torch.nn.ConvTranspose2d,  # 100*1*1 -> 256*4*4
                                    'params': {'in_channels': 100, 'out_channels': 256, 'kernel_size': 4,
                                               'stride': 1, 'padding': 0, 'bias': False}},
                                   {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 256}},
                                   {'type': torch.nn.ReLU, 'params': {'inplace': True}},
                                   {'type': torch.nn.ConvTranspose2d,  # 256*4*4 -> 128*7*7
                                    'params': {'in_channels': 256, 'out_channels': 128, 'kernel_size': 3,
                                               'stride': 2, 'padding': 1, 'bias': False}},
                                   {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 128}},
                                   {'type': torch.nn.ReLU, 'params': {'inplace': True}},
                                   {'type': torch.nn.ConvTranspose2d,  # 128*7*7 -> 64*14*14
                                    'params': {'in_channels': 128, 'out_channels': 64, 'kernel_size': 4,
                                               'stride': 2, 'padding': 1, 'bias': False}},
                                   {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 64}},
                                   {'type': torch.nn.ReLU, 'params': {'inplace': True}},
                                   {'type': torch.nn.ConvTranspose2d,  # 64*14*14 -> 1*28*28
                                    'params': {'in_channels': 64, 'out_channels': 1, 'kernel_size': 4,
                                               'stride': 2, 'padding': 1, 'bias': False}},
                                   {'type': torch.nn.Tanh, 'params': {}}]}
  def __init__(self, config):
    super().__init__()
    # H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    CNNGenerator.BASE_CONFIG['layers_config'][0]['params']['in_channels'] = config.get('latent_vector_size', 100)
    CNNGenerator.BASE_CONFIG['layers_config'][-2]['params']['out_channels'] = config.get('n_channels', 1)
    self.config = {**CNNGenerator.BASE_CONFIG, **config}

    network = []
    for layer_config in self.config['layers_config']:
      network.append(layer_config['type'](**layer_config['params']))
    self.network = torch.nn.Sequential(*network)

    self._reset_weights()

  def forward(self, x):
    return self.network(x)
  
  def _reset_weights(self):
    self.network.apply(weights_init)


class CNNDiscriminator(torch.nn.Module):
  BASE_CONFIG = {'layers_config': [{'type': torch.nn.Conv2d,  # 1*28*28 -> 64*14*14
                                    'params': {'in_channels': 1, 'out_channels': 64, 'kernel_size': 4,
                                               'stride': 2, 'padding': 1, 'bias': False}},
                                   {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2, 'inplace': True}},
                                   {'type': torch.nn.Conv2d,  # 64*14*14 -> 128*7*7
                                    'params': {'in_channels': 64, 'out_channels': 128, 'kernel_size': 4,
                                               'stride': 2, 'padding': 1, 'bias': False}},
                                   {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 128}},
                                   {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2, 'inplace': True}},
                                   {'type': torch.nn.Conv2d,  # 128*7*7 -> 256*4*4
                                    'params': {'in_channels': 128, 'out_channels': 256, 'kernel_size': 3,
                                               'stride': 2, 'padding': 1, 'bias': False}},
                                   {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 256}},
                                   {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2, 'inplace': True}},
                                   {'type': torch.nn.Conv2d,  # 256*4*4 -> 1*1*1
                                    'params': {'in_channels': 256, 'out_channels': 1, 'kernel_size': 4,
                                               'stride': 1, 'padding': 0, 'bias': False}},
                                   {'type': torch.nn.Sigmoid, 'params': {}}]}
  def __init__(self, config):
    super().__init__()
    # H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    CNNDiscriminator.BASE_CONFIG['layers_config'][0]['params']['in_channels'] = config.get('n_channels', 1)
    CNNDiscriminator.BASE_CONFIG['layers_config'][-2]['params']['out_channels'] = config.get('n_classes', 1)
    self.config = {**CNNDiscriminator.BASE_CONFIG, **config}

    network = []
    for layer_config in self.config['layers_config']:
      network.append(layer_config['type'](**layer_config['params']))
    self.network = torch.nn.Sequential(*network)

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
  BASE_CONFIG = {'body_config': [{'type': torch.nn.Conv2d,  # 1*28*28 -> 64*14*14
                                  'params': {'in_channels': 1, 'out_channels': 64, 'kernel_size': 4,
                                             'stride': 2, 'padding': 1, 'bias': False}},
                                 {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2, 'inplace': True}},
                                 {'type': torch.nn.Conv2d,  # 64*14*14 -> 128*7*7
                                  'params': {'in_channels': 64, 'out_channels': 128, 'kernel_size': 4,
                                             'stride': 2, 'padding': 1, 'bias': False}},
                                 {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 128}},
                                 {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2, 'inplace': True}},
                                 {'type': torch.nn.Conv2d,  # 128*7*7 -> 256*4*4
                                  'params': {'in_channels': 128, 'out_channels': 256, 'kernel_size': 3,
                                             'stride': 2, 'padding': 1, 'bias': False}},
                                 {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 256}},
                                 {'type': torch.nn.LeakyReLU, 'params': {'negative_slope': 0.2, 'inplace': True}}],
                  'heads_config': [[{'type': torch.nn.Conv2d, # 256*4*4 -> 1*1*1
                                     'params': {'in_channels': 256, 'out_channels': 1, 'kernel_size': 4,
                                                'stride': 1, 'padding': 0, 'bias': False}},
                                    {'type': torch.nn.Sigmoid, 'params': {}}],
                                   [{'type': torch.nn.Conv2d, # 256*4*4 -> 10*1*1
                                     'params': {'in_channels': 256, 'out_channels': 10, 'kernel_size': 4,
                                                'stride': 1, 'padding': 0, 'bias': False}}]]}
  def __init__(self, config):
    super().__init__()
    # H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    ACCNNDiscriminator.BASE_CONFIG['body_config'][0]['params']['in_channels'] = config.get('n_channels', 1)
    ACCNNDiscriminator.BASE_CONFIG['heads_config'][1][0]['params']['out_channels'] = config.get('n_classes', 10)
    self.config = {**ACCNNDiscriminator.BASE_CONFIG, **config}

    body = []
    for layer_config in self.config['body_config']:
      body.append(layer_config['type'](**layer_config['params']))
    self.body = torch.nn.Sequential(*body)

    heads = []
    for head in self.config['heads_config']:
      head_layers = []
      for layer_config in head:
        head_layers.append(layer_config['type'](**layer_config['params']))
      heads.append(torch.nn.Sequential(*head_layers))
      
    self.heads = torch.nn.ModuleList(heads)

    self._reset_weights()

  def forward(self, x):
    out = self.body(x)
    return [head(out) for head in self.heads]
  
  def _reset_weights(self):
    self.body.apply(weights_init)
    for head in self.heads:
      head.apply(weights_init)


if __name__ == '__main__':
  mlp_generator = MLPGenerator({})
  print('\nMLPGenerator input=[4, 100]}')  # out = [4, 784]
  print(f'MLPGenerator out={mlp_generator(torch.randn(4, 100)).shape}')

  mlp_discriminator = MLPDiscriminator({})
  print('\nMLPDiscriminator input=[4, 28*28]')  # out = [4, 1]
  print(f'MLPDiscriminator out={mlp_discriminator(torch.randn(4, 28*28)).shape}')

  cnn_generator = CNNGenerator({})
  print('\nCNNGenerator input=[4, 100, 1, 1]')  # out = [4, 1, 28, 28]
  print(f'CNNGenerator out={cnn_generator(torch.randn(4, 100, 1, 1)).shape}')

  cnn_discriminator = CNNDiscriminator({})
  print('\nCNNDiscriminator input=[4, 1, 28, 28]')  # out = [4, 1, 1, 1]
  print(f'CNNDiscriminator out={cnn_discriminator(torch.randn(4, 1, 28, 28)).shape}')

  cnn_discriminator = CNNDiscriminator({'n_classes': 10})
  print('\nCNNDiscriminator input=[4, 1, 28, 28]')  # out = [4, 10, 1, 1]
  print(f'CNNDiscriminator out={cnn_discriminator(torch.randn(4, 1, 28, 28)).shape}')

  c_cnn_discriminator = ConditionalCNNDiscriminator({})
  print('\nConditionalCNNDiscriminator input=[4, 1, 28, 28], [4]')  # out = [4, 1, 1, 1]
  print(f'ConditionalCNNDiscriminator out={c_cnn_discriminator(torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,))).shape}')

  c_cnn_generator = ConditionalCNNGenerator({})
  print('\nConditionalCNNGenerator input=[4, 100, 1, 1], [4]')  # out = [4, 1, 28, 28]
  print(f'ConditionalCNNGenerator out={c_cnn_generator(torch.randn(4, 100, 1, 1), torch.randint(0, 10, (4,))).shape}')

  ac_cnn_discriminator = ACCNNDiscriminator({})
  print('\nACCNNDiscriminator input=[4, 1, 28, 28]')  # out = [4, 1, 1, 1], [4, 10, 1, 1]
  out_real_fake, out_classif = ac_cnn_discriminator(torch.randn(4, 1, 28, 28))
  print(f'ACCNNDiscriminator out_real_fake={out_real_fake.shape} | out_classif={out_classif.shape}')