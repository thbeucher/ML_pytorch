import math
import torch


def orthogonal_loss(t):
  # eq (2) from https://arxiv.org/abs/2112.00384
  n = t.shape[0]
  normed_codes = torch.nn.functional.normalize(t, p=2, dim=-1)
  identity = torch.eye(n, device=t.device)
  cosine_sim = torch.einsum('i d, j d -> i j', normed_codes, normed_codes)
  return ((cosine_sim - identity) ** 2).sum() / (n ** 2)


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
    config['merging_type'] = config.get('merging_type', 'cat')  # cat or mul
    default_lat_vect_size = 100 + 100 if config['merging_type'] == 'cat' else 100
    config['latent_vector_size'] = config.get('latent_vector_size', default_lat_vect_size)
    super().__init__(config)
    self.embed = torch.nn.Embedding(config.get('n_classes', 10), config.get('embedding_size', 100))
  
  def forward(self, x, labels):
    embedding = self.embed(labels).unsqueeze(-1).unsqueeze(-1)
    inp = torch.cat([x, embedding], dim=1) if self.config['merging_type'] == 'cat' else torch.mul(x, embedding)
    return self.network(inp)


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
  # H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
  BASE_CONFIG = {'body_config': [
    # 1*28*28 -> 32*26*26
    {'type': torch.nn.Conv2d, 'params': {'in_channels': 1, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 32}},
    # 32*26*26 -> 32*24*24
    {'type': torch.nn.Conv2d, 'params': {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 32}},
    # 32*24*24 -> 32*12*12
    {'type': torch.nn.Conv2d, 'params': {'in_channels': 32, 'out_channels': 32, 'kernel_size': 5, 'stride': 2, 'padding': 2}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 32}},

    {'type': torch.nn.Dropout, 'params': {'p': 0.4}},
    # 32*12*12 -> 64*10*10
    {'type': torch.nn.Conv2d, 'params': {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 64}},
    # 64*10*10 -> 64*8*8
    {'type': torch.nn.Conv2d, 'params': {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 64}},
    # 64*8*8 -> 64*4*4
    {'type': torch.nn.Conv2d, 'params': {'in_channels': 64, 'out_channels': 64, 'kernel_size': 5, 'stride': 2, 'padding': 2}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 64}},

    {'type': torch.nn.Dropout, 'params': {'p': 0.4}},
    # 64*4*4 -> 128*1*1
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


# Modified version from https://nbviewer.org/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
class VectorQuantizer(torch.nn.Module):
  BASE_CONFIG = {'n_embeddings': 512, 'embedding_dim': 64, 'commitment_cost': 0.25, 'decay': 0.99, 'eps': 1e-5,
                 'ema': False, 'orthogonal_loss': 0.}
  def __init__(self, config):
    super().__init__()
    self.config = {**VectorQuantizer.BASE_CONFIG, **config}
    
    self.embedding = torch.nn.Embedding(self.config['n_embeddings'], self.config['embedding_dim'])

    if self.config['ema']:
      self.embedding.weight.data.normal_()
      # parameters in register_buffer appeared in state_dict but not in model.parameters()
      # so the optimizer will not update them
      self.register_buffer('ema_cluster_size', torch.zeros(self.config['n_embeddings']))
      self.ema_w = torch.nn.Parameter(torch.Tensor(self.config['n_embeddings'], self.config['embedding_dim']))
      self.ema_w.data.normal_()
    else:
      self.embedding.weight.data.uniform_(-1/self.config['n_embeddings'], 1/self.config['n_embeddings'])
    

  def forward(self, inputs):
    # convert inputs from BCHW -> BHWC
    inputs = inputs.permute(0, 2, 3, 1).contiguous()
    input_shape = inputs.shape
    
    # Flatten input
    flat_input = inputs.view(-1, self.config['embedding_dim'])
    
    # Calculate distances
    distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                + torch.sum(self.embedding.weight**2, dim=1)
                - 2 * torch.matmul(flat_input, self.embedding.weight.T))
        
    # Encoding
    encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # [B*H*W, 1]
    encodings = torch.zeros(encoding_indices.shape[0], self.config['n_embeddings'], device=inputs.device)
    encodings.scatter_(1, encoding_indices, 1)  # [B*H*W, n_embeddings]
    
    # Quantize and unflatten
    quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

    # Use EMA to update the embedding vectors
    if self.config['ema'] and self.training:
      self.ema_cluster_size = self.ema_cluster_size * self.config['decay'] + (1 - self.config['decay']) * torch.sum(encodings, 0)
      
      # Laplace smoothing of the cluster size
      n = torch.sum(self.ema_cluster_size.data)
      self.ema_cluster_size = ((self.ema_cluster_size + self.config['eps']) 
                                / (n + self.config['n_embeddings'] * self.config['eps']) * n)
      
      dw = torch.matmul(encodings.T, flat_input)
      self.ema_w = torch.nn.Parameter(self.ema_w * self.config['decay'] + (1 - self.config['decay']) * dw)
      
      self.embedding.weight = torch.nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))
    
    # Loss
    e_latent_loss = torch.nn.functional.mse_loss(quantized.detach(), inputs)
    if self.config['ema']:
      loss = self.config['commitment_cost'] * e_latent_loss
    else:
      q_latent_loss = torch.nn.functional.mse_loss(quantized, inputs.detach())
      loss = q_latent_loss + self.config['commitment_cost'] * e_latent_loss
    
    if self.config['orthogonal_loss'] > 0:
      loss += orthogonal_loss(self.embedding.weight) * self.config['orthogonal_loss']
    
    quantized = inputs + (quantized - inputs).detach()
    avg_probs = torch.mean(encodings, dim=0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    
    # convert quantized from BHWC -> BCHW
    return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerGSSoft(torch.nn.Module):
  BASE_CONFIG = {'n_embeddings': 128, 'embedding_dim': 8, 'latent_dim': 8,
                 'commitment_cost': 0.25, 'decay': 0.99, 'eps': 1e-5, 'ema': False}
  def __init__(self, config):
    super().__init__()
    self.config = {**VectorQuantizerGSSoft.BASE_CONFIG, **config}

    self.embedding = torch.nn.Parameter(torch.Tensor(self.config['latent_dim'],
                                                     self.config['n_embeddings'],
                                                     self.config['embedding_dim']))
    torch.nn.init.uniform_(self.embedding, -1/self.config['n_embeddings'], 1/self.config['n_embeddings'])

  def forward(self, x):
    B, C, H, W = x.size()
    N, M, D = self.embedding.size()
    assert C == N * D

    x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)
    x_flat = x.reshape(N, -1, D)

    distances = torch.baddbmm(torch.sum(self.embedding ** 2, dim=2).unsqueeze(1) +
                              torch.sum(x_flat ** 2, dim=2, keepdim=True),
                              x_flat, self.embedding.transpose(1, 2),
                              alpha=-2.0, beta=1.0)
    distances = distances.view(N, B, H, W, M)

    dist = torch.distributions.RelaxedOneHotCategorical(0.2, logits=-distances)
    if self.training:
      samples = dist.rsample().view(N, -1, M)
    else:
      samples = torch.argmax(dist.probs, dim=-1)
      samples = torch.nn.functional.one_hot(samples, M).float()
      samples = samples.view(N, -1, M)

    quantized = torch.bmm(samples, self.embedding)
    quantized = quantized.view_as(x)

    KL = dist.probs * (dist.logits + math.log(M))
    KL[(dist.probs == 0).expand_as(KL)] = 0
    KL = KL.sum(dim=(0, 2, 3, 4)).mean()

    avg_probs = torch.mean(samples, dim=1)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))

    return KL, quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W), perplexity.sum(), samples


class ResidualCNN(torch.nn.Module):
  BASE_CONFIG = {'layers_config': [
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},
    {'type': torch.nn.Conv2d,
     'params': {'in_channels': 128, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},
    {'type': torch.nn.Conv2d,
     'params': {'in_channels': 32, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'bias': False}}]}
  def __init__(self, config):
    super().__init__()
    self.config = {**ResidualCNN.BASE_CONFIG, **config}
    self.network = sequential_constructor(self.config['layers_config'])
  
  def forward(self, x):
    return x + self.network(x)


class VQVAEEncoder(torch.nn.Module):
  # H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
  BASE_CONFIG = {'layers_config': [
    # 1*28*28 -> 64*14*14
    {'type': torch.nn.Conv2d, 'params': {'in_channels': 1, 'out_channels': 64, 'kernel_size': 4, 'stride': 2, 'padding': 1}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},
    # 64*14*14 -> 128*7*7
    {'type': torch.nn.Conv2d, 'params': {'in_channels': 64, 'out_channels': 128, 'kernel_size': 4, 'stride': 2, 'padding': 1}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},
    # 128*7*7 -> 128*7*7
    {'type': torch.nn.Conv2d, 'params': {'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1}},
    # 2 Residual blocks
    {'type': ResidualCNN, 'params': {'config': {}}},
    {'type': ResidualCNN, 'params': {'config': {}}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}}]}
  def __init__(self, config):
    super().__init__()
    VQVAEEncoder.BASE_CONFIG['layers_config'][0]['params']['in_channels'] = config.get('n_channels', 1)
    self.config = {**VQVAEEncoder.BASE_CONFIG, **config}
    self.network = sequential_constructor(self.config['layers_config'])
  
  def forward(self, x):
    return self.network(x)


class VQVAEDecoder(torch.nn.Module):
  # H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
  BASE_CONFIG = {'layers_config': [
    # 64*7*7 -> 128*7*7
    {'type': torch.nn.Conv2d, 'params': {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1}},
    # 2 Residual blocks
    {'type': ResidualCNN, 'params': {'config': {}}},
    {'type': ResidualCNN, 'params': {'config': {}}},
    # 128*7*7 -> 64*14*14
    {'type': torch.nn.ConvTranspose2d, 'params': {'in_channels': 128, 'out_channels': 64, 'kernel_size': 4, 'stride': 2, 'padding': 1}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},
    # 64*14*14 -> 3*28*28
    {'type': torch.nn.ConvTranspose2d, 'params': {'in_channels': 64, 'out_channels': 1, 'kernel_size': 4, 'stride': 2, 'padding': 1}}]}
  def __init__(self, config):
    super().__init__()
    VQVAEDecoder.BASE_CONFIG['layers_config'][0]['params']['in_channels'] = config.get('embedding_dim', 64)
    VQVAEDecoder.BASE_CONFIG['layers_config'][-1]['params']['out_channels'] = config.get('n_channels', 1)
    self.config = {**VQVAEDecoder.BASE_CONFIG, **config}
    self.network = sequential_constructor(self.config['layers_config'])
  
  def forward(self, x):
    return self.network(x)


class VQVAEModel(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.encoder = VQVAEEncoder(config.get('encoder_config', {}))
    self.pre_vq_conv = torch.nn.Conv2d(**config.get('pre_vq_conv_config', {'in_channels': 128, 'out_channels': 64,
                                                                           'kernel_size': 1, 'stride': 1, 'padding': 0}))
    self.vq = VectorQuantizer(config.get('vq_config', {}))
    self.decoder = VQVAEDecoder(config.get('decoder_config', {}))
  
  def forward(self, x):
    z = self.encoder(x)  # [128, 1, 28, 28] -> [128, 128, 7, 7]
    z = self.pre_vq_conv(z)  # -> [128, 64, 7, 7]
    loss, quantized, perplexity, encodings = self.vq(z)
    x_rec = self.decoder(quantized)  # [128, 64, 7, 7] -> [128, 1, 28, 28]
    return loss, x_rec, perplexity, encodings, quantized


class VQVAE2Residual(torch.nn.Module):
  BASE_CONVS_CONFIG = [[[128, 32, 3, 1, 1], torch.nn.ReLU],
                       [[32, 128, 1, 1, 0], torch.nn.ReLU]]
  def __init__(self, config={}, bias=False):
    '''
    config : dict like {'convs_config': [[conv_config=[in_chan, out_chan, kernel, stride, padding], activation_fn], ...]}
    '''
    super().__init__()
    blocks = []
    for conv_conf, act_fn in config.get('convs_config', VQVAE2Residual.BASE_CONVS_CONFIG):
      blocks.append(torch.nn.Conv2d(*conv_conf, bias=bias))
      if act_fn is not None:
        blocks.append(act_fn())
    self.network = torch.nn.Sequential(*blocks)
  
  def forward(self, x):  # [B, C, H, W]
    return self.network(x)


class VQVAE2Encoder(torch.nn.Module):
  # H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
  BASE_CONVS_CONFIG = [[[1, 64, 4, 2, 1], torch.nn.ReLU],
                       [[64, 128, 4, 2, 1], torch.nn.ReLU],
                       [[128, 128, 3, 1, 1], torch.nn.ReLU]]
  BASE_RESCONVS_CONFIG = [{}, {}]
  def __init__(self, config={}):
    '''
    config : dict like {'down_convs_config': [[[in_chan, out_chan, kernel, stride, padding], activation_fn], ...],
                        'residual_convs_config': [{'convs_config': [[i, o, k, s, p], act_fn]}, ...]}
    '''
    super().__init__()
    blocks = []
    for dconv_conf, act_fn in config.get('down_convs_config', VQVAE2Encoder.BASE_CONVS_CONFIG):
      blocks.append(torch.nn.Conv2d(*dconv_conf))
      if config.get('batch_norm', False):
        blocks.append(torch.nn.BatchNorm2d(dconv_conf[1]))
      if act_fn is not None:
        blocks.append(act_fn())

    for res_conv_conf in config.get('residual_convs_config', VQVAE2Encoder.BASE_RESCONVS_CONFIG):
      blocks.append(VQVAE2Residual(res_conv_conf))
    
    self.network = torch.nn.Sequential(*blocks)
  
  def forward(self, x):
    return self.network(x)


class VQVAE2Decoder(torch.nn.Module):
  BASE_CONVS_CONFIG = [[64, 128, 3, 1, 1]]
  BASE_RESCONVS_CONFIG = [{}, {}]
  BASE_TRANSCONVS_CONFIG = [[[128, 64, 4, 2, 1], torch.nn.ReLU],
                            [[64, 1, 4, 2, 1], None]]
  def __init__(self, config={}):
    super().__init__()
    blocks = []
    for conv_conf in config.get('convs_config', VQVAE2Decoder.BASE_CONVS_CONFIG):
      blocks.append(torch.nn.Conv2d(*conv_conf))

    for res_conv_conf in config.get('residual_convs_config', VQVAE2Decoder.BASE_RESCONVS_CONFIG):
      blocks.append(VQVAE2Residual(res_conv_conf))
    
    for tconv_conf, act_fn in config.get('transpose_convs_config', VQVAE2Decoder.BASE_TRANSCONVS_CONFIG):
      blocks.append(torch.nn.ConvTranspose2d(*tconv_conf))
      if config.get('batch_norm', False):
        blocks.append(torch.nn.BatchNorm2d(tconv_conf[1]))
      if act_fn is not None:
        blocks.append(act_fn())
    
    self.network = torch.nn.Sequential(*blocks)
  
  def forward(self, x):
    return self.network(x)


class VQVAE2(torch.nn.Module):
  # H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
  def __init__(self, config={}):
    super().__init__()
    self.encoder_bottom = VQVAE2Encoder(config.get('encoder_bottom_config', {}))
    self.encoder_top = VQVAE2Encoder(config.get('encoder_top_config', {'down_convs_config': [[[128, 64, 3, 2, 1], torch.nn.ReLU],
                                                                                             [[64, 128, 3, 1, 1], torch.nn.ReLU]]}))
    self.pre_vq_top = torch.nn.Conv2d(*config.get('pre_vq_top_config', [128, 64, 1, 1, 0]))
    self.pre_vq_bottom = torch.nn.Conv2d(*config.get('pre_vq_bottom_config', [192, 64, 1, 1, 0]))
    self.vq_top = VectorQuantizer(config.get('vq_top_config', {}))
    self.vq_bottom = VectorQuantizer(config.get('vq_bottom_config', {}))
    self.decoder_top = VQVAE2Decoder(config.get('decoder_top_config', {'transpose_convs_config': [[[128, 64, 3, 2, 1], None]]}))
    self.decoder_bottom = VQVAE2Decoder(config.get('decoder_bottom_config', {'convs_config': [[128, 128, 3, 1, 1]]}))
    self.upsample_top = torch.nn.ConvTranspose2d(*config.get('upsample_top_config', [64, 64, 3, 2, 1]))
  
  def forward(self, x):
    z_bottom = self.encoder_bottom(x)  # [B, C=1, H=28, W=28] -> [B, 128, 7, 7]
    z_top = self.encoder_top(z_bottom)  # [B, 128, 7, 7] -> [B, 128, 4, 4]
    loss_top, quantized_top, pt, et = self.vq_top(self.pre_vq_top(z_top))  # [B, 64, 4, 4]
    rec_top = self.decoder_top(quantized_top)  # [B, 64, 7, 7]
    z_bottom = torch.cat([rec_top, z_bottom], 1)  # [B, 192, 7, 7]
    loss_bottom, quantized_bottom, pb, eb = self.vq_bottom(self.pre_vq_bottom(z_bottom))  # [B, 64, 7, 7]
    upsampled_quantized_top = self.upsample_top(quantized_top)  # [B, 64, 7, 7]
    quantized = torch.cat([upsampled_quantized_top, quantized_bottom], 1)  # [B, 128, 7, 7]
    x_rec = self.decoder_bottom(quantized)  # [B, 1, 28, 28]
    return loss_top + loss_bottom, x_rec, pt + pb, None, quantized


class VQVAETransformerModel(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.encoder = VQVAEEncoder(config.get('encoder_config', {}))
    self.pre_vq_conv = torch.nn.Conv2d(**config.get('pre_vq_conv_config', {'in_channels': 128, 'out_channels': 64,
                                                                           'kernel_size': 1, 'stride': 1, 'padding': 0}))
    self.pre_q_t = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(**config.get('pre_q_t', {'d_model': 64,
                                                                                                         'nhead': 4,
                                                                                                         'dim_feedforward': 256})),
                                               config.get('pre_q_t_n_layers', 2))
    self.vq = VectorQuantizer(config.get('vq_config', {}))
    self.post_q_t = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(**config.get('pre_q_t', {'d_model': 64,
                                                                                                          'nhead': 4,
                                                                                                          'dim_feedforward': 256})),
                                                config.get('post_q_t_n_layers', 2))
    self.decoder = VQVAEDecoder(config.get('decoder_config', {}))
  
  def forward(self, x):
    z = self.encoder(x)  # [128, 1, 28, 28] -> [128, 128, 7, 7]
    z = self.pre_vq_conv(z)  # -> [128, 64, 7, 7]
    z = self.pre_q_t(z)
    loss, quantized, perplexity, encodings = self.vq(z)
    quantized = self.post_q_t(quantized)
    x_rec = self.decoder(quantized)  # [128, 64, 7, 7] -> [128, 1, 28, 28]
    return loss, x_rec, perplexity, encodings, quantized


class MLPHead(torch.nn.Module):
  BASE_CONFIG = {'layers_config': [{'type': torch.nn.Flatten, 'params': {}},  #          3136
                                   {'type': torch.nn.Linear,  'params': {'in_features': 64*7*7, 'out_features': 1024}},
                                   {'type': torch.nn.ReLU,    'params': {'inplace': True}},
                                   {'type': torch.nn.Linear,  'params': {'in_features': 1024, 'out_features': 10}}]}
  def __init__(self, config):
    MLPHead.BASE_CONFIG['layers_config'][1]['params']['in_features'] = config.get('in_features', 3136)
    MLPHead.BASE_CONFIG['layers_config'][-1]['params']['out_features'] = config.get('n_classes', 10)
    super().__init__()
    self.config = {**MLPHead.BASE_CONFIG, **config}
    self.network = sequential_constructor(self.config['layers_config'])
  
  def forward(self, x):
    return self.network(x)


class TransformerHead(torch.nn.Module):
  BASE_CONFIG = {'layers_config': [{'type': torch.nn.Flatten,                 'params': {'start_dim': -2}},
                                   {'type': torch.nn.Linear,                  'params': {'in_features': 7*7, 'out_features': 32}},
                                   {'type': torch.nn.TransformerEncoderLayer, 'params': {'d_model': 32, 'nhead': 8,
                                                                                         'dim_feedforward': 2048, 'dropout': 0.1,
                                                                                         'batch_first': True}},
                                   {'type': torch.nn.Flatten,                 'params': {}},
                                   {'type': torch.nn.Linear,                  'params':{'in_features': 2048, 'out_features': 10}}]}
  def __init__(self, config):
    TransformerHead.BASE_CONFIG['layers_config'][-1]['params']['out_features'] = config.get('n_classes', 10)
    super().__init__()
    self.config = {**TransformerHead.BASE_CONFIG, **config}
    self.network = sequential_constructor(self.config['layers_config'])
  
  def forward(self, x):
    return self.network(x)


class VQVAEClassifierModel(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.encoder = VQVAEEncoder(config.get('encoder_config', {}))
    self.pre_vq_conv = torch.nn.Conv2d(**config.get('pre_vq_conv_config', {'in_channels': 128, 'out_channels': 64,
                                                                           'kernel_size': 1, 'stride': 1, 'padding': 0}))
    self.vq = config.get('vq_type', VectorQuantizer)(config.get('vq_config', {}))
    self.decoder = VQVAEDecoder(config.get('decoder_config', {}))
    self.classifier = config.get('classifier_type', MLPHead)(config.get('classifier_config', {}))
  
  def forward(self, x):
    z = self.encoder(x)  # [128, 1, 28, 28] -> [128, 128, 7, 7]
    z = self.pre_vq_conv(z)  # -> [128, 64, 7, 7]
    loss, quantized, perplexity, encodings = self.vq(z)
    x_rec = self.decoder(quantized)  # [128, 64, 7, 7] -> [128, 1, 28, 28]
    out = self.classifier(quantized)
    return loss, x_rec, perplexity, encodings, quantized, out


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
            {'type': MNISTClassifier,             'args': {}},
            {'type': VQVAEEncoder,                'args': {}},
            {'type': VQVAEDecoder,                'args': {}},
            {'type': MLPHead,                     'args': {}}]
  
  inputs = [ {'type': torch.randn, 'args': (4, 100)},
             {'type': torch.randn, 'args': (4, 28*28)},
             {'type': torch.randn, 'args': (4, 100, 1, 1)},
             {'type': torch.randn, 'args': (4, 1, 28, 28)},
             {'type': torch.randn, 'args': (4, 1, 28, 28)},
            [{'type': torch.randn, 'args': (4, 1, 28, 28)}, {'type': torch.randint, 'args': [0, 10, (4,)]}],
            [{'type': torch.randn, 'args': (4, 100, 1, 1)}, {'type': torch.randint, 'args': [0, 10, (4,)]}],
             {'type': torch.randn, 'args': (4, 1, 28, 28)},
             {'type': torch.randn, 'args': (4, 1, 28, 28)},
             {'type': torch.randn, 'args': (4, 1, 28, 28)},
             {'type': torch.randn, 'args': (4, 64, 7, 7)},
             {'type': torch.randn, 'args': (4, 64, 7, 7)}]
  
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