import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import cnn_layers as cl


def get_embedding_block(n_embeddings, embedding_dim):
  return nn.Sequential(nn.Embedding(n_embeddings, embedding_dim),
                       nn.Linear(embedding_dim, embedding_dim),
                       nn.SiLU(True))


def get_linear_net(input_dim, hidden_dim, output_dim):
  return nn.Sequential(nn.Linear(input_dim, hidden_dim),
                       nn.ReLU(True),
                       nn.Linear(hidden_dim, hidden_dim),
                       nn.ReLU(True),
                       nn.Linear(hidden_dim, output_dim))


class CNNAE(nn.Module):
  '''AutoEncoder that take an image and try to reconstruct it'''
  CONFIG = {'skip_connection': True,
            'linear_bottleneck': False,
            'add_noise_bottleneck': False,
            'add_noise_encoder': False,
            'add_enc_attn': False,
            'add_dec_attn': True,
            'noise_prob': 0.5,
            'noise_std': 0.1,
            'latent_dim': 128,
            'encoder_archi': 'CNNEncoder'}
  def __init__(self, config={}):
    super().__init__()
    self.config = {**CNNAE.CONFIG, **config}

    self.down = cl.CNN_LAYERS[self.config['encoder_archi']](**self.config)

    if self.config['linear_bottleneck']:
      self.embedder = nn.Linear(256*4*4, self.config['latent_dim'])      # H=32,W=32 for cifar10
      self.fc_dec = nn.Linear(self.config['latent_dim'], 256*4*4)

    self.up = cl.CNNDecoder(add_attn=self.config['add_dec_attn'])

    if self.config['add_noise_encoder'] or self.config['add_noise_bottleneck']:
      self.noise_layer = cl.NoiseLayer(p=self.config['noise_prob'], std=self.config['noise_std'])
  
  def forward(self, x, return_latent=False):
    d1, d2, d3 = self.down(x)

    if self.config['linear_bottleneck']:
      d3 = self.fc_dec(self.embedder(d3.flatten(1))).view(d3.shape)
    
    if self.config['add_noise_bottleneck']:
      d3 = self.noise_layer(d3)

    u3 = self.up(d3,
                 d2 if self.config['skip_connection'] else None,
                 d1 if self.config['skip_connection'] else None)

    if return_latent:
      return u3, d3
    return u3


class TimeEmbedding(nn.Module):
  """
  Classical sinusoidal time embedding (like in diffusion/transformers).
  """
  def __init__(self, dim, max_positions=10000):
    super().__init__()
    self.dim = dim
    self.max_positions = max_positions

  def forward(self, t):
    """
    t: (batch,) in [0,1]
    returns: (batch, dim)
    """
    t = t * self.max_positions
    half = self.dim // 2

    freqs = torch.exp(
        torch.arange(half, device=t.device) * -(math.log(self.max_positions) / (half - 1))
    )
    emb = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if self.dim % 2 == 1:
        emb = F.pad(emb, (0,1))

    return emb


class WorldModelFlowUnet(nn.Module):
  def __init__(self, img_chan=3, time_dim=64, x_start=False,
               add_action=True, n_actions=5, action_dim=8,
               add_is=False, is_n_values=36, is_dim=32,  # is = internal_state
               add_ds=False, ds_n_values=2, ds_dim=64,  # ds = done_signal
               ):
    super().__init__()
    self.time_emb = nn.Sequential(TimeEmbedding(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU())
    condition_dim = time_dim 

    self.add_action = add_action
    if add_action:
      self.action_emb = nn.Sequential(nn.Embedding(n_actions, action_dim), nn.Linear(action_dim, action_dim), nn.SiLU())
      condition_dim += action_dim

    self.add_is = add_is
    if add_is:
      self.is_emb = nn.Sequential(nn.Embedding(is_n_values, is_dim), nn.Linear(is_dim, is_dim), nn.SiLU())
      condition_dim += 2 * is_dim
    
    self.add_ds = add_ds
    if add_ds:
      self.ds_emb = nn.Sequential(nn.Embedding(ds_n_values, ds_dim), nn.Linear(ds_dim, ds_dim), nn.SiLU())
      condition_dim += ds_dim

    self.x_start = x_start
    start_dim = 2*img_chan if x_start else img_chan
    self.init_conv = nn.Conv2d(start_dim + condition_dim, 64, 3, 1, 1)

    self.down1 = cl.EnhancedResidualFullBlock(64, 128, pooling=True, cond_emb=condition_dim)
    self.down2 = cl.EnhancedResidualFullBlock(128, 256, pooling=True, cond_emb=condition_dim)
    self.down3 = cl.EnhancedResidualFullBlock(256, 512, pooling=True, cond_emb=condition_dim, groups=1)

    self.up1 = cl.EnhancedResidualFullBlock(512, 256, upscaling=True, cond_emb=condition_dim, groups=1)
    self.up2 = cl.EnhancedResidualFullBlock(256, 128, upscaling=True, cond_emb=condition_dim)
    self.up3 = cl.EnhancedResidualFullBlock(128, 64, upscaling=True, cond_emb=condition_dim)
    
    self.final_conv = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.SiLU(),
                                    nn.Conv2d(32, img_chan, 3, 1, 1))
  
  def forward(self, x, t, condition={}):  # image, time, condition
    B, C, H, W = x.shape

    t_emb = self.time_emb(t.squeeze(-1))         # -> [B, time_dim]
    all_embed = [t_emb]

    if self.add_action:
      a_emb = self.action_emb(condition['action'].squeeze(-1))  # -> [B, action_dim]
      all_embed += [a_emb]
    
    if self.add_is:
      is_emb = self.is_emb(condition['internal_state'])  # [B, n_is=2] -> [B, n_is=2, is_dim=32]
      all_embed += [is_emb.flatten(1)]  # [B, 2*32]
    
    if self.add_ds:
      ds_emb = self.ds_emb(condition['done_signal'].squeeze(-1))  # [B, ds_dim=64]
      all_embed += [ds_emb]

    all_embed = torch.cat(all_embed, dim=1)  # -> [B, 104]

    all_cond = [all_embed[:, :, None, None].expand(-1, -1, H, W)]
    if self.x_start:
      all_cond = [condition['x_cond']] + all_cond
    
    x = torch.cat([x] + all_cond, dim=1)

    x = self.init_conv(x)                        # -> [B, 64, H, W]

    d1 = self.down1(x, c=all_embed)              # -> [B, 128, H/2, W/2]
    d2 = self.down2(d1, c=all_embed)             # -> [B, 256, H/4, W/4]
    d3 = self.down3(d2, c=all_embed)             # -> [B, 512, H/8, W/8]

    u1 = self.up1(d3, c=all_embed)               # -> [B, 256, H/4, W/4]
    u2 = self.up2(u1 + d2, c=all_embed)          # -> [B, 128, H/2, W/2]
    u3 = self.up3(u2 + d1, c=all_embed)          # -> [B, 64, H, W]

    velocity = self.final_conv(u3)               # -> [B, img_chan, H, W]
    return velocity


class GoalPolicy(nn.Module):
  def __init__(self, a_dim=5, is_dim=32, is_n_values=37, hidden=256):
    super().__init__()
    self.is_emb = get_embedding_block(is_n_values, is_dim)
    self.net = get_linear_net(2 * is_dim * 3, hidden, a_dim)

  def forward(self, is_c, is_g):  # [B, 2], [B, 2]
    is_c_emb = self.is_emb(is_c).flatten(1)  # -> [B, 2, 32] -> [B, 64]
    is_g_emb = self.is_emb(is_g).flatten(1)  # -> [B, 2, 32] -> [B, 64]
    x = torch.cat([is_c_emb, is_g_emb, is_g_emb - is_c_emb], dim=-1)  # [B, 192]
    return self.net(x)  # [B, 5]
  
  def sample(self, states, goals):
    # ---- Policy forward pass π(a | s, g) ----
    a_logits = self.forward(states, goals)
    dist = torch.distributions.Categorical(logits=a_logits)
    # Sample action (on-policy)
    actions = dist.sample()
    # Get log probs for RL learning
    log_probs = dist.log_prob(actions)
    return actions, log_probs, dist.entropy()


class GoalValue(nn.Module):
  def __init__(self, is_dim=32, is_n_values=37, hidden=256):
    super().__init__()
    self.is_emb = get_embedding_block(is_n_values, is_dim)
    self.net = get_linear_net(2 * is_dim * 2, hidden, 1)

  def forward(self, is_c, is_g):
    is_c_emb = self.is_emb(is_c).flatten(1)  # -> [B, 2, 32] -> [B, 64]
    is_g_emb = self.is_emb(is_g).flatten(1)  # -> [B, 2, 32] -> [B, 64]
    return self.net(torch.cat([is_c_emb, is_g_emb], dim=-1)).squeeze(-1)  # [B, 1] -> [B]


class WGANGP(nn.Module):
  CONFIG = {
    'autoencoder_config': {'skip_connection': True,
                           'linear_bottleneck': False,
                           'add_noise_bottleneck': True,
                           'add_noise_encoder': False,
                           'noise_prob': 1.0,
                           'add_enc_attn': False,
                           'add_dec_attn': True}
  }
  def __init__(self, config={}):
    super().__init__()
    self.config = {**WGANGP.CONFIG, **config}

    self.auto_encoder = CNNAE(self.config['autoencoder_config'])
    self.critic = cl.Critic()