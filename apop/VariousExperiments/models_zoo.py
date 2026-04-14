# This file contains a collection of neural network models for various experiments.
#
# Available classes:
# - SequentialBottleneck: An encoder-decoder model with a sequential bottleneck.
# - CNNAE: A convolutional autoencoder for image reconstruction.
# - CNNAENSP: An autoencoder that also predicts the next state in the embedding space.
# - TimeEmbedding: A module for sinusoidal time embedding.
# - WorldModelFlowUnet: A U-Net based world model for flow prediction.
# - GoalPolicy: A policy network that outputs an action given a current and goal state.
# - GoalValue: A value network that estimates the value of a state-goal pair.
# - WGANGP: A Wasserstein GAN with Gradient Penalty.
# - ISPredictor: Predicts an internal state from an image.
# - ISPredictorFromPatchIndex: Predicts an internal state from a patch index.
# - EpisodeCompressor: Compresses a sequence of embeddings into a single embedding.
# - MemoryBankFuturStatePredictor: Predicts a future state using a memory bank of past episodes.
# - AlteredPredictor: A Vision Transformer (ViT) that predicts which image patches are likely to change.
# - AlterationPredictor: A Transformer model that predicts the exact pixel-level changes for patches identified as changing.
# - ObjectPredictor: A Vision Transformer (ViT) that predicts which image patch contains a specified object.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence

import cnn_layers as cl
from vision_transformer.vit import *
from helpers_zoo import random_patch_mask


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


class SequentialBottleneck(nn.Module):
  CONFIG = {
    'add_dec_attn':  True,
    'embedding_dim': 128,
    'n_embeddings':  1,
  }
  def __init__(self, config={}):
    super().__init__()
    self.config = {**SequentialBottleneck.CONFIG, **config}
    self.encoder = cl.BigCNNEncoder()
    # --- CONV VERSION ---
    self.embedder = nn.Conv2d(256, self.config['embedding_dim'] * self.config['n_embeddings'], 4)
    self.dec_embedder = nn.ConvTranspose2d(self.config['embedding_dim'], 256, 4)
    # --------------------
    # --- LINEAR VERSION ---
    # embeddings = [nn.Linear(256*4*4, self.config['embedding_dim']) for _ in range(self.config['n_embeddings'])]
    # self.embedders = nn.ModuleList(embeddings)
    # self.dec_embedder = nn.Linear(self.config['embedding_dim'], 256*4*4)
    # ----------------------
    self.decoder = cl.CNNDecoder(add_attn=self.config['add_dec_attn'])
  
  def forward(self, x, return_latent=False):
    d1, d2, d3 = self.encoder(x)
    # --- LINEAR VERSION ---
    # out = torch.zeros_like(x)
    # for embedder in self.embedders:
    #   emb = embedder(d3.flatten(1))
    #   dec_emb = self.dec_embedder(emb)
    #   out += self.decoder(dec_emb.view(d3.shape))
    # ----------------------
    # --- CONV VERSION ---
    embs = self.embedder(d3)  # [B, n_embeddings * embedding_dim, 1, 1]
    dec_embs = self.dec_embedder(embs.view(x.shape[0] * self.config['n_embeddings'],
                                           self.config['embedding_dim'], 1, 1))
    decoded = self.decoder(dec_embs)
    out = decoded.view(x.shape[0], self.config['n_embeddings'], 3, 32, 32).sum(dim=1)
    # --------------------
    if return_latent:
      return out, d3
    return out


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
    print(f'CNNAE: {self.config}')

    self.down = cl.CNN_LAYERS[self.config['encoder_archi']](**self.config)

    if self.config['linear_bottleneck']:
      self.embedder = nn.Linear(256*4*4, self.config['latent_dim'])      # H=32,W=32 for cifar10
      self.fc_dec = nn.Linear(self.config['latent_dim'], 256*4*4)

    self.up = cl.CNNDecoder(add_attn=self.config['add_dec_attn'])

    if self.config['add_noise_encoder'] or self.config['add_noise_bottleneck']:
      self.noise_layer = cl.NoiseLayer(p=self.config['noise_prob'], std=self.config['noise_std'])
  
  def forward(self, x, return_latent=False, return_all=False):
    d1, d2, d3 = self.down(x)

    latent = d3
    if self.config['linear_bottleneck']:
      latent = self.embedder(latent.flatten(1))
    
    if self.config['add_noise_bottleneck']:
      latent = self.noise_layer(latent)

    u3 = self.up(self.fc_dec(latent).view(d3.shape) if self.config['linear_bottleneck'] else latent,
                 d2 if self.config['skip_connection'] else None,
                 d1 if self.config['skip_connection'] else None)

    if return_all:
      return u3, (d1, d2, d3)
    if return_latent:
      return u3, latent
    return u3

  def decode(self, latent, d2=None, d1=None):
    if self.config['linear_bottleneck']:
      x = self.fc_dec(latent).view(latent.shape[0], 256, 4, 4)
    else:
      x = latent
    return self.up(x, d2, d1)


class CNNAENSP(nn.Module):
  '''AutoEncoder that take an image and reconstruct it and predict the next state in the embedding space'''
  CONFIG = {
    'ae_config': {'encoder_archi': 'BigCNNEncoder', 'skip_connection': True, 'linear_bottleneck': True,
                  'latent_dim': 128},
    'n_actions': 5, 'action_dim': 8, 'is1_n_values': 19, 'is2_n_values': 37, 'is_dim': 16,
    'nsp_hidden_dim': 256,
    }
  def __init__(self, config={}):
    super().__init__()
    self.config = {**CNNAENSP.CONFIG, **config}
    self.ae = CNNAE(self.config['ae_config'])
    self.action_emb = get_embedding_block(self.config['n_actions'], self.config['action_dim'])
    self.is_emb = get_embedding_block(max(self.config['is1_n_values'], self.config['is2_n_values']),
                                      self.config['is_dim'])
    self.nsp = get_linear_net(self.ae.config['latent_dim'] + self.config['action_dim'] + 2*self.config['is_dim'],
                              self.config['nsp_hidden_dim'], self.ae.config['latent_dim'])
  
  @torch.no_grad()
  def get_embedding(self, image):
    _, latent = self.ae(image, return_latent=True)
    return latent
  
  @torch.no_grad()
  def infer(self, image):
    return self.get_embedding(image)
  
  def forward(self, image, condition={}):
    rec, latent = self.ae(image, return_latent=True)
    action_emb = self.action_emb(condition.get('action', torch.zeros(image.shape[0],
                                                                     dtype=torch.long,
                                                                     device=image.device)).squeeze(-1))
    is_emb = self.is_emb(condition.get('internal_state', torch.zeros(image.shape[0],
                                                                     dtype=torch.long,
                                                                     device=image.device))).flatten(1)
    nsp_input = torch.cat([latent.flatten(1), action_emb, is_emb], dim=-1)
    pred_next_latent = self.nsp(nsp_input)
    return rec, pred_next_latent


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
               add_other_cond=False, other_cond_dim=128,  # any other embedding to conditioned the model
               ):
    super().__init__()
    print(f'WorldModelFlowUnet: {add_action=} | {add_is=} | {add_ds=} | {add_other_cond=}')
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
    
    self.add_other_cond = add_other_cond
    if add_other_cond:
      condition_dim += other_cond_dim

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
    
    if self.add_other_cond:
      all_embed += [condition['other']]

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


class ISPredictor(nn.Module):
  """
  Predicts an internal_state from:
    - image: [B, 3, 32, 32]

  Outputs:
    - internal_state logits for each internal dimension
  """
  def __init__(self, is1_n_values=19, is2_n_values=37, is_dim=32, hidden_dim=512, done_info=False,
               rnn_goal_prediction=False, rnn_hidden=256, num_layers=2, is_rnn=True,
               layer_norm_predictor=False):
    super().__init__()
    # --------------------------------------------------
    # Image encoder
    # --------------------------------------------------
    self.image_encoder = nn.Sequential(
      nn.Conv2d(3, 64, 3, stride=2, padding=1),   # [B, 64, 16, 16]
      nn.BatchNorm2d(64),
      nn.SiLU(),

      nn.Conv2d(64, 128, 3, stride=2, padding=1), # [B, 128, 8, 8]
      nn.BatchNorm2d(128),
      nn.SiLU(),

      nn.Conv2d(128, 256, 3, stride=2, padding=1),# [B, 256, 4, 4]
      nn.BatchNorm2d(256),
      nn.SiLU(),

      nn.Conv2d(256, hidden_dim, 3, padding=1),  # [B, 512, 4, 4]
      nn.BatchNorm2d(hidden_dim),
      nn.SiLU(),
    )

    self.pool = nn.AdaptiveAvgPool2d(1)  # → [B, hidden_dim]

    # --------------------------------------------------
    # LSTM for temporal goal prediction
    # --------------------------------------------------
    self.rnn_goal_predictor = None
    if rnn_goal_prediction:
      self.rnn_goal_predictor = nn.LSTM(
        input_size=hidden_dim + 2*is_dim,
        hidden_size=rnn_hidden,
        num_layers=num_layers,
        batch_first=True,
        bidirectional=True,
      )
      # --------------------------------------------------
      # Internal state embedding
      # --------------------------------------------------
      self.is_rnn = is_rnn
      if is_rnn:
        self.is_emb = get_embedding_block(
          n_embeddings=max(is1_n_values, is2_n_values),
          embedding_dim=is_dim,
        )

    # --------------------------------------------------
    # Goal prediction heads
    # --------------------------------------------------
    self.is1_head = nn.Linear(hidden_dim, is1_n_values)
    self.is2_head = nn.Linear(hidden_dim, is2_n_values)

    self.layer_norm_predictor = layer_norm_predictor
    if layer_norm_predictor:
      self.layer_norm_is1 = nn.LayerNorm(is1_n_values)
      self.layer_norm_is2 = nn.LayerNorm(is2_n_values)

    self.done_info = done_info
    if self.done_info:
      self.goal1_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2),
                                      nn.SiLU(True),
                                      nn.Dropout(p=0.7),
                                      nn.Linear(hidden_dim//2, hidden_dim))
      self.goal2_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2),
                                      nn.SiLU(True),
                                      nn.Dropout(p=0.7),
                                      nn.Linear(hidden_dim//2, hidden_dim))

  def forward(self, image, condition={}):
    """image: [B, 3, 32, 32] or [B, T, 3, 32, 32]"""
    if self.rnn_goal_predictor:
      assert len(image.shape) == 5, 'ImageISPredictor in RNN mode but no temporal dimension found in given image'
      B, T, C, H, W = image.shape
      image = image.view(B*T, C, H, W)

    img_feat = self.image_encoder(image)
    img_feat = self.pool(img_feat).flatten(1)         # [B, hidden_dim]

    if self.rnn_goal_predictor:
      img_feat = img_feat.view(B, T, -1)  # Restore time dimension

      if self.is_rnn:
        is_emb = self.is_emb(condition['internal_state']).flatten(-2)  # [B, T, 2*32]
        img_feat = torch.cat([img_feat, is_emb], dim=-1)

      rnn_out, _ = self.rnn_goal_predictor(img_feat)  # [B, T, rnn_hidden]
      img_feat = rnn_out
      # img_feat = rnn_out[:, -1, :]                    # [B, rnn_hidden]

    is1_logits = self.is1_head(img_feat)             # [B, is1_n_values]
    is2_logits = self.is2_head(img_feat)             # [B, is2_n_values]

    if self.layer_norm_predictor:
      is1_logits = self.layer_norm_is1(is1_logits)
      is2_logits = self.layer_norm_is2(is2_logits)

    if self.done_info:
      isg1_logits = self.goal1_head(img_feat)
      isg2_logits = self.goal2_head(img_feat)
      return is1_logits, is2_logits, isg1_logits, isg2_logits

    return is1_logits, is2_logits


class ISPredictorFromPatchIndex(nn.Module):
  """
  Predicts an internal_state from:
    - patch_index: [B]

  Outputs:
    - internal_state logits for each internal dimension
  """
  def __init__(self, patch_index_n_values=256, is1_n_values=19, is2_n_values=37, patch_index_dim=32,
               hidden_dim=256, net_hidden_dim=128):
    super().__init__()
    self.patch_index_emb = get_embedding_block(patch_index_n_values, patch_index_dim)

    self.net = get_linear_net(patch_index_dim, net_hidden_dim, hidden_dim)

    self.is1_head = nn.Linear(hidden_dim, is1_n_values)
    self.is2_head = nn.Linear(hidden_dim, is2_n_values)

  def forward(self, patch_index):
    patch_index_emb = self.patch_index_emb(patch_index)

    x = self.net(patch_index_emb)

    is1_logits = self.is1_head(x)
    is2_logits = self.is2_head(x)

    return is1_logits, is2_logits


class EpisodeCompressor(nn.Module):
  def __init__(self, embed_dim):
    """
    Compresses a sequence of embeddings (an episode) into a single embedding.
    """
    super(EpisodeCompressor, self).__init__()
    # GRU is excellent for temporal sequence compression.
    # It takes sequences of embed_dim and outputs a final hidden state of embed_dim.
    self.gru = nn.GRU(
      input_size=embed_dim, 
      hidden_size=embed_dim, 
      batch_first=True
    )
  
  def forward(self, episodes, episode_lengths):
    """
    Args:
        episodes: Tensor of shape [Flat_Batch, Max_Episode_Len, Embed_Dim]
        episode_lengths: Tensor of shape [Flat_Batch] containing actual integer lengths.
    Returns:
        compressed_episodes: Tensor of shape [Flat_Batch, Embed_Dim]
    """
    # 1. Pack the padded sequence
    # enforce_sorted=False allows us to pass lengths in any order
    packed_episodes = pack_padded_sequence(
      episodes, 
      episode_lengths.cpu(), # Lengths must be on CPU for packing
      batch_first=True, 
      enforce_sorted=False
    )
    
    # 2. Pass through GRU
    # The GRU automatically retrieves the state at the *true* end of each sequence
    _, final_hidden_state = self.gru(packed_episodes)
    
    # 3. Squeeze the layer dimension: [1, Flat_Batch, Embed_Dim] -> [Flat_Batch, Embed_Dim]
    compressed_episodes = final_hidden_state.squeeze(0)
    
    return compressed_episodes


class MemoryBankFuturStatePredictor(nn.Module):
  def __init__(self, embed_dim, num_heads, num_layers, hidden_dim, max_memory_size):
    """
    Args:
        embed_dim: Dimension of the state and goal embeddings.
        num_heads: Number of attention heads in the Transformer.
        num_layers: Number of Transformer encoder layers.
        hidden_dim: Hidden dimension size for the feedforward network (MLP).
        max_memory_size: The bounded maximum size of the memory buffer.
    """
    super(MemoryBankFuturStatePredictor, self).__init__()
    
    self.embed_dim = embed_dim
    self.max_memory_size = max_memory_size

    self.episode_compressor = EpisodeCompressor(embed_dim)
    
    # Learnable embeddings to distinguish the current state from memory tokens
    self.state_token_type = nn.Parameter(torch.randn(1, 1, embed_dim))
    self.memory_token_type = nn.Parameter(torch.randn(1, 1, embed_dim))
    
    # Transformer Encoder
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=embed_dim, 
      nhead=num_heads, 
      dim_feedforward=hidden_dim,
      batch_first=True, # Ensure batch is the first dimension [B, SeqLen, D]
      dropout=0.1
    )
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    # Output MLP to predict the final goal embedding
    self.output_head = nn.Sequential(
      nn.Linear(embed_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, embed_dim)
    )
  
  def forward(self, current_state, memory_bank_episodes, episode_lengths, memory_padding_mask=None):
    """
    Args:
      current_state: [B, D]
      memory_bank_episodes: [B, M, E, D] 
      episode_lengths: [B, M] containing the actual length of each stored episode
      memory_padding_mask: [B, M] (True means ignore memory slot)
    Returns:
      predicted_goal: [B, D]
    """
    B, M, E, D = memory_bank_episodes.shape
    
    # 1. Flatten the Batch and Memory dimensions
    flat_episodes = memory_bank_episodes.reshape(B * M, E, D)
    flat_lengths = episode_lengths.reshape(B * M)
    
    # 2. Protect against length 0 for empty memory slots
    # (pack_padded_sequence needs length > 0. The transformer mask handles ignoring it later).
    flat_lengths = torch.clamp(flat_lengths, min=1)
    
    # 3. Compress the episodes using true lengths
    flat_compressed_memory = self.episode_compressor(flat_episodes, flat_lengths)
    
    # 4. Reshape back to Batch and Memory dimensions
    memory_bank = flat_compressed_memory.reshape(B, M, D)
    
    # 5. Add token types and concatenate
    state_token = current_state.unsqueeze(1) + self.state_token_type  
    memory_tokens = memory_bank + self.memory_token_type  
    sequence = torch.cat([state_token, memory_tokens], dim=1)         
    
    # 6. Apply Transformer masking
    if memory_padding_mask is not None:
      state_mask = torch.zeros((B, 1), dtype=torch.bool, device=current_state.device)
      full_padding_mask = torch.cat([state_mask, memory_padding_mask], dim=1) 
    else:
      full_padding_mask = None
        
    # 7. Predict
    encoded_sequence = self.transformer_encoder(sequence, src_key_padding_mask=full_padding_mask)

    # 8. Extract the state token's representation (the first token)
    state_representation = encoded_sequence[:, 0, :] # [B, D]
    
    # 9. Predict the goal embedding
    predicted_goal = self.output_head(state_representation)
    
    return predicted_goal


class AlteredPredictor(nn.Module):
  """
  AlteredPredictor: A Vision Transformer (ViT) model that predicts which image patches are likely to change
  given the current image patches and a sequence of actions.

  This model processes image patches through a transformer encoder, incorporates action embeddings,
  and outputs a probability for each patch indicating whether it will be altered in the next state.
  Used as the first step in the two-step next state prediction pipeline to identify regions of change.

  ARCHITECTURE OVERVIEW:
  1. Patchify: Splits input image into non-overlapping patches (e.g., 32x32 image → 8x8=64 patches of 4x4 pixels)
  2. Patch Embedding: Projects each patch from pixel space to embedding space
  3. Positional Embedding: Adds 2D sinusoidal positional encoding to preserve spatial information
  4. Action Embedding: Encodes the action sequence into a single embedding token
  5. Internal State Embedding: Optionally encodes the robot's internal state (angle, position)
  6. Transformer Encoder: Processes all tokens (patches + action + state) through multi-head attention
  7. Change Prediction Head: Binary classification (Sigmoid) to predict if each patch will change
  8. Internal State Prediction Heads: Predicts the next internal state values

  INPUT SHAPES:
  - patch: [B, n_patchs=64, N=48] - flattened image patches (B=batch_size, N=patch_dim=channels*ph*pw)
  - action: [B, 1] - discrete action indices (one action per forward pass)
  - internal_state: [B, 2] - two internal state values, angle joint1 and angle joint2, discretized into bins

  OUTPUT SHAPES:
  - preds: [B, n_patchs=64, 1] - probability [0,1] that each patch will change
  - next_is1_logits: [B, is1_n_values=19] - logits for angle prediction (19 discrete bins)
  - next_is2_logits: [B, is2_n_values=37] - logits for position prediction (37 discrete bins)

  PURPOSE IN TWO-STEP PIPELINE:
  This model acts as a filter that identifies which patches are relevant for detailed prediction.
  By predicting changes efficiently at the patch level, it reduces computational load for the
  more expensive AlterationPredictor which only processes changed patches.
  """
  def __init__(
    self,
    *,
    image_size,       # int or tuple (H, W): spatial resolution of the input image (e.g., 32x32)
    patch_size,       # int or tuple (Ph, Pw): spatial size of each image patch (e.g., 4x4)
    dim,              # int: embedding dimension of each patch/token (e.g., 64)
    depth,            # int: number of Transformer encoder blocks (e.g., 2)
    heads,            # int: number of attention heads per Transformer block (e.g., 4)
    mlp_dim,          # int: hidden dimension of the feed-forward (MLP) layer (e.g., 128)
    channels=3,       # int: number of input image channels (3 for RGB)
    dim_head=64,      # int: dimension of each attention head (e.g., 64)
    dropout=0.,       # float: dropout rate inside attention and MLP layers
    emb_dropout=0.,   # float: dropout rate applied to patch embeddings
    n_actions=5,      # int: number of discrete actions
    action_dim=8,     # int: intermediate embedding dimension for actions
    is1_n_values=19,  # number of discrete values for internal state 1 (angle joint1)
    is2_n_values=37,  # number of discrete values for internal state 2 (angle joint2)
    is_emb_dim=16     # int: intermediate embedding dimension for internal states
  ):
    super().__init__()
    # === SPATIAL SETUP ===
    # Store image and patch dimensions
    self.channels = channels
    self.image_height, self.image_width = pair(image_size)
    self.patch_height, self.patch_width = pair(patch_size)

    # Calculate dimensions derived from image and patch size
    self.patch_dim = channels * self.patch_height * self.patch_width

    # Calculate grid dimensions: how many patches along each axis
    self.grid = [(self.image_height // self.patch_height), (self.image_width // self.patch_width)]
    self.n_patchs = self.grid[0] * self.grid[1]  # Total patches: e.g., 64

    # === ACTION EMBEDDING ===
    # Converts discrete action ID (0-4) → action_dim → dim embedding
    # Discrete action becomes continuous representation that can be added to other embeddings
    self.action_emb = nn.Sequential(
      nn.Embedding(n_actions, action_dim),    # [B, 1] → [B, 1, action_dim]
      nn.Linear(action_dim, dim), nn.SiLU())  # [B, 1, action_dim] → [B, 1, dim]

    # === INTERNAL STATE EMBEDDING ===
    # Embed internal state (joint angles): converts discrete bin index to continuous
    self.is1_emb = nn.Sequential(
      nn.Embedding(is1_n_values, is_emb_dim),  # is1_n_values = 19 (0-18 inclusive)
      nn.Linear(is_emb_dim, is_emb_dim),
      nn.SiLU()
    )
    self.is2_emb = nn.Sequential(
      nn.Embedding(is2_n_values, is_emb_dim),  # is2_n_values = 37 (0-36 inclusive)
      nn.Linear(is_emb_dim, is_emb_dim),
      nn.SiLU()
    )
    # Project concatenated internal state embeddings to model dimension
    # Concatenates is1_emb + is2_emb (is_emb_dim*2) and projects to dim
    self.is_proj = nn.Linear(2 * is_emb_dim, dim)
    
    # === INTERNAL STATE PREDICTION HEADS ===
    # Predict next internal state values
    self.predict_next_is1 = nn.Linear(dim, is1_n_values)
    self.predict_next_is2 = nn.Linear(dim, is2_n_values)
    
    # === PATCH EMBEDDING ===
    # Rearrange: Convert image [B, C, H, W] → patches [B, n_patchs, patch_dim]
    # Example: [B, 3, 32, 32] → [B, 64, 48] (64 patches of 3*4*4=48 pixels)
    self.patchify = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width)
    # Project patch pixels to embedding dimension
    self.to_patch_embedding = nn.Sequential(#nn.LayerNorm(self.patch_dim),
                                            nn.Linear(self.patch_dim, dim),  # [B, n_patchs, patch_dim] → [B, n_patchs, dim]
                                            nn.LayerNorm(dim))               # Normalize for training stability

    # === POSITIONAL EMBEDDING ===
    # Pre-compute 2D sinusoidal positional embeddings for the entire grid
    # This encodes "which patch is where" using sine/cosine functions
    # Result: [n_patchs=64, dim] tensor where each row is the embedding for patch at (h, w)
    self.pos_embedding = posemb_sincos_2d(
      h = self.image_height // self.patch_height,
      w = self.image_width // self.patch_width,
      dim = dim,
    ) 

    # Dropout applied after adding positional embeddings
    self.dropout = nn.Dropout(emb_dropout)

    # === TRANSFORMER ENCODER ===
    # Multi-head self-attention blocks that allow patches to exchange information
    # Processes: [B, n_patchs+1(+1), dim] → [B, n_patchs+1(+1), dim]
    self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    # === CHANGE PREDICTION HEAD ===
    # Binary classification head: for each patch, output probability [0, 1] of change
    # Architecture: dim → 2*dim (expansion) → 1 (binary) with Sigmoid activation
    self.find_changed_patch = nn.Sequential(nn.Linear(dim, 2*dim),
                                            nn.ReLU(True),
                                            nn.Linear(2*dim, 1),
                                            nn.Sigmoid())

    # === UNPATCHIFY ===
    # Reconstruct image from patches for visualization
    # [B, n_patchs, patch_dim] → [B, C, H, W]
    self.unpatchify = Rearrange(
      'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
      h=self.image_height // self.patch_height,
      w=self.image_width // self.patch_width,
      p1=self.patch_height,
      p2=self.patch_width,
      c=channels
    )

  def forward(self, patch, action, internal_state=None):
    """
    COMPUTATION FLOW:
    1. Embed patches to embedding dimension
    2. Add positional information
    3. Embed and concatenate action
    4. (Optional) Embed and concatenate internal state
    5. Process through Transformer
    6. Extract patch tokens and predict change probability
    7. Predict next internal state
    
    Args:
      patch: [B, n_patchs=64, patch_dim=48] - flattened image patches
      action: [B, 1] - discrete action indices
      internal_state: [B, 2] - (optional) internal state bins
    
    Returns:
      preds: [B, n_patchs=64, 1] - change probabilities for each patch
      next_is1_logits: [B, is1_n_values] - next joint1 angle logits
      next_is2_logits: [B, is2_n_values] - next joint2 angle logits
    """
    # Step 1: Project patches from pixel space to embedding space
    # [B, n_patchs, patch_dim] → [B, n_patchs, dim]
    patch = self.to_patch_embedding(patch)

    # Step 2: Add positional embeddings to preserve spatial information
    # Positional encoding tells each patch "where are you in the grid"
    pos_emb = self.pos_embedding.to(patch.device, dtype=patch.dtype)
    # Apply dropout after adding positional info for regularization
    # [B, n_patchs, dim] + [n_patchs, dim] = [B, n_patchs, dim]
    patch = self.dropout(patch + pos_emb)

    # Step 3: Embed and process action
    # [B, 1] → [B, 1, dim]
    action_emb = self.action_emb(action)
    # Collect all tokens to be processed by Transformer
    tokens = [patch, action_emb]

    # Step 4: (Optional) Embed internal state
    if internal_state is not None:
      # [B, 1] → [B, 1, is_emb_dim] each, then concatenate → [B, 1, 2*is_emb_dim]
      internal_emb = torch.cat([self.is1_emb(internal_state[:, 0]), self.is2_emb(internal_state[:, 1])], dim=-1)
      # Project to model dimension: [B, 1, 2*is_emb_dim] → [B, 1, dim]
      internal_emb = self.is_proj(internal_emb).unsqueeze(1)  # [B, 1, dim]
      tokens.append(internal_emb)

    # Step 5: Concatenate all tokens
    # Result: [B, n_patchs+1] or [B, n_patchs+2] depending on whether internal_state provided
    # Each token is [B, dim]
    patch = torch.cat(tokens, dim=1)  # [B, n_patchs+1(+1), dim]

    # Step 6: Process through Transformer
    # Self-attention allows patches to communicate with each other and with action/state
    # [B, n_patchs+1(+1), dim] → [B, n_patchs+1(+1), dim]
    patch = self.transformer(patch)

    # Step 7: Extract patch tokens (remove action/internal state tokens)
    # Keep only the first n_patchs tokens for patch-level predictions
    # [B, n_patchs, dim]
    patchs_wo_extra = patch[:, :self.n_patchs]  # Exclude action/internal tokens

    # Step 8: Predict which patches will change
    # For each patch embedding, predict if it will change: [B, n_patchs, dim] → [B, n_patchs, 1]
    # Output is probability [0, 1] due to Sigmoid
    preds = self.find_changed_patch(patchs_wo_extra)  # -> [B, n_patchs, 1]

    # Step 9: Predict next internal state
    # Use the internal state token if available (richer information), else use action token
    if internal_state is not None:
        # Use the internal state token (after Transformer processing)
        next_is_token = internal_emb.squeeze(1)  # [B, 1, dim] → [B, dim]
    else:
        # Fallback to action token if no internal state provided
        next_is_token = action_emb.squeeze(1)  # [B, 1, dim] → [B, dim]
    
    # Predict next internal state values as logits (unnormalized probabilities)
    # [B, dim] → [B, is1_n_values] and [B, is2_n_values]
    next_is1_logits = self.predict_next_is1(next_is_token)  # [B, is1_n_values]
    next_is2_logits = self.predict_next_is2(next_is_token)  # [B, is2_n_values]

    return preds, next_is1_logits, next_is2_logits


class AlterationPredictor(nn.Module):
  """
  AlterationPredictor: A Transformer model that predicts the exact pixel-level changes (alterations)
  for image patches identified as changing by the AlteredPredictor.

  This model takes selected patches, their positional embeddings, and actions as input,
  processes them through a transformer, and outputs the predicted altered patch pixels.
  Used as the second step in the two-step pipeline to generate precise next-state predictions.

  ARCHITECTURE OVERVIEW:
  1. Patch Embedding: Projects selected patches to embedding space
  2. Positional Embedding (from indices): Encodes spatial location of selected patches in original grid
  3. Action Embedding: Encodes action sequence
  4. Internal State Embedding: Optionally encodes robot state
  5. Transformer: Processes only the relevant patches + action + state
  6. Pixel Reconstruction: Reconstructs pixel values for altered patches
  7. Internal State Prediction: Predicts next internal state

  KEY DIFFERENCE FROM AlteredPredictor:
  - Works on a SUBSET of patches (only those marked as changed)
  - Receives SPATIAL INDICES of selected patches to compute correct positional embeddings
  - More expressive (larger network) for detailed pixel-level prediction
  - Focuses computation only where needed (computational efficiency)

  INPUT SHAPES:
  - patch: [B, M<=n_patchs, patch_dim=48] - only selected patches (M varies per batch)
  - patch_indices: [B, M] - spatial indices of selected patches in the 8x8 grid (0-63)
  - action: [B, n_actions_seq] - flattened sequence of actions
  - internal_state: [B, 2] - internal state values

  OUTPUT SHAPES:
  - patch: [B, M, patch_dim=48] - reconstructed pixel values for selected patches
  - next_is1_logits: [B, is1_n_values] - logits for next internal state 1
  - next_is2_logits: [B, is2_n_values] - logits for next internal state 2

  PURPOSE IN TWO-STEP PIPELINE:
  This model generates fine-grained predictions for patches identified as changed.
  By only processing changed patches, it's much more efficient than processing all 64 patches.
  """
  def __init__(self, patch_dim=48, dim=64, depth=4, heads=8, dim_head=32, mlp_dim=128, dropout=0.0,
               n_actions=5, action_dim=8, is1_n_values=19, is2_n_values=37, is_emb_dim=16,
               grid_h=8, grid_w=8):
    super().__init__()
    # === PATCH EMBEDDING ===
    # Projects pixel patch to embedding space
    self.patch_embedder = nn.Sequential(nn.LayerNorm(patch_dim),
                                        nn.Linear(patch_dim, dim),
                                        nn.LayerNorm(dim))
    # === ACTION EMBEDDING ===
    # Same as AlteredPredictor
    self.action_embedder = nn.Sequential(nn.Embedding(n_actions, action_dim), nn.Linear(action_dim, dim), nn.SiLU())
    
    # === INTERNAL STATE EMBEDDING ===
        # Same as AlteredPredictor
    self.is1_emb = nn.Sequential(
      nn.Embedding(is1_n_values, is_emb_dim),  # is1_n_values = 19 (0-18 inclusive)
      nn.Linear(is_emb_dim, is_emb_dim),
      nn.SiLU()
    )
    self.is2_emb = nn.Sequential(
      nn.Embedding(is2_n_values, is_emb_dim),  # is2_n_values = 37 (0-36 inclusive)
      nn.Linear(is_emb_dim, is_emb_dim),
      nn.SiLU()
    )
    self.is_proj = nn.Linear(2 * is_emb_dim, dim)

    # === GRID STORAGE FOR POSITIONAL ENCODING ===
    # Store grid dimensions to compute positional embeddings from patch indices
    self.grid_h = grid_h
    self.grid_w = grid_w
    self.dim = dim

    # === TRANSFORMER ===
    # Process selected patches + action + state through Transformer
    # Note: Unlike AlteredPredictor, we don't have a static positional embedding here
    # Positional info is provided dynamically based on selected patch indices
    self.main = Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)
    
    # === PIXEL RECONSTRUCTION ===
    # Project embedding back to pixel space to reconstruct altered patches
    # [B, M, dim] → [B, M, patch_dim=48]
    self.to_patch_pixels = nn.Sequential(nn.Linear(dim, patch_dim))#, nn.Tanh())
    
    # === INTERNAL STATE PREDICTION ===
    # Predict next internal state
    self.predict_next_is1 = nn.Linear(dim, is1_n_values)
    self.predict_next_is2 = nn.Linear(dim, is2_n_values)
  
  def compute_pos_emb_from_indices(self, indices, dtype):
    """
    Compute positional embeddings from patch spatial indices.

    Instead of using pre-computed embeddings for all patches and then selecting them,
    we compute embeddings ONLY for the selected patches based on their actual grid positions.
    This ensures the model knows the true spatial location of each patch.

    EXAMPLE:
    If patches [5, 15, 42] are selected from 64 patches in an 8x8 grid:
    - Patch 5 is at grid position (0, 5) → gets positional embedding for (0, 5)
    - Patch 15 is at grid position (1, 7) → gets positional embedding for (1, 7)
    - Patch 42 is at grid position (5, 2) → gets positional embedding for (5, 2)
    
    The model then sees these three patches with correct positional information,
    even though they appear at positions [0, 1, 2] in the input sequence.
    
    Args:
      indices: [B, M] tensor of patch indices
      dtype: data type for embeddings
      
    Returns:
      pos_emb: [B, M, dim] positional embeddings
    """
    return posemb_sincos_2d_from_indices(
      indices,
      grid_h=self.grid_h,
      grid_w=self.grid_w,
      dim=self.dim,
      dtype=dtype
    )
  
  def forward(self, patch, patch_indices, action, internal_state=None):
    """
    COMPUTATION FLOW:
    1. Compute positional embeddings from patch spatial indices
    2. Embed selected patches
    3. Add positional information to patch embeddings
    4. Embed action sequence
    5. (Optional) Embed internal state
    6. Process through Transformer
    7. Reconstruct pixel values
    8. Predict next internal state
    
    Args:
        patch: [B, M<=n_patchs, patch_dim=48] - selected patches only
        patch_indices: [B, M] - spatial indices in original grid
        action: [B, n_actions_seq] - action sequence
        internal_state: [B, 2] - (optional) internal state
    
    Returns:
        patch: [B, M, patch_dim=48] - reconstructed altered pixels
        next_is1_logits: [B, is1_n_values] - next state 1 logits
        next_is2_logits: [B, is2_n_values] - next state 2 logits
    """
    # Step 1: Compute positional embeddings from spatial indices and add them
    # Get correct positional embeddings based on actual grid positions
    # [B, M] → [B, M, dim] (positional embedding for each selected patch)
    pos_emb = self.compute_pos_emb_from_indices(patch_indices, patch.dtype)

    # Step 2-3: Embed patches to embedding space
    # Compute positional embeddings based on actual spatial indices
    # [B, M, patch_dim] → [B, M, dim]
    patch_emb = self.patch_embedder(patch) + pos_emb  # [B, M, dim]

    # Step 4: Embed action
    # [B, n_actions_seq] → [B, n_actions_seq, dim]
    # Note: action is a sequence of actions, not just one action
    action_emb = self.action_embedder(action)
    # Collect tokens for Transformer
    tokens = [patch_emb, action_emb]
    
    # Step 5: (Optional) Embed internal state
    if internal_state is not None:
      # Embed and concatenate internal state components
      internal_emb = torch.cat([self.is1_emb(internal_state[:, 0]), self.is2_emb(internal_state[:, 1])], dim=-1)
      # [B, 1, 2*is_emb_dim] → [B, 1, dim]
      internal_emb = self.is_proj(internal_emb).unsqueeze(1)
      tokens.append(internal_emb)

    # Step 6: Concatenate all tokens
    # Result: [B, M+n_actions_seq(+1), dim]
    patch = torch.cat(tokens, dim=1)

    # Step 7: Process through Transformer
    # [B, M+n_actions_seq(+1), dim] → [B, M+n_actions_seq(+1), dim]
    patch = self.main(patch)

    # Step 8: Extract patch tokens (remove action/state tokens)
    # Keep only the first M tokens for patches
    # [B, M, dim]
    patch_wo_extra = patch[:, :patch_emb.size(1)]  # Exclude action/internal tokens

    # Step 9: Reconstruct pixel values from embeddings
    # [B, M, dim] → [B, M, patch_dim=48]
    patch = self.to_patch_pixels(patch_wo_extra)  # [B, M, N=channels*patch_height*patch_width=3*4*4=48]
    
    # Step 10: Predict next internal state
    # Predict next internal state using the internal token if available, else action token
    if internal_state is not None:
        # Use the internal state token
        # [B, 1, dim] → [B, dim]
        next_is_token = internal_emb.squeeze(1)
    else:
        # Fallback to action token
        # Handle both single action and multiple actions
        next_is_token = action_emb[:, 0] if action_emb.size(1) == 1 else action_emb.mean(dim=1)  # [B, dim]
    
    # [B, dim] → [B, is1_n_values] and [B, is2_n_values]
    next_is1_logits = self.predict_next_is1(next_is_token)  # [B, is1_n_values]
    next_is2_logits = self.predict_next_is2(next_is_token)  # [B, is2_n_values]
    
    return patch, next_is1_logits, next_is2_logits


class ObjectPredictor(nn.Module):
  """
  ObjectPredictor: A Vision Transformer (ViT) model that predicts which image patch contains a specified object
  (e.g., the robot's hand or a target), given the current image, the robot's internal state, and an object ID.

  This model processes image patches through a transformer encoder, incorporates embeddings for internal state
  and the specified object, and outputs a probability distribution over the patches, indicating the most likely
  location of the object.

  ARCHITECTURE OVERVIEW:
  1. Patchify: Splits input image into non-overlapping patches.
  2. Patch Embedding: Projects each patch from pixel space to embedding space.
  3. Positional Embedding: Adds 2D sinusoidal positional encoding to preserve spatial information.
  4. Internal State Embedding: Encodes the robot's internal state (e.g., joint angles).
  5. Object Embedding: Encodes the ID of the object to be located (e.g., 0 for hand, 1 for target).
  6. Transformer Encoder: Processes all tokens (patches + state + object) through multi-head attention.
  7. Prediction Head: A classifier that outputs a probability distribution over all patches for the object's location.

  INPUT SHAPES:
  - patch: [B, n_patchs, N] - flattened image patches.
  - internal_state: [B, 2] - two internal state values.
  - object_id: [B, 1] - ID of the object to locate.

  OUTPUT SHAPES:
  - preds: [B, n_patchs] - probability for each patch to contain the specified object.
  """
  def __init__(
    self,
    *,
    image_size,
    patch_size,
    dim,
    depth,
    heads,
    mlp_dim,
    channels=3,
    dim_head=64,
    dropout=0.,
    emb_dropout=0.,
    is1_n_values=19,
    is2_n_values=37,
    is_emb_dim=16,
    patch_mask_ratio=0.25,
  ):
    super().__init__()
    # === SPATIAL SETUP ===
    self.channels = channels
    self.image_height, self.image_width = pair(image_size)
    self.patch_height, self.patch_width = pair(patch_size)
    self.patch_dim = channels * self.patch_height * self.patch_width
    self.grid = [(self.image_height // self.patch_height), (self.image_width // self.patch_width)]
    self.n_patchs = self.grid[0] * self.grid[1]
    self.patch_mask_ratio = patch_mask_ratio

    # === INTERNAL STATE EMBEDDING ===
    self.is1_emb = nn.Sequential(
      nn.Embedding(is1_n_values, is_emb_dim),
      nn.Linear(is_emb_dim, is_emb_dim),
      nn.SiLU()
    )
    self.is2_emb = nn.Sequential(
      nn.Embedding(is2_n_values, is_emb_dim),
      nn.Linear(is_emb_dim, is_emb_dim),
      nn.SiLU()
    )
    self.is_proj = nn.Linear(2 * is_emb_dim, dim)

    # === PATCH EMBEDDING ===
    self.patchify = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width)
    self.to_patch_embedding = nn.Sequential(
        nn.Linear(self.patch_dim, dim),
        nn.LayerNorm(dim)
    )

    # === POSITIONAL EMBEDDING ===
    self.pos_embedding = posemb_sincos_2d(
      h = self.image_height // self.patch_height,
      w = self.image_width // self.patch_width,
      dim = dim,
    ) 
    self.dropout = nn.Dropout(emb_dropout)

    # === TRANSFORMER ===
    self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    # === PREDICTION HEADS ===
    # Multi-head architecture: one head per object
    self.find_hand_patch = nn.Sequential(
        nn.Linear(dim, 2*dim),
        nn.ReLU(True),
        nn.Linear(2*dim, 1)
    )
    self.find_target_patch = nn.Sequential(
        nn.Linear(dim, 2*dim),
        nn.ReLU(True),
        nn.Linear(2*dim, 1)
    )

    # === UNPATCHIFY ===
    self.unpatchify = Rearrange(
      'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
      h=self.image_height // self.patch_height,
      w=self.image_width // self.patch_width,
      p1=self.patch_height,
      p2=self.patch_width,
      c=channels
    )

  def forward(self, patch, internal_state):
    """
    COMPUTATION FLOW:
    1. Embed patches and add positional information.
    2. Embed internal state.
    3. Concatenate all tokens and process through Transformer.
    4. Extract patch tokens and pass through separate heads for hand and target prediction.
    
    Args:
      patch: [B, n_patchs, patch_dim] - flattened image patches
      internal_state: [B, 2] - internal state bins
    
    Returns:
      hand_logits: [B, n_patchs] - logits for hand location
      target_logits: [B, n_patchs] - logits for target location
    """
    # Step 1: Embed patches and add positional info
    patch = self.to_patch_embedding(patch)
    pos_emb = self.pos_embedding.to(patch.device, dtype=patch.dtype)
    patch = self.dropout(patch + pos_emb)

    # Step 2 (Optional): Randomly mask some patches during training for regularization
    if self.training and self.patch_mask_ratio > 0:
      patch = random_patch_mask(patch, self.patch_mask_ratio)

    # Step 3: Embed internal state
    internal_emb = torch.cat([self.is1_emb(internal_state[:, 0]), self.is2_emb(internal_state[:, 1])], dim=-1)
    internal_emb = self.is_proj(internal_emb).unsqueeze(1)

    # Step 4: Concatenate and process through Transformer
    tokens = [patch, internal_emb]
    patch_tokens = torch.cat(tokens, dim=1)
    processed_tokens = self.transformer(patch_tokens)

    # Step 5: Extract patch embeddings and predict with separate heads
    patch_embeddings = processed_tokens[:, :self.n_patchs]

    # Step 6: Predict hand and target locations using separate heads
    hand_logits = self.find_hand_patch(patch_embeddings).squeeze(-1)  # [B, n_patchs]
    target_logits = self.find_target_patch(patch_embeddings).squeeze(-1)  # [B, n_patchs]

    return hand_logits, target_logits, patch_embeddings
