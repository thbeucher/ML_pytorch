"""
Two-Step Next State Predictor Experiments Script

This script implements a hierarchical two-step approach for predicting next states in a reinforcement learning environment.
It uses two specialized models to efficiently model state transitions:

1. AlteredPredictor: A Vision Transformer (ViT) that predicts which image patches are likely to change
   given the current image and a sequence of actions. It outputs a binary mask indicating altered patches.

2. AlterationPredictor: A Transformer model that predicts the exact pixel-level changes (alterations)
   for the patches identified as changing by the AlteredPredictor.

The system is trained end-to-end using a replay buffer filled with transitions from a RobotArmEnv.
The two-step process allows for efficient prediction by focusing computation on relevant image regions.

Key Features:
- Patch-based processing with ViT for spatial understanding.
- Sequence-aware action conditioning.
- Training on image deltas (changes) for stability.
- Evaluation through imagined trajectory generation.
- TensorBoard logging for monitoring training progress.

Environment: Custom RobotArmEnv from gymnasium.
Training: Supervised learning on collected transitions, with separate optimizers for each predictor.
"""

import os
import sys
import copy
import json
import torch
import random
import logging
import warnings
import numpy as np
import torch.nn as nn
import gymnasium as gym

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from torch.utils.tensorboard import SummaryWriter

from vision_transformer.vit import *
from replay_buffer import ReplayBuffer, ReplayBufferDataset

sys.path.append('../../../robot/')
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AlteredPredictor(nn.Module):
    """
    AlteredPredictor: A Vision Transformer (ViT) model that predicts which image patches are likely to change
    given the current image patches and a sequence of actions.

    This model processes image patches through a transformer encoder, incorporates action embeddings,
    and outputs a probability for each patch indicating whether it will be altered in the next state.
    Used as the first step in the two-step next state prediction pipeline to identify regions of change.
    """
    def __init__(
        self,
        *,
        image_size,   # int or tuple (H, W): spatial resolution of the input image
        patch_size,   # int or tuple (Ph, Pw): spatial size of each image patch
        dim,          # int: embedding dimension of each patch/token
        depth,        # int: number of Transformer encoder blocks
        heads,        # int: number of attention heads per Transformer block
        mlp_dim,      # int: hidden dimension of the feed-forward (MLP) layer
        channels=3,   # int: number of input image channels (3 for RGB)
        dim_head=64,  # int: dimension of each attention head
        dropout=0.,   # float: dropout rate inside attention and MLP layers
        emb_dropout=0.,# float: dropout rate applied to patch embeddings
        n_actions=5,
        action_dim=8,
        is1_n_values=19,  # number of discrete values for internal state 1 (e.g., angle bins)
        is2_n_values=37,  # number of discrete values for internal state 2 (e.g., position bins)
        is_emb_dim=16
      ):
        super().__init__()
        self.channels = channels
        self.image_height, self.image_width = pair(image_size)
        self.patch_height, self.patch_width = pair(patch_size)

        self.patch_dim = channels * self.patch_height * self.patch_width

        self.grid = [(self.image_height // self.patch_height), (self.image_width // self.patch_width)]
        self.n_patchs = self.grid[0] * self.grid[1]

        self.action_emb = nn.Sequential(nn.Embedding(n_actions, action_dim), nn.Linear(action_dim, dim), nn.SiLU())

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
        
        self.predict_next_is1 = nn.Linear(dim, is1_n_values)
        self.predict_next_is2 = nn.Linear(dim, is2_n_values)
        
        self.patchify = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width)
        self.to_patch_embedding = nn.Sequential(#nn.LayerNorm(self.patch_dim),
                                                nn.Linear(self.patch_dim, dim),
                                                nn.LayerNorm(dim))

        self.pos_embedding = posemb_sincos_2d(
          h = self.image_height // self.patch_height,
          w = self.image_width // self.patch_width,
          dim = dim,
        ) 
    
        self.dropout = nn.Dropout(emb_dropout)
    
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.find_changed_patch = nn.Sequential(nn.Linear(dim, 2*dim),
                                                nn.ReLU(True),
                                                nn.Linear(2*dim, 1),
                                                nn.Sigmoid())

        self.unpatchify = Rearrange(
          'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
          h=self.image_height // self.patch_height,
          w=self.image_width // self.patch_width,
          p1=self.patch_height,
          p2=self.patch_width,
          c=channels
        )

    def forward(self, patch, action, internal_state=None):
        # [B, n_patchs=64, N=48], [B, 1], [B, internal_state_dim]
        patch = self.to_patch_embedding(patch)  # -> [B, n_patchs, dim]
    
        pos_emb = self.pos_embedding.to(patch.device, dtype=patch.dtype)
        patch = self.dropout(patch + pos_emb)

        action_emb = self.action_emb(action)  # [B, 1, dim]
        tokens = [patch, action_emb]
    
        if internal_state is not None:
          internal_emb = torch.cat([self.is1_emb(internal_state[:, 0]), self.is2_emb(internal_state[:, 1])], dim=-1)
          internal_emb = self.is_proj(internal_emb).unsqueeze(1)  # [B, 1, dim]
          tokens.append(internal_emb)

        patch = torch.cat(tokens, dim=1)  # [B, n_patchs+1(+1), dim]
    
        patch = self.transformer(patch)

        patchs_wo_extra = patch[:, :self.n_patchs]  # Exclude action/internal tokens
        preds = self.find_changed_patch(patchs_wo_extra)  # -> [B, n_patchs, 1]
    
        # Predict next internal state using the internal token if available, else action token
        if internal_state is not None:
            next_is_token = internal_emb.squeeze(1)  # [B, dim]
        else:
            next_is_token = action_emb.squeeze(1)  # [B, dim]
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
  """
  def __init__(self, patch_dim=48, dim=64, depth=4, heads=8, dim_head=32, mlp_dim=128, dropout=0.0,
               n_actions=5, action_dim=8, is1_n_values=19, is2_n_values=37, is_emb_dim=16):
    super().__init__()
    self.patch_embedder = nn.Sequential(nn.LayerNorm(patch_dim),
                                        nn.Linear(patch_dim, dim),
                                        nn.LayerNorm(dim))
    self.action_embedder = nn.Sequential(nn.Embedding(n_actions, action_dim), nn.Linear(action_dim, dim), nn.SiLU())
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
    self.pos_embedding = posemb_sincos_2d(h=8, w=8, dim=dim) 
    self.main = Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)
    self.to_patch_pixels = nn.Sequential(nn.Linear(dim, patch_dim))#, nn.Tanh())
    
    self.predict_next_is1 = nn.Linear(dim, is1_n_values)
    self.predict_next_is2 = nn.Linear(dim, is2_n_values)
  
  def forward(self, patch, pos_emb, action, internal_state=None):  # [B, M<=n_patchs, N=48], [B, M, dim], [B, n_a]
    patch_emb = self.patch_embedder(patch) + pos_emb
    action_emb = self.action_embedder(action)
    tokens = [patch_emb, action_emb]
    
    if internal_state is not None:
      internal_emb = torch.cat([self.is1_emb(internal_state[:, 0]), self.is2_emb(internal_state[:, 1])], dim=-1)
      internal_emb = self.is_proj(internal_emb).unsqueeze(1)  # [B, 1, dim]
      tokens.append(internal_emb)

    patch = torch.cat(tokens, dim=1)  # [B, M+n_a(+1), dim]

    patch = self.main(patch)  # [B, M+n_a(+1), dim]

    patch_wo_extra = patch[:, :patch_emb.size(1)]  # Exclude action/internal tokens
    patch = self.to_patch_pixels(patch_wo_extra)
    # [B, M, N=channels*patch_height*patch_width=3*4*4=48]
    
    # Predict next internal state using the internal token if available, else action token
    if internal_state is not None:
        next_is_token = internal_emb.squeeze(1)  # [B, dim]
    else:
        next_is_token = action_emb[:, 0] if action_emb.size(1) == 1 else action_emb.mean(dim=1)  # [B, dim]
    next_is1_logits = self.predict_next_is1(next_is_token)  # [B, is1_n_values]
    next_is2_logits = self.predict_next_is2(next_is_token)  # [B, is2_n_values]
    
    return patch, next_is1_logits, next_is2_logits


class NSPTrainer:  # NextStatePredictor
  """
  NSPTrainer: Trainer class for the Two-Step Next State Predictor system.

  This class manages the training and evaluation of the AlteredPredictor and AlterationPredictor models.
  It handles environment interaction, replay buffer management, model instantiation, optimization,
  and logging. The trainer fills a replay buffer with environment transitions, trains the models
  to predict next states hierarchically, and evaluates performance through imagined trajectory generation.
  Used for experiments in model-based reinforcement learning with efficient state prediction.
  """
  CONFIG = {'image_size':                        256,
            'resize_to':                         32,
            'image_chan':                        3,
            'normalize_image':                   True,
            'time_dim':                          64,
            'internal_state_dim':                2,
            'internal_state_n_values':           (90//5+1, 180//5+1),  # max_angle / angle_step +1 for inclusive
            'action_dim':                        1,
            'action_n_values':                   5,
            'action_emb':                        8,
            'replay_buffer_size':                10_000,
            'replay_buffer_device':              'cpu',
            'render_mode':                       'rgb_array',
            'n_epochs':                          100,
            'batch_size':                        128,
            'use_tf_logger':                     True,
            'save_dir':                          'NSP_experiments/',
            'log_dir':                           'runs/',
            'exp_name':                          'nsp_twosteps_big_rollout_actionIS',
            'seed':                              42,
            'rollout_size':                      30,
            'use_internal_state':                True,
            'internal_state_emb_dim':            16,
            }
  def __init__(self, config={}):
    self.config = {**NSPTrainer.CONFIG, **config}
    self.device = torch.device('cuda' if torch.cuda.is_available() else
                               'mps' if torch.backends.mps.is_available() else
                               'cpu')
    logger.info(f'Using device: {self.device}')

    if self.config['normalize_image']:
      self.clamp_denorm_fn = lambda x: (x.clamp(-1, 1) + 1) / 2
    self.get_train_params = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)

    self.set_seed()

    self.env = gym.make("gymnasium_env:RobotArmEnv", render_mode=self.config['render_mode'], cropping=True)

    save_dir_run = os.path.join(self.config['save_dir'], self.config['exp_name'], self.config['log_dir'])
    self.tf_logger = SummaryWriter(save_dir_run) if self.config['use_tf_logger'] else None

    self.instanciate_model()
    self.set_trainer_utils()
    self.dump_config()
  
  def dump_config(self):
    with open(os.path.join(self.config['save_dir'],
                           self.config['exp_name'],
                           f"{self.config['exp_name']}_CONFIG.json"), 'w') as f:
      json.dump(self.config, f)
  
  def set_seed(self):
    # Set seeds for reproducibility
    torch.manual_seed(self.config['seed'])
    np.random.seed(self.config['seed'])
    random.seed(self.config['seed'])
    if self.device.type == 'cuda':
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  
  def save_model(self, model_to_save, model_name):
    save_dir = os.path.join(self.config['save_dir'], self.config['exp_name'])
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}.pt")
    torch.save({model_name: model_to_save.state_dict()}, path)

  def load_model(self, model_to_load, model_name):
    path = os.path.join(self.config['save_dir'],
                        self.config['exp_name'],
                        f"{model_name}.pt")
    if os.path.isfile(path):
      model = torch.load(path, map_location=self.device)
      model_to_load.load_state_dict(model[model_name])
      print(f'Model loaded successfully from {path}...')
    else:
      print(f'File {path} not found... No loaded model.')

  def instanciate_model(self):
    self.altered_predictor = AlteredPredictor(
      image_size=32,
      patch_size=4,
      dim=64,
      depth=2,
      heads=4,
      mlp_dim=128,
      dim_head=32,
      channels=3,
      is1_n_values=self.config['internal_state_n_values'][0],
      is2_n_values=self.config['internal_state_n_values'][1],
      is_emb_dim=self.config['internal_state_emb_dim']
    ).to(self.device)
    print(f'Instanciate altered_predictor with {self.get_train_params(self.altered_predictor):,} params')
    self.alteration_predictor = AlterationPredictor(
      patch_dim=self.altered_predictor.patch_dim,
      dim=256,
      depth=8,
      heads=12,
      dim_head=64,
      mlp_dim=1024,
      dropout=0.0,
      is1_n_values=self.config['internal_state_n_values'][0],
      is2_n_values=self.config['internal_state_n_values'][1],
      is_emb_dim=self.config['internal_state_emb_dim']
    ).to(self.device)
    print(f'Instanciate alteration_predictor with {self.get_train_params(self.alteration_predictor):,} params')
  
  def set_trainer_utils(self):
    resize_img = True if self.config['resize_to'] != self.config['image_size'] else False
    self.replay_buffer = ReplayBuffer(self.config['internal_state_dim'],
                                      self.config['action_dim'],
                                      self.config['image_size'],
                                      resize_to=self.config['resize_to'] if resize_img else None,
                                      normalize_img=self.config['normalize_image'],
                                      capacity=self.config['replay_buffer_size'],
                                      device=self.config['replay_buffer_device'],
                                      target_device=self.device)
    self.altered_opt = torch.optim.AdamW(self.altered_predictor.parameters(), lr=1e-4,
                                         weight_decay=1e-4, betas=(0.9, 0.999))
    self.alteration_opt = torch.optim.AdamW(self.alteration_predictor.parameters(), lr=1e-4,
                                            weight_decay=1e-4, betas=(0.9, 0.999))
    self.bce_criterion = nn.BCELoss()
    self.mse_criterion = nn.MSELoss()
    
  def fill_memory(self, random_act=True, max_episode_steps=50):
    print('Filling memory buffer...')
    obs, _ = self.env.reset()
    img = self.env.render()
    episode_step = 0
    for _ in tqdm(range(self.config['replay_buffer_size'])):
      action = random.randint(0, 4) if random_act else 0  #TODO use policy
      next_obs, reward, terminated, truncated, info = self.env.step(action)
      next_img = self.env.render()
      episode_step += 1
      self.replay_buffer.add(obs//5,action, img, reward, terminated or episode_step > max_episode_steps,
                             next_obs//5, next_img)
      obs, img = next_obs, next_img

      if terminated or episode_step > max_episode_steps:
        obs, _ = self.env.reset()
        img = self.env.render()
        episode_step = 0
    
  def get_patch(self, patch, mask):
    """
    patch: [B, Np, N]
    mask:  [B, Np] (bool)

    returns:
        padded: [B, M, N]
        counts: [B] (number of selected patches per batch)
    """
    B, Np, N = patch.shape

    mask = mask.bool()
    counts = mask.sum(dim=1)  # [B], count of selected patches per batch
    M = counts.max().item()

    # Early exit if no patches are selected
    if M == 0:
      return patch.new_zeros((B, 0, N)), counts

    # Compute an index rank for each selected patch
    ranks = torch.cumsum(mask.to(torch.int64), dim=1) - 1  # [B, Np]

    # Extract the flat list of selected patch indices
    batch_idx, patch_idx = mask.nonzero(as_tuple=True)  # [total_selected]
    # Map of selected patches to their target position in the output
    target_pos = ranks[batch_idx, patch_idx]          # [total_selected]

    selected = patch[batch_idx, patch_idx]            # [total_selected, N]

    padded = patch.new_zeros((B, M, N))
    padded[batch_idx, target_pos] = selected

    return padded, counts
  
  def get_next_image(self, patch, patch_predicted, patch_mask):
    """
    patch:           [B, Np, N]      original patches
    patch_predicted: [B, M, N]       predicted patches (only for selected ones)
    patch_mask:      [B, Np] (bool)  which patches were selected

    returns:
        patch_next:  [B, Np, N]
    """
    patch_mask = patch_mask.bool()

    # If no patches are predicted to change, return original image patches.
    if not patch_mask.any():
      return patch.clone()

    # Determine the order of predicted patches (same as get_patch)
    ranks = torch.cumsum(patch_mask.to(torch.int64), dim=1) - 1  # [B, Np]
    batch_idx, patch_idx = patch_mask.nonzero(as_tuple=True)
    target_pos = ranks[batch_idx, patch_idx]

    patch_next = patch.clone()
    patch_next[batch_idx, patch_idx] = patch_predicted[batch_idx, target_pos]
    return patch_next
  
  def add_highlight(self, mask, img):
    # --- Convert patch mask → pixel mask --- #
    # M: [B, 64, 1] → [B, 64]
    patch_mask = mask.squeeze(-1).bool()
    # reshape to patch grid
    grid_height, grid_width = self.altered_predictor.grid
    patch_mask = patch_mask.view(-1, grid_height, grid_width)  # [B, 8, 8]
    # expand each patch to pixels
    pixel_mask = patch_mask.repeat_interleave(self.altered_predictor.patch_height, dim=1)\
                            .repeat_interleave(self.altered_predictor.patch_width, dim=2)
    # pixel_mask: [B, 32, 32]

    # --- Create green overlay --- #
    img_vis = img.clone()
    green = torch.zeros_like(img_vis)
    green[:, 1, :, :] = 1.0   # pure green channel

    # --- Blend image + green filter --- #
    alpha = 0.5  # strength of highlight
    mask = pixel_mask.unsqueeze(1)  # [B, 1, 32, 32]
    img_vis = torch.where(
      mask,
      (1 - alpha) * img_vis + alpha * green,
      img_vis
    )
    return img_vis
  
  def sample_training_batch(self):
    """Sample a batch of episodes for training."""
    return self.replay_buffer.sample_episode_batch(self.config['batch_size'], self.config['rollout_size'])

  def compute_altered_loss(self, patch, action, patch_next_image, internal_state=None, next_internal_state=None):
    """Compute loss for the AlteredPredictor."""
    # [B, n_patchs, 1], [B, is1_n], [B, is2_n]
    patch_change_pred, next_is1_logits, next_is2_logits = self.altered_predictor(patch, action, internal_state)
    patch_change_pred_mask = (patch_change_pred > 0.5)  # [B, n_patchs, 1]

    # Construct changed patch target
    patch_diff = (patch - patch_next_image).abs().sum(dim=2, keepdim=True)
    altered_patch_target = patch_diff > 1e-3  # [B, n_patchs, 1]

    altered_loss = self.bce_criterion(patch_change_pred, altered_patch_target)
    
    # Add next internal state loss if available
    if next_internal_state is not None:
        ce_criterion = nn.CrossEntropyLoss()
        next_is1_loss = ce_criterion(next_is1_logits, next_internal_state[:, 0].long())
        next_is2_loss = ce_criterion(next_is2_logits, next_internal_state[:, 1].long())
        altered_loss += next_is1_loss + next_is2_loss
    
    return altered_loss, patch_change_pred_mask, altered_patch_target

  def compute_alteration_loss(self, patch, patch_next_image, patch_change_gt_mask, action, internal_state=None, next_internal_state=None):
    """Compute loss for the AlterationPredictor."""
    patch_to_pred, counts = self.get_patch(patch, patch_change_gt_mask)  # [B, M, N], M
    pos_emb = self.alteration_predictor.pos_embedding.to(patch.device, dtype=patch.dtype)  # [n_patchs, dim]
    patch_to_pred_pos_emb, _ = self.get_patch(pos_emb.unsqueeze(0).repeat(patch.shape[0], 1, 1),
                                              patch_change_gt_mask)
    # -> [B, M, dim]

    if counts.max() > 0:  # if there is some patch to change
      patch_predicted, next_is1_logits, next_is2_logits = self.alteration_predictor(patch_to_pred.detach(),
                                                                                    patch_to_pred_pos_emb.detach(),
                                                                                    action.detach(),
                                                                                    internal_state.detach() if internal_state is not None else None)
      patch_next_image_predicted = self.get_next_image(patch,
                                                       patch_predicted,
                                                       patch_change_gt_mask)
      # -> [B, n_patchs, N]

      # Get patch to reconstruct
      patch_target, _ = self.get_patch(patch_next_image, patch_change_gt_mask)

      # remove padded time to match patch_target
      alteration_loss = self.mse_criterion(patch_predicted[:, :patch_target.size(1)], patch_target)
      
      # Add next internal state loss if available
      if next_internal_state is not None:
          ce_criterion = nn.CrossEntropyLoss()
          next_is1_loss = ce_criterion(next_is1_logits, next_internal_state[:, 0].long())
          next_is2_loss = ce_criterion(next_is2_logits, next_internal_state[:, 1].long())
          alteration_loss += next_is1_loss + next_is2_loss
    else:
      patch_next_image_predicted = patch
      alteration_loss = 0.0

    return alteration_loss, patch_next_image_predicted, counts

  def update_models(self, altered_loss, alteration_loss):
    """Update the models using their respective optimizers."""
    # TRAIN altered predictor
    self.altered_opt.zero_grad()
    altered_loss.backward()
    self.altered_opt.step()

    # TRAIN alteration predictor
    self.alteration_opt.zero_grad()
    alteration_loss.backward()
    self.alteration_opt.step()

  def compute_metrics(self, patch_change_pred_mask, altered_patch_target, alteration_loss, counts):
    """Compute training metrics."""
    altered_accuracy = (patch_change_pred_mask == altered_patch_target).float().mean()
    return altered_accuracy, alteration_loss.item() if alteration_loss else 0.0, counts.float().mean().item() if counts.max() > 0 else 0.0

  def log_epoch(self, epoch, mean_altered_loss, mean_alteration_loss, mean_accuracy, mean_n_patch_to_modify, image, next_image, patch_next_image_predicted):
    """Log metrics and images for the epoch."""
    self.tf_logger.add_scalar('altered_loss', mean_altered_loss, epoch)
    self.tf_logger.add_scalar('alteration_loss', mean_alteration_loss, epoch)
    self.tf_logger.add_scalar('altered_accuracy', mean_accuracy, epoch)
    self.tf_logger.add_scalar('mean_n_patch_to_modify', int(mean_n_patch_to_modify), epoch)

    self.tf_logger.add_images(
      'alterationNet_reconstructed_image',
      torch.cat([image[:8],
                 next_image[:8],
                 self.altered_predictor.unpatchify(patch_next_image_predicted[:8])], dim=0),
      global_step=epoch
    )

  def save_best_models(self, mean_accuracy, best_accuracy, mean_alteration_loss, best_loss):
    """Save models if they are the best so far."""
    if mean_accuracy > best_accuracy:
      self.save_model(self.altered_predictor, 'altered_predictor')
    if mean_alteration_loss < best_loss:
      self.save_model(self.alteration_predictor, 'alteration_predictor')
    return max(mean_accuracy, best_accuracy), min(mean_alteration_loss, best_loss)

  def generate_imagined_trajectory(self, image, epoch):
    """Generate and log an imagined trajectory for evaluation."""
    self.altered_predictor.eval()
    self.alteration_predictor.eval()
    with torch.no_grad():
      imagined_traj = [image[:1]]
      start_img_idx, current_history = 0, 0
      next_restart = random.randint(0, self.config['rollout_size']-1)
      fake_actions = []
      current_internal_state = torch.zeros(1, 2, dtype=torch.long, device=self.device)  # Initial state
      for fake_action in torch.as_tensor([[[1]]]*12 + [[[3]]]*11, device=self.device, dtype=torch.long):
        fake_actions.append(fake_action)
        if current_history > next_restart:
          start_img_idx += next_restart
          current_history = 0
          next_restart = random.randint(0, self.config['rollout_size']-1)
          fake_actions = [fake_action]
        current_history += 1
        fake_action = torch.cat(fake_actions, dim=-1)

        patch = self.altered_predictor.patchify(imagined_traj[start_img_idx])

        # get patches that are expected to change
        patch_change_pred, next_is1_logits, next_is2_logits = self.altered_predictor(patch, fake_action, current_internal_state)

        # Update internal state for next step
        current_internal_state = torch.stack([torch.argmax(next_is1_logits, dim=1), torch.argmax(next_is2_logits, dim=1)], dim=1)

        # get mask to select only predicted changed patches
        patch_change_pred_mask = (patch_change_pred > 0.5).squeeze(-1)

        # select only predicted changed patches
        patch_to_pred, counts = self.get_patch(patch, patch_change_pred_mask)
        pos_emb = self.alteration_predictor.pos_embedding.to(patch.device, dtype=patch.dtype)
        patch_to_pred_pos_emb, _ = self.get_patch(pos_emb.unsqueeze(0), patch_change_pred_mask)

        # make prediction on next patches if there is some to predict
        if counts.max() > 0:  # if there is some patch to change
          patch_predicted, _, _ = self.alteration_predictor(patch_to_pred,
                                                            patch_to_pred_pos_emb,
                                                            fake_action,
                                                            current_internal_state)
          patch_next_image_predicted = self.get_next_image(patch,
                                                           patch_predicted,
                                                           patch_change_pred_mask)
        else:  # if no patches to change, next image is the original one
          patch_next_image_predicted = patch
        # Clean the predicted image
        cleaned_img = self.altered_predictor.unpatchify(patch_next_image_predicted)
        imagined_traj.append(cleaned_img)

    self.tf_logger.add_images('generated_world_model_prediction',
                              torch.cat(imagined_traj, dim=0),
                              global_step=epoch)
    self.altered_predictor.train()
    self.alteration_predictor.train()

  def train_alter_predictor(self):
    self.altered_predictor.train()
    self.alteration_predictor.train()

    best_loss = torch.inf
    best_accuracy = 0.0

    pbar = tqdm(range(self.config['n_epochs']))
    for epoch in pbar:
      mean_accuracy = 0.0
      mean_altered_loss = 0.0
      mean_alteration_loss = 0.0
      mean_n_patch_to_modify = 0.0
      for i in tqdm(range(10), leave=False):
        episode = self.sample_training_batch()

        image = episode['image'][:, 0]
        # Random time step to predict from direct next step to as far as rollout_size
        # to have a variety of difficulty in the training samples
        for t in torch.randint(1, self.config['rollout_size'], (5,)).tolist():
          next_image = episode['next_image'][:, t-1]
          action = episode['action'][:, :t].flatten(1)
          internal_state = episode['internal_state'][:, t-1] if self.config['use_internal_state'] else None
          next_internal_state = episode['next_internal_state'][:, t-1] if self.config['use_internal_state'] else None

          patch = self.altered_predictor.patchify(image)  # [B, n_patchs=64, N=48]
          patch_next_image = self.altered_predictor.patchify(next_image)  # [B, n_patchs, N]

          # Compute losses
          altered_loss, patch_change_pred_mask, altered_patch_target = self.compute_altered_loss(patch,
                                                                                                 action,
                                                                                                 patch_next_image,
                                                                                                 internal_state if self.config['use_internal_state'] else None,
                                                                                                 next_internal_state if self.config['use_internal_state'] else None)
          patch_change_gt_mask = altered_patch_target.squeeze(-1)
          alteration_loss, patch_next_image_predicted, counts = self.compute_alteration_loss(patch,
                                                                                             patch_next_image,
                                                                                             patch_change_gt_mask,
                                                                                             action,
                                                                                             internal_state if self.config['use_internal_state'] else None,
                                                                                             next_internal_state if self.config['use_internal_state'] else None)

          # Update models
          self.update_models(altered_loss, alteration_loss)

        # Compute metrics
        altered_accuracy, alt_loss_val, n_patch_val = self.compute_metrics(patch_change_pred_mask, altered_patch_target, alteration_loss, counts)

        mean_altered_loss += (altered_loss.item() - mean_altered_loss) / (i + 1)
        if alteration_loss:
          mean_alteration_loss += (alt_loss_val - mean_alteration_loss) / (i + 1)
        mean_accuracy += (altered_accuracy.item() - mean_accuracy) / (i + 1)
        if counts.max() > 0:
          mean_n_patch_to_modify += (n_patch_val - mean_n_patch_to_modify) / (i + 1)

      # Log epoch
      self.log_epoch(epoch, mean_altered_loss, mean_alteration_loss, mean_accuracy, mean_n_patch_to_modify, image, next_image, patch_next_image_predicted)

      # Save best models
      best_accuracy, best_loss = self.save_best_models(mean_accuracy, best_accuracy, mean_alteration_loss, best_loss)

      # Generate imagined trajectory
      self.generate_imagined_trajectory(image, epoch)

      pbar.set_description(f'Mean_acc: {mean_accuracy:.3f}')
  
  def train(self):
    self.fill_memory()
    self.train_alter_predictor()


if __name__ == '__main__':
  trainer = NSPTrainer()
  trainer.load_model(trainer.altered_predictor, 'altered_predictor')
  trainer.load_model(trainer.alteration_predictor, 'alteration_predictor')
  trainer.train()