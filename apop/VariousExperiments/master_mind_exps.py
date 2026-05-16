import os
import sys
import torch
import random
import logging
import warnings
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F

from tqdm import tqdm
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

import models_zoo as mz
import helpers_zoo as hz

from replay_buffer import ReplayBuffer
from vision_transformer.vit import Transformer


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# get access to robot environments
sys.path.append('../../../robot/')
warnings.filterwarnings("ignore")


class NextEmbeddingPredictorV2(nn.Module):
  #TODO to finish
  CONFIG = {
    'memory_size': 32,
    'emb_dim':     128,
    'depth':       2,
    'heads':       4,
    'dim_head':    32,
    'mlp_dim':     256,
    'dropout':     0.0,
    'action_emb':  8,
  }
  def __init__(self, config={}, device='cpu'):
    super(NextEmbeddingPredictorV2, self).__init__()
    self.config = {**NextEmbeddingPredictorV2.CONFIG, **config}
    self.memory = torch.zeros(
      (self.config['memory_size'], self.config['emb_dim'] + self.config['action_emb']),
      device=device
    )
    self.latent_encoder = Transformer(
      dim=self.config['emb_dim'],
      depth=self.config['depth'],
      heads=self.config['heads'],
      dim_head=self.config['dim_head'],
      mlp_dim=self.config['mlp_dim'],
      dropout=self.config['dropout'],
    )

  def forward(self, embedding, action_emb):
    # Update memory
    self.memory = torch.cat([self.memory[1:], torch.cat([embedding, action_emb], dim=-1)], dim=0)
    latent = self.latent_encoder(self.memory)
    return latent


class InternalStatePredictor(nn.Module):
  CONFIG = {
    'emb_dim':      256,
    'action_emb':   16,
    'hidden_dim':   256,
    'is1_n_values': 19, 
    'is2_n_values': 37,
    'is_emb_dim':   32,
  }
  def __init__(self, config={}):
    super().__init__()
    self.config = {**InternalStatePredictor.CONFIG, **config}
    self.net = mz.get_linear_net(self.config['emb_dim'], self.config['hidden_dim'], self.config['hidden_dim'])

    self.is1_head = nn.Linear(self.config['hidden_dim'], self.config['is1_n_values'])
    self.is2_head = mz.get_linear_net(self.config['hidden_dim'], self.config['hidden_dim']//2, self.config['is2_n_values'])

    enriched_dim_size = 2 * self.config['is_emb_dim'] + self.config['action_emb']
    self.next_net = mz.get_linear_net(enriched_dim_size, 2 * self.config['hidden_dim'], self.config['hidden_dim'])
    self.nis1_head = nn.Linear(self.config['hidden_dim'], self.config['is1_n_values'])
    self.nis2_head = mz.get_linear_net(self.config['hidden_dim'], 2 * self.config['hidden_dim'], self.config['is2_n_values'])
  
  def get_is(self, emb):
    x = self.net(emb)  # [B, 128] -> [B, 256]
    return self.is1_head(x), self.is2_head(x)
  
  def get_next_is(self, is1_emb, is2_emb, action_emb):
    x = self.next_net(torch.cat([is1_emb, is2_emb, action_emb], dim=-1))
    return self.nis1_head(x), self.nis2_head(x)

  def forward(self, emb, action_emb, is1_emb, is2_emb):
    is1_logits, is2_logits = self.get_is(emb)
    nis1_logits, nis2_logits = self.get_next_is(is1_emb, is2_emb, action_emb)
    return is1_logits, is2_logits, nis1_logits, nis2_logits


class WorldEncoder(nn.Module):
  CONFIG = {
    'emb_dim':           256,
    'action_emb':        8,
    'is_emb_dim':        16,
    'token_dim':         64,
    'n_patches':         16*16,
    'patch_emb_dim':     64,
    'action_n_values':   5,
    'is1_n_values':      19,
    'is2_n_values':      37,
    'ae_config': {
      'encoder_archi': 'BigCNNEncoder',
      'skip_connection': True, 
      'linear_bottleneck': True,
      'latent_dim': 256,
    },
  }
  def __init__(self, config={}):
    super().__init__()
    self.config = {**WorldEncoder.CONFIG, **config}
    # === ACTION EMBEDDING ===
    self.action_emb = nn.Sequential(
      nn.Embedding(self.config['action_n_values'], self.config['action_emb']),
      nn.Linear(self.config['action_emb'], self.config['action_emb']),
      nn.SiLU()
    )
    # === INTERNAL STATE EMBEDDING ===
    self.is1_emb = nn.Sequential(
      nn.Embedding(self.config['is1_n_values'], self.config['is_emb_dim']),
      nn.Linear(self.config['is_emb_dim'], self.config['is_emb_dim']),
      nn.SiLU()
    )
    self.is2_emb = nn.Sequential(
      nn.Embedding(self.config['is2_n_values'], self.config['is_emb_dim']),
      nn.Linear(self.config['is_emb_dim'], self.config['is_emb_dim']),
      nn.SiLU()
    )
    # === CNN Auto-Encoder ===
    self.ae = mz.CNNAE(self.config['ae_config'])
    self.emb_enricher = mz.Transformer(dim=self.config['token_dim'], depth=2, heads=4, dim_head=32, mlp_dim=128, dropout=0.0)
    # === Next Embedding Predictor ===
    self.nep = mz.get_linear_net(
      self.config['emb_dim'] + self.config['action_emb'],
      2 * self.config['emb_dim'],
      self.config['emb_dim']
    )
    # === (Next) Internal State Predictor ===
    self.isp = InternalStatePredictor(config=self.config)
    # === Object Predictors ===
    self.pos_emb = nn.Embedding(self.config['n_patches'], self.config['token_dim'])
    self.is_scaler = nn.Linear(self.config['is_emb_dim'], self.config['token_dim'])
    self.emb_scaler = nn.Linear(self.config['emb_dim'], self.config['token_dim'])
    self.action_scaler = nn.Linear(self.config['action_emb'], self.config['token_dim'])
    # C1 = d1 = [B, 64, 16, 16] -> [B, 16*16, 64]
    self.object_enricher = mz.Transformer(dim=self.config['token_dim'], depth=2, heads=4, dim_head=32, mlp_dim=128, dropout=0.0)
    self.find_hand_patch = nn.Sequential(
      nn.Linear(self.config['token_dim'], 2 * self.config['emb_dim']),
      nn.ReLU(True),
      nn.Linear(2 * self.config['emb_dim'], 1)
    )
    self.patch_emb = nn.Embedding(self.config['n_patches'], self.config['patch_emb_dim'])
    self.find_target_patch = nn.Sequential(
      nn.Linear(self.config['token_dim'], 2 * self.config['emb_dim']),
      nn.ReLU(True),
      nn.Linear(2 * self.config['emb_dim'], 1)
    )
    # === Internal State from Patch ===
    self.patch_to_is_emb = nn.Embedding(self.config['n_patches'], self.config['patch_emb_dim'])
    self.patch_to_is_net = nn.Sequential(
        nn.Linear(self.config['patch_emb_dim'], 128),
        nn.SiLU(),
        nn.Linear(128, self.config['is1_n_values'] + self.config['is2_n_values'])
    )
  
  def get_is_from_patch_idx(self, patch_idx):
    return self.patch_to_is_net(self.patch_to_is_emb(patch_idx))
  
  def scale_embs(self, emb, is1_emb, is2_emb, action_emb=None, only_action=False):
    scaled_embs = []

    if action_emb is not None:
      action_emb_scale = self.action_scaler(action_emb).unsqueeze(1)  # [B, 1, 64]
      scaled_embs.append(action_emb_scale)

      if only_action:
        return action_emb_scale
    
    emb_scale = self.emb_scaler(emb).unsqueeze(1)         # [B, 128] -> [B, 1, 64]
    is1_emb_scale = self.is_scaler(is1_emb).unsqueeze(1)  # [B, 16]  -> [B, 1, 64]
    is2_emb_scale = self.is_scaler(is2_emb).unsqueeze(1)

    return [emb_scale, is1_emb_scale, is2_emb_scale] + scaled_embs

  def _get_base_representation(self, image, internal_state):
    rec, (d1, d2, d3), emb = self.ae(image, return_all=True)
    emb = self.emb_enricher(emb.view(-1, 4, self.config['token_dim'])).view(-1, 256)

    is1_emb = self.is1_emb(internal_state[:, 0])
    is2_emb = self.is2_emb(internal_state[:, 1])

    # C1 = d1 = [B, 64, 16, 16] -> [B, 16*16, 64]
    patchs = d1.flatten(2).transpose(1, 2) + self.pos_emb(torch.arange(0, self.config['n_patches'], device=d1.device))
    return rec, d1, emb, is1_emb, is2_emb, patchs

  def get_state_representation(self, image, internal_state):
    _, _, emb, is1_emb, is2_emb, patchs = self._get_base_representation(image, internal_state)

    scaled_embs = self.scale_embs(emb, is1_emb, is2_emb)
    patchs_enriched = self.object_enricher(torch.cat([patchs] + scaled_embs, dim=1))[:, :-len(scaled_embs)]
    hand_pred = self.find_hand_patch(patchs_enriched).squeeze(-1).argmax(-1)

    return emb, is1_emb, is2_emb, patchs, patchs_enriched, scaled_embs, hand_pred

  def forward(self, image, action, internal_state):
    action_emb = self.action_emb(action).squeeze(1)
    rec, _, emb, is1_emb, is2_emb, patchs = self._get_base_representation(image, internal_state)
    
    emb_action = torch.cat([emb, action_emb], dim=-1)         # -> [B, 256+16]

    # 2) Predict Next Embedding
    next_emb = self.nep(emb_action)

    # 3) Predict Internal State and Next Internal State
    isp1, isp2, nisp1, nisp2 = self.isp(emb, action_emb, is1_emb, is2_emb)

    # 4) Predict Object Position - object = hand & target
    scaled_embs = self.scale_embs(emb, is1_emb, is2_emb)
    patchs_enriched = self.object_enricher(torch.cat([patchs] + scaled_embs, dim=1))[:, :-len(scaled_embs)]

    hand_pred = self.find_hand_patch(patchs_enriched).squeeze(-1)
    target_pred = self.find_target_patch(patchs_enriched).squeeze(-1)

    # Predict next hand position
    with torch.no_grad():
      next_is1_emb = self.is1_emb(nisp1.argmax(dim=-1))
      next_is2_emb = self.is2_emb(nisp2.argmax(dim=-1))
    
    scaled_embs_next = self.scale_embs(next_emb, next_is1_emb, next_is2_emb, action_emb)
    next_patchs_enriched = self.object_enricher(torch.cat([patchs] + scaled_embs_next, dim=1))[:, :-len(scaled_embs_next)]
    next_hand_pred = self.find_hand_patch(next_patchs_enriched).squeeze(-1)
      
    return emb, rec, next_emb, isp1, isp2, nisp1, nisp2, hand_pred, target_pred, next_hand_pred


class Actor(nn.Module):
  CONFIG = {
    'is_emb_dim':      16,
    'hidden_dim':      512,
    'action_n_values': 5,
  }
  def __init__(self, config={}):
    super().__init__()
    self.config = {**Actor.CONFIG, **config}
    self.net = mz.get_linear_net(
      4 * self.config['is_emb_dim'],
      self.config['hidden_dim'],
      self.config['action_n_values']
    )

  def forward(self, current_is_emb, goal_is_emb):
    # current_is_emb: concatenated (is1_emb, is2_emb)
    # goal_is_emb: concatenated (goal_is1_emb, goal_is2_emb)
    x = torch.cat([current_is_emb, goal_is_emb], dim=-1)
    return self.net(x)

  def sample(self, current_is_emb, goal_is_emb):
    # ---- Policy forward pass π(a | s, g) ----
    a_logits = self.forward(current_is_emb, goal_is_emb)
    dist = torch.distributions.Categorical(logits=a_logits)
    # Sample action (on-policy)
    actions = dist.sample()
    # Get log probs for RL learning
    log_probs = dist.log_prob(actions)
    return actions, log_probs, dist.entropy()


class Critic(nn.Module):
  CONFIG = {
    'is_emb_dim':      16,
    'hidden_dim':      512,
  }
  def __init__(self, config={}):
    super().__init__()
    self.config = {**Critic.CONFIG, **config}
    self.net = mz.get_linear_net(
      4 * self.config['is_emb_dim'],
      self.config['hidden_dim'],
      1
    )

  def forward(self, current_is_emb, goal_is_emb):
    # state: concatenated (is1_emb, is2_emb)
    # goal: concatenated (goal_is1_emb, goal_is2_emb)
    x = torch.cat([current_is_emb, goal_is_emb], dim=-1)
    return self.net(x).squeeze(-1)


class MasterMind:
  CONFIG = {
    'save_dir':                          'experiments/',
    'exp_name':                          'master_mind_ppo',
    'use_tf_logger':                     True,
    'load_model':                        True,
    # === Replay Buffer & Models info ===
    'replay_buffer_device':              'cpu',   # Store on CPU to save GPU memory
    'image_size':                        256,     # Original environment image size (RGB)
    'resize_to':                         32,      # Resize to for memory efficiency (256 → 32)
    'normalize_image':                   True,    # Normalize to [-1, 1]
    'internal_state_dim':                2,       # Robot state: [angle_joint1, angle_joint2]
    'internal_state_n_values':           (90//5+1, 180//5+1),  # max_angle / angle_step +1 for inclusive
    'action_dim':                        1,       # Single action per step
    'n_train_episodes':                  128,
    'n_test_episodes':                   10,
    'max_ep_len':                        60,
    'is_emb_dim':                        16,      # int: intermediate embedding dimension for internal states
    'n_patches':                         256,     # patchs of size 2*2 -> (32/2)*(32/2)
    'patch_emb_dim':                     64,
    'token_dim':                         64,      # Embedding size use in Transforme enrichers
    # === Training hyperparameters ===
    'batch_size':                        128,
  }
  def __init__(self, config={}):
    self.config = {**MasterMind.CONFIG, **config}
    self.device = torch.device('cuda' if torch.cuda.is_available() else
                               'mps' if torch.backends.mps.is_available() else
                               'cpu')
    self.patch_to_internal_states = defaultdict(set)
    self.get_train_params = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    
    # === TENSORBOARD LOGGING ===
    # Setup TensorBoard logger for monitoring
    self.save_dir = os.path.join(self.config['save_dir'], self.config['exp_name'])
    os.makedirs(self.save_dir, exist_ok=True)
    save_dir_run = os.path.join(self.save_dir, 'runs/')
    self.tf_logger = SummaryWriter(save_dir_run) if self.config.get('use_tf_logger', False) else None

    # === UTILS ===
    hz.dump_json_data(self.save_dir, self.config['exp_name'], self.config)
    hz.set_seed()
    self.set_env()

    # === Models ===
    self.instanciate_models()
    # Replay Buffer - train / test - Optimizers
    self.set_utils()

    if self.config['load_model']:
      model_path = os.path.join(self.save_dir, f"{self.config['exp_name']}.pt")
      self.load_models(model_path)
  
  def set_env(self, render_mode='rgb_array'):
    self.env = gym.make(
      "gymnasium_env:RobotArmEnv",
      render_mode=render_mode,
      cropping=True
    )
  
  def instanciate_models(self):
    # === WORLD ENCODER ===
    # CNN Auto-Encoder with Residual Connection and Linear Bottleneck
    # NEP (NextEmbeddingPredictor)
    # ISWM (InternalStateWorlModel) - ISP - NISP
    # OPP (ObjectPositionPredictor)
    # TP (TargetPredictor)
    self.we = WorldEncoder(config=self.config).to(self.device)
    logger.info(f'WorldEncoder instanciate with n_params={self.get_train_params(self.we):,}')
    # === Actor Network ===
    self.actor = Actor(config=self.config).to(self.device)
    logger.info(f'Actor instanciate with n_params={self.get_train_params(self.actor):,}')
    # === Critic Network ===
    self.critic = Critic(config=self.config).to(self.device)
    logger.info(f'Critic instanciate with n_params={self.get_train_params(self.critic):,}')
  
  def set_utils(self):
    resize_img = True if self.config['resize_to'] != self.config['image_size'] else False
    self.train_buffer = ReplayBuffer(
      self.config['internal_state_dim'],
      self.config['action_dim'],
      self.config['image_size'],
      resize_to=self.config['resize_to'] if resize_img else None,
      normalize_img=self.config['normalize_image'],
      capacity=self.config['n_train_episodes'] * self.config['max_ep_len'],
      device=self.config['replay_buffer_device'],
      target_device=self.device
    )
    self.test_buffer = ReplayBuffer(
      self.config['internal_state_dim'],
      self.config['action_dim'],
      self.config['image_size'],
      resize_to=self.config['resize_to'] if resize_img else None,
      normalize_img=self.config['normalize_image'],
      capacity=self.config['n_test_episodes'] * self.config['max_ep_len'],
      device=self.config['replay_buffer_device'],
      target_device=self.device
    )
    self.we_optimizer = torch.optim.AdamW([
      {'params': self.we.parameters(), 'lr': 1e-4},
    ])
    self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=1e-4)
    self.ppo_actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=1e-4)
    self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=1e-4)

    self.hand_condition = lambda frames: (frames[:, 2, :, :] > frames[:, 0, :, :]) & \
                                          (frames[:, 2, :, :] > frames[:, 1, :, :]) & \
                                          (frames[:, 2, :, :] > 0.1)
    self.train_buffer.set_hand_condition(self.hand_condition)
    self.test_buffer.set_hand_condition(self.hand_condition)
  
  def save_models(self, path):
    """Saves the state of all models and optimizers."""
    # logger.info(f"Saving models to {path}")
    torch.save({
      'we_state_dict': self.we.state_dict(),
      'actor_state_dict': self.actor.state_dict(),
      'critic_state_dict': self.critic.state_dict(),
      'we_optimizer_state_dict': self.we_optimizer.state_dict(),
      'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
      'ppo_actor_optimizer_state_dict': self.ppo_actor_optimizer.state_dict(),
      'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
    }, path)

  def load_models(self, path):
    """Loads the state of all models and optimizers."""
    if not os.path.exists(path):
      logger.warning(f"Model checkpoint not found at {path}. Starting from scratch.")
      return
    logger.info(f"Loading models from {path}")
    checkpoint = torch.load(path, map_location=self.device)
    self.we.load_state_dict(checkpoint['we_state_dict'])
    self.actor.load_state_dict(checkpoint['actor_state_dict'])
    self.critic.load_state_dict(checkpoint['critic_state_dict'])
    self.we_optimizer.load_state_dict(checkpoint['we_optimizer_state_dict'])
    self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    self.ppo_actor_optimizer.load_state_dict(checkpoint['ppo_actor_optimizer_state_dict'])
    self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
  
  def fill_memory(self, replay_buffer, act='random', n_episodes=128, max_episode_steps=60):
    logger.info(f'Filling memory buffer... ({act=}) ({n_episodes=} | {max_episode_steps=})')
    obs, _ = self.env.reset()
    img = self.env.render()

    for _ in tqdm(range(n_episodes)):
      episode_step = 0
      for _ in range(max_episode_steps):
        if act == 'policy':
          action = random.randint(0, 4)
        elif act == 'best':
          action = self.env.unwrapped.get_best_action()
        else:
          action = random.randint(0, 4)

        next_obs, reward, terminated, truncated, info = self.env.step(action)
        next_img = self.env.render()

        episode_step += 1

        replay_buffer.add(obs//5, action, img, reward, terminated or episode_step >= max_episode_steps,
                          next_obs//5, next_img)
        obs, img = next_obs, next_img

        if terminated or episode_step >= max_episode_steps:
          obs, _ = self.env.reset()
          img = self.env.render()
          episode_step = 0
          break
  
  def compute_metrics(self, batch, isp1, isp2, nisp1, nisp2, hand_pred, target_pred, hand_patch_gt, target_patch_gt, next_hand_pred, next_hand_patch_gt):
    is1_acc = (isp1.argmax(dim=1) == batch['internal_state'][:, 0]).float().mean().item()
    is2_acc = (isp2.argmax(dim=1) == batch['internal_state'][:, 1]).float().mean().item()
    nis1_acc = (nisp1.argmax(dim=1) == batch['next_internal_state'][:, 0]).float().mean().item()
    nis2_acc = (nisp2.argmax(dim=1) == batch['next_internal_state'][:, 1]).float().mean().item()

    hand_patch_acc = 0.0
    valid_hand_mask = hand_patch_gt != -1
    if valid_hand_mask.any():
      hand_patch_acc = (hand_pred[valid_hand_mask].squeeze(-1).argmax(dim=1) == hand_patch_gt[valid_hand_mask]).float().mean().item()

    next_hand_patch_acc = 0.0
    valid_next_hand_mask = next_hand_patch_gt != -1
    if valid_next_hand_mask.any():
      next_hand_patch_acc = (next_hand_pred[valid_next_hand_mask].squeeze(-1).argmax(dim=1) == next_hand_patch_gt[valid_next_hand_mask]).float().mean().item()

    target_patch_acc = 0.0
    valid_target_mask = target_patch_gt != -1
    if valid_target_mask.any():
      target_patch_acc = (target_pred[valid_target_mask].squeeze(-1).argmax(dim=1) == target_patch_gt[valid_target_mask]).float().mean().item()

    return is1_acc, is2_acc, nis1_acc, nis2_acc, hand_patch_acc, target_patch_acc, next_hand_patch_acc
  
  def _reset_logs(self):
    self.losses = defaultdict(list)
    self.accuracies = defaultdict(list)
    self.actor_metrics = defaultdict(list)

  def _update_logs(self, step_losses, step_accuracies):
    for name, value in step_losses.items():
      self.losses[name].append(value.item())
    for name, value in step_accuracies.items():
      self.accuracies[name].append(value)

  def log_metrics(self, epoch, image, rec, hand_pred, target_pred, next_hand_pred, hand_patch_gt_full, target_patch_gt_full, next_hand_patch_gt_full, prefix='train'):
    if self.tf_logger is None:
      return

    for name, val in self.losses.items():
      self.tf_logger.add_scalar(f'{prefix}_{name}', np.mean(val), epoch)

    if self.accuracies:
      accuracy_scalars = {name: np.mean(val) for name, val in self.accuracies.items()}
      self.tf_logger.add_scalars(f'{prefix}/accuracies', accuracy_scalars, epoch)
    
    self.tf_logger.add_images(
      f'{prefix}_reconstructed_image',
      torch.cat([image[:8], rec[:8]], dim=0),
      global_step=epoch,
      dataformats='NCHW'
    )

    # --- Visualize object predictions ---
    def draw_patch(img, patch_idx, color):
      if patch_idx < 0:
        return img
      img_copy = img.clone()
      patch_size = 2
      grid_size = 16
      patch_y = patch_idx // grid_size
      patch_x = patch_idx % grid_size
      start_x = patch_x * patch_size
      start_y = patch_y * patch_size
      for c_idx, c_val in enumerate(color):
        img_copy[c_idx, start_y:start_y+patch_size, start_x:start_x+patch_size] = c_val
      return img_copy

    n_samples = 8
    images_to_log_ori = []
    images_to_log_viz = []
    
    hand_preds_indices = hand_pred[:n_samples].argmax(dim=1)
    target_preds_indices = target_pred[:n_samples].argmax(dim=1)
    next_hand_preds_indices = next_hand_pred[:n_samples].argmax(dim=1)

    colors = {
      'hand_gt': (0, 0, 1),        # Blue
      'hand_pred': (0, 1, 1),       # Cyan
      'next_hand_gt': (0, 0.5, 0.5),  # Dark Cyan
      'next_hand_pred': (0, 1, 0),    # Green
      'target_gt': (1, 0, 0),      # Red
      'target_pred': (1, 1, 0),     # Yellow
    }

    for i in range(n_samples):
      img_orig = (image[i] * 0.5) + 0.5 # De-normalize to [0, 1]
      
      img_viz = img_orig.clone()
      # Draw hand GT (blue) and Pred (cyan)
      img_viz = draw_patch(img_viz, hand_patch_gt_full[i], colors['hand_gt'])
      img_viz = draw_patch(img_viz, hand_preds_indices[i], colors['hand_pred'])
      # Draw next hand GT (dark cyan) and Pred (green)
      img_viz = draw_patch(img_viz, next_hand_patch_gt_full[i], colors['next_hand_gt'])
      img_viz = draw_patch(img_viz, next_hand_preds_indices[i], colors['next_hand_pred'])
      # Draw target GT (red) and Pred (yellow)
      img_viz = draw_patch(img_viz, target_patch_gt_full[i], colors['target_gt'])
      img_viz = draw_patch(img_viz, target_preds_indices[i], colors['target_pred'])
      
      images_to_log_ori.append(img_orig)
      images_to_log_viz.append(img_viz)

    images_to_log = images_to_log_ori + images_to_log_viz

    if images_to_log:
      self.tf_logger.add_images(
        f'{prefix}_object_predictions',
        torch.stack(images_to_log),
        global_step=epoch,
        dataformats='NCHW'
      )
  
  def _get_patch_from_img(self, img_tensor, condition):
    pos = hz.find_object_center(img_tensor, condition)
    valid_mask = ~torch.isnan(pos).any(dim=1)
    gt_full = torch.full((img_tensor.shape[0],), -1, dtype=torch.long, device=self.device)
    if valid_mask.any():
      valid_pos = pos[valid_mask]
      # The image is 32x32, and the patch grid is 16x16. Patch size is 2x2.
      patch_x = (valid_pos[:, 0] / 2).long()
      patch_y = (valid_pos[:, 1] / 2).long()
      gt = patch_y * 16 + patch_x
      gt_full[valid_mask] = gt
    return gt_full

  def _fill_patch_to_internal_states_mapping(self):
    logger.info("Filling patch_to_internal_states mapping...")
    buffers = [self.train_buffer, self.test_buffer]
    for buffer in buffers:
      if buffer.size == 0:
        continue
      images = buffer.image[:buffer.size]
      internal_states = buffer.internal_state[:buffer.size]

      # Process in batches to avoid large tensors on device
      batch_size = self.config['batch_size']
      for i in tqdm(range(0, len(images), batch_size), leave=False, desc="Filling patch map"):
        batch_images = images[i:i+batch_size].to(self.device)
        batch_internal_states = internal_states[i:i+batch_size].to(self.device)
        
        hand_patch_indices = self._get_patch_from_img(batch_images, self.hand_condition)
        
        valid_mask = hand_patch_indices != -1
        if valid_mask.any():
          valid_indices = hand_patch_indices[valid_mask].cpu().numpy()
          valid_states = batch_internal_states[valid_mask].cpu().numpy()
          
          for patch_idx, state in zip(valid_indices, valid_states):
            # Convert state to a tuple to be able to use it in a set to avoid duplicates
            state_tuple = tuple(state)
            self.patch_to_internal_states[patch_idx].add(state_tuple)

  def _compute_object_prediction_loss_and_gt(self, image, prediction, condition):
    gt_full = self._get_patch_from_img(image, condition)
    loss = self._compute_loss_from_gt(prediction, gt_full)
    return loss, gt_full
  
  def _compute_loss_from_gt(self, pred, gt_full):
    loss = torch.tensor(0.0, device=self.device)
    valid_mask = gt_full != -1
    if valid_mask.any():
      valid_pred = pred[valid_mask]
      valid_gt = gt_full[valid_mask]
      loss = F.cross_entropy(valid_pred, valid_gt)
    return loss

  def _compute_specialized_target_loss(self, buffer, batch_size):
    target_patch_loss = torch.tensor(0.0, device=self.device)
    target_batch = buffer.sample_from_successful_episodes(batch_size, distinct_episodes=True)
    if target_batch:
      # A new forward pass is needed for the specialized batch

      # We only need the target prediction from this forward pass
      _, _, _, _, _, _, _, _, t_target_pred, _ = self.we(
          target_batch['image'], target_batch['action'], target_batch['internal_state']
      )
      
      # Calculate loss only on this specialized batch
      target_patch_loss = self._compute_loss_from_gt(t_target_pred, target_batch['target_patch_gt'])
    return target_patch_loss
  
  def _compute_is_from_patch_accuracies(self, is1_pred_fp, is2_pred_fp, valid_patches):
    is1_pred_argmax = is1_pred_fp.argmax(dim=1)
    is2_pred_argmax = is2_pred_fp.argmax(dim=1)
    
    correct_is1 = 0
    correct_is2 = 0
    n_valid_patches_in_map = 0

    for i, patch_idx in enumerate(valid_patches):
      patch_item = patch_idx.item()
      possible_states = self.patch_to_internal_states.get(patch_item)
      if possible_states:
        n_valid_patches_in_map += 1
        possible_is1s = {s[0] for s in possible_states}
        possible_is2s = {s[1] for s in possible_states}

        if is1_pred_argmax[i].item() in possible_is1s:
          correct_is1 += 1
        if is2_pred_argmax[i].item() in possible_is2s:
          correct_is2 += 1
    
    is1_from_patch_acc = correct_is1 / n_valid_patches_in_map if n_valid_patches_in_map > 0 else 0.
    is2_from_patch_acc = correct_is2 / n_valid_patches_in_map if n_valid_patches_in_map > 0 else 0.
    return is1_from_patch_acc, is2_from_patch_acc
  
  def _compute_is_from_patch_loss_n_accuracies(self, batch, hand_patch_gt_full):
    is_from_patch_loss = torch.tensor(0.0, device=self.device)
    is1_from_patch_acc, is2_from_patch_acc = 0., 0.
    valid_mask = hand_patch_gt_full != -1
    if valid_mask.any():
      is_from_patch_pred = self.we.get_is_from_patch_idx(hand_patch_gt_full[valid_mask])
      is1_pred_fp = is_from_patch_pred[:, :self.config['internal_state_n_values'][0]]
      is2_pred_fp = is_from_patch_pred[:, self.config['internal_state_n_values'][0]:]
      
      is1_gt_fp = batch['internal_state'][valid_mask][:, 0].long()
      is2_gt_fp = batch['internal_state'][valid_mask][:, 1].long()

      loss1 = F.cross_entropy(is1_pred_fp, is1_gt_fp)
      loss2 = F.cross_entropy(is2_pred_fp, is2_gt_fp)
      is_from_patch_loss = loss1 + loss2

      is1_from_patch_acc, is2_from_patch_acc = self._compute_is_from_patch_accuracies(
        is1_pred_fp, is2_pred_fp, hand_patch_gt_full[valid_mask])
    return is_from_patch_loss, is1_from_patch_acc, is2_from_patch_acc

  def train_we(self, epoch, n_steps=10):
    '''
      * Image Reconstruction
      * Next Embedding Prediction
      * Internal State Prediction
      * Next Internal State Prediction
      * Hand & Target Prediction
    '''
    self.we.train()

    self._reset_logs()
    batch_losses = []
    for step in tqdm(range(n_steps), leave=False):
      batch = self.train_buffer.sample(self.config['batch_size'], distinct_episodes=True)

      image = batch['image']

      _, rec, next_emb, isp1, isp2, nisp1, nisp2, hand_pred, target_pred, next_hand_pred = self.we(
        image, batch['action'], batch['internal_state'])

      # === Reconstruction Loss ===
      rec_loss = F.mse_loss(rec, image)

      # === Next Embedding Prediction Loss ===
      with torch.no_grad():
        _, target_emb = self.we.ae(batch['next_image'], return_latent=True)
      nep_loss = F.mse_loss(next_emb, target_emb.detach())

      # === Internal State Prediction Loss ===
      is1_loss = F.cross_entropy(isp1, batch['internal_state'][:, 0].long())
      is2_loss = F.cross_entropy(isp2, batch['internal_state'][:, 1].long())

      # === Next Internal State Prediction Loss ===
      nis1_loss = F.cross_entropy(nisp1, batch['next_internal_state'][:, 0].long())
      nis2_loss = F.cross_entropy(nisp2, batch['next_internal_state'][:, 1].long())

      # === Hand and Target Prediction Loss ===
      hand_patch_loss, hand_patch_gt_full = self._compute_object_prediction_loss_and_gt(image, hand_pred, self.hand_condition)
      next_hand_patch_loss, next_hand_patch_gt_full = self._compute_object_prediction_loss_and_gt(
        batch['next_image'], next_hand_pred, self.hand_condition
      )
      target_patch_loss = self._compute_specialized_target_loss(self.train_buffer, self.config['batch_size'])
      target_patch_gt_full = batch['target_patch_gt']

      # === Internal State from Patch Loss ===
      is_from_patch_loss, is1_from_patch_acc, is2_from_patch_acc = self._compute_is_from_patch_loss_n_accuracies(batch, hand_patch_gt_full)

      loss = rec_loss + nep_loss + is1_loss + is2_loss + nis1_loss + nis2_loss + hand_patch_loss + target_patch_loss + next_hand_patch_loss + is_from_patch_loss
      self.we_optimizer.zero_grad()
      loss.backward()
      self.we_optimizer.step()

      step_losses = {
        'rec_loss': rec_loss, 'nep_loss': nep_loss, 'is1_loss': is1_loss, 'is2_loss': is2_loss,
        'nis1_loss': nis1_loss, 'nis2_loss': nis2_loss, 'hand_patch_loss': hand_patch_loss,
        'target_patch_loss': target_patch_loss, 'next_hand_patch_loss': next_hand_patch_loss,
        'is_from_patch_loss': is_from_patch_loss,
      }
      is1_acc, is2_acc, nis1_acc, nis2_acc, hand_patch_acc, target_patch_acc, next_hand_patch_acc = self.compute_metrics(
        batch, isp1, isp2, nisp1, nisp2, hand_pred, target_pred, hand_patch_gt_full, target_patch_gt_full,
        next_hand_pred, next_hand_patch_gt_full
      )
      step_accuracies = {
        'is1_acc': is1_acc, 'is2_acc': is2_acc, 'nis1_acc': nis1_acc, 'nis2_acc': nis2_acc,
        'hand_patch_acc': hand_patch_acc, 'target_patch_acc': target_patch_acc,
        'next_hand_patch_acc': next_hand_patch_acc, 'is1_from_patch_acc': is1_from_patch_acc,
        'is2_from_patch_acc': is2_from_patch_acc
      }
      self._update_logs(step_losses, step_accuracies)

      batch_losses.append(loss.item())

    # === Log losses & metrics ===
    self.log_metrics(epoch, image, rec, hand_pred, target_pred, next_hand_pred, hand_patch_gt_full, target_patch_gt_full, next_hand_patch_gt_full, prefix='train')

    return np.mean(batch_losses)

  @torch.no_grad()
  def eval_we(self, epoch, n_steps=5):
    self.we.eval()

    self._reset_logs()
    batch_losses = []
    for step in tqdm(range(n_steps), leave=False):
      batch = self.test_buffer.sample(self.config['n_test_episodes'], distinct_episodes=True)

      image = batch['image']

      _, rec, next_emb, isp1, isp2, nisp1, nisp2, hand_pred, target_pred, next_hand_pred = self.we(
        image, batch['action'], batch['internal_state'])

      # === Reconstruction Loss ===
      rec_loss = F.mse_loss(rec, image)

      # === Next Embedding Prediction Loss ===
      _, target_emb = self.we.ae(batch['next_image'], return_latent=True)
      nep_loss = F.mse_loss(next_emb, target_emb.detach())

      # === Internal State Prediction Loss ===
      is1_loss = F.cross_entropy(isp1, batch['internal_state'][:, 0].long())
      is2_loss = F.cross_entropy(isp2, batch['internal_state'][:, 1].long())

      # === Next Internal State Prediction Loss ===
      nis1_loss = F.cross_entropy(nisp1, batch['next_internal_state'][:, 0].long())
      nis2_loss = F.cross_entropy(nisp2, batch['next_internal_state'][:, 1].long())

      # === Hand and Target Prediction Loss ===
      hand_patch_loss, hand_patch_gt_full = self._compute_object_prediction_loss_and_gt(
        image, hand_pred, self.hand_condition)
      next_hand_patch_loss, next_hand_patch_gt_full = self._compute_object_prediction_loss_and_gt(
        batch['next_image'], next_hand_pred, self.hand_condition
      )
      target_patch_loss = self._compute_specialized_target_loss(self.test_buffer, self.config['n_test_episodes'])
      target_patch_gt_full = batch['target_patch_gt']
      
      # === Internal State from Patch Loss ===
      is_from_patch_loss, is1_from_patch_acc, is2_from_patch_acc = self._compute_is_from_patch_loss_n_accuracies(batch, hand_patch_gt_full)

      loss = rec_loss + nep_loss + is1_loss + is2_loss + nis1_loss + nis2_loss + hand_patch_loss + target_patch_loss + next_hand_patch_loss + is_from_patch_loss
      batch_losses.append(loss.item())

      step_losses = {
        'rec_loss': rec_loss, 'nep_loss': nep_loss, 'is1_loss': is1_loss, 'is2_loss': is2_loss,
        'nis1_loss': nis1_loss, 'nis2_loss': nis2_loss, 'hand_patch_loss': hand_patch_loss,
        'target_patch_loss': target_patch_loss, 'next_hand_patch_loss': next_hand_patch_loss,
        'is_from_patch_loss': is_from_patch_loss,
      }
      is1_acc, is2_acc, nis1_acc, nis2_acc, hand_patch_acc, target_patch_acc, next_hand_patch_acc = self.compute_metrics(
        batch, isp1, isp2, nisp1, nisp2, hand_pred, target_pred, hand_patch_gt_full, target_patch_gt_full,
        next_hand_pred, next_hand_patch_gt_full
      )
      step_accuracies = {
        'is1_acc': is1_acc, 'is2_acc': is2_acc, 'nis1_acc': nis1_acc, 'nis2_acc': nis2_acc,
        'hand_patch_acc': hand_patch_acc, 'target_patch_acc': target_patch_acc,
        'next_hand_patch_acc': next_hand_patch_acc, 'is1_from_patch_acc': is1_from_patch_acc,
        'is2_from_patch_acc': is2_from_patch_acc
      }
      self._update_logs(step_losses, step_accuracies)

    # === Log losses & metrics ===
    self.log_metrics(epoch, image, rec, hand_pred, target_pred, next_hand_pred, hand_patch_gt_full, target_patch_gt_full, next_hand_patch_gt_full, prefix='test')

    return np.mean(batch_losses)
  
  def get_goal_is_emb(self, target_patch_idx):
    goal_is_logits = self.we.get_is_from_patch_idx(target_patch_idx)
    goal_is1_idx = goal_is_logits[:, :self.config['internal_state_n_values'][0]].argmax(-1)
    goal_is2_idx = goal_is_logits[:, self.config['internal_state_n_values'][0]:].argmax(-1)
    goal_is1_emb = self.we.is1_emb(goal_is1_idx)
    goal_is2_emb = self.we.is2_emb(goal_is2_idx)
    return torch.stack([goal_is1_idx, goal_is2_idx], dim=1), torch.cat([goal_is1_emb, goal_is2_emb], dim=-1)

  @torch.no_grad()
  def evaluate_policy_in_imagination(self):
    self.actor.eval();self.we.eval()

    # --- Get starting point from buffer ---
    batch = self.test_buffer.get_first_states()

    # --- Retrieves embeddings, Get Goal Internal State ---
    is1_emb = self.we.is1_emb(batch['internal_state'][:, 0])
    is2_emb = self.we.is2_emb(batch['internal_state'][:, 1])

    goal_is_idx, goal_is_emb = self.get_goal_is_emb(batch['target_patch_gt'])

    # --- Run Episodes ---
    batch_size = batch['image'].shape[0]
    active_episodes = torch.ones(batch_size, dtype=torch.bool, device=self.device)
    steps_to_success = torch.full((batch_size,), 0, dtype=torch.long, device=self.device)

    for step in tqdm(range(self.config['max_ep_len']), leave=False):
      if not active_episodes.any():
        break

      current_is_emb = torch.cat([is1_emb, is2_emb], dim=-1)
      # ---- Policy forward pass π(a | s, g) ----
      action, _, _ = self.actor.sample(current_is_emb, goal_is_emb)  # [B], [B], [B]

      # --- Get next states --- next_is1, next_is2
      action_emb = self.we.action_emb(action)  # [B, 8]

      nisp1, nisp2 = self.we.isp.get_next_is(is1_emb, is2_emb, action_emb)  # [B, 19], [B, 37]
      nisp1_idx, nisp2_idx = nisp1.argmax(-1), nisp2.argmax(-1)
      is1_emb = self.we.is1_emb(nisp1_idx)  # next_is1_emb
      is2_emb = self.we.is2_emb(nisp2_idx)  # next_is2_emb

      # --- Termination criteria ---
      reached = (torch.stack([nisp1_idx, nisp2_idx], dim=1) == goal_is_idx).all(dim=1)
      
      newly_reached = active_episodes & reached
      if newly_reached.any():
        steps_to_success[newly_reached] = step + 1
      
      active_episodes.logical_and_(~reached)

    total_successes = (steps_to_success > 0).sum().item()
    success_rate = total_successes / batch_size if batch_size > 0 else 0.0
    avg_steps = steps_to_success[steps_to_success > 0].float().mean().item() if total_successes > 0 else 0.0

    self.actor.train();self.we.train()
    return success_rate, avg_steps
  
  @torch.no_grad()
  def evaluate_policy(self, n_episodes=10):
    self.actor.eval()

    total_successes = 0
    total_steps_to_success = []
    total_target_reached = 0
    total_steps_to_reached_target = []

    for _ in tqdm(range(n_episodes), leave=False):
      obs, _ = self.env.reset()
      img = self.env.render()
      internal_state, _, image, _, _, _, _ = self.train_buffer.prepare_data(obs//5, None, img)

      emb, is1_emb, is2_emb, patchs, patchs_enriched, scaled_embs, hand_pred = self.we.get_state_representation(
        image.unsqueeze(0).to(self.device), internal_state.unsqueeze(0).to(self.device))
      target_pred = self.we.find_target_patch(patchs_enriched).squeeze(-1).argmax(-1)  # [1]
      
      goal_is_idx, goal_is_emb = self.get_goal_is_emb(target_pred)

      for step in range(self.config['max_ep_len']):
        current_is_emb = torch.cat([is1_emb, is2_emb], dim=-1)
        action_logits = self.actor(current_is_emb, goal_is_emb)  # [1, 5]
        action = torch.distributions.Categorical(logits=action_logits).sample()    # [1]

        next_obs, reward, terminated, truncated, info = self.env.step(action.item())
        next_img = self.env.render()

        if terminated:
          total_successes += 1
          total_steps_to_success.append(step)
          total_target_reached += 1
          total_steps_to_reached_target.append(step)
          break

        internal_state, _, _, _, _, _, _ = self.train_buffer.prepare_data(next_obs//5, None, next_img)
        internal_state = internal_state.to(self.device).unsqueeze(0)

        if (internal_state == goal_is_idx).all():
          total_target_reached += 1
          total_steps_to_reached_target.append(step)
          break

        is1_emb = self.we.is1_emb(internal_state[:, 0])
        is2_emb = self.we.is2_emb(internal_state[:, 1])
    
    success_rate = total_successes / n_episodes
    avg_steps = np.mean(total_steps_to_success) if total_steps_to_success else 0.0
    success_rate_predtarget = total_target_reached / n_episodes
    avg_steps_predtarget = np.mean(total_steps_to_reached_target) if total_steps_to_reached_target else 0.0

    self.actor.train()
    return success_rate, avg_steps, success_rate_predtarget, avg_steps_predtarget
  
  def ppo_train_actor(self, n_epochs=500):
    self.we.eval()
    self.actor.train(); self.critic.train()
    self._reset_logs()

    pbar = tqdm(range(n_epochs), desc='Phase 2 (PPO)')
    for epoch in pbar:
      traj = defaultdict(list)
      
      # --- Get starting point from buffer ---
      batch = self.train_buffer.get_first_states()

      # --- Get target patch position
      with torch.no_grad():
        # --- Retrieves embeddings, Get hand and target position ---
        # emb: [B, 256], is1_emb: [B, 16], patchs_enriched: [B, 256, 64], hand_pred: [B]
        emb, is1_emb, is2_emb, patchs, patchs_enriched, scaled_embs, hand_pred = self.we.get_state_representation(
          batch['image'], batch['internal_state'])
        target_pred = self.we.find_target_patch(patchs_enriched).squeeze(-1).argmax(-1)  # [B]
        # replace target value where true value available
        target_pred = torch.where(batch['target_patch_gt'] != -1, batch['target_patch_gt'], target_pred)

        current_is_emb = torch.cat([is1_emb, is2_emb], dim=-1)
        goal_is_idx, goal_is_emb = self.get_goal_is_emb(target_pred)

        # --- Collect Episodes ---
        active_episodes = torch.ones(batch['image'].shape[0], dtype=torch.bool, device=self.device)
        for _ in tqdm(range(self.config['max_ep_len']), leave=False):
          # ---- Policy forward pass π(a | s, g) ----
          action, log_prob, entropy = self.actor.sample(current_is_emb, goal_is_emb)  # [B], [B], [B]

          # --- Goal-conditioned value estimate V(s, g) ---
          value = self.critic(current_is_emb, goal_is_emb)

          # --- Get next states --- next_is1, next_is2
          action_emb = self.we.action_emb(action)  # [B, 8]

          nisp1, nisp2 = self.we.isp.get_next_is(is1_emb, is2_emb, action_emb)  # [B, 19], [B, 37]
          nisp1_idx, nisp2_idx = nisp1.argmax(-1), nisp2.argmax(-1)
          is1_emb = self.we.is1_emb(nisp1_idx)  # next_is1_emb
          is2_emb = self.we.is2_emb(nisp2_idx)  # next_is2_emb

          # --- Termination criteria, reward ---
          reached = (goal_is_idx == torch.stack([nisp1_idx, nisp2_idx], dim=1)).all(dim=1)
          # The reward is 0 if the goal is reached, and -0.1 otherwise (sparse reward).
          reward = torch.where(reached, 0.0, -0.1)
          # The mask is 0 when the episode is terminal (goal reached), and 1 otherwise.
          mask = (~reached).float() * active_episodes.float()
          active_episodes.logical_and_(~reached)

          # --- Save step ---
          traj['states'].append(current_is_emb)
          traj['goals'].append(goal_is_emb)
          traj['actions'].append(action)
          traj['rewards'].append(reward)
          traj['log_probs'].append(log_prob)
          traj['values'].append(value)
          traj['masks'].append(mask)

          current_is_emb = torch.cat([is1_emb, is2_emb], dim=-1)

          if not active_episodes.any(): break
      
      # Bootstrap value V(s_T, g) for GAE
      with torch.no_grad():
        final_val = self.critic(current_is_emb, goal_is_emb)

      policy_loss, value_loss = hz.ppo_update(
        self.actor, self.critic, self.ppo_actor_optimizer, self.critic_optimizer, traj, final_val
      )
      
      if (epoch + 1) % 10 == 0:
        success_rate_imagination, avg_steps_imagination = self.evaluate_policy_in_imagination()
        success_rate, avg_steps, success_rate_predtarget, avg_steps_predtarget = self.evaluate_policy()

        if self.tf_logger:
          self.tf_logger.add_scalar('ppo_policy_loss', policy_loss, epoch)
          self.tf_logger.add_scalar('ppo_value_loss', value_loss, epoch)
          self.tf_logger.add_scalar('ppo_eval_success_rate', success_rate, epoch)
          self.tf_logger.add_scalar('ppo_eval_avg_steps', avg_steps, epoch)
          self.tf_logger.add_scalar('ppo_eval_success_rate_predtarget', success_rate_predtarget, epoch)
          self.tf_logger.add_scalar('ppo_eval_avg_steps_predtarget_to_success', avg_steps_predtarget, epoch)
          self.tf_logger.add_scalar('ppo_eval_success_rate_imagination', success_rate_imagination, epoch)
          self.tf_logger.add_scalar('ppo_eval_avg_steps_to_success_imagination', avg_steps_imagination, epoch)

        descr = f'PPO: p_loss={policy_loss:.4f}, v_loss={value_loss:.4f}, succ={success_rate:.2f}, steps={avg_steps:.2f}'
        descr += f' | sr_target={success_rate_predtarget:.2f} | steps={avg_steps_predtarget:.2f}'
        descr += f' | sr_i={success_rate_imagination:.2f} | steps={avg_steps_imagination:.2f}'
        pbar.set_description(descr)

  def train(self):
    # === PHASE 1 ===
    self.fill_memory(self.train_buffer, n_episodes=self.config['n_train_episodes'])
    self.fill_memory(self.test_buffer, n_episodes=self.config['n_test_episodes'], act='best')

    self._fill_patch_to_internal_states_mapping()

    pbar = tqdm(range(1000), desc='Phase 1')
    eval_loss = 0.0
    best_loss = float('inf')
    patience = 0
    save_at_epoch = 0
    for epoch in pbar:
      train_loss = self.train_we(epoch)

      if (epoch + 1) % 10 == 0:
        eval_loss = self.eval_we(epoch)

        if eval_loss < best_loss:
          self.save_models(os.path.join(self.save_dir, f"{self.config['exp_name']}.pt"))
          best_loss = eval_loss
          patience = 0
          save_at_epoch = epoch
        else:
          patience += 1
        
        if patience > 20:
          logger.info(f'The validation loss did not improve over the last 200 epochs -> training stopped')
          break
      
      pbar.set_description(f'Phase 1: {train_loss=:.4f} - {eval_loss=:.4f} - {save_at_epoch=}')

    # === PHASE 2 ===
    self.ppo_train_actor()


if __name__ == '__main__':
  mm = MasterMind()
  mm.train()
  # TODO use RNN for next embedding prediction and add multistep training