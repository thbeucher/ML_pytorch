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
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

import models_zoo as mz
import helpers_zoo as hz

from replay_buffer import ReplayBuffer


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# get access to robot environments
sys.path.append('../../../robot/')
warnings.filterwarnings("ignore")


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
    # === Internal State Predictor ===
    self.isp_net = mz.get_linear_net(self.config['emb_dim'], self.config['emb_dim'], self.config['emb_dim'])
    self.isp_is1_head = nn.Linear(self.config['emb_dim'], self.config['is1_n_values'])
    self.isp_is2_head = mz.get_linear_net(self.config['emb_dim'], self.config['emb_dim'] // 2, self.config['is2_n_values'])
    # === Object Predictors ===
    self.pos_emb = nn.Embedding(self.config['n_patches'], self.config['token_dim'])
    self.is_scaler = nn.Linear(self.config['is_emb_dim'], self.config['token_dim'])
    self.emb_scaler = nn.Linear(self.config['emb_dim'], self.config['token_dim'])
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
  
  def _get_next_internal_state(self, internal_state, action):
    next_internal_state = internal_state.clone().long()
    action = action.squeeze(-1)

    # action 1: increase is1
    next_internal_state[action == 1, 0] += 1
    # action 2: decrease is1
    next_internal_state[action == 2, 0] -= 1
    # action 3: increase is2
    next_internal_state[action == 3, 1] += 1
    # action 4: decrease is2
    next_internal_state[action == 4, 1] -= 1
    
    is1_max = self.config['is1_n_values'] - 1
    is2_max = self.config['is2_n_values'] - 1

    next_internal_state[:, 0] = torch.clamp(next_internal_state[:, 0], 0, is1_max)
    next_internal_state[:, 1] = torch.clamp(next_internal_state[:, 1], 0, is2_max)

    return next_internal_state

  def scale_embs(self, emb, is1_emb, is2_emb):
    emb_scale = self.emb_scaler(emb).unsqueeze(1)         # [B, 128] -> [B, 1, 64]
    is1_emb_scale = self.is_scaler(is1_emb).unsqueeze(1)  # [B, 16]  -> [B, 1, 64]
    is2_emb_scale = self.is_scaler(is2_emb).unsqueeze(1)

    return [emb_scale, is1_emb_scale, is2_emb_scale]

  def _get_base_representation(self, image, internal_state):
    rec, (d1, d2, d3), emb = self.ae(image, return_all=True)
    emb = self.emb_enricher(emb.view(-1, 4, self.config['token_dim'])).view(-1, 256)

    is1_emb = self.is1_emb(internal_state[:, 0])
    is2_emb = self.is2_emb(internal_state[:, 1])

    # C1 = d1 = [B, 64, 16, 16] -> [B, 16*16, 64]
    patchs = d1.flatten(2).transpose(1, 2) + self.pos_emb(torch.arange(0, self.config['n_patches'], device=d1.device))
    return rec, d1, emb, is1_emb, is2_emb, patchs
  
  def get_target_prediction(self, image, internal_state):
    _, _, emb, is1_emb, is2_emb, patchs = self._get_base_representation(image, internal_state)
    scaled_embs = self.scale_embs(emb, is1_emb, is2_emb)
    patchs_enriched = self.object_enricher(torch.cat([patchs] + scaled_embs, dim=1))[:, :-len(scaled_embs)]
    target_pred = self.find_target_patch(patchs_enriched).squeeze(-1)
    return target_pred

  def forward(self, image, action, internal_state):
    # --- Encode, Embed ---
    action_emb = self.action_emb(action).squeeze(1)
    rec, _, emb, is1_emb, is2_emb, patchs = self._get_base_representation(image, internal_state)
    
    emb_action = torch.cat([emb, action_emb], dim=-1)         # -> [B, 256+16]

    # --- Predict Next Embedding ---
    next_emb = self.nep(emb_action)

    # --- Predict Internal State ---
    x_isp = self.isp_net(emb)
    isp1, isp2 = self.isp_is1_head(x_isp), self.isp_is2_head(x_isp)

    # --- Predict Object Position - object = hand & target ---
    scaled_embs = self.scale_embs(emb, is1_emb, is2_emb)
    patchs_enriched = self.object_enricher(torch.cat([patchs] + scaled_embs, dim=1))[:, :-len(scaled_embs)]

    hand_pred = self.find_hand_patch(patchs_enriched).squeeze(-1)
    target_pred = self.find_target_patch(patchs_enriched).squeeze(-1)

    # Predict next hand position
    with torch.no_grad():
      next_internal_state = self._get_next_internal_state(internal_state, action)
      next_is1_emb = self.is1_emb(next_internal_state[:, 0])
      next_is2_emb = self.is2_emb(next_internal_state[:, 1])

    scaled_embs_next = self.scale_embs(next_emb, next_is1_emb, next_is2_emb)
    next_patchs_enriched = self.object_enricher(torch.cat([patchs] + scaled_embs_next, dim=1))
    next_hand_pred = self.find_hand_patch(next_patchs_enriched[:, :-len(scaled_embs_next)]).squeeze(-1)
      
    return emb, rec, next_emb, isp1, isp2, hand_pred, target_pred, next_hand_pred
  

class MasterMind:
  CONFIG = {
    'save_dir':                          'experiments/',
    'exp_name':                          'master_mind_cleaned',
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
    # ISWM (InternalStateWorlModel) - ISP
    # OPP (ObjectPositionPredictor)
    # TP (TargetPredictor)
    self.we = WorldEncoder(config=self.config).to(self.device)
    logger.info(f'WorldEncoder instanciate with n_params={self.get_train_params(self.we):,}')
  
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
      'we_optimizer_state_dict': self.we_optimizer.state_dict(),
    }, path)

  def load_models(self, path):
    """Loads the state of all models and optimizers."""
    if not os.path.exists(path):
      logger.warning(f"Model checkpoint not found at {path}. Starting from scratch.")
      return
    logger.info(f"Loading models from {path}")
    checkpoint = torch.load(path, map_location=self.device)
    self.we.load_state_dict(checkpoint['we_state_dict'])
    self.we_optimizer.load_state_dict(checkpoint['we_optimizer_state_dict'])
  
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
  
  def compute_metrics(self, batch, isp1, isp2, hand_pred, target_pred, hand_patch_gt, target_patch_gt, next_hand_pred, next_hand_patch_gt):
    is1_acc = (isp1.argmax(dim=1) == batch['internal_state'][:, 0]).float().mean().item()
    is2_acc = (isp2.argmax(dim=1) == batch['internal_state'][:, 1]).float().mean().item()

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

    return is1_acc, is2_acc, hand_patch_acc, target_patch_acc, next_hand_patch_acc
  
  def _reset_logs(self):
    self.losses = defaultdict(list)
    self.accuracies = defaultdict(list)
    self.actor_metrics = defaultdict(list)

  def _update_logs(self, step_losses, step_accuracies):
    for name, value in step_losses.items():
      self.losses[name].append(value.item())
    for name, value in step_accuracies.items():
      self.accuracies[name].append(value)

  def draw_patch(self, img, patch_idx, color):
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
  
  def log_metrics(self, epoch, image, rec, hand_pred, target_pred, hand_patch_gt, target_patch_gt, prefix='train'):
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
    n_samples = 8
    images_to_log_ori = []
    images_to_log_viz = []
    
    hand_preds_indices = hand_pred[:n_samples].argmax(dim=1)
    target_preds_indices = target_pred[:n_samples].argmax(dim=1)

    colors = {
      'hand_gt': (0, 0, 1),        # Blue
      'hand_pred': (0, 1, 1),       # Cyan
      'target_gt': (0, 1, 0),      # Green
      'target_pred': (1, 1, 0),     # Yellow
    }

    for i in range(n_samples):
      img_orig = (image[i] * 0.5) + 0.5 # De-normalize to [0, 1]
      
      img_viz = img_orig.clone()
      # Draw hand GT (blue) and Pred (cyan)
      img_viz = self.draw_patch(img_viz, hand_patch_gt[i], colors['hand_gt'])
      img_viz = self.draw_patch(img_viz, hand_preds_indices[i], colors['hand_pred'])
      # Draw target GT (red) and Pred (yellow)
      img_viz = self.draw_patch(img_viz, target_patch_gt[i], colors['target_gt'])
      img_viz = self.draw_patch(img_viz, target_preds_indices[i], colors['target_pred'])
      
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
    gt = torch.full((img_tensor.shape[0],), -1, dtype=torch.long, device=self.device)
    if valid_mask.any():
      valid_pos = pos[valid_mask]
      # The image is 32x32, and the patch grid is 16x16. Patch size is 2x2.
      patch_x = (valid_pos[:, 0] / 2).long()
      patch_y = (valid_pos[:, 1] / 2).long()
      patch_indices = patch_y * 16 + patch_x
      gt[valid_mask] = patch_indices
    return gt

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
    gt = self._get_patch_from_img(image, condition)
    loss = self._compute_loss_from_gt(prediction, gt)
    return loss, gt
  
  def _compute_loss_from_gt(self, pred, gt):
    loss = torch.tensor(0.0, device=self.device)
    valid_mask = gt != -1
    if valid_mask.any():
      valid_pred = pred[valid_mask]
      valid_gt = gt[valid_mask]
      loss = F.cross_entropy(valid_pred, valid_gt)
    return loss

  def _compute_specialized_target_loss(self, buffer, batch_size):
    target_patch_loss = torch.tensor(0.0, device=self.device)
    target_batch = buffer.sample_from_successful_episodes(batch_size, distinct_episodes=True)
    if target_batch:
      # A new forward pass is needed for the specialized batch
      # We only need the target prediction from this forward pass
      *_, t_target_pred, _ = self.we(target_batch['image'], target_batch['action'], target_batch['internal_state'])
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
  
  def _compute_is_from_patch_loss_n_accuracies(self, batch, hand_patch_gt):
    is_from_patch_loss = torch.tensor(0.0, device=self.device)
    is1_from_patch_acc, is2_from_patch_acc = 0., 0.
    valid_mask = hand_patch_gt != -1
    if valid_mask.any():
      is_from_patch_pred = self.we.get_is_from_patch_idx(hand_patch_gt[valid_mask])
      is1_pred_fp = is_from_patch_pred[:, :self.config['internal_state_n_values'][0]]
      is2_pred_fp = is_from_patch_pred[:, self.config['internal_state_n_values'][0]:]
      
      is1_gt_fp = batch['internal_state'][valid_mask][:, 0].long()
      is2_gt_fp = batch['internal_state'][valid_mask][:, 1].long()

      loss1 = F.cross_entropy(is1_pred_fp, is1_gt_fp)
      loss2 = F.cross_entropy(is2_pred_fp, is2_gt_fp)
      is_from_patch_loss = loss1 + loss2

      is1_from_patch_acc, is2_from_patch_acc = self._compute_is_from_patch_accuracies(
        is1_pred_fp, is2_pred_fp, hand_patch_gt[valid_mask])
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

      _, rec, next_emb, isp1, isp2, hand_pred, target_pred, next_hand_pred = self.we(
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

      # === Hand and Target Prediction Loss ===
      hand_patch_loss, hand_patch_gt = self._compute_object_prediction_loss_and_gt(image, hand_pred, self.hand_condition)
      next_hand_patch_loss, next_hand_patch_gt = self._compute_object_prediction_loss_and_gt(
        batch['next_image'], next_hand_pred, self.hand_condition
      )
      target_patch_loss = self._compute_specialized_target_loss(self.train_buffer, self.config['batch_size'])
      target_patch_gt = batch['target_patch_gt']

      # === Internal State from Patch Loss ===
      is_fp_loss, is1_fp_acc, is2_fp_acc = self._compute_is_from_patch_loss_n_accuracies(batch, hand_patch_gt)

      loss = rec_loss + nep_loss + is1_loss + is2_loss + hand_patch_loss + target_patch_loss + next_hand_patch_loss + is_fp_loss
      self.we_optimizer.zero_grad()
      loss.backward()
      self.we_optimizer.step()

      step_losses = {
        'rec_loss': rec_loss, 'nep_loss': nep_loss, 'is1_loss': is1_loss, 'is2_loss': is2_loss,
        'hand_patch_loss': hand_patch_loss, 'target_patch_loss': target_patch_loss,
        'next_hand_patch_loss': next_hand_patch_loss, 'is_from_patch_loss': is_fp_loss,
      }
      is1_acc, is2_acc, hand_patch_acc, target_patch_acc, next_hand_patch_acc = self.compute_metrics(
        batch, isp1, isp2, hand_pred, target_pred, hand_patch_gt, target_patch_gt,
        next_hand_pred, next_hand_patch_gt
      )
      step_accuracies = {
        'is1_acc': is1_acc, 'is2_acc': is2_acc, 'hand_patch_acc': hand_patch_acc,
        'target_patch_acc': target_patch_acc, 'next_hand_patch_acc': next_hand_patch_acc,
        'is1_from_patch_acc': is1_fp_acc, 'is2_from_patch_acc': is2_fp_acc
      }
      self._update_logs(step_losses, step_accuracies)

      batch_losses.append(loss.item())

    # === Log losses & metrics ===
    self.log_metrics(epoch, image, rec, hand_pred, target_pred, hand_patch_gt, target_patch_gt, prefix='train')

    return np.mean(batch_losses)

  @torch.no_grad()
  def eval_we(self, epoch):
    self.we.eval()

    self._reset_logs()
    batch = self.test_buffer.sample(self.config['n_test_episodes'], distinct_episodes=True)

    image = batch['image']

    _, rec, next_emb, isp1, isp2, hand_pred, target_pred, next_hand_pred = self.we(
      image, batch['action'], batch['internal_state'])

    # === Reconstruction Loss ===
    rec_loss = F.mse_loss(rec, image)

    # === Next Embedding Prediction Loss ===
    _, target_emb = self.we.ae(batch['next_image'], return_latent=True)
    nep_loss = F.mse_loss(next_emb, target_emb.detach())

    # === Internal State Prediction Loss ===
    is1_loss = F.cross_entropy(isp1, batch['internal_state'][:, 0].long())
    is2_loss = F.cross_entropy(isp2, batch['internal_state'][:, 1].long())

    # === Hand and Target Prediction Loss ===
    hand_patch_loss, hand_patch_gt = self._compute_object_prediction_loss_and_gt(
      image, hand_pred, self.hand_condition)
    next_hand_patch_loss, next_hand_patch_gt = self._compute_object_prediction_loss_and_gt(
      batch['next_image'], next_hand_pred, self.hand_condition
    )
    target_patch_loss = self._compute_specialized_target_loss(self.test_buffer, self.config['n_test_episodes'])
    target_patch_gt = batch['target_patch_gt']
    
    # === Internal State from Patch Loss ===
    is_fp_loss, is1_fp_acc, is2_fp_acc = self._compute_is_from_patch_loss_n_accuracies(batch, hand_patch_gt)

    loss = rec_loss + nep_loss + is1_loss + is2_loss + hand_patch_loss + target_patch_loss + next_hand_patch_loss + is_fp_loss

    step_losses = {
      'rec_loss': rec_loss, 'nep_loss': nep_loss, 'is1_loss': is1_loss, 'is2_loss': is2_loss,
      'hand_patch_loss': hand_patch_loss, 'target_patch_loss': target_patch_loss,
      'next_hand_patch_loss': next_hand_patch_loss, 'is_from_patch_loss': is_fp_loss,
    }
    is1_acc, is2_acc, hand_patch_acc, target_patch_acc, next_hand_patch_acc = self.compute_metrics(
      batch, isp1, isp2, hand_pred, target_pred, hand_patch_gt, target_patch_gt,
      next_hand_pred, next_hand_patch_gt
    )
    step_accuracies = {
      'is1_acc': is1_acc, 'is2_acc': is2_acc, 'hand_patch_acc': hand_patch_acc,
      'target_patch_acc': target_patch_acc, 'next_hand_patch_acc': next_hand_patch_acc,
      'is1_from_patch_acc': is1_fp_acc, 'is2_from_patch_acc': is2_fp_acc
    }
    self._update_logs(step_losses, step_accuracies)

    # === Log losses & metrics ===
    self.log_metrics(epoch, image, rec, hand_pred, target_pred, hand_patch_gt, target_patch_gt, prefix='test')

    return loss.item()
    
  def get_actions_to_goal(self, current_is, goal_is):
    """
    Calculates the sequence of actions to get from the current internal state to the goal internal state.

    Args:
      current_is (torch.Tensor): The current internal state [is1, is2].
      goal_is (torch.Tensor): The goal internal state [is1, is2].

    Returns:
      list: A list of actions to perform.
    """
    actions = []
    
    # --- Actions for is1 ---
    is1_diff = goal_is[0] - current_is[0]
    if is1_diff > 0:
      actions.extend([1] * is1_diff)  # Action 1: increase is1
    elif is1_diff < 0:
      actions.extend([2] * abs(is1_diff))  # Action 2: decrease is1
      
    # --- Actions for is2 ---
    is2_diff = goal_is[1] - current_is[1]
    if is2_diff > 0:
      actions.extend([3] * is2_diff)  # Action 3: increase is2
    elif is2_diff < 0:
      actions.extend([4] * abs(is2_diff))  # Action 4: decrease is2
      
    return actions
  
  def perform_actions_sequence(self, env, actions):
    distances = []
    for action in actions:
      obs, reward, terminated, truncated, info = env.step(action)
      distances.append(round(info['distance_to_target']))
      if terminated:
        break
    return terminated, obs, distances
  
  @torch.no_grad()
  def evaluate_target_predictor(self, n_episodes=10):
    self.we.eval()

    total_successes = 0
    for ep in tqdm(range(n_episodes), leave=False):
      obs, _ = self.env.reset()
      img = self.env.render()
      internal_state, _, image, _, _, _, _ = self.train_buffer.prepare_data(obs//5, None, img)
      image, internal_state = image.to(self.device), internal_state.to(self.device)
      # --- Predict Target Patch Index ---
      target_patch = self.we.get_target_prediction(image.unsqueeze(0), internal_state.unsqueeze(0))
      top_patches = target_patch.topk(3)[-1].squeeze(0)

      for patch_idx in top_patches:
        # --- Predict Goal Internal State from Target Patch Index ---
        goal_is_logits = self.we.get_is_from_patch_idx(patch_idx.unsqueeze(0))
        goal_is1_idxs = goal_is_logits[:, :self.config['internal_state_n_values'][0]].topk(3)[-1].squeeze(0)
        goal_is2_idxs = goal_is_logits[:, self.config['internal_state_n_values'][0]:].topk(3)[-1].squeeze(0)

        for goal_is1_idx, goal_is2_idx in zip(goal_is1_idxs, goal_is2_idxs):
          # --- Get actions sequence ---
          actions = self.get_actions_to_goal(internal_state, torch.stack([goal_is1_idx, goal_is2_idx]))
          # --- Perform actions ---
          terminated, obs, distances = self.perform_actions_sequence(self.env, actions)
          if terminated:
            total_successes += 1
            break
        if terminated:
          break
      if not terminated:
        print(f'{distances=}')
        print(f'Top 5 predicted target patches: {top_patches.tolist()}')

        # Visualization
        img_to_show = (image.squeeze(0).cpu() * 0.5) + 0.5  # De-normalize and move to CPU
        
        # Define colors for top predictions
        colors = [(1, 1, 0), (0, 1, 0), (0.9, 0.6, 0), (0.7, 0.1, 0.1), (0.8, 0.2, 0)]  # Yellow, Green, Orange, Dark Red, Reddish

        img_viz = img_to_show.clone()
        for i, patch_idx in enumerate(top_patches):
          img_viz = self.draw_patch(img_viz, patch_idx.item(), colors[i])

        plt.imshow(img_viz.permute(1, 2, 0).numpy())
        plt.title(f"Episode {ep+1}: Target Prediction (Not Terminated)")
        plt.show()
    print(f'total_successes: {total_successes}/{n_episodes}')

    self.we.train()

  def save_episode_gif(self, replay_buffer, filename, episode_index=None):
    """
    Retrieves a full episode from the replay buffer and saves it as a GIF.

    Args:
        replay_buffer (ReplayBuffer): The replay buffer to sample from.
        filename (str): The path to save the GIF file.
        episode_index (int, optional): The specific episode ID to retrieve. 
                                       If None, a random episode is sampled. Defaults to None.
    """
    if replay_buffer.size == 0:
      logger.warning("Replay buffer is empty. Cannot save episode GIF.")
      return

    episode_ids_tensor = None
    if episode_index is not None:
      episode_ids_tensor = torch.tensor([episode_index], device=replay_buffer.device)
    
    # Sample a single, full-length episode
    # We use max_ep_len to ensure we get the whole episode, which will be padded if shorter
    batch = replay_buffer.sample_episode_batch(
      batch_size=1,
      episode_length=self.config['max_ep_len'],
      random_window=False, # Get the last part of the episode if it's too long
      episode_ids=episode_ids_tensor
    )

    # Get the actual length of the sampled episode
    episode_len = batch['episode_size'][0].item()

    if episode_len == 0:
      logger.warning("Sampled an empty episode. Cannot save GIF.")
      return

    # Get all images from the episode trajectory
    # Shape: [T, C, H, W]
    episode_images = batch['image'][0, :episode_len]

    # Get the final 'next_image' of the last state in the episode
    # Shape: [C, H, W]
    last_next_image = batch['next_image'][0, episode_len - 1]

    # Combine the sequence of images with the final next_image
    all_images = list(episode_images) + [last_next_image]

    # De-normalize images if they were normalized for the buffer
    if self.config['normalize_image']:
      all_images = [(img * 0.5) + 0.5 for img in all_images]

    # Create and save the GIF
    hz.create_gif_from_images(all_images, filename, duration=150)
    logger.info(f"Saved episode GIF to {filename}")

  def train(self):
    # === PHASE 1 ===
    logger.info('Phase 1: Train World Encoder-Model')
    self.fill_memory(self.train_buffer, n_episodes=self.config['n_train_episodes'])
    logger.info(f'Successful episodes: {len(self.train_buffer.successful_episodes)}/{self.train_buffer.current_episode_id}')
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
          save_at_epoch = epoch + 1
        else:
          patience += 1
        
        if patience > 20:
          logger.info(f'The validation loss did not improve over the last 200 epochs -> training stopped')
          break
      
      pbar.set_description(f'Phase 1: {train_loss=:.4f} - {eval_loss=:.4f} - {save_at_epoch=}')

    # === PHASE 2 ===
    self.evaluate_target_predictor()


if __name__ == '__main__':
  mm = MasterMind()
  mm.train()
  
  # --- Save a GIF of a random episode from the test buffer ---
  gif_save_path = os.path.join(mm.save_dir, "test_episode.gif")
  mm.save_episode_gif(mm.test_buffer, gif_save_path)
  # TODO use RNN for next embedding prediction and add multistep training
