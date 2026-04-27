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
    # self.is2_head = nn.Linear(self.config['hidden_dim'], self.config['is2_n_values'])
    self.is2_head = mz.get_linear_net(self.config['hidden_dim'], self.config['hidden_dim']//2, self.config['is2_n_values'])

    enriched_dim_size = self.config['hidden_dim'] + 2 * self.config['is_emb_dim'] + self.config['action_emb']
    self.next_net = mz.get_linear_net(enriched_dim_size, 2*self.config['hidden_dim'], self.config['hidden_dim'])
    self.nis1_head = nn.Linear(self.config['hidden_dim'], self.config['is1_n_values'])
    # self.nis2_head = nn.Linear(self.config['hidden_dim'], self.config['is2_n_values'])
    self.nis2_head = mz.get_linear_net(self.config['hidden_dim'], self.config['hidden_dim']*2, self.config['is2_n_values'])

  def forward(self, emb, action_emb, is1_emb, is2_emb):
    x = self.net(emb)  # [B, 128] -> [B, 256]

    is1_logits = self.is1_head(x)
    is2_logits = self.is2_head(x)

    # x_enriched = torch.cat([x, action_emb, is1_emb, is2_emb], dim=-1)
    x_enriched = self.next_net(torch.cat([x, action_emb, is1_emb, is2_emb], dim=-1))

    nis1_logits = self.nis1_head(x_enriched)
    nis2_logits = self.nis2_head(x_enriched)

    return is1_logits, is2_logits, nis1_logits, nis2_logits


class WorldEncoder(nn.Module):
  CONFIG = {
    'emb_dim':    256,
    'action_emb': 16,
    'is_emb_dim': 32,    
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
    # === CNN Auto-Encoder ===
    self.ae = mz.CNNAE(self.config['ae_config'])
    self.emb_enricher = mz.Transformer(dim=64, depth=2, heads=4, dim_head=32, mlp_dim=128, dropout=0.0)
    # === Next Embedding Predictor ===
    self.nep = mz.get_linear_net(
      self.config['emb_dim'] + self.config['action_emb'],
      2 * self.config['emb_dim'],
      self.config['emb_dim']
    )
    # === (Next) Internal State Predictor ===
    self.isp = InternalStatePredictor(config=self.config)
    # === Object Predictors ===
    self.pos_emb = nn.Embedding(16*16, 64)
    self.is_scaler = nn.Linear(self.config['is_emb_dim'], 64)
    self.emb_scaler = nn.Linear(self.config['emb_dim'], 64)
    # C1 = d1 = [B, 64, 16, 16] -> [B, 16*16, 64]
    self.object_enricher = mz.Transformer(dim=64, depth=2, heads=4, dim_head=32, mlp_dim=128, dropout=0.0)
    self.find_hand_patch = nn.Sequential(
      nn.Linear(64, 2 * self.config['emb_dim']),
      nn.ReLU(True),
      nn.Linear(2 * self.config['emb_dim'], 1)
    )
    self.find_target_patch = nn.Sequential(
      nn.Linear(64, 2 * self.config['emb_dim']),
      nn.ReLU(True),
      nn.Linear(2 * self.config['emb_dim'], 1)
    )
  
  def forward(self, image, action_emb, is1_emb, is2_emb):
    # 1) Embed image and reconstruct it
    rec, (d1, d2, d3), emb = self.ae(image, return_all=True)  # [B, 3, 32, 32] -> [B, 256], ([B, 64, 16, 16])
    # Enrich embedding
    emb = self.emb_enricher(emb.view(-1, 4, 64)).view(-1, 256)
    emb_action = torch.cat([emb, action_emb], dim=-1)         # -> [B, 256+16]

    # 2) Predict Next Embedding
    next_emb = self.nep(emb_action)

    # 3) Predict Internal State and Next Internal State
    isp1, isp2, nisp1, nisp2 = self.isp(emb, action_emb, is1_emb, is2_emb)

    # 4) Predict Object Position - object = hand & target
    # C1 = d1 = [B, 64, 16, 16] -> [B, 16*16, 64]
    patchs = d1.flatten(2).transpose(1, 2) + self.pos_emb(torch.arange(0, 256, device=d1.device))

    emb_scale = self.emb_scaler(emb).unsqueeze(1)         # [B, 128] -> [B, 1, 64]
    is1_emb_scale = self.is_scaler(is1_emb).unsqueeze(1)  # [B, 16]  -> [B, 1, 64]
    is2_emb_scale = self.is_scaler(is2_emb).unsqueeze(1)

    patchs = self.object_enricher(torch.cat([patchs, emb_scale, is1_emb_scale, is2_emb_scale], dim=1))[:, :-3]

    hand_pred = self.find_hand_patch(patchs)
    target_pred = self.find_target_patch(patchs)

    return rec, next_emb, isp1, isp2, nisp1, nisp2, hand_pred, target_pred



class MasterMind:
  CONFIG = {
    'save_dir':                          'experiments/',
    'exp_name':                          'master_mind_exp1_emb-n-isa_enricher_deeperIS2p2_handtarget',
    'use_tf_logger':                     True,
    # === Replay Buffer & Models info ===
    'replay_buffer_device':              'cpu',   # Store on CPU to save GPU memory
    'image_size':                        256,     # Original environment image size (RGB)
    'resize_to':                         32,      # Resize to for memory efficiency (256 → 32)
    'normalize_image':                   True,    # Normalize to [-1, 1]
    'internal_state_dim':                2,       # Robot state: [angle_joint1, angle_joint2]
    'internal_state_n_values':           (90//5+1, 180//5+1),  # max_angle / angle_step +1 for inclusive
    'action_dim':                        1,       # Single action per step
    'action_n_values':                   5,       # 5 possible actions (0-4)
    'action_emb':                        8,       # Action embedding dimension
    'n_train_episodes':                  128,
    'n_test_episodes':                   10,
    'max_ep_len':                        60,
    'is1_n_values':                      19,  # number of discrete values for internal state 1 (angle joint1)
    'is2_n_values':                      37,  # number of discrete values for internal state 2 (angle joint2)
    'is_emb_dim':                        16,  # int: intermediate embedding dimension for internal states
    # === Training hyperparameters ===
    'batch_size':                        128,
  }
  def __init__(self, config={}):
    self.config = {**MasterMind.CONFIG, **config}
    self.device = torch.device('cuda' if torch.cuda.is_available() else
                               'mps' if torch.backends.mps.is_available() else
                               'cpu')
    logger.info(f'Using device: {self.device}')
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
    # GPA (GoalPolicyActor)
    self.instanciate_models()
    # Replay Buffer - train / test - Optimizers
    self.set_utils()
  
  def set_env(self, render_mode='rgb_array'):
    self.env = gym.make(
      "gymnasium_env:RobotArmEnv",
      render_mode=render_mode,
      cropping=True
    )
  
  def instanciate_models(self):
    # === ACTION EMBEDDING ===
    # Converts discrete action ID (0-4) → action_dim → dim embedding
    # Discrete action becomes continuous representation that can be added to other embeddings
    self.action_emb = nn.Sequential(
      nn.Embedding(self.config['action_n_values'], self.config['action_emb']),  # [B, 1] → [B, 1, action_dim]
      nn.Linear(self.config['action_emb'], self.config['action_emb']),          # [B, 1, action_dim] → [B, 1, dim]
      nn.SiLU()
    ).to(self.device)
    # === INTERNAL STATE EMBEDDING ===
    # Embed internal state (joint angles): converts discrete bin index to continuous
    self.is1_emb = nn.Sequential(
      nn.Embedding(self.config['is1_n_values'], self.config['is_emb_dim']),  # is1_n_values = 19 (0-18 inclusive)
      nn.Linear(self.config['is_emb_dim'], self.config['is_emb_dim']),
      nn.SiLU()
    ).to(self.device)
    self.is2_emb = nn.Sequential(
      nn.Embedding(self.config['is2_n_values'], self.config['is_emb_dim']),  # is2_n_values = 37 (0-36 inclusive)
      nn.Linear(self.config['is_emb_dim'], self.config['is_emb_dim']),
      nn.SiLU()
    ).to(self.device)
    # === WORLD ENCODER ===
    # CNN Auto-Encoder with Residual Connection and Linear Bottleneck
    # NEP (NextEmbeddingPredictor)
    # ISWM (InternalStateWorlModel) - ISP - NISP
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
    # self.brain_optimizer = torch.optim.AdamW([
    #   {'params': self.action_emb.parameters(), 'lr': 1e-4},
    #   {'params': self.is1_emb.parameters(), 'lr': 1e-4},
    #   {'params': self.is2_emb.parameters(), 'lr': 1e-4},
    #   {'params': self.we.parameters(), 'lr': 1e-4},
    # ])
    self.brain_optimizer = torch.optim.AdamW(
      [p for m in [self.action_emb, self.is1_emb, self.is2_emb, self.we] for p in m.parameters()],
      lr=1e-4
    )
  
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
  
  def compute_metrics(self, batch, isp1, isp2, nisp1, nisp2, hand_pred, target_pred, hand_patch_gt, target_patch_gt):
    is1_acc = (isp1.argmax(dim=1) == batch['internal_state'][:, 0]).float().mean().item()
    is2_acc = (isp2.argmax(dim=1) == batch['internal_state'][:, 1]).float().mean().item()
    nis1_acc = (nisp1.argmax(dim=1) == batch['next_internal_state'][:, 0]).float().mean().item()
    nis2_acc = (nisp2.argmax(dim=1) == batch['next_internal_state'][:, 1]).float().mean().item()

    hand_patch_acc = 0.0
    valid_hand_mask = hand_patch_gt != -1
    if valid_hand_mask.any():
        hand_patch_acc = (hand_pred[valid_hand_mask].squeeze(-1).argmax(dim=1) == hand_patch_gt[valid_hand_mask]).float().mean().item()

    target_patch_acc = 0.0
    valid_target_mask = target_patch_gt != -1
    if valid_target_mask.any():
        target_patch_acc = (target_pred[valid_target_mask].squeeze(-1).argmax(dim=1) == target_patch_gt[valid_target_mask]).float().mean().item()

    return is1_acc, is2_acc, nis1_acc, nis2_acc, hand_patch_acc, target_patch_acc

  def log_metrics(self, epoch, image, rec, hand_pred, target_pred, hand_patch_gt_full, target_patch_gt_full, prefix='train'):
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
    
    hand_preds_indices = hand_pred[:n_samples].squeeze(-1).argmax(dim=1)
    target_preds_indices = target_pred[:n_samples].squeeze(-1).argmax(dim=1)

    colors = {
        'hand_gt': (0, 0, 1),    # Blue
        'hand_pred': (0, 1, 1),   # Cyan
        'target_gt': (1, 0, 0),  # Red
        'target_pred': (1, 1, 0), # Yellow
    }

    for i in range(n_samples):
        img_orig = (image[i] * 0.5) + 0.5 # De-normalize to [0, 1]
        
        img_viz = img_orig.clone()
        # Draw hand GT (blue) and Pred (cyan)
        img_viz = draw_patch(img_viz, hand_patch_gt_full[i], colors['hand_gt'])
        img_viz = draw_patch(img_viz, hand_preds_indices[i], colors['hand_pred'])
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

  def _compute_object_prediction_loss_and_gt(self, image, prediction, condition):
    pos = hz.find_object_center(image, condition)
    valid_mask = ~torch.isnan(pos).any(dim=1)

    loss = torch.tensor(0.0, device=self.device)
    gt_full = torch.full((image.shape[0],), -1, dtype=torch.long, device=self.device)

    if valid_mask.any():
        valid_pos = pos[valid_mask]
        
        # The image is 32x32, and the patch grid is 16x16. Patch size is 2x2.
        patch_x = (valid_pos[:, 0] / 2).long()
        patch_y = (valid_pos[:, 1] / 2).long()
        gt = patch_y * 16 + patch_x
        gt_full[valid_mask] = gt
        
        valid_pred = prediction[valid_mask].squeeze(-1)
        loss = F.cross_entropy(valid_pred, gt)
        
    return loss, gt_full

  def _reset_logs(self):
    self.losses = defaultdict(list)
    self.accuracies = defaultdict(list)

  def _update_logs(self, step_losses, step_accuracies):
    for name, value in step_losses.items():
        self.losses[name].append(value.item())
    for name, value in step_accuracies.items():
        self.accuracies[name].append(value)

  def train_step(self, epoch, n_steps=10):
    self.action_emb.train();self.is1_emb.train();self.is2_emb.train();self.we.train()

    hand_condition = lambda frames: (frames[:, 2, :, :] > frames[:, 0, :, :]) & (frames[:, 2, :, :] > frames[:, 1, :, :]) & (frames[:, 2, :, :] > 0.1)
    target_condition = lambda frames: (frames[:, 0, :, :] > frames[:, 1, :, :]) & (frames[:, 0, :, :] > frames[:, 2, :, :]) & (frames[:, 0, :, :] > 0.1)

    self._reset_logs()
    batch_losses = []
    for step in tqdm(range(n_steps), leave=False):
      batch = self.train_buffer.sample(self.config['batch_size'], distinct_episodes=True)

      image = batch['image']
      action_emb = self.action_emb(batch['action']).squeeze(1)  # -> [B, 1, 8]
      is1_emb = self.is1_emb(batch['internal_state'][:, 0])     # -> [B, 16]
      is2_emb = self.is2_emb(batch['internal_state'][:, 1])

      rec, next_emb, isp1, isp2, nisp1, nisp2, hand_pred, target_pred = self.we(image, action_emb, is1_emb, is2_emb)

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
      hand_patch_loss, hand_patch_gt_full = self._compute_object_prediction_loss_and_gt(image, hand_pred, hand_condition)
      target_patch_loss, target_patch_gt_full = self._compute_object_prediction_loss_and_gt(image, target_pred, target_condition)

      loss = rec_loss + nep_loss + is1_loss + is2_loss + nis1_loss + nis2_loss + hand_patch_loss + target_patch_loss
      self.brain_optimizer.zero_grad()
      loss.backward()
      self.brain_optimizer.step()

      step_losses = {
          'rec_loss': rec_loss, 'nep_loss': nep_loss, 'is1_loss': is1_loss, 'is2_loss': is2_loss,
          'nis1_loss': nis1_loss, 'nis2_loss': nis2_loss, 'hand_patch_loss': hand_patch_loss,
          'target_patch_loss': target_patch_loss
      }
      is1_acc, is2_acc, nis1_acc, nis2_acc, hand_patch_acc, target_patch_acc = self.compute_metrics(
          batch, isp1, isp2, nisp1, nisp2, hand_pred, target_pred, hand_patch_gt_full, target_patch_gt_full
      )
      step_accuracies = {
          'is1_acc': is1_acc, 'is2_acc': is2_acc, 'nis1_acc': nis1_acc, 'nis2_acc': nis2_acc,
          'hand_patch_acc': hand_patch_acc, 'target_patch_acc': target_patch_acc
      }
      self._update_logs(step_losses, step_accuracies)

      batch_losses.append(loss.item())

    # === Log losses & metrics ===
    self.log_metrics(epoch, image, rec, hand_pred, target_pred, hand_patch_gt_full, target_patch_gt_full, prefix='train')

    return np.mean(batch_losses)

  @torch.no_grad()
  def eval_step(self, epoch, n_steps=5):
    self.action_emb.eval();self.is1_emb.eval();self.is2_emb.eval();self.we.eval()

    hand_condition = lambda frames: (frames[:, 2, :, :] > frames[:, 0, :, :]) & (frames[:, 2, :, :] > frames[:, 1, :, :]) & (frames[:, 2, :, :] > 0.1)
    target_condition = lambda frames: (frames[:, 0, :, :] > frames[:, 1, :, :]) & (frames[:, 0, :, :] > frames[:, 2, :, :]) & (frames[:, 0, :, :] > 0.1)

    self._reset_logs()
    batch_losses = []
    for step in tqdm(range(n_steps), leave=False):
      batch = self.test_buffer.sample(self.config['n_test_episodes'], distinct_episodes=True)

      image = batch['image']
      action_emb = self.action_emb(batch['action']).squeeze(1)  # -> [B, 8]
      is1_emb = self.is1_emb(batch['internal_state'][:, 0])     # -> [B, 16]
      is2_emb = self.is2_emb(batch['internal_state'][:, 1])

      rec, next_emb, isp1, isp2, nisp1, nisp2, hand_pred, target_pred = self.we(image, action_emb, is1_emb, is2_emb)

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
        image, hand_pred, hand_condition)
      target_patch_loss, target_patch_gt_full = self._compute_object_prediction_loss_and_gt(
        image, target_pred, target_condition)

      loss = rec_loss + nep_loss + is1_loss + is2_loss + nis1_loss + nis2_loss + hand_patch_loss + target_patch_loss
      batch_losses.append(loss.item())

      step_losses = {
          'rec_loss': rec_loss, 'nep_loss': nep_loss, 'is1_loss': is1_loss, 'is2_loss': is2_loss,
          'nis1_loss': nis1_loss, 'nis2_loss': nis2_loss, 'hand_patch_loss': hand_patch_loss,
          'target_patch_loss': target_patch_loss
      }
      is1_acc, is2_acc, nis1_acc, nis2_acc, hand_patch_acc, target_patch_acc = self.compute_metrics(
          batch, isp1, isp2, nisp1, nisp2, hand_pred, target_pred, hand_patch_gt_full, target_patch_gt_full
      )
      step_accuracies = {
          'is1_acc': is1_acc, 'is2_acc': is2_acc, 'nis1_acc': nis1_acc, 'nis2_acc': nis2_acc,
          'hand_patch_acc': hand_patch_acc, 'target_patch_acc': target_patch_acc
      }
      self._update_logs(step_losses, step_accuracies)

    # === Log losses & metrics ===
    self.log_metrics(epoch, image, rec, hand_pred, target_pred, hand_patch_gt_full, target_patch_gt_full, prefix='test')

    return np.mean(batch_losses)

  def train(self):
    # === PHASE 1 ===
    self.fill_memory(self.train_buffer, n_episodes=self.config['n_train_episodes'])
    self.fill_memory(self.test_buffer, n_episodes=self.config['n_test_episodes'], act='best')

    pbar = tqdm(range(1000), desc='Phase 1')
    eval_loss = 0.0
    for epoch in pbar:
      train_loss = self.train_step(epoch)

      if epoch % 50:
        eval_loss = self.eval_step(epoch)
      
      pbar.set_description(f'Phase 1: {train_loss=:.4f} - {eval_loss=:.4f}')


if __name__ == '__main__':
  mm = MasterMind()
  mm.train()