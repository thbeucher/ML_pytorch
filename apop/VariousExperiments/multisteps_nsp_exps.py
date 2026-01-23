import math
import torch
import pygame
import random
import imageio
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import deque
from pytorch_msssim import SSIM
from torchvision.utils import make_grid

from flow_matching_exps import flow_matching_loss, rk45_sampling
from next_state_predictor_exps import NextStatePredictorTrainer, WorldModelFlowUnet
from models_zoo import CNNAE, get_embedding_block, GoalPolicy, GoalValue, ISPredictor


class NISPredictor(nn.Module):
  def __init__(self, n_actions=5, action_dim=8, is1_n_values=19, is2_n_values=37, is_dim=32, mlp_dim=64):
    super().__init__()
    self.action_emb = get_embedding_block(n_actions, action_dim)
    self.is_emb = get_embedding_block(max(is1_n_values, is2_n_values), is_dim)

    self.main = nn.Sequential(nn.Linear(action_dim + 2*is_dim, mlp_dim), nn.SiLU(True),
                              nn.Linear(mlp_dim, mlp_dim), nn.SiLU(True))

    self.predictor1 = nn.Linear(mlp_dim, is1_n_values)
    self.predictor2 = nn.Linear(mlp_dim, is2_n_values)
  
  def forward(self, action, internal_state):
    if len(action.shape) > 1:
      action = action.squeeze(-1)                             # [B, 1] -> [B]
    action_emb = self.action_emb(action)                      # -> [B, 8]
    is_emb = self.is_emb(internal_state).flatten(1)           # -> [B, 2, 32] -> [B, 64]
    main = self.main(torch.cat([action_emb, is_emb], dim=1))  # [B, 72] -> [B, 64]
    is1 = self.predictor1(main)                               # -> [B, 18]
    is2 = self.predictor2(main)                               # -> [B, 36]
    return is1, is2


class ImageISGoalPredictor(nn.Module):
  """
  Predicts an internal_state_goal from:
    - image: [B, 3, 32, 32]
    - internal_state: [B, 2] (discrete)
  Outputs:
    - goal logits for each internal dimension
  """
  def __init__(self, is1_n_values=19, is2_n_values=37, is_dim=32, hidden_dim=512, mlp_dim=256):
    super().__init__()
    # --------------------------------------------------
    # Image encoder (same spirit as ImageToIS)
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

      nn.Conv2d(256, hidden_dim, 3, padding=1),  # [B, 512, 2, 2]
      nn.BatchNorm2d(hidden_dim),
      nn.SiLU(),
    )

    self.pool = nn.AdaptiveAvgPool2d(1)  # → [B, hidden_dim]

    # --------------------------------------------------
    # Internal state embedding
    # --------------------------------------------------
    self.is_emb = get_embedding_block(
      n_embeddings=max(is1_n_values, is2_n_values),
      embedding_dim=is_dim,
    )

    # --------------------------------------------------
    # Fusion MLP
    # --------------------------------------------------
    fusion_dim = hidden_dim + 4 * is_dim

    self.fusion = nn.Sequential(
      nn.Linear(fusion_dim, mlp_dim),
      nn.SiLU(),
      nn.Linear(mlp_dim, mlp_dim),
      nn.SiLU(),
    )

    # --------------------------------------------------
    # single scalar head (ENERGY / SCORE)
    # --------------------------------------------------
    self.score_head = nn.Linear(mlp_dim, 1)
  
  def forward(self, image, internal_state, internal_state_goal):
    """
    image: [B, 3, 32, 32]
    internal_state: [B, 2]
    """
    # ---- Image features ----
    img_feat = self.image_encoder(image)
    img_feat = self.pool(img_feat).flatten(1)  # [B, hidden_dim]

    # ---- Internal state embedding ----
    is_emb = self.is_emb(internal_state).flatten(1)  # [B, 2*is_dim]
    is_g_emb = self.is_emb(internal_state_goal).flatten(1)

    # ---- Fuse ----
    h = self.fusion(torch.cat([img_feat, is_emb, is_g_emb], dim=1))

    # ---- Generate score ----
    return self.score_head(h).squeeze(-1)  # [B]


class NSPTrainer(NextStatePredictorTrainer):
  '''
  I: image
  a: action
  IS: Internal State
  NIS: NextInternalState
  1) (IS, a) ---internal_world_model---> NIS
  2)    IS   ---world_model------------> Ip
  3)    Ip   ---image_cleaner----------> I
  '''
  CONFIG = {'exp_name':                                 'multisteps_nsp',
            'replay_buffer_size':                       10_000,
            # INTERNAL STATE
            'internal_state_dim':                       2,
            'internal_state_n_values':                  (90//5+1, 180//5+1),  # max_angle / angle_step
            'internal_state_emb':                       32,
            'internal_world_model_max_train_steps':     100,  # 10_000
            'internal_world_model_batch_size':          128,
            # IMAGE CLEANER
            'image_cleaner_max_train_steps':            5,  # 300
            'image_cleaner_batch_size':                 128,
            'lambda_mse':                               0.8,
            'lambda_ssim':                              0.2,
            # WORLD MODEL (Some already setted in NextStatePredictorTrainer)
            'world_model_max_train_steps':              30,  # 3_000
            # INTERNAL STATE GOAL POLICY
            'isgp_n_updates':                           300,  # 5_000
            'isgp_batch_size':                          128,
            'isgp_max_horizon':                         50,
            'isgp_gamma':                               0.99,
            'isgp_eval_every':                          50,
            'isgp_lmbda':                               0.95,  # 0 = 1-step TD, 1 = Monte-Carlo
            'isgp_entropy_coef_start':                  0.01,
            'isgp_entropy_coef_end':                    0.01 / 10,
            'isgp_curriculum_start':                    0.2,
            'isgp_curriculum_end':                      0.8,
            'isgp_her_prob':                            0.2,
            'isgp_use_HER':                             False,  # Hindsight Experience Replay -> relabeling end goal
            # IMAGE GOAL PREDICTOR
            'image_goal_predictor_max_train_steps':     200,
            # INTERNAL STATE PREDICTOR
            'internal_state_predictor_max_train_steps': 100,
            'isgs_max_train_steps':                     1_000,
            }
  def __init__(self, config={}):
    self.config = {**NSPTrainer.CONFIG, **config}
    super().__init__(self.config)
  
  def instanciate_model(self):
    self.world_model = WorldModelFlowUnet(img_chan=self.config['image_chan'],
                                          time_dim=self.config['time_dim'],
                                          add_action=False,
                                          add_is=True,
                                          add_ds=True,
                                          is_n_values=max(self.config['internal_state_n_values']),
                                          is_dim=self.config['internal_state_emb']).to(self.device)
    print(f'Instanciate Worl_Model (trainable parameters: {self.get_train_params(self.world_model):,})')
    self.internal_world_model = NISPredictor(n_actions=self.config['action_n_values'],
                                            action_dim=self.config['action_emb'],
                                            is1_n_values=self.config['internal_state_n_values'][0],
                                            is2_n_values=self.config['internal_state_n_values'][1],
                                            is_dim=self.config['internal_state_emb']).to(self.device)
    print('Instanciate Internal_Worl_Model (trainable parameters:'\
           f'{self.get_train_params(self.internal_world_model):,})')
    self.image_cleaner = CNNAE({'encoder_archi': 'BigCNNEncoder'}).to(self.device)
    print(f'Instanciate ImageCleaner (trainable parameters: {self.get_train_params(self.image_cleaner):,})')
    self.goal_policy = GoalPolicy(a_dim=self.config['action_n_values'],
                                  is_n_values=sum(self.config['internal_state_n_values']),
                                  is_dim=self.config['internal_state_emb']).to(self.device)
    print(f'Instanciate GoalPolicy (trainable parameters: {self.get_train_params(self.goal_policy):,})')
    self.goal_value = GoalValue(is_dim=self.config['internal_state_emb'],
                                is_n_values=sum(self.config['internal_state_n_values'])).to(self.device)
    print(f'Instanciate GoalPolicy (trainable parameters: {self.get_train_params(self.goal_policy):,})')
    self.is_predictor = ISPredictor().to(self.device)
    print(f'Instanciate ISPredictor (trainable parameters: {self.get_train_params(self.is_predictor):,})')
    # self.goal_is_predictor = ImageISPredictor(done_info=True).to(self.device)
    # print(f'Instanciate GoalISPredictor (trainable parameters: {self.get_train_params(self.goal_is_predictor):,})')
    # self.img_goal_predictor = CNNAE({'encoder_archi': 'BigCNNEncoder'}).to(self.device)
    # print(f'Instanciate ImgGoalPredictor (trainable parameters: {self.get_train_params(self.img_goal_predictor):,})')
    # self.is_goal_scorer = ImageISGoalPredictor().to(self.device)
    # print(f'Instanciate ISGoalScorer (trainable parameters: {self.get_train_params(self.is_goal_scorer):,})')
    self.goal_image_predictor = WorldModelFlowUnet(img_chan=self.config['image_chan'],
                                                   time_dim=self.config['time_dim'],
                                                   add_action=False,
                                                   add_is=False,
                                                   add_ds=False,
                                                   x_start=True).to(self.device)
    print(f'Instanciate GoalImagePredictor (trainable parameters: {self.get_train_params(
      self.goal_image_predictor):,})')
    # self.rnn_gisp = ImageISPredictor(rnn_goal_prediction=True).to(self.device)
    # print(f'Instanciate RNNGoalImagePredictor (trainable parameters: {self.get_train_params(self.rnn_gisp):,})')
  
  def set_training_utils(self):
    super().set_training_utils()
    self.mse_criterion = nn.MSELoss()
    self.cross_entropy_criterion = nn.CrossEntropyLoss()
    self.ssim_criterion = SSIM(data_range=1, size_average=True, channel=3)
      
  @torch.no_grad()
  def show_imagined_goal_policy(self):
    self.goal_policy.eval()
    self.internal_world_model.eval()
    self.world_model.eval()

    self.fill_memory(replay_buffer_size=200)

    batch = self.replay_buffer.sample(2)

    start = batch['image'][:1]
    goal = batch['image'][1:]

    traj = torch.zeros(24, 3, 32, 32)
    traj[0] = goal
    traj[1] = start

    is_c = batch['internal_state'][:1]
    is_g = batch['internal_state'][1:]

    for i in range(2, 24):
      # Get action
      a = self.goal_policy(is_c, is_g).argmax(-1)

      # Get next internal_state
      is1_logits, is2_logits = self.internal_world_model(a, is_c)
      is_next = torch.stack([is1_logits.argmax(-1), is2_logits.argmax(-1)], dim=1)

      # Get predicted world image
      condition = {'internal_state': is_next}
      condition['done_signal'] = batch['done'][:1]
      x1_pred = rk45_sampling(
        self.world_model,
        device=self.device,
        n_samples=is_next.shape[0],
        condition=condition,
        n_steps=4
      )
      x1_pred = self.clamp_denorm_fn(x1_pred[-1].detach())

      # Clean the image
      img = self.image_cleaner(x1_pred)
      traj[i] = img.cpu()

      reached = (is_next == is_g).all(dim=1)
      if reached.all():
        print(f'Reached in {i-2} actions')
        break

      is_c = is_next
    
    grid = make_grid(traj, nrow=8, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()

    frames = [
      (frame.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
      for frame in traj
    ]
    imageio.mimsave("sequence.gif", frames, fps=10)
  
  def compute_gae(self, rewards, values, masks, next_value, gamma, lmbda):
    T = rewards.size(0)

    # ---- Generalized Advantage Estimation (GAE) ----
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0

    for t in reversed(range(T)):
      # Bootstrap with V(s_{t+1}, g)
      next_values = next_value if t == T - 1 else values[t + 1]
      # TD error δ_t
      delta = rewards[t] + gamma * next_values * masks[t] - values[t]
      # GAE recursion
      last_gae = delta + gamma * lmbda * masks[t] * last_gae
      advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns

  def reinforce_update_is_goal_networks(self, trajs, next_value, gamma, lmbda, entropy_coef):
    # Convert lists to tensors [T, B]
    rewards = torch.stack(trajs['rewards'])
    values = torch.stack(trajs['values'])
    masks = torch.stack(trajs['masks'])
    log_probs = torch.stack(trajs['log_probs'])
    entropies = torch.stack(trajs['entropies'])

    advantages, returns = self.compute_gae(rewards, values, masks, next_value, gamma, lmbda)

    # Normalize advantages for stability
    advantages = advantages.view(-1)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = advantages.view(returns.shape)

    # --- Policy Loss ---
    loss_policy = -(log_probs * advantages.detach()).mean() - entropy_coef * entropies.mean()

    # --- Value Loss ---
    loss_value = self.mse_criterion(values, returns.detach())

    # Update
    self.opt_policy.zero_grad()
    loss_policy.backward()
    self.opt_policy.step()

    self.opt_value.zero_grad()
    loss_value.backward()
    self.opt_value.step()

    return loss_policy.item(), loss_value.item()

  def ppo_update_is_goal_networks(self, traj, next_value, gamma=0.99, lmbda=0.95, clip_eps=0.2,
                                  value_coef=0.5, entropy_coef=0.01, n_epochs=4, max_grad_norm=0.5):
    """
    Perform PPO updates using a collected rollout.

    PPO key ideas implemented here:
    - GAE advantage estimation
    - Clipped policy objective
    - Clipped value loss
    - Multiple epochs over the same rollout
    """
    # ---- Stack rollout tensors: shape [T, B] ----
    rewards = torch.stack(traj['rewards'])
    values = torch.stack(traj['values'])
    masks = torch.stack(traj['masks'])

    advantages, returns = self.compute_gae(rewards, values, masks, next_value, gamma, lmbda)

    # ---- Flatten time and batch dimensions ----
    states = torch.cat(traj['states'])
    goals = torch.cat(traj['goals'])
    actions = torch.cat(traj['actions'])
    old_log_probs = torch.cat(traj['log_probs']).detach()
    old_values = torch.cat(traj['values']).detach()

    advantages = advantages.view(-1)
    returns = returns.view(-1)

    # --- Normalize advantages ---
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # --- PPO epochs ---
    for _ in range(n_epochs):
      # Recompute policy π(a | s, g) with CURRENT parameters
      logits = self.goal_policy(states, goals)
      dist = torch.distributions.Categorical(logits=logits)

      new_log_probs = dist.log_prob(actions)
      entropy = dist.entropy().mean()

      # Recompute value V(s, g)
      new_values = self.goal_value(states, goals).squeeze(-1)

      # ---- PPO policy loss (clipped surrogate) ----
      ratio = torch.exp(new_log_probs - old_log_probs)
      surr1 = ratio * advantages
      surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
      policy_loss = -torch.min(surr1, surr2).mean()

      # ---- PPO value loss (clipped) ----
      value_clipped = old_values + \
        (new_values - old_values).clamp(-clip_eps, clip_eps)

      value_loss = torch.max(
        (new_values - returns).pow(2),
        (value_clipped - returns).pow(2)
      ).mean()

      loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

      self.opt_policy.zero_grad()
      self.opt_value.zero_grad()
      loss.backward()
      # Gradient clipping prevents large destructive updates
      torch.nn.utils.clip_grad_norm_(
        list(self.goal_policy.parameters()) +
        list(self.goal_value.parameters()),
        max_grad_norm
      )
      self.opt_policy.step()
      self.opt_value.step()

    return policy_loss.item(), value_loss.item()
  
  def long_goal_fraction(self, mean_success_rate, linear_ramp=False):
    """
    Maps success rate to fraction of batch that uses longer (obtain by permutation across batch) goals.
    """
    start = self.config['isgp_curriculum_start']   # e.g. 0.2
    end   = self.config['isgp_curriculum_end']     # e.g. 0.8

    if mean_success_rate <= start:
        return 0.0
    if mean_success_rate >= end:
        return 1.0

    goal_fraction = (mean_success_rate - start) / (end - start)

    if linear_ramp:
      # Linear ramp (can swap for sigmoid)
      return goal_fraction
    
    # Exponential ramp
    alpha = self.config.get('isgp_curriculum_exp_alpha', 2.0)
    return float(1.0 - math.exp(-alpha * goal_fraction))
    
  def collect_is_goal_rollout(self, batch, mean_success_rate, max_horizon):
    """
    Collect an on-policy rollout using the current goal-conditioned policy.

    IMPORTANT:
    - This function must NOT keep computation graphs.
    - PPO replays data, not graphs.
    """
    # Current internal state s_t
    is_c = batch['internal_state']  # internal_state_current [B, 2]
    is_g = batch['next_internal_state'].clone()

    # Fraction of long goals
    p_long = self.long_goal_fraction(mean_success_rate)

    if p_long > 0.0:
      B = is_c.size(0)

      # Which batch elements get long goals
      long_mask = torch.rand(B, device=is_c.device) < p_long

      if long_mask.any():
        perm = torch.randperm(B, device=is_c.device)
        is_g[long_mask] = is_g[perm][long_mask]

    # Buffers for trajectory
    trajs = {'states': [], 'goals': [], 'actions': [], 'log_probs': [], 'values': [],
             'rewards': [], 'masks': [], 'entropies': []}

    for t in range(max_horizon):
      # ---- Policy forward pass π(a | s, g) ----
      a, log_p, ent = self.goal_policy.sample(is_c, is_g)

      # Goal-conditioned value estimate V(s, g)
      val = self.goal_value(is_c, is_g).squeeze(-1)

      # Internal World model is frozen and used only to simulate transitions
      with torch.no_grad():
        # Predict next state
        is1_logits, is2_logits = self.internal_world_model(a, is_c)
        # Discrete next internal state
        is_next = torch.stack([is1_logits.argmax(-1), is2_logits.argmax(-1)], dim=1)
        
        reached = (is_next == is_g).all(dim=1)
        reward = torch.where(reached, 0.0, -0.1)
        # Mask = 0 when terminal (goal reached), 1 otherwise
        mask = (~reached).float()

      # Detaching is CRITICAL: PPO must not reuse graphs
      trajs['states'].append(is_c.detach())
      trajs['goals'].append(is_g.detach())
      trajs['actions'].append(a.detach())
      trajs['log_probs'].append(log_p.detach())
      trajs['values'].append(val.detach())
      trajs['rewards'].append(reward.detach())
      trajs['masks'].append(mask.detach())
      trajs['entropies'].append(ent.detach())

      is_c = is_next.detach()
      if reached.all(): break

    # Bootstrap value V(s_T, g) for GAE
    with torch.no_grad():
      final_val = self.goal_value(is_next, is_g).squeeze(-1).detach()
  
    return trajs, final_val, reached
  
  def train_is_goal_policy(self):
    print('Training Internal State Goal policy...')
    self.goal_policy.train()
    self.goal_value.train()

    # Freeze internal state world model
    self.internal_world_model.eval()
    for p in self.internal_world_model.parameters():
      p.requires_grad = False
    
    # instanciate optimizers
    self.opt_policy = torch.optim.Adam(self.goal_policy.parameters(), lr=1e-4)
    self.opt_value = torch.optim.Adam(self.goal_value.parameters(), lr=1e-4)

    # training loop
    self.long_goal = False
    entropy_coef = self.config['isgp_entropy_coef_start']
    best_success_rate = -torch.inf
    mean_policy_loss = 0.0
    mean_value_loss = 0.0
    mean_success_rate = 0.0
    mean_success_rate_window = deque(maxlen=100)
    pbar = tqdm(range(self.config['isgp_n_updates']))
    for step in pbar:
      batch = self.replay_buffer.sample(self.config['isgp_batch_size'])

      # Collect ON-POLICY rollout
      trajs, final_val, reached = self.collect_is_goal_rollout(batch,
                                                               mean_success_rate,
                                                              #  self.config['isgp_mean_success_rate_curriculum'],
                                                               self.config['isgp_max_horizon'])
      
      # --- 2. Update Networks ---
      # loss_policy, loss_value = self.update_is_goal_networks(trajs, final_val, gamma, lmbda, entropy_coef)
      loss_policy, loss_value = self.ppo_update_is_goal_networks(trajs,
                                                                 final_val,
                                                                 gamma=self.config['isgp_gamma'],
                                                                 lmbda=self.config['isgp_lmbda'],
                                                                 clip_eps=0.2,
                                                                 value_coef=0.5,
                                                                 entropy_coef=entropy_coef,
                                                                 n_epochs=4)

      # Gradual decay
      entropy_coef = max(self.config['isgp_entropy_coef_end'],
                         self.config['isgp_entropy_coef_start'] * (1 - step / self.config['isgp_n_updates']))

      mean_policy_loss += (loss_policy - mean_policy_loss) / (step + 1)
      mean_value_loss += (loss_value - mean_value_loss) / (step + 1)
      # mean_success_rate += (reached.float().mean() - mean_success_rate) / (step + 1)
      mean_success_rate_window.append(reached.float().mean().item())
      mean_success_rate = sum(mean_success_rate_window) / len(mean_success_rate_window)
    
      if mean_success_rate > best_success_rate:
        self.save_model(self.goal_policy, 'goal_policy')
        self.save_model(self.goal_value, 'goal_value')
        best_success_rate = mean_success_rate
      
      if self.tf_logger is not None:
        self.tf_logger.add_scalar('policy_loss', mean_policy_loss, step)
        self.tf_logger.add_scalar('value_loss', mean_value_loss, step)
        self.tf_logger.add_scalar('success_rate', mean_success_rate, step)
      
      if mean_success_rate > mean_success_rate:
        mean_success_rate = mean_success_rate
        self.save_model(self.goal_policy, 'goal_policy')
        self.save_model(self.goal_value, 'goal_value')
      
      descr = f"P_Loss: {loss_policy:.3f} | V_Loss: {loss_value:.3f} | SuccessRate: {mean_success_rate:.2f}"
      pbar.set_description(descr)
  
  def train_image_cleaner(self, image_generator):
    print('Training ImageCleaner...')
    self.image_cleaner.train()
    image_generator.eval()

    self.ic_opt = torch.optim.AdamW(self.image_cleaner.parameters(), lr=1e-3)

    best_loss = torch.inf
    mean_rec_loss = 0.0
    mean_mse_loss = 0.0
    mean_ssim_loss = 0.0
    pbar = tqdm(range(self.config['image_cleaner_max_train_steps']))
    for step in pbar:
      batch = self.replay_buffer.sample(self.config['image_cleaner_batch_size'])
      
      # --------------------------------------------------------------------------- #
      # STEP1: retrieves the predicted image of the world model from internal state #
      # --------------------------------------------------------------------------- #
      with torch.no_grad():
        condition = {'internal_state': batch['internal_state'],
                     'done_signal': batch['done'],
                     'x_cond': batch['image']}
        x1_pred = rk45_sampling(
          image_generator,
          device=self.device,
          n_samples=batch['image'].shape[0],
          condition=condition,
          n_steps=4
        )
        x1_pred = self.clamp_denorm_fn(x1_pred[-1].detach())

      # -------------------------------- #
      # STEP2: Clean the predicted image #
      # -------------------------------- #
      rec = self.image_cleaner(x1_pred)
      target = self.clamp_denorm_fn(batch['image'])

      mse_loss = self.mse_criterion(rec, target)
      ssim_loss = 1 - self.ssim_criterion(rec, target)
      rec_loss = self.config['lambda_mse'] * mse_loss + self.config['lambda_ssim'] * ssim_loss

      self.ic_opt.zero_grad()
      rec_loss.backward()
      self.ic_opt.step()

      mean_mse_loss += (mse_loss.item() - mean_mse_loss) / (step + 1)
      mean_ssim_loss += (ssim_loss.item() - mean_ssim_loss) / (step + 1)
      mean_rec_loss += (rec_loss.item() - mean_rec_loss) / (step + 1)

      if self.tf_logger is not None:
        self.tf_logger.add_scalar('ic_reconstruction_loss', mean_rec_loss, step)
        self.tf_logger.add_scalar('ic_mse_loss', mean_mse_loss, step)
        self.tf_logger.add_scalar('ic_ssim_loss', mean_ssim_loss, step)

        if step % 10 == 0:
          ori_fm_rec = torch.cat([batch['image'][:8], x1_pred[:8], rec[:8]], dim=0)
          self.tf_logger.add_images(f'image_cleaned', ori_fm_rec, global_step=step//10)
      
      if mean_rec_loss < best_loss:
        best_loss = mean_rec_loss
        self.save_model(self.image_cleaner, 'image_cleaner')

      pbar.set_description(f'Loss: {mean_rec_loss:.4f}')
  
  def train_world_model(self):
    print('Training world model...')
    self.world_model.train()

    best_loss = torch.inf
    mean_loss = 0.0

    pbar = tqdm(range(self.config['world_model_max_train_steps']))
    for step in pbar:
      batch = self.replay_buffer.sample(self.config['world_model_batch_size'])
      x1 = batch['image']  # target distribution is the image to predict
      condition = {'internal_state': batch['internal_state'],
                   'done_signal': batch['done']}

      # ------------------------------------------------------------------------------------------ #
      # STEP1: Train the world_model to produce a corresponding image with provided internal_state #
      # ------------------------------------------------------------------------------------------ #
      # Moving random (gaussian) distribution to x1 distribution based on provided internal state  #
      loss = flow_matching_loss(self.world_model, x1, condition=condition,
                                weighted_time_sampling=self.config['weighted_time_sampling'],
                                noise_scale=self.config['noise_scale'])

      self.wm_optimizer.zero_grad()
      loss.backward()
      self.wm_optimizer.step()

      mean_loss += (loss.item() - mean_loss) / (step + 1)

      if mean_loss < best_loss:
        self.save_model(self.world_model, 'world_model')
        best_loss = mean_loss
      
      if self.tf_logger:
        self.tf_logger.add_scalar('worlmodel_projection_fm_loss', mean_loss, step)
        
        if step % self.config['world_model_check_pred_loss_every'] == 0:
          x1_pred = rk45_sampling(
            self.world_model,
            device=self.device,
            n_samples=x1.shape[0],
            condition=condition,
            n_steps=4
          )
          x1_pred = x1_pred[-1].clamp(-1, 1)

          self.tf_logger.add_scalar('world_model_projection_pred_loss',
                                    torch.nn.functional.mse_loss(x1_pred, x1),
                                    step // self.config['world_model_check_pred_loss_every'])
          self.tf_logger.add_images('generated_world_model_projection_prediction',
                                    torch.cat([x1[:8], x1_pred[:8]], dim=0),
                                    global_step=step)

      pbar.set_description(f'Loss: {mean_loss:.6f}')
  
  def train_internal_world_model(self):
    print('Training Internal World Model...')
    self.internal_world_model.train()

    self.iwm_opt = torch.optim.AdamW(self.internal_world_model.parameters(), lr=1e-2)

    best_loss = torch.inf
    mean_loss = 0.0
    mean_acc1, mean_acc2 = 0.0, 0.0

    pbar = tqdm(range(self.config['internal_world_model_max_train_steps']))
    for step in pbar:
      batch = self.replay_buffer.sample(self.config['internal_world_model_batch_size'])

      # ---------------------------------------------------------------------------------- #
      # STEP1: Based on current internal_state and action, predict the next_internal_state #
      # ---------------------------------------------------------------------------------- #
      nis1_pred, nis2_pred = self.internal_world_model(batch['action'], batch['internal_state'])  # [B, 18], [B, 36]

      target = batch['next_internal_state']
      is1_target, is2_target = target[:, 0], target[:, 1]

      loss1 = self.cross_entropy_criterion(nis1_pred, is1_target)
      loss2 = self.cross_entropy_criterion(nis2_pred, is2_target)
      loss = loss1 + loss2

      self.iwm_opt.zero_grad()
      loss.backward()
      self.iwm_opt.step()

      mean_loss += (loss.item() - mean_loss) / (step + 1)
      # Accuracies
      pred1 = nis1_pred.argmax(dim=1)  # [B]
      pred2 = nis2_pred.argmax(dim=1)  # [B]
      acc1 = (pred1 == is1_target).float().mean()
      acc2 = (pred2 == is2_target).float().mean()
      mean_acc1 += (acc1.item() - mean_acc1) / (step + 1)
      mean_acc2 += (acc2.item() - mean_acc2) / (step + 1)

      if mean_loss < best_loss:
        self.save_model(self.internal_world_model, 'internal_world_model')
        best_loss = mean_loss
      
      if self.tf_logger:
        self.tf_logger.add_scalar('iwm_loss', mean_loss, step)
        self.tf_logger.add_scalar('iwm_accuracy1', mean_acc1, step)
        self.tf_logger.add_scalar('iwm_accuracy2', mean_acc2, step)
      
      pbar.set_description(f'Loss: {mean_loss:.6f} | Mean_acc: {(mean_acc1+mean_acc2)/2:.2f}')
  
  def train_is_predictor(self, add_gip=True):
    print('Training Internal State Predictor')
    self.is_predictor.train()
    self.is_predictor_opt = torch.optim.AdamW(self.is_predictor.parameters(), lr=3e-4)

    self.goal_image_predictor.eval()

    best_loss = torch.inf
    mean_loss = 0.0
    mean_acc1, mean_acc2 = 0.0, 0.0
    mean_goal_acc1, mean_goal_acc2 = 0.0, 0.0

    pbar = tqdm(range(self.config['internal_state_predictor_max_train_steps']))
    for step in pbar:
      batch = self.replay_buffer.sample(self.config['image_cleaner_batch_size'])
      goal_batch = self.replay_buffer.sample_image_is_goal_batch(self.config.get('isp_batch_size', 128))

      # Infer internal_state from replay_buffer images
      g1_logits, g2_logits = self.is_predictor(self.clamp_denorm_fn(batch['image']))

      loss = (
        self.cross_entropy_criterion(g1_logits, batch['internal_state'][:, 0]) +
        self.cross_entropy_criterion(g2_logits, batch['internal_state'][:, 1])
      )

      # Infer internal_state from reconstructed goal image
      loss_gip = None
      if add_gip:
        condition = {'x_cond': goal_batch['image']}
        x1_pred = rk45_sampling(
          self.goal_image_predictor,
          device=self.device,
          n_samples=goal_batch['image'].shape[0],
          condition=condition,
          n_steps=2
          )
        with torch.no_grad():
          goal_img = self.image_cleaner(self.clamp_denorm_fn(x1_pred[-1]))
        self.tf_logger.add_images('goal_image', torch.cat([goal_batch['image'][:8],
                                                           goal_batch['goal_image'][:8],
                                                           goal_img[:8]], dim=0), global_step=step)
        pg1_logits, pg2_logits = self.is_predictor(goal_img.detach())
        loss_gip = (
          self.cross_entropy_criterion(pg1_logits, goal_batch['goal_internal_state'][:, 0]) +
          self.cross_entropy_criterion(pg2_logits, goal_batch['goal_internal_state'][:, 1])
        )
      if loss_gip is not None:
        loss = 0.2 * loss + 0.8 * loss_gip

      self.is_predictor_opt.zero_grad()
      loss.backward()
      self.is_predictor_opt.step()

      mean_loss += (loss.item() - mean_loss) / (step + 1)
      # Accuracies
      acc1 = (g1_logits.argmax(dim=1) == batch['internal_state'][:, 0]).float().mean().item()
      acc2 = (g2_logits.argmax(dim=1) == batch['internal_state'][:, 1]).float().mean().item()
      mean_acc1 += (acc1 - mean_acc1) / (step + 1)
      mean_acc2 += (acc2 - mean_acc2) / (step + 1)

      goal_acc1 = (pg1_logits.argmax(dim=1) == goal_batch['goal_internal_state'][:, 0]).float().mean().item()
      goal_accg2 = (pg2_logits.argmax(dim=1) == goal_batch['goal_internal_state'][:, 1]).float().mean().item()
      mean_goal_acc1 += (goal_acc1 - mean_goal_acc1) / (step + 1)
      mean_goal_acc2 += (goal_accg2 - mean_goal_acc2) / (step + 1)

      if mean_loss < best_loss:
        self.save_model(self.is_predictor, 'is_predictor')
        best_loss = mean_loss
      
      if self.tf_logger:
        self.tf_logger.add_scalar('isp_loss', mean_loss, step)
        self.tf_logger.add_scalar('isp_accuracy1', mean_acc1, step)
        self.tf_logger.add_scalar('isp_accuracy2', mean_acc2, step)
      
      mean_acc = (mean_acc1 + mean_acc2) / 2
      mean_goal_acc = (mean_goal_acc1 + mean_goal_acc2) / 2
      pbar.set_description(f'Loss: {mean_loss:.6f} | Mean_acc: {mean_acc:.2f} | Mean_goal_acc: {mean_goal_acc:.2f}')

  def train_goal_is_predictor(self):
    print('Training Goal Internal State Predictor')
    self.goal_is_predictor.train()
    self.gis_predictor_opt = torch.optim.AdamW(self.goal_is_predictor.parameters(), lr=3e-4)

    best_loss = torch.inf
    mean_loss = 0.0
    mean_acc1, mean_acc2 = 0.0, 0.0
    mean_goal_acc1, mean_goal_acc2 = 0.0, 0.0

    B = self.config.get('gisp_batch_size', 128)
    pbar = tqdm(range(self.config.get('gisp_max_train_steps', 100)))
    for step in pbar:
      batch = self.replay_buffer.sample_image_is_goal_batch(B)

      if step == 0:
        n_valid_ep = self.replay_buffer.n_finished_episode
        n_ep = self.replay_buffer.current_episode_id
        print(f'Number of episode with a reached target end: {n_valid_ep}/{n_ep}')  

      is1_logits, is2_logits, isg1_logits, isg2_logits = self.goal_is_predictor(batch['image'])

      # --- Current internal_state from image ---
      loss = (
        self.cross_entropy_criterion(is1_logits, batch['internal_state'][:, 0]) +
        self.cross_entropy_criterion(is2_logits, batch['internal_state'][:, 1])
      )

      # --- Goal internal_state ---
      goal_loss = (
        self.cross_entropy_criterion(isg1_logits, batch['goal_internal_state'][:, 0]) +
        self.cross_entropy_criterion(isg2_logits, batch['goal_internal_state'][:, 1])
      )

      loss = loss + goal_loss

      self.gis_predictor_opt.zero_grad()
      loss.backward()
      self.gis_predictor_opt.step()

      mean_loss += (loss.item() - mean_loss) / (step + 1)
      # Accuracies
      acc1 = (is1_logits.argmax(-1) == batch['internal_state'][:, 0]).float().mean().item()
      acc2 = (is2_logits.argmax(-1) == batch['internal_state'][:, 1]).float().mean().item()
      mean_acc1 += (acc1 - mean_acc1) / (step + 1)
      mean_acc2 += (acc2 - mean_acc2) / (step + 1)

      goal_acc1 = (isg1_logits.argmax(-1) == batch['goal_internal_state'][:, 0]).float().mean().item()
      goal_acc2 = (isg2_logits.argmax(-1) == batch['goal_internal_state'][:, 1]).float().mean().item()
      mean_goal_acc1 += (goal_acc1 - mean_goal_acc1) / (step + 1)
      mean_goal_acc2 += (goal_acc2 - mean_goal_acc2) / (step + 1)

      if mean_loss < best_loss:
        self.save_model(self.goal_is_predictor, 'goal_is_predictor')
        best_loss = mean_loss
      
      if self.tf_logger:
        self.tf_logger.add_scalar('gisp_loss', mean_loss, step)
        self.tf_logger.add_scalar('gisp_accuracy1', mean_acc1, step)
        self.tf_logger.add_scalar('gisp_accuracy2', mean_acc2, step)
        self.tf_logger.add_scalar('gisp_goal_accuracy1', mean_goal_acc1, step)
        self.tf_logger.add_scalar('gisp_goal_accuracy2', mean_goal_acc2, step)
      
      mean_acc = (mean_acc1 + mean_acc2) / 2
      mean_goal_acc = (mean_goal_acc1 + mean_goal_acc2) / 2
      pbar.set_description(f'Loss: {mean_loss:.6f} | Mean_acc: {mean_acc:.2f} | Mean_goal_acc: {mean_goal_acc:.2f}')

  def train_image_goal_predictor(self):  # Currently overfit
    print('Training Image Goal Predictor...')
    self.img_goal_predictor.train()
    self.img_goal_opt = torch.optim.AdamW(self.img_goal_predictor.parameters(), lr=3e-4)

    best_loss = torch.inf
    mean_loss = 0.0

    pbar = tqdm(range(self.config['image_goal_predictor_max_train_steps']))
    for step in pbar:
      batch = self.replay_buffer.sample_image_is_goal_batch(self.config.get('igp_batch_size', 128))

      if step == 0:
        n_valid_ep = self.replay_buffer.n_finished_episode
        n_ep = self.replay_buffer.current_episode_id
        print(f'Number of episode with a reached target end: {n_valid_ep}/{n_ep}')

      image = batch["image"]
      img_goal = batch['goal_image']

      rec = self.img_goal_predictor(image)
      loss = self.mse_criterion(rec, img_goal)

      self.img_goal_opt.zero_grad()
      loss.backward()
      self.img_goal_opt.step()

      mean_loss += (loss.item() - mean_loss) / (step + 1)

      if mean_loss < best_loss:
        self.save_model(self.img_goal_predictor, 'img_goal_predictor')
        best_loss = mean_loss
      
      if self.tf_logger:
        self.tf_logger.add_scalar('img_goal_loss', mean_loss, step)

        if step % 10 == 0:
          ori_rec = torch.cat([image[:8], img_goal[:8], rec[:8]], dim=0)
          self.tf_logger.add_images(f'generated_goal_image', ori_rec, global_step=step//10)
      
      pbar.set_description(f'Loss: {mean_loss:.6f}')

  def train_is_goal_scorer(self):  # Currently underfit ~30% accuracy
    print('Training IS Goal Predictor...')
    self.is_goal_scorer.train()
    self.isgp_opt = torch.optim.AdamW(self.is_goal_scorer.parameters(), lr=1e-4)

    self.all_goals = self.build_all_goals()

    best_loss = torch.inf
    mean_loss = 0.0
    mean_acc1, mean_acc2 = 0.0, 0.0

    pbar = tqdm(range(self.config['isgs_max_train_steps']))
    for step in pbar:
      batch = self.replay_buffer.sample_image_is_goal_batch(self.config.get('isgs_batch_size', 128))

      image = batch["image"]
      is_c = batch["internal_state"]
      is_g_pos = batch["goal_internal_state"]

      # ---- counterfactual negative goals ----
      # is_g_neg = batch['fake_goal_internal_state']

      # ---- score both ----
      s_pos = self.is_goal_scorer(image, is_c, is_g_pos)
      # s_neg = self.is_goal_scorer(image, is_c, is_g_neg)

      # expand inputs for fake goals
      B, K, _ = batch["fake_goal_internal_state"].shape

      image_rep = image.unsqueeze(1).repeat(1, K, 1, 1, 1).view(B*K, 3, 32, 32)
      is_c_rep = is_c.unsqueeze(1).repeat(1, K, 1).view(B*K, 2)

      # Use K/2 from same episode and K/2 from other episode
      # fg_from_ep = batch['fake_goal_internal_state'][:, :K//2]
      # perm = torch.randperm(B, device=self.device)
      # fg_from_other_ep = batch['fake_goal_internal_state'][:, K//2:][perm]
      # fake_goals = torch.cat([fg_from_ep, fg_from_other_ep], dim=1).view(B*K, 2)
      fake_goals = batch["fake_goal_internal_state"].view(B*K, 2)

      s_neg = self.is_goal_scorer(image_rep, is_c_rep, fake_goals)
      s_neg = s_neg.view(B, K)

      # Constrastive loss
      # loss = -torch.log(torch.sigmoid(1.0 * (s_pos - s_neg))).mean()
      # loss = -torch.log(torch.sigmoid(1.0 * (s_pos.unsqueeze(1) - s_neg))).mean()
      # loss = F.relu(1.0 - (s_pos.unsqueeze(1) - s_neg)).mean()
      loss = torch.nn.functional.softplus(-(s_pos.unsqueeze(1) - s_neg)/0.1).mean()
      # temperature = 1.0
      # logits = torch.cat([s_pos.unsqueeze(1), s_neg], dim=1)
      # labels = torch.zeros(B, dtype=torch.long, device=logits.device)
      # loss = F.cross_entropy(logits / temperature, labels)

      self.isgp_opt.zero_grad()
      loss.backward()
      self.isgp_opt.step()

      mean_loss += (loss.item() - mean_loss) / (step + 1)

      # Find best goal based on learned scorer
      if step % 100 == 0:
        # pred_goal, score = self.infer_best_goal_discrete(image, is_c, self.all_goals)
        # acc1 = (pred_goal[:, 0] == is_g_pos[:, 0]).float().mean().item()
        # acc2 = (pred_goal[:, 1] == is_g_pos[:, 1]).float().mean().item()
        acc_joint, acc1, acc2 = self.compute_topk_goal_accuracy(
            image=image,
            is_c=is_c,
            is_g_true=is_g_pos,
            all_goals=self.all_goals,
            k=30
        )
        mean_acc1 += (acc1 - mean_acc1) / (step//100 + 1)
        mean_acc2 += (acc2 - mean_acc2) / (step//100 + 1)

      if mean_loss < best_loss:
        self.save_model(self.is_goal_scorer, 'is_goal_scorer')
        best_loss = mean_loss
      
      if self.tf_logger:
        self.tf_logger.add_scalar('isgs_loss', mean_loss, step)
        if step % 100 == 0:
          self.tf_logger.add_scalar('isgs_accuracy1', mean_acc1, step)
          self.tf_logger.add_scalar('isgs_accuracy2', mean_acc2, step)

      pbar.set_description(f'Loss: {mean_loss:.6f} | Mean_acc: {(mean_acc1+mean_acc2)/2:.2f}')

  def train_rnn_goal_is_predictor(self):
    print('Training IS Goal Predictor (RNN)...')
    self.rnn_gisp.train()
    self.rnngisp_opt = torch.optim.AdamW(self.rnn_gisp.parameters(), lr=1e-4)

    best_loss = torch.inf
    mean_loss = 0.0
    mean_acc1, mean_acc2 = 0.0, 0.0

    pbar = tqdm(range(self.config.get('rnngisp_max_train_steps', 500)))
    for step in pbar:
      # ---- sample episode batch ----
      episode_length = random.randint(1, self.config.get('rnngisp_max_episode_length', 50))
      batch = self.replay_buffer.sample_episode_batch(self.config.get('rnngisp_batch_size', 32),
                                                      episode_length,
                                                      random_window=False, success_reward=10)

      image_seq = batch['image']                     # [B, T, 3, 32, 32]
      internal_state_seq = batch['internal_state']   # [B, T, 2]
      condition = {'internal_state': internal_state_seq}

      # ---- forward ----
      g1_logits, g2_logits = self.rnn_gisp(image_seq, condition=condition)  # [B, T, 19or37]

      # ---- target goal: final internal state ----
      B, T, *_ = image_seq.shape
      goal = internal_state_seq[:, -1]               # [B, 2]
      goal1 = goal[:, 0].long()
      goal2 = goal[:, 1].long()
      goal1 = goal1.unsqueeze(1).expand(B, T)  # [B, T]
      goal2 = goal2.unsqueeze(1).expand(B, T)  # [B, T]

      # ---- classification loss ----
      loss1 = F.cross_entropy(g1_logits.reshape(B*T, -1), goal1.flatten())
      loss2 = F.cross_entropy(g2_logits.reshape(B*T, -1), goal2.flatten())
      loss = loss1 + loss2

      # ---- optimize ----
      self.rnngisp_opt.zero_grad()
      loss.backward()
      self.rnngisp_opt.step()

      mean_loss += (loss.item() - mean_loss) / (step + 1)

      # ---- accuracy ----
      acc1 = (g1_logits.argmax(dim=-1) == goal1).float().mean()
      acc2 = (g2_logits.argmax(dim=-1) == goal2).float().mean()
      mean_acc1 += (acc1.item() - mean_acc1) / (step + 1)
      mean_acc2 += (acc2.item() - mean_acc2) / (step + 1)

      # ---- save best ----
      if mean_loss < best_loss:
        self.save_model(self.rnn_gisp, 'rnn_gisp')
        best_loss = mean_loss

      # ---- logging ----
      if self.tf_logger:
        self.tf_logger.add_scalar('rnngisp_loss', mean_loss, step)
        self.tf_logger.add_scalar('rnngisp_accuracy1', mean_acc1, step)
        self.tf_logger.add_scalar('rnngisp_accuracy2', mean_acc2, step)

      pbar.set_description(f'Loss: {mean_loss:.6f} | Mean_acc: {(mean_acc1+mean_acc2)/2:.2f}')

  def train_goal_image_predictor(self):  # Flow | CURRENTLY OVERFIT
    print('Training Goal Image Predictor...')
    self.goal_image_predictor.train()
    self.gip_opt = torch.optim.AdamW(self.goal_image_predictor.parameters(), lr=1e-4, weight_decay=0.1)

    best_loss = torch.inf
    mean_loss = 0.0

    pbar = tqdm(range(self.config.get('gip_max_train_steps', 100)))
    for step in pbar:
      batch = self.replay_buffer.sample_image_is_goal_batch(self.config.get('igp_batch_size', 128))
      x1 = batch['goal_image']  # target distribution is the image to predict
      condition = {'x_cond': batch['image']}

      if step == 0:
        n_valid_ep = self.replay_buffer.n_finished_episode
        n_ep = self.replay_buffer.current_episode_id
        print(f'Number of episode with a reached target end: {n_valid_ep}/{n_ep}')  

      # ------------------------------------------------------------------------------------------ #
      # STEP1: Train the world_model to produce a corresponding image with provided internal_state #
      # ------------------------------------------------------------------------------------------ #
      # Moving random (gaussian) distribution to x1 distribution based on provided internal state  #
      loss = flow_matching_loss(self.goal_image_predictor, x1, condition=condition,
                                weighted_time_sampling=self.config['weighted_time_sampling'],
                                noise_scale=self.config['noise_scale'])

      self.gip_opt.zero_grad()
      loss.backward()
      self.gip_opt.step()

      mean_loss += (loss.item() - mean_loss) / (step + 1)

      if mean_loss < best_loss:
        self.save_model(self.goal_image_predictor, 'goal_image_predictor')
        best_loss = mean_loss
      
      if self.tf_logger:
        self.tf_logger.add_scalar('goal_image_predictor_fm_loss', mean_loss, step)
        
        if step % self.config.get('gip_check_pred_loss_every', 20) == 0:
          x1_pred = rk45_sampling(
            self.goal_image_predictor,
            device=self.device,
            n_samples=x1.shape[0],
            condition=condition,
            n_steps=2
          )
          x1_pred = x1_pred[-1]
          with torch.no_grad():
            x1_pred_cleaned = self.image_cleaner(self.clamp_denorm_fn(x1_pred))

          self.tf_logger.add_scalar('goal_image_predictor_rec_loss',
                                    torch.nn.functional.mse_loss(x1_pred, x1),
                                    step // self.config['world_model_check_pred_loss_every'])
          self.tf_logger.add_images('gip_generated_goal_image',
                                    torch.cat([self.clamp_denorm_fn(batch['image'][:8]),
                                               self.clamp_denorm_fn(x1[:8]),
                                               self.clamp_denorm_fn(x1_pred[:8]),
                                               x1_pred_cleaned[:8]], dim=0),
                                    global_step=step)

      pbar.set_description(f'Loss: {mean_loss:.6f}')

  def train(self):
    self.fill_memory()
    self.train_internal_world_model()
    self.train_world_model()
    self.train_image_cleaner(self.world_model)
    self.train_is_goal_policy()

    # self.train_goal_is_predictor()
    # self.train_is_goal_scorer()
    # self.train_image_goal_predictor()

    self.train_goal_image_predictor()
    self.train_image_cleaner(self.goal_image_predictor)
    self.train_is_predictor()
    # self.train_rnn_goal_is_predictor()
  
  @torch.no_grad()
  def show_imagined_trajectory(self):
    self.fill_memory(replay_buffer_size=200)
    self.internal_world_model.eval()
    self.world_model.eval()
    self.image_cleaner.eval()

    batch = self.replay_buffer.sample(1)

    imagined_traj = [batch['image'][:1]]
    internal_state = batch['internal_state'][:1]  # [1, 2], action = [1, 1]
    for action in torch.as_tensor([[[1]]]*12 + [[[3]]]*11, device=self.device, dtype=torch.long):
      # ----------------------------------------------------------------- #
      # STEP1: from internal_state and action -> find next_internal_state #
      # ----------------------------------------------------------------- #
      nis1_pred, nis2_pred = self.internal_world_model(action, internal_state)  # [B, 18], [B, 36]
      pred1 = nis1_pred.argmax(dim=1)
      pred2 = nis2_pred.argmax(dim=1)
      internal_state = torch.stack([pred1, pred2], dim=1)

      # ----------------------------------------------------------------- #
      # STEP2: from internal_state -> get image from world model          #
      # ----------------------------------------------------------------- #
      x1_pred = rk45_sampling(
        self.world_model,
        device=self.device,
        n_samples=internal_state.shape[0],
        condition={'internal_state': internal_state, 'done_signal': batch['done']},
        n_steps=10
      )
      x1_pred = self.clamp_denorm_fn(x1_pred[-1])

      # ------------------------------------------------------------------------- #
      # STEP3: from imagined image -> clean world model image to fit with reality #
      # ------------------------------------------------------------------------- #
      rec = self.image_cleaner(x1_pred)

      imagined_traj.append(rec)

    self.tf_logger.add_images('generated_traj_multisteps_nsp', torch.cat(imagined_traj, dim=0))
  
  def build_all_goals(self):
    g1 = torch.arange(19, device=self.device)
    g2 = torch.arange(37, device=self.device)

    grid_g1, grid_g2 = torch.meshgrid(g1, g2, indexing='ij')
    all_goals = torch.stack([grid_g1.flatten(), grid_g2.flatten()], dim=1)  # [703, 2]

    return all_goals
  
  @torch.no_grad()
  def infer_best_goal_discrete(self, image, is_c, all_goals):
    """
    image: [B, C, H, W]
    is_c:  [B, is_dim]
    all_goals: [N, goal_dim]
    """
    B = image.shape[0]
    N = all_goals.shape[0]

    img_expanded = image.repeat_interleave(N, dim=0) # [B*N, 3, 32, 32]
    is_c_expanded = is_c.repeat_interleave(N, dim=0)
    goals_expanded = all_goals.repeat(B, 1)        # [B*N, 2]

    scores = self.is_goal_scorer(img_expanded, is_c_expanded, goals_expanded)
    scores = scores.view(B, N)

    max_scores_vals, max_indices = torch.max(scores, dim=1)
    best_goals = all_goals[max_indices]
    return best_goals, max_scores_vals

  @torch.no_grad()
  def compute_topk_goal_accuracy(self, image, is_c, is_g_true, all_goals, k=5):
    """
    image:      [B, 3, 32, 32]
    is_c:       [B, 2]
    is_g_true:  [B, 2]
    all_goals:  [N, 2]   (703 goals)
    k:          top-k

    Returns:
        acc_joint:  fraction where true goal is in top-k (both dims correct jointly)
        acc1:       top-k accuracy for dim1
        acc2:       top-k accuracy for dim2
    """
    B = image.shape[0]
    N = all_goals.shape[0]

    # Expand for scoring all goals
    img_expanded = image.repeat_interleave(N, dim=0)     # [B*N, 3, 32, 32]
    is_c_expanded = is_c.repeat_interleave(N, dim=0)    # [B*N, 2]
    goals_expanded = all_goals.repeat(B, 1)             # [B*N, 2]

    # Score all candidate goals
    scores = self.is_goal_scorer(img_expanded, is_c_expanded, goals_expanded)
    scores = scores.view(B, N)

    # Top-k indices per sample
    topk_scores, topk_idx = torch.topk(scores, k=k, dim=1)   # [B, k]
    topk_goals = all_goals[topk_idx]                         # [B, k, 2]

    # ---- Joint top-k accuracy (exact goal in top-k set) ----
    is_g_true_exp = is_g_true.unsqueeze(1)                   # [B, 1, 2]
    match = (topk_goals == is_g_true_exp).all(dim=-1)        # [B, k]
    acc_joint = match.any(dim=1).float().mean().item()

    # ---- Per-dimension top-k accuracy ----
    # dim 1
    acc1 = (topk_goals[..., 0] == is_g_true[:, 0].unsqueeze(1)).any(dim=1).float().mean().item()
    # dim 2
    acc2 = (topk_goals[..., 1] == is_g_true[:, 1].unsqueeze(1)).any(dim=1).float().mean().item()

    return acc_joint, acc1, acc2

  @torch.no_grad()
  def autoplay(self):
    # self.img_goal_predictor.eval()
    self.goal_image_predictor.eval()
    self.is_predictor.eval()
    self.goal_policy.eval()
    # self.rnn_gisp.eval()
    # self.goal_is_predictor.eval()
    self.image_cleaner.eval()

    pygame.init()  # Initialize pygame for keyboard input
    plt.ion()   # interactive mode (non-blocking)

    fig, ax = plt.subplots(figsize=(3, 3))
    goal_im = ax.imshow(np.zeros((32, 32, 3), dtype=np.uint8))
    ax.set_title("Predicted Goal Image")
    ax.axis("off")
    fig.canvas.draw()
    fig.show()

    running = True
    force_reset = False

    obs, _ = self.env.reset()
    img = self.env.render()

    def tensor_to_uint8(img):
      img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

      if img.min() < 0:
          img = (img + 1) / 2

      img = np.clip(img, 0, 1)
      img = (img * 255).astype(np.uint8)
      return img

    def format_data(obs, img):
      obs, _, img, *_ = self.replay_buffer.prepare_data(obs//5, None, img, None, None, None, None)
      obs, img = obs.unsqueeze(0).to(self.device), img.unsqueeze(0).to(self.device)
      return obs, img  # [1, 2], [1, 3, 32, 32]

    def find_goal(img, obs):
      # --- From GoalInternalStatePredictor --- #
      # _, _, g1_logits, g2_logits  = self.goal_is_predictor(img)
      # img_goal = None
      # --------------------------------------- #
      # img_goal = self.img_goal_predictor(img)
      # --- From GoalImagePredictor --- #
      condition = {'x_cond': img}
      img_goal = rk45_sampling(
        self.goal_image_predictor,
        device=self.device,
        n_samples=img.shape[0],
        condition=condition,
        n_steps=4
      )
      img_goal = self.image_cleaner(self.clamp_denorm_fn(img_goal[-1]))
      g1_logits, g2_logits = self.is_predictor(img_goal)
      # ------------------------------- #
      goal = torch.stack([g1_logits.argmax(-1), g2_logits.argmax(-1)], dim=1)  # [1, 2]
      # --- From RNN IS Goal Predictor --- #
      # g1_logits, g2_logits = self.rnn_gisp(img, condition={'internal_state': obs})
      # g1_logits, g2_logits = g1_logits.argmax(-1)[:, -1], g2_logits.argmax(-1)[:, -1]
      # goal = torch.stack([g1_logits, g2_logits], dim=1)
      # img_goal = None
      # ---------------------------------- #
      return goal, img_goal

    # --- Agent goal prediction ---
    obs, img = format_data(obs, img)

    # image_history = deque([img], maxlen=self.config.get('rnngisp_max_episode_length', 50))
    # obs_history = deque([obs], maxlen=self.config.get('rnngisp_max_episode_length', 50))

    # img = torch.stack(list(image_history), dim=1)  # [1, T, 3, 32, 32]
    # obs = torch.stack(list(obs_history), dim=1)  # [1, T, 2]

    goal, img_goal = find_goal(img, obs)

    while running:
      # Capture keyboard events
      for event in pygame.event.get():
        if event.type == pygame.QUIT:  # Close window
          running = False
        elif event.type == pygame.KEYDOWN:
          if event.key == pygame.K_q:  # Quit when 'Q' is pressed
            running = False
          elif event.key == pygame.K_r:
            force_reset = True

      # --- Agent action ---
      action, *_ = self.goal_policy.sample(obs, goal)
      # action, *_ = self.goal_policy.sample(obs[:, -1], goal)

      # --- Perform action ---
      obs, reward, terminated, truncated, info = self.env.step(action)
      img = self.env.render()

      # --- Format data ---
      obs, img = format_data(obs, img)

      # image_history.append(img)
      # obs_history.append(obs)
      # img = torch.stack(list(image_history), dim=1)
      # obs = torch.stack(list(obs_history), dim=1)

      # goal, img_goal = find_goal(img, obs)

      print(f'{terminated=} | {obs=} | {goal=}')
      if terminated or truncated or force_reset:
        force_reset = False
        print('RESET ENV!')
        # image_history.clear()
        # obs_history.clear()

        obs, _ = self.env.reset(options={'only_target': True})
        img = self.env.render()
        obs, img = format_data(obs, img)

        # image_history.append(img)
        # img = torch.stack(list(image_history), dim=1)

        # obs_history.append(obs)
        # obs = torch.stack(list(obs_history), dim=1)

        # --- Agent goal prediction ---
        goal, img_goal = find_goal(img, obs)

      if img_goal is not None:
        goal_np = tensor_to_uint8(img_goal)

        goal_im.set_data(goal_np)
        ax.draw_artist(ax.patch)
        ax.draw_artist(goal_im)

        fig.canvas.blit(ax.bbox)
        fig.canvas.flush_events()

    self.env.close()
    pygame.quit()


def get_args():
  parser = argparse.ArgumentParser(description='Next state predictor experiments')
  parser.add_argument('--trainer', '-t', type=str, default='nsp')
  parser.add_argument('--load_model', '-lm', action='store_true')
  parser.add_argument('--train_model', '-tm', action='store_true')
  parser.add_argument('--eval_model', '-em', action='store_true')
  parser.add_argument('--save_model', '-sm', action='store_true')
  parser.add_argument('--play_model', '-pm', action='store_true')
  parser.add_argument('--show_imagined_trajectory', '-sit', action='store_true')
  parser.add_argument('--show_imagined_goal_trajectory', '-sigt', action='store_true')
  parser.add_argument('--force_human_view', '-fhv', action='store_true')
  parser.add_argument('--experiment_name', '-en', type=str, default=None)
  return parser.parse_args()


if __name__ == '__main__':
  '''
  Current SETUP:
    1) Fill memory with random play
    2) Train internal_world_model -> use internal_state and action to predict next_internal_state
    3) Train world_model -> Flow Model that create image conditionned on internal_state
    4) Train image_cleaner -> Remove noise from the image generated by the world_model
    5) Train internal_state_goal_policy -> Get action based on internal_state and internal_state_goal
    6) Train internal_state_predictor -> use image and predict internal_state
    7) Train image_goal_predictor -> use current image to creates goal image
  '''
  trainers = {'nsp': NSPTrainer}
  args = get_args()

  config = {} if args.experiment_name is None else {'exp_name': args.experiment_name}
  config['render_mode'] = 'human' if args.play_model or args.force_human_view else 'rgb_array'

  print(f'Trainer: {args.trainer}')
  trainer = trainers[args.trainer](config)

  if args.load_model:
    trainer.load_model(trainer.world_model, 'world_model')
    trainer.load_model(trainer.internal_world_model, 'internal_world_model')
    trainer.load_model(trainer.image_cleaner, 'image_cleaner')
    trainer.load_model(trainer.goal_policy, 'goal_policy')
    trainer.load_model(trainer.goal_value, 'goal_value')
    trainer.load_model(trainer.is_predictor, 'is_predictor')
    # trainer.load_model(trainer.goal_is_predictor, 'goal_is_predictor')
    # trainer.load_model(trainer.img_goal_predictor, 'img_goal_predictor')
    # trainer.load_model(trainer.is_goal_scorer, 'is_goal_scorer')
    trainer.load_model(trainer.goal_image_predictor, 'goal_image_predictor')
    # trainer.load_model(trainer.rnn_gisp, 'rnn_gisp')
  
  if args.train_model:
    print('Start training...')
    trainer.train()
  
  if args.show_imagined_trajectory:
    print('Show Imagined Trajectory...')
    trainer.show_imagined_trajectory()
  
  if args.show_imagined_goal_trajectory:
    trainer.show_imagined_goal_policy()
  
  if args.play_model:
    print('Start autoplay...')
    trainer.autoplay()