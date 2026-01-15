import torch
import imageio
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from pytorch_msssim import SSIM
from torchvision.utils import make_grid

from encoding_exps import CNNAE
from flow_matching_exps import flow_matching_loss, rk45_sampling
from next_state_predictor_exps import NextStatePredictorTrainer, WorldModelFlowUnet


class GoalPolicy(nn.Module):
  def __init__(self, a_dim=5, is_dim=32, is_n_values=37, hidden=256):
    super().__init__()
    self.is_emb = nn.Sequential(nn.Embedding(is_n_values, is_dim),
                                nn.Linear(is_dim, is_dim),
                                nn.SiLU(True))
    self.net = nn.Sequential(
      nn.Linear(2 * is_dim * 3, hidden),
      nn.ReLU(True),
      nn.Linear(hidden, hidden),
      nn.ReLU(True),
      nn.Linear(hidden, a_dim)
    )

  def forward(self, is_c, is_g):  # [B, 2], [B, 2]
    is_c_emb = self.is_emb(is_c).flatten(1)  # -> [B, 2, 32] -> [B, 64]
    is_g_emb = self.is_emb(is_g).flatten(1)  # -> [B, 2, 32] -> [B, 64]
    x = torch.cat([is_c_emb, is_g_emb, is_g_emb - is_c_emb], dim=-1)  # [B, 192]
    return self.net(x)  # [B, 5]


class GoalValue(nn.Module):
  def __init__(self, is_dim=32, is_n_values=37, hidden=256):
    super().__init__()
    self.is_emb = nn.Sequential(nn.Embedding(is_n_values, is_dim),
                                nn.Linear(is_dim, is_dim),
                                nn.SiLU(True))
    self.net = nn.Sequential(
      nn.Linear(2 * is_dim * 2, hidden),
      nn.ReLU(True),
      nn.Linear(hidden, hidden),
      nn.ReLU(True),
      nn.Linear(hidden, 1)
    )

  def forward(self, is_c, is_g):
    is_c_emb = self.is_emb(is_c).flatten(1)  # -> [B, 2, 32] -> [B, 64]
    is_g_emb = self.is_emb(is_g).flatten(1)  # -> [B, 2, 32] -> [B, 64]
    return self.net(torch.cat([is_c_emb, is_g_emb], dim=-1)).squeeze(-1)  # [B, 1] -> [B]


class NISPredictor(nn.Module):
  def __init__(self, n_actions=5, action_dim=8, is1_n_values=19, is2_n_values=37, is_dim=32, mlp_dim=64):
    super().__init__()
    self.action_emb = nn.Sequential(nn.Embedding(n_actions, action_dim),
                                    nn.Linear(action_dim, action_dim),
                                    nn.SiLU(True))
    self.is_emb = nn.Sequential(nn.Embedding(max(is1_n_values, is2_n_values), is_dim),
                                nn.Linear(is_dim, is_dim),
                                nn.SiLU(True))

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


class ImageToIS(nn.Module):
  def __init__(self, is1_n_values=18, is2_n_values=36, hidden_dim=512):
      super().__init__()

      self.encoder = nn.Sequential(
          # [B, 3, 32, 32]
          nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, 16, 16]
          nn.BatchNorm2d(64),
          nn.SiLU(),

          nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 128, 8, 8]
          nn.BatchNorm2d(128),
          nn.SiLU(),

          nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # [B, 256, 4, 4]
          nn.BatchNorm2d(256),
          nn.SiLU(),

          nn.Conv2d(256, hidden_dim, kernel_size=3, stride=1, padding=1),  # [B, 512, 4, 4]
          nn.BatchNorm2d(hidden_dim),
          nn.SiLU(),
      )

      # Global Average Pooling → [B, hidden_dim]
      self.pool = nn.AdaptiveAvgPool2d(1)

      # Two prediction heads
      self.head1 = nn.Linear(hidden_dim, is1_n_values)
      self.head2 = nn.Linear(hidden_dim, is2_n_values)

  def forward(self, x):
    """
    x: [B, 3, 32, 32]
    """
    h = self.encoder(x)
    h = self.pool(h).flatten(1)

    out1 = self.head1(h)  # [B, 18]
    out2 = self.head2(h)  # [B, 36]

    return out1, out2


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
  CONFIG = {'exp_name':                             'threesteps_nsp',
            'replay_buffer_size':                   10_000,
            # INTERNAL STATE
            'internal_state_dim':                   2,
            'internal_state_n_values':              (90//5+1, 180//5+1),  # max_angle / angle_step
            'internal_state_emb':                   32,
            'internal_world_model_max_train_steps': 50,  # 10_000
            'internal_world_model_batch_size':      128,
            # IMAGE CLEANER
            'image_cleaner_max_train_steps':        5,  # 300
            'image_cleaner_batch_size':             128,
            'lambda_mse':                           0.8,
            'lambda_ssim':                          0.2,
            'world_model_max_train_steps':          30,}  # 3_000
  def __init__(self, config={}):
    super().__init__(NSPTrainer.CONFIG)
    self.config = {**self.config, **config}
  
  def instanciate_model(self):
    self.world_model = WorldModelFlowUnet(img_chan=self.config['image_chan'],
                                          time_dim=self.config['time_dim'],
                                          add_action=False,
                                          add_is=True,
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
  
  @torch.no_grad()
  def show_goal_policy(self):
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
      print(f'action={a.item()}')

      # Get next internal_state
      is1_logits, is2_logits = self.internal_world_model(a, is_c)
      is_next = torch.stack([is1_logits.argmax(-1), is2_logits.argmax(-1)], dim=1)

      # Get predicted world image
      condition = {'internal_state': is_next}
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

  def reinforce_update_is_goal_networks(self, trajs, next_value, gamma, lmbda, entropy_coef):
    # Convert lists to tensors [T, B]
    rewards = torch.stack(trajs['rewards'])
    values = torch.stack(trajs['values'])
    masks = torch.stack(trajs['masks'])
    log_probs = torch.stack(trajs['log_probs'])
    entropies = torch.stack(trajs['entropies'])

    T = rewards.size(0)
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0

    # --- GAE Calculation ---
    for t in reversed(range(T)):
      if t == T - 1:
        next_non_terminal = masks[t]
        next_values = next_value
      else:
        next_non_terminal = masks[t]
        next_values = values[t + 1]
      
      # TD Error (delta)
      delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
      # GAE: Recursive smoothing of TD errors
      advantages[t] = last_gae_lam = delta + gamma * lmbda * next_non_terminal * last_gae_lam
    
    returns = advantages + values.detach()

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

    T, B = rewards.shape

    # ---- Generalized Advantage Estimation (GAE) ----
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0

    for t in reversed(range(T)):
      # Bootstrap with V(s_{t+1}, g)
      next_val = next_value if t == T - 1 else values[t + 1]
      # TD error δ_t
      delta = rewards[t] + gamma * next_val * masks[t] - values[t]
      # GAE recursion
      last_gae = delta + gamma * lmbda * masks[t] * last_gae
      advantages[t] = last_gae

    returns = advantages + values

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
  
  def collect_is_goal_rollout(self, batch, mean_success_rate, mean_success_rate_curriculum, long_goal, max_horizon):
    """
    Collect an on-policy rollout using the current goal-conditioned policy.

    IMPORTANT:
    - This function must NOT keep computation graphs.
    - PPO replays data, not graphs.
    """
    # Current internal state s_t
    is_c = batch['internal_state']  # internal_state_current [B, 2]

    # Current internal state goal g
    if mean_success_rate > mean_success_rate_curriculum or long_goal:
      perm = torch.randperm(is_c.size(0))
      is_g = batch['next_internal_state'][perm]
      long_goal = True
    else:
      is_g = batch['next_internal_state']  # internal_state_goal [B, 2]

    # Buffers for trajectory
    trajs = {'states': [], 'goals': [], 'actions': [], 'log_probs': [], 'values': [],
              'rewards': [], 'masks': [], 'entropies': []}

    for t in range(max_horizon):
      # ---- Policy forward pass π(a | s, g) ----
      a_logits = self.goal_policy(is_c, is_g)
      dist = torch.distributions.Categorical(logits=a_logits)
      # Sample action (on-policy)
      a = dist.sample()
      
      log_p = dist.log_prob(a)
      # Goal-conditioned value estimate V(s, g)
      val = self.goal_value(is_c, is_g).squeeze(-1)
      ent = dist.entropy()

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
  
    return trajs, final_val, reached, long_goal
  
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

    # training params
    n_updates = 2_000  # 5_000
    batch_size = 128
    max_horizon = 50
    gamma = 0.99
    eval_every = 100
    lmbda = 0.95  # 0 = 1-step TD, 1 = Monte-Carlo
    entropy_coef = 0.01
    entropy_coef_start = entropy_coef
    entropy_coef_end = entropy_coef / 10
    mean_success_rate_curriculum = 0.7
    long_goal = False

    # training loop
    best_success_rate = -torch.inf
    mean_policy_loss = 0.0
    mean_value_loss = 0.0
    mean_success_rate = 0.0
    pbar = tqdm(range(n_updates))
    for step in pbar:
      batch = self.replay_buffer.sample(batch_size)

      # Collect ON-POLICY rollout
      trajs, final_val, reached, long_goal = self.collect_is_goal_rollout(batch,
                                                                          mean_success_rate,
                                                                          mean_success_rate_curriculum,
                                                                          long_goal,
                                                                          max_horizon)
      
      # --- 2. Update Networks ---
      # loss_policy, loss_value = self.update_is_goal_networks(trajs, final_val, gamma, lmbda, entropy_coef)
      loss_policy, loss_value = self.ppo_update_is_goal_networks(
        trajs,
        final_val,
        gamma=gamma,
        lmbda=lmbda,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=entropy_coef,
        n_epochs=4,
      )

      entropy_coef = max(entropy_coef_end, entropy_coef_start * (1 - step / n_updates)) # Gradual decay

      mean_policy_loss += (loss_policy - mean_policy_loss) / (step + 1)
      mean_value_loss += (loss_value - mean_value_loss) / (step + 1)
      mean_success_rate += (reached.float().mean() - mean_success_rate) / (step//eval_every + 1)
    
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

  def train_image_cleaner(self):
    print('Training ImageCleaner...')
    self.image_cleaner.train()
    self.world_model.eval()

    self.ic_opt = torch.optim.AdamW(self.image_cleaner.parameters(), lr=1e-3)
    self.mse_criterion = nn.MSELoss()
    self.ssim_criterion = SSIM(data_range=1, size_average=True, channel=3)

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
        condition = {'internal_state': batch['internal_state']}
        x1_pred = rk45_sampling(
          self.world_model,
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
      condition = {'internal_state': batch['internal_state']}

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
    self.iwm_criterion = nn.CrossEntropyLoss()

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

      loss1 = self.iwm_criterion(nis1_pred, is1_target)
      loss2 = self.iwm_criterion(nis2_pred, is2_target)
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
      
      pbar.set_description(f'Loss: {mean_loss:.6f}')
  
  def train(self):
    self.fill_memory()
    self.train_internal_world_model()
    self.train_world_model()
    self.train_image_cleaner()
    self.train_is_goal_policy()
  
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
        condition={'internal_state': internal_state},
        n_steps=10
      )
      x1_pred = self.clamp_denorm_fn(x1_pred[-1])

      # ------------------------------------------------------------------------- #
      # STEP3: from imagined image -> clean world model image to fit with reality #
      # ------------------------------------------------------------------------- #
      rec = self.image_cleaner(x1_pred)

      imagined_traj.append(rec)

    self.tf_logger.add_images('generated_traj_threesteps_nsp', torch.cat(imagined_traj, dim=0))


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

  if args.play_model:
    print('Start autoplay...')
    trainer.autoplay()
  
  if args.train_model:
    print('Start training...')
    trainer.train()
  
  if args.show_imagined_trajectory:
    print('Show Imagined Trajectory...')
    trainer.show_imagined_trajectory()
  
  if args.show_imagined_goal_trajectory:
    trainer.show_goal_policy()