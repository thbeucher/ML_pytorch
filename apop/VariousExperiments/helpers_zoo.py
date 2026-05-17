"""
This module contains a collection of helper functions for various machine learning experiments.

| Function                | Description                                                               |
|-------------------------|---------------------------------------------------------------------------|
| `exponential_schedule`  | Generates an exponential learning rate schedule.                          |
| `linear_schedule`       | Generates a linear learning rate schedule.                                |
| `cosine_schedule`       | Generates a cosine learning rate schedule.                                |
| `sample_time`           | Samples time for flow matching, biased towards smaller values.            |
| `flow_matching_loss`    | Calculates the flow matching loss for training a flow-based model.        |
| `euler_sampling`        | Performs sampling using the Euler method for a flow-based model.          |
| `rk45_step`             | Performs a single step of the Runge-Kutta 45 (Dormand-Prince) method.     |
| `rk45_sampling`         | Performs sampling using the RK45 method for a flow-based model.           |
| `gradient_penalty`      | Calculates the gradient penalty for a critic in a GAN (WGAN-GP).          |
| `random_patch_mask`     | Randomly masks out a percentage of patch embeddings.                      |
| `mask_specific_patch`   | Masks a specific patch in the patch embeddings.                           |
| `create_gif_from_images`| Creates a GIF from a list of images.                                      |
| `set_seed`              | Sets the random seed for reproducibility.                                 |
| `compute_gae`           | Computes the Generalized Advantage Estimation (GAE) for a trajectory.     |
| `ppo_update`            | Performs a Proximal Policy Optimization (PPO) update.                     |
| `dump_json_data`        | Saves configuration data to a JSON file for reproducibility.              |
| `find_object_center`    | Finds the geometric center of pixels matching a given color condition.    |
"""
import os
import json
import math
import torch
import random
import imageio
import numpy as np
import torch.nn.functional as F

from torch.autograd import grad
from typing import Callable


def exponential_schedule(epoch, n_epochs_decay, start=0.1, end=0.75):
  if end == 0.0:
    end = 1e-6
  op_check = min if end > start else max
  return round(op_check(end, start * (end / start) ** (epoch / (n_epochs_decay - 1))), 2)


def linear_schedule(epoch, n_epochs, start, end):
    t = min(epoch / (n_epochs - 1), 1.0)
    return round(start + t * (end - start), 2)


def cosine_schedule(epoch, n_epochs, start, end):
    t = min(epoch / (n_epochs - 1), 1.0)
    return round(end + (start - end) * 0.5 * (1 + math.cos(math.pi * t)), 2)


def sample_time(batch_size, device):
  u = torch.rand(batch_size, device=device)
  t = 1 - u**2   # bias toward small t
  return t[:, None]


def flow_matching_loss(model, x1, x0=None, condition=None, weighted_time_sampling=False, noise_scale=1.0):
  '''
  Training logic of flow model:
  1) Sample random points from our source and target distributions, and pair the points (x1 and x0)
  2) Sample random times between 0 and 1
  3) Calculate the locations where these points would be at those times
      if they were moving at constant velocity from source to target locations (interpolation, xt)
  4) Calculate what velocity they would have at those locations if they were moving at constant velocity (x1-x0)
  5) Train the network to predict these velocities – which will end up “seeking the mean”
      when the network has to do the same for many, many points.
  '''
  if x0 is None:
    # ---- Sample Gaussian noise as x0 ----
    x0 = torch.randn_like(x1) * noise_scale

  # ---- Sample t uniformly or weighted ----
  if weighted_time_sampling:
    t = sample_time(x1.size(0), x1.device)
  else:
    t = torch.rand(x1.size(0), 1, device=x1.device)
  t_expand = t[:, :, None, None]

  # ---- The target vector field (x1 - x0) ----
  v_target = x1 - x0

  # ---- Interpolate x_t = (1-t)x0 + t x1 ----
  xt = (1 - t_expand) * x0 + t_expand * x1

  # ---- Predict velocity and compute loss ----
  v_pred = model(xt, t, condition)
  return F.mse_loss(v_pred, v_target)


@torch.no_grad()
def euler_sampling(model, device, x=None, n_samples=8, n_steps=10, get_step=5, condition=None, clamp=True):
  model.eval()
  samples = []

  if x is None:
    # Start from pure Gaussian noise
    x = torch.randn(n_samples, 3, 32, 32).to(device)

  # Euler integration from t=0 → t=1
  dt = 1.0 / n_steps
  for i, step in enumerate(range(n_steps)):
    t = torch.full((x.shape[0], 1), step * dt, device=device)
    v = model(x, t, condition)  # vector field
    x = x + v * dt  # Euler update

    if clamp:
      x = torch.clamp(x, -3.0, 3.0)

    if i % get_step == 0 and n_steps > 1:
      samples.append(x.clone())

  return samples + [x]


@torch.no_grad()
def rk45_step(f, t, x, dt):
  """
  One Dormand–Prince RK45 step.
  Args:
      f : function f(t, x) -> dx/dt
      t : scalar float tensor
      x : tensor (B, C, H, W)
      dt : step size (float)
  """
  k1 = f(t, x)
  k2 = f(t + dt*1/5, x + dt*(1/5)*k1)
  k3 = f(t + dt*3/10, x + dt*(3/40*k1 + 9/40*k2))
  k4 = f(t + dt*4/5, x + dt*(44/45*k1 - 56/15*k2 + 32/9*k3))
  k5 = f(t + dt*8/9, x + dt*(19372/6561*k1 - 25360/2187*k2 + 64448/6561*k3 - 212/729*k4))
  k6 = f(t + dt, x + dt*(9017/3168*k1 - 355/33*k2 + 46732/5247*k3 + 49/176*k4 - 5103/18656*k5))

  # 5th-order solution
  x_next = x + dt * (35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)
  return x_next


@torch.no_grad()
def rk45_sampling(model, device, x=None, n_samples=8, n_steps=10, get_step=5, condition=None, clamp=True):
  model.eval()

  def ode_fn(t, x):
    t_tensor = torch.full((x.shape[0], 1), t, device=device)
    return model(x, t_tensor, condition)

  if x is None:
    # Initial noise
    x = torch.randn(n_samples, 3, 32, 32, device=device)

  t = 0.0
  dt = 1.0 / n_steps

  samples = []
  for i in range(n_steps):
    x = rk45_step(ode_fn, t, x, dt)
    t += dt

    if clamp:
      x = torch.clamp(x, -3.0, 3.0)
    
    if i % get_step == 0 and n_steps > 1:
      samples.append(x.clone())

  return samples + [x]


def gradient_penalty(critic, real, fake, device, gp_lambda=10.0):
  batch_size = real.size(0)
  # Random weight for interpolation between real and fake
  alpha = torch.rand(batch_size, 1, 1, 1, device=device)
  interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

  critic_interpolates = critic(interpolates)
  # For autograd.grad to work we need a scalar for each sample — use ones
  ones = torch.ones_like(critic_interpolates, device=device)

  gradients = grad(
    outputs=critic_interpolates,
    inputs=interpolates,
    grad_outputs=ones,
    create_graph=True,
    retain_graph=True,
    only_inputs=True,
  )[0]

  gradients = gradients.view(batch_size, -1)
  grad_norm = gradients.norm(2, dim=1)
  gp = gp_lambda * ((grad_norm - 1) ** 2).mean()
  return gp


def random_patch_mask(patch_embeddings, mask_prob=0.25):
  """
  Randomly zeros out a percentage of patches.
  Args:
      patch_embeddings: [B, N, D] tensor
      mask_prob: Float between 0 and 1 (0.25 = mask 25% of patches)
  """
  B, N, D = patch_embeddings.shape
  if mask_prob <= 0:
    return patch_embeddings

  # Create a random mask of 1s and 0s
  # bernoulli(0.75) gives 1 with 75% probability (the kept patches)
  mask = torch.bernoulli(torch.full((B, N, 1), 1 - mask_prob)).to(patch_embeddings.device)
  
  # Apply mask and scale to maintain activation magnitude (like Dropout)
  return (patch_embeddings * mask) / (1 - mask_prob)


def mask_specific_patch(patch_embeddings, patch_index):
  """
  Zeros out the exact patch where the object is.
  patch_index: [B] tensor of indices
  """
  B, N, D = patch_embeddings.shape
  # Create a mask of ones
  mask = torch.ones((B, N, 1), device=patch_embeddings.device)
  
  # Set the specific patch index to zero for each batch item
  batch_indices = torch.arange(B, device=patch_embeddings.device)
  mask[batch_indices, patch_index, 0] = 0
  
  return patch_embeddings * mask


def create_gif_from_images(images, filename, duration=100):
  """
  Creates a GIF from a list of PIL images or numpy arrays.
  """
  pil_images = []
  for img in images:
    if isinstance(img, torch.Tensor):
      img = img.permute(1, 2, 0).cpu().numpy()
    if img.dtype == np.float32 or img.dtype == np.float64:
      img = (img * 255).astype(np.uint8)
    pil_images.append(img)

  imageio.mimsave(filename, pil_images, duration=duration)


def set_seed(seed=42, device='cuda'):
  """Set random seeds for reproducibility across PyTorch, NumPy, and Python."""
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  # For deterministic behavior on GPU (may be slower)
  if device == 'cuda':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def compute_gae(rewards: torch.Tensor,
                values: torch.Tensor,
                masks: torch.Tensor,
                next_value: torch.Tensor,
                gamma: float,
                lmbda: float,
                normalize: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
  """
  Computes the Generalized Advantage Estimation (GAE) for a trajectory.

  GAE is a method for estimating the advantage function, which is used to update the policy in PPO.
  It's a trade-off between unbiased but high-variance Monte-Carlo returns and biased but low-variance TD-returns.

  Args:
    rewards (torch.Tensor): Tensor of rewards for each step in the trajectory. Shape: [T, B]
    values (torch.Tensor): Tensor of value estimates for each state. Shape: [T, B]
    masks (torch.Tensor): Tensor of masks for each step (0 for terminal states, 1 otherwise). Shape: [T, B]
    next_value (torch.Tensor): Value estimate for the state after the last step. Shape: [B]
    gamma (float): The discount factor.
    lmbda (float): The GAE lambda parameter (λ). Controls the bias-variance trade-off.
                   λ=0 corresponds to 1-step TD error, λ=1 corresponds to Monte-Carlo returns.
    normalize (bool): If True, normalize the advantages to have zero mean and unit variance. This often stabilizes training.

  Returns:
    advantages (torch.Tensor): The computed advantages for each step. Shape: [T, B]
    returns (torch.Tensor): The computed returns (targets for the value function). Shape: [T, B]
  """
  T = rewards.size(0)
  advantages = torch.zeros_like(rewards)
  last_gae = torch.zeros_like(next_value)

  for t in reversed(range(T)):
    # If the current state is terminal, the value of the next state is 0.
    # Otherwise, if it's the last step of the trajectory, we bootstrap with `next_value`.
    # Otherwise, we use the value of the actual next state from the rollout.
    next_values = next_value if t == T - 1 else values[t + 1]

    # Calculate the TD error (δ_t)
    # δ_t = r_t + γ * V(s_{t+1}) * mask_t - V(s_t)
    delta = rewards[t] + gamma * next_values * masks[t] - values[t]

    # GAE recursion: A_t = δ_t + γ * λ * mask_t * A_{t+1}
    last_gae = delta + gamma * lmbda * masks[t] * last_gae
    advantages[t] = last_gae

  # Returns are the advantages plus the value estimates
  # R_t = A_t + V(s_t)
  returns = advantages + values

  advantages = advantages.flatten()
  returns = returns.flatten()

  if normalize:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

  return advantages, returns


def ppo_update(policy_net: torch.nn.Module,
               value_net: torch.nn.Module,
               policy_optimizer: torch.optim.Optimizer,
               value_optimizer: torch.optim.Optimizer,
               traj: dict,
               next_value: torch.Tensor,
               gamma: float = 0.99,
               gae_lambda: float = 0.95,
               clip_eps: float = 0.2,
               value_coef: float = 0.5,
               entropy_coef: float = 0.01,
               n_epochs: int = 4,
               n_minibatches: int = 4,
               max_grad_norm: float = 0.5,
               normalize_advantage: bool = True):
    """
    Performs a PPO (Proximal Policy Optimization) update for the policy and value networks.

    This function implements the core PPO algorithm, including:
    - GAE advantage estimation.
    - Clipped surrogate policy objective.
    - Clipped value loss (optional, but standard).
    - Entropy bonus for exploration.
    - Multiple update epochs over the same rollout data for improved sample efficiency.
    - Minibatching for more stable and efficient updates.

    Args:
      policy_net (torch.nn.Module): The policy network to be updated.
      value_net (torch.nn.Module): The value network to be updated.
      policy_optimizer (torch.optim.Optimizer): The optimizer for the policy network.
      value_optimizer (torch.optim.Optimizer): The optimizer for the value network.
      traj (dict): A dictionary containing the collected trajectory data. Must contain:
                   'rewards', 'values', 'masks', 'states', 'goals', 'actions', 'log_probs'.
      next_value (torch.Tensor): The value estimate for the last state in the trajectory.
      gamma (float): The discount factor.
      gae_lambda (float): The GAE lambda parameter.
      clip_eps (float): The PPO clipping parameter (ε).
      value_coef (float): The coefficient for the value loss.
      entropy_coef (float): The coefficient for the entropy bonus.
      n_epochs (int): The number of optimization epochs to run on the rollout data.
      n_minibatches (int): The number of minibatches to split the data into for each epoch.
      max_grad_norm (float): The maximum norm for gradient clipping to prevent large updates.
      normalize_advantage (bool): Whether to normalize the advantages.

    Returns:
        A tuple containing the mean policy loss and mean value loss over the update epochs.
    """
    # ---- Stack rollout tensors from list to tensor: shape [T, B, ...] ----
    rewards = torch.stack(traj['rewards'])
    values = torch.stack(traj['values'])
    masks = torch.stack(traj['masks'])

    # ---- Compute advantages and returns using GAE ----
    advantages, returns = compute_gae(
      rewards, values, masks, next_value, gamma, gae_lambda, normalize_advantage)

    # ---- Flatten time and batch dimensions for easier processing in epochs ----
    # Filter out the steps of episodes that finished before the last episode that finished
    # We want to include the terminal step, but not any steps after it.
    # A 'valid_step' at time `t` is one where the episode was not done at `t-1`.
    T, B = masks.shape
    # We shift the masks by one timestep and pad the beginning with ones (as all episodes are active at t=0)
    padded_masks = torch.ones(T + 1, B, device=masks.device)
    padded_masks[1:] = masks
    # The cumulative product will propagate the first zero, marking all subsequent steps as invalid.
    valid_steps = torch.cumprod(padded_masks, dim=0)[:-1, :].bool()
    valid_steps_flat = valid_steps.flatten()

    states = torch.cat(traj['states'])[valid_steps_flat]
    goals = torch.cat(traj['goals'])[valid_steps_flat]
    actions = torch.cat(traj['actions'])[valid_steps_flat]
    old_log_probs = torch.cat(traj['log_probs']).squeeze(-1)[valid_steps_flat]
    old_values = torch.cat(traj['values']).flatten()[valid_steps_flat]
    advantages = advantages[valid_steps_flat]
    returns = returns[valid_steps_flat]

    # --- PPO update epochs ---
    batch_size = len(states)

    policy_losses = []
    value_losses = []

    for _ in range(n_epochs):
      # Shuffle data at the start of each epoch
      indices = torch.randperm(batch_size, device=states.device)

      for mb_indices in torch.chunk(indices, n_minibatches):
        # --- Get minibatch data ---
        mb_states = states[mb_indices]
        mb_goals = goals[mb_indices]
        mb_actions = actions[mb_indices]
        mb_old_log_probs = old_log_probs[mb_indices]
        mb_advantages = advantages[mb_indices]
        mb_returns = returns[mb_indices]
        mb_old_values = old_values[mb_indices]

        # Recompute policy π(a | s, g) with CURRENT network parameters
        logits = policy_net(mb_states, mb_goals)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(mb_actions)
        entropy = dist.entropy().mean()

        # Recompute value V(s, g) with CURRENT network parameters
        new_values = value_net(mb_states, mb_goals).squeeze(-1)

        # ---- PPO Policy Loss (Clipped Surrogate Objective) ----
        # ratio = π_new(a|s) / π_old(a|s)
        ratio = torch.exp(new_log_probs - mb_old_log_probs)
        surr1 = ratio * mb_advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # ---- PPO Value Loss (Clipped) ----
        # This helps to prevent the value function from changing too quickly.
        value_pred_clipped = mb_old_values + torch.clamp(new_values - mb_old_values, -clip_eps, clip_eps)
        value_loss_unclipped = (new_values - mb_returns).pow(2)
        value_loss_clipped = (value_pred_clipped - mb_returns).pow(2)
        value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

        # ---- Total Loss & Update ----
        # Note: We assume the policy and value networks are separate.
        # The policy loss is the standard PPO clipped surrogate objective minus an entropy bonus.
        policy_total_loss = policy_loss - entropy_coef * entropy
        # The value loss is scaled by a coefficient.
        value_total_loss = value_loss * value_coef

        # Update policy network
        policy_optimizer.zero_grad()
        policy_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
        policy_optimizer.step()

        # Update value network
        value_optimizer.zero_grad()
        value_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
        value_optimizer.step()

        policy_losses.append(policy_total_loss.item())
        value_losses.append(value_total_loss.item())

    return np.mean(policy_losses), np.mean(value_losses)


def dump_json_data(save_dir, exp_name, data):
  """Save configuration to JSON file for reproducibility."""
  with open(os.path.join(save_dir, f"{exp_name}.json"), 'w') as f:
    json.dump(data, f)


def find_object_center(
    frames: torch.Tensor,
    color_condition: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
  """
  Finds the geometric center (mean) of pixels that match a given color condition for a batch of frames.

  Args:
      frames (torch.Tensor): A tensor of shape [B, C, H, W].
      color_condition (Callable): A function that takes the frames tensor and returns a boolean mask
                                  of shape [B, H, W] where True indicates a pixel of interest.

  Returns:
      torch.Tensor: A tensor of shape [B, 2] containing the (x, y) coordinates of the center for each frame.
                    If no object is found in a frame, the coordinates for that frame will be NaN.
  """
  mask = color_condition(frames)
  batch_size = frames.shape[0]
  centers = torch.zeros(batch_size, 2, device=frames.device, dtype=torch.float32)

  for i in range(batch_size):
    # nonzero(as_tuple=True) is the replacement for np.where(condition)
    pixels = mask[i].nonzero(as_tuple=True)
    if pixels[0].numel() > 0:
      y_coords, x_coords = pixels
      center_y = torch.mean(y_coords.float())
      center_x = torch.mean(x_coords.float())
      centers[i, 0] = center_x
      centers[i, 1] = center_y
    else:
      centers[i, :] = float('nan')
  
  return centers
