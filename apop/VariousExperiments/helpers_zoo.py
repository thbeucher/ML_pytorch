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
"""
import os
import json
import math
import torch
import random
import numpy as np
import torch.nn.functional as F

from torch.autograd import grad
from typing import Callable, Optional, Tuple


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
  import imageio
  import numpy as np
  
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
