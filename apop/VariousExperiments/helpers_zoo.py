import math
import torch
import torch.nn.functional as F

from torch.autograd import grad


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