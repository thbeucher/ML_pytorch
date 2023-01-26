import os
import ast
import gym
import sys
import torch
import random
import logging
import argparse
import itertools
import numpy as np
import torchvision.transforms as tvt

from tqdm import tqdm
from collections import deque
from torchvision.utils import make_grid, save_image

sys.path.append('../../../robot/')

sys.path.append(os.path.abspath(__file__).replace('USS-GL/rec_exps.py', ''))
import utils as u
import brain_exps as be


class AutoEncoder(torch.nn.Module):
  def __init__(self, gdn_act=False, add_body_infos=False):
    super().__init__()
    self.encoder = be.ImageEmbedder(gdn_act=gdn_act)
    self.decoder = be.ImageReconstructor(gdn_act=gdn_act, n_input_features=258 if add_body_infos else 256)
  
  def forward(self, x, body_infos=None):  # [B, C, H, W], [B, 2]
    code = self.encoder(x)  # -> [B, 256]
    if body_infos is not None:
      code = torch.cat([code, body_infos], dim=1)  # [B, 258]
    return self.decoder(code)


def get_game_env(game_view=False):
  env = gym.make('gym_robot_arm:robot-arm-v1')
  env.config['in_background'] = not game_view
  render_mode = 'human' if game_view else 'robot'
  env.reset()
  env.render(mode=render_mode)
  return env, render_mode


def get_state(my_env):
  screen = tvt.functional.to_tensor(my_env.get_screen())
  screen_cropped = tvt.functional.crop(screen, 140, 115, 180, 245)
  screen_resized = tvt.functional.resize(screen_cropped, [180, 180])
  return screen_resized


def random_apply_transform(batch, p=0.5):  # [B, C, H, W]
  mask = torch.rand(batch.size(0))
  mask1 = (mask > p) & (mask <= 0.75)
  mask2 = mask > 0.75

  if mask1.sum() > 0:
    batch[mask1] = u.add_noise(batch[mask1])

  if mask2.sum() > 0:
    batch[mask2] = tvt.functional.gaussian_blur(batch[mask2], kernel_size=(5, 9), sigma=(0.1, 5))
  
  return batch, mask


def get_min_maxmin_joints():
  MIN_ANGLE0, MAX_ANGLE0 = 0, 90  # joint1
  MIN_ANGLE1, MAX_ANGLE1 = 0, 180  # joint2
  MAX_MIN_ANGLE0 = MAX_ANGLE0 - MIN_ANGLE0
  MAX_MIN_ANGLE1 = MAX_ANGLE1 - MIN_ANGLE1
  MINS = torch.FloatTensor([MIN_ANGLE0, MIN_ANGLE1])
  MAXS_MINS = torch.FloatTensor([MAX_MIN_ANGLE0, MAX_MIN_ANGLE1])
  return MINS, MAXS_MINS


def compute_separate_loss(states, rec_states, criterion, white_coef=0.9):
  white_mask = states.sum(1, keepdim=True).repeat(1, 3, 1, 1) == 3
  white_loss = criterion(rec_states[white_mask], states[white_mask])

  other_mask = ~white_mask
  other_loss = criterion(rec_states[other_mask], states[other_mask])

  loss = (1 - white_coef) * other_loss + white_coef * white_loss
  return loss


def collect_states(n_states, max_ep_len=50):
  env, render_mode = get_game_env(game_view=False)

  states = [get_state(env)]

  for i in range(n_states-1):
    action = random.randint(0, 4)
    joints_angle, reward, target_reached, _ = env.step(action)

    env.render(mode=render_mode)

    states.append(get_state(env))

    if target_reached or i % max_ep_len == 0:
      env.reset(to_reset='target')
  
  env.close()
  
  return states


def train_model(model, optimizer, criterion, n_epochs=10, batch_size=32, memory_size=1024, vp=None, start_epoch=0):
  states = collect_states(memory_size)
  random.shuffle(states)

  for epoch in tqdm(range(start_epoch, n_epochs + start_epoch)):
    losses = []
    for i in tqdm(range(0, memory_size, batch_size), leave=False):
      batch = torch.stack(states[i:i+batch_size])

      rec_batch = model(batch)

      optimizer.zero_grad()
      loss = criterion(rec_batch, batch)
      loss.backward()
      optimizer.step()

      losses.append(loss.item())
    
    epoch_loss = np.mean(losses)
    logging.info(f'Epoch {epoch} | loss={epoch_loss:.5f}')

    if vp is not None:
      vp.line_plot('Loss', 'Train', f'Training Loss={type(criterion).__name__}', epoch, epoch_loss, x_label='Epoch')

      tmp = torch.empty(6, 3, 180, 180)
      tmp[:3] = batch[:3]
      tmp[3:] = rec_batch[:3]
      vp.image_plot('img', make_grid(tmp, nrow=3), opts={'title': 'Image_n_reconstruction'})
    
    random.shuffle(states)


def reconstruction_experiment(use_visdom=True, batch_size=32, gdn_act=False, lr=1e-3, loss_type='ms_ssim',
                              n_epochs=10, memory_size=1024):
  if use_visdom:
    vp = u.VisdomPlotter()
  
  model = AutoEncoder(gdn_act=gdn_act)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  criterion = be.MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3) if loss_type == 'ms_ssim' else torch.nn.MSELoss()

  start_epoch = 0
  while True:
    train_model(model, optimizer, criterion, n_epochs=n_epochs, batch_size=batch_size, memory_size=memory_size, vp=vp,
                start_epoch=start_epoch)
    start_epoch += n_epochs


def visdom_plotting(vp, epoch, epoch_loss, loss_type, batch, rec_batch, gdn_act):
  vp.line_plot('Loss', 'Train', f'Training Loss={loss_type}', epoch, epoch_loss, x_label='Epoch')

  tmp = torch.empty(6, 3, 180, 180)
  tmp[:3] = batch[:3]
  tmp[3:] = rec_batch[:3]
  vp.image_plot('img', make_grid(tmp, nrow=3), opts={'title': 'Image_n_reconstruction'})
  save_image(make_grid(tmp, nrow=3), f'rec_img_exps_loss_{loss_type}_gdn_{gdn_act}.png')


def train_model2(model, optimizer, criterion, states, batch_size=32, n_epochs=5, start_epoch=0, vp=None,
                 gdn_act=False, device=None, add_augmentation=False, body_infos=None, use_separate_loss=True):
  if device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  if body_infos is None:
    random.shuffle(states)
    states = list(states)
  else:
    states, body_infos = u.shuffle_lists(list(states), list(body_infos))

  for epoch in tqdm(range(start_epoch, n_epochs + start_epoch)):
    losses = []
    for i in tqdm(range(0, len(states), batch_size), leave=False):
      batch_states = torch.stack(states[i:i+batch_size]).to(device)
      batch_body_infos = torch.stack(body_infos[i:i+batch_size]).to(device) if body_infos is not None else None

      if batch_states.size(0) <= 1:
        continue

      if add_augmentation:
        batch_augmented, _ = random_apply_transform(batch_states.clone())

      rec_batch = model(batch_augmented if add_augmentation else batch_states, body_infos=batch_body_infos)

      optimizer.zero_grad()
      loss = compute_separate_loss(batch_states, rec_batch, criterion) if use_separate_loss else criterion(rec_batch, batch_states)
      loss.backward()
      optimizer.step()

      losses.append(loss.item())

    epoch_loss = np.mean(losses)
    logging.info(f'Epoch {epoch} | loss={epoch_loss:.5f}')

    if vp is not None:
      visdom_plotting(vp, epoch, epoch_loss, type(criterion).__name__, batch_states.cpu(), rec_batch.cpu(), gdn_act)


def reconstruction_experiment2(use_visdom=True, batch_size=32, gdn_act=False, lr=1e-3, loss_type='ms_ssim',
                               n_epochs=5, memory_size=1024, max_ep_len=200, add_augmentation=False, add_body_infos=False,
                               use_separate_loss=True):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  vp = u.VisdomPlotter() if use_visdom else None
  
  model = AutoEncoder(gdn_act=gdn_act, add_body_infos=add_body_infos).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  criterion = be.MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3) if loss_type == 'ms_ssim' else torch.nn.MSELoss()
  
  env, render_mode = get_game_env(game_view=False)

  start_epoch = 0
  current_ep_len = 1
  target_reached = False
  MINS, MAXS_MINS = get_min_maxmin_joints()

  states = deque([get_state(env)], maxlen=memory_size)
  body_infos = deque([(torch.FloatTensor(env.joints_angle) - MINS) / MAXS_MINS], maxlen=memory_size)
  while True:
    action = random.randint(0, 4)
    joints_angle, reward, target_reached, _ = env.step(action)

    env.render(mode=render_mode)

    states.append(get_state(env))
    body_infos.append((torch.FloatTensor(joints_angle) - MINS) / MAXS_MINS)

    if target_reached or current_ep_len % max_ep_len == 0:
      train_model2(model, optimizer, criterion, states, batch_size=batch_size, n_epochs=n_epochs, start_epoch=start_epoch,
                   vp=vp, gdn_act=gdn_act, device=device, add_augmentation=add_augmentation,
                   body_infos=body_infos if add_body_infos else None, use_separate_loss=use_separate_loss)

      start_epoch += n_epochs
      current_ep_len = 0

      env.reset(to_reset='target')
    
    current_ep_len += 1
  
  env.close()


if __name__ == '__main__':
  # Observations :
  # -> if the learning batch is composed of followed step it will not be able to learn as
  #    the sgd assumption on iid is more than not respected
  # -> same in reconstruction_experiment2, if we not use random.shuffle, it also struggle to converge
  # -> perform training on small subset of same element before gradually increase dataset size
  #    seems to allow a fast convergence (reconstruction_experiment2 >> reconstruction_experiment)
  #    small overfitting to start learning is beneficial?
  # -> reducing n_epochs on reconstruction_experiment2 seems to allow a faster convergence,
  #    it may be because it avoid temporary small overfitting?
  #    But leaving n_epochs higher allow the network to more easily reconstruct the red target
  # -> reconstruction_experiment2 converge in ~20 epochs achieved in ~6min
  # -> reconstruction_experiment struggle to reconstruct the red target and is not able to do it
  #    even after 20 epochs achieved in ~10min
  # -> ms_ssim loss is slower and struggle to reconstruct the red target
  # -> gde activation with same hyperparameter seems to make convergence slower and is computationnally heavier
  # -> between 2 runs, the reconstruction of the red point vary a lot and often it struggles to reconstruct it
  # -> By using a separate loss for white pixels and other pixels and keeping a ratio giving more importance
  #    to the white loss (as there is way more white pixels) allow a fast convergence to a good reconstruction
  argparser = argparse.ArgumentParser(prog='rec_exps.py', description='')
  argparser.add_argument('--log_file', default='_tmp_rec_exps_logs.txt', type=str)
  argparser.add_argument('--use_visdom', default=True, type=ast.literal_eval)
  argparser.add_argument('--save_model', default=True, type=ast.literal_eval)
  argparser.add_argument('--load_model', default=True, type=ast.literal_eval)
  argparser.add_argument('--gdn_act', default=False, type=ast.literal_eval)
  argparser.add_argument('--add_augmentation', default=False, type=ast.literal_eval)
  argparser.add_argument('--add_body_infos', default=False, type=ast.literal_eval)
  argparser.add_argument('--use_separate_loss', default=True, type=ast.literal_eval)
  argparser.add_argument('--seed', default=42, type=int)
  argparser.add_argument('--n_epochs', default=5, type=int)
  argparser.add_argument('--batch_size', default=32, type=int)
  argparser.add_argument('--max_ep_len', default=200, type=int)
  argparser.add_argument('--memory_size', default=1024, type=int)
  argparser.add_argument('--lr', default=1e-3, type=float)
  argparser.add_argument('--loss_type', default='mse', type=str, choices=['mse', 'ms_ssim'])
  args = argparser.parse_args()

  logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')
  
  rep = input('Start training? (y or n): ')
  if rep == 'y':
    # seeding for reproducibility
    random.seed(args.seed * args.seed)
    torch.manual_seed(args.seed)

    u.dump_argparser_parameters(args)

    # reconstruction_experiment(use_visdom=args.use_visdom, batch_size=args.batch_size, gdn_act=args.gdn_act,
    #                           lr=args.lr, loss_type=args.loss_type, n_epochs=args.n_epochs, memory_size=args.memory_size)
    reconstruction_experiment2(use_visdom=args.use_visdom, batch_size=args.batch_size, gdn_act=args.gdn_act, lr=args.lr,
                               loss_type=args.loss_type, n_epochs=args.n_epochs, memory_size=args.memory_size,
                               max_ep_len=args.max_ep_len, add_augmentation=args.add_augmentation,
                               add_body_infos=args.add_body_infos, use_separate_loss=args.use_separate_loss)
  

  # Example to update lr if threshold
  # if loss < 0.005:
  #   for param_group in optimizer.param_groups:
  #     if param_group['lr'] == 1e-3:
  #       param_group['lr'] = 1e-4
  #       print('!LR UPDATED!!!!!!!!!!!!!!!!')