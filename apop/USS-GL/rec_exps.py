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
  def __init__(self, gdn_act=False):
    super().__init__()
    self.encoder = be.ImageEmbedder(gdn_act=gdn_act)
    self.decoder = be.ImageReconstructor(gdn_act=gdn_act)
  
  def forward(self, x):
    return self.decoder(self.encoder(x))


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


def reconstruction_experiment2(use_visdom=True, batch_size=32, gdn_act=False, lr=1e-3, loss_type='ms_ssim',
                               n_epochs=5, memory_size=1024, max_ep_len=200):
  if use_visdom:
    vp = u.VisdomPlotter()
  
  model = AutoEncoder(gdn_act=gdn_act)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  criterion = be.MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3) if loss_type == 'ms_ssim' else torch.nn.MSELoss()
  
  env, render_mode = get_game_env(game_view=False)

  start_epoch = 0
  current_ep_len = 1
  target_reached = False

  batch_states = deque([get_state(env)], maxlen=memory_size)
  while True:
    action = random.randint(0, 4)
    joints_angle, reward, target_reached, _ = env.step(action)

    env.render(mode=render_mode)

    batch_states.append(get_state(env))

    if target_reached or current_ep_len % max_ep_len == 0:
      random.shuffle(batch_states)

      for epoch in tqdm(range(start_epoch, n_epochs + start_epoch)):
        losses = []
        for i in tqdm(range(0, len(batch_states), batch_size), leave=False):
          batch = torch.stack(list(itertools.islice(batch_states, i, i+batch_size)))

          if batch.size(0) <= 1:
            continue

          rec_batch = model(batch)

          optimizer.zero_grad()
          loss = criterion(rec_batch, batch)
          loss.backward()
          optimizer.step()

          losses.append(loss.item())

        epoch_loss = np.mean(losses)
        logging.info(f'Epoch {epoch} | loss={epoch_loss:.5f}')

        if use_visdom:
          vp.line_plot('Loss', 'Train', f'Training Loss={loss_type}', epoch, epoch_loss, x_label='Epoch')

          tmp = torch.empty(6, 3, 180, 180)
          tmp[:3] = batch[:3]
          tmp[3:] = rec_batch[:3]
          vp.image_plot('img', make_grid(tmp, nrow=3), opts={'title': 'Image_n_reconstruction'})
          save_image(make_grid(tmp, nrow=3), f'rec_img_exps_loss_{loss_type}_gdn_{gdn_act}.png')

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
  # -> reconstruction_experiment2 converge in ~20 epochs achieved in ~6min
  # -> reconstruction_experiment struggle to reconstruct the red target and is not able to do it
  #    even after 20 epochs achieved in ~10min
  # -> ms_ssim loss is slower and struggle to reconstruct the red target
  # -> gde activation with same hyperparameter seems to make convergence slower and is computationnally heavier
  argparser = argparse.ArgumentParser(prog='rec_exps.py', description='')
  argparser.add_argument('--log_file', default='_tmp_rec_exps_logs.txt', type=str)
  argparser.add_argument('--use_visdom', default=True, type=ast.literal_eval)
  argparser.add_argument('--save_model', default=True, type=ast.literal_eval)
  argparser.add_argument('--load_model', default=True, type=ast.literal_eval)
  argparser.add_argument('--gdn_act', default=False, type=ast.literal_eval)
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
                               max_ep_len=args.max_ep_len)