import os
import sys
import gym
import time
import torch
import pygame
import visdom
import random
import logging
import argparse
import threading
import numpy as np
import torchvision.transforms as tvt

from tqdm import tqdm
from collections import deque
from torchvision.io import read_image
from torchvision.utils import make_grid, save_image

sys.path.append('../../../robot/')
sys.path.append(os.path.abspath(__file__).replace('USS-GL/simple_goal_exps.py', ''))

import models.gan_vae_divers as gvd


class ReplayMemory(object):
  def __init__(self, capacity):
    self.memory = deque([], maxlen=capacity)

  def push(self, data):
    self.memory.append(data)

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)


class GlobalTrainer(object):
  BASE_CONFIG = {'batch_size': 32}
  def __init__(self, config):
    self.config = {**GlobalTrainer.BASE_CONFIG, **config}

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.instanciate_model()
    self.instanciate_optimizers_n_criterions()

    self.loss_plot = None
  
  def instanciate_model(self):
    # vfl = visual_feature_learner
    # with batch normalization, it learns faster but creates some artefacts
    vfl_residualCNN_config = {'layers_config': [
      {'type': torch.nn.ReLU, 'params': {'inplace': True}},
      {'type': torch.nn.Conv2d, 'params': {'in_channels': 64, 'out_channels': 16, 'kernel_size': 3, 'stride': 1,
                                           'padding': 1, 'bias': False}},
      {'type': torch.nn.ReLU, 'params': {'inplace': True}},
      {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 16}},
      {'type': torch.nn.Conv2d, 'params': {'in_channels': 16, 'out_channels': 64, 'kernel_size': 1, 'stride': 1,
                                           'padding': 0, 'bias': False}}]}
    vfl_encoder_config = {'layers_config': [
      # 3*180*180 -> 32*90*90
      {'type': torch.nn.Conv2d, 'params': {'in_channels': 3, 'out_channels': 32, 'kernel_size': 4, 'stride': 2, 'padding': 1}},
      {'type': torch.nn.ReLU, 'params': {'inplace': True}},
      {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 32}},
      # 32*90*90 -> 64*45*45
      {'type': torch.nn.Conv2d, 'params': {'in_channels': 32, 'out_channels': 64, 'kernel_size': 4, 'stride': 2, 'padding': 1}},
      {'type': torch.nn.ReLU, 'params': {'inplace': True}},
      {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 64}},
      # 64*45*45 -> 64*45*45
      {'type': torch.nn.Conv2d, 'params': {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}},
      # 2 Residual blocks
      {'type': gvd.ResidualCNN, 'params': {'config': vfl_residualCNN_config}},
      {'type': gvd.ResidualCNN, 'params': {'config': vfl_residualCNN_config}},
      {'type': torch.nn.ReLU, 'params': {'inplace': True}}]}
    vfl_pre_vq_conv_config = {'in_channels': 64, 'out_channels': 32, 'kernel_size': 1, 'stride': 1, 'padding': 0}
    vfl_decoder_config = {'layers_config': [
      # 32*45*45 -> 64*45*45
      {'type': torch.nn.Conv2d, 'params': {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}},
      # 2 Residual blocks
      {'type': gvd.ResidualCNN, 'params': {'config': vfl_residualCNN_config}},
      {'type': gvd.ResidualCNN, 'params': {'config': vfl_residualCNN_config}},
      # 64*45*45 -> 32*90*90
      {'type': torch.nn.ConvTranspose2d, 'params': {'in_channels': 64, 'out_channels': 32, 'kernel_size': 4,
                                                    'stride': 2, 'padding': 1}},
      {'type': torch.nn.ReLU, 'params': {'inplace': True}},
      # 32*90*90 -> 3*180*180
      {'type': torch.nn.ConvTranspose2d, 'params': {'in_channels': 32, 'out_channels': 3, 'kernel_size': 4,
                                                    'stride': 2, 'padding': 1}}]}
    self.visual_feature_learner = gvd.VQVAEModel({'encoder_config': vfl_encoder_config,
                                                  'pre_vq_conv_config': vfl_pre_vq_conv_config,
                                                  'vq_config': {'n_embeddings': 512, 'embedding_dim': 32},
                                                  'decoder_config': vfl_decoder_config}).to(self.device)
  
  def instanciate_optimizers_n_criterions(self):
    self.vfl_optimizer = torch.optim.AdamW(self.visual_feature_learner.parameters())
    self.vfl_criterion = torch.nn.MSELoss()
    self.vfl_replay_memory = ReplayMemory(12800)
  
  def get_state(self):
    screen = tvt.functional.to_tensor(env.get_screen())
    screen_cropped = tvt.functional.crop(screen, 140, 115, 180, 245)
    screen_resized = tvt.functional.resize(screen_cropped, [180, 180])
    return screen_resized
  
  def plot_loss(self, loss, iteration):
    if self.loss_plot is None:
      self.loss_plot = vis.line(X=np.array([iteration, iteration]), Y=np.array([loss.item(), loss.item()]), win='loss',
                                opts={'ylabel': 'loss', 'xlabel': 'iteration', 'title': 'loss evolution'})
    else:
      vis.line(X=np.array([iteration]), Y=np.array([loss.item()]), win='loss', update='append')
  
  def train(self):
    # fill the replay memory to have minimal batch size
    state = self.get_state()
    for _ in range(self.config['batch_size']):
      self.vfl_replay_memory.push(state)

    for i in tqdm(range(1000)):
      action = random.randint(0, 4)
      env.step(action)
      state = self.get_state()
      self.vfl_replay_memory.push(state)

      # Learn visual features
      states_batch = torch.stack(self.vfl_replay_memory.sample(self.config['batch_size'])).to(self.device)
      vq_loss, state_rec, perplexity, encodings, quantized = self.visual_feature_learner(states_batch)
      self.vfl_optimizer.zero_grad()
      rec_loss = self.vfl_criterion(state_rec, states_batch)
      loss = vq_loss + rec_loss
      loss.backward()
      self.vfl_optimizer.step()

      # Plot images target and reconstructed using visdom
      vis.image(make_grid(states_batch[:6].cpu(), nrow=3), win='state', opts={'title': 'state'})
      # Strangely, it plots black image so an hack is to save the image first then load it to pass it to visdom
      # vis.image(make_grid(state_rec[:6].cpu(), nrow=3), win='rec', opts={'title': 'reconstructed'})
      save_image(make_grid(state_rec[:6].cpu(), nrow=3), 'test_make_grid.png')
      vis.image(read_image('test_make_grid.png'), win='rec', opts={'title': 'reconstructed'})

      self.plot_loss(loss, i)


def stream_img(win_id=None):
  screen = torch.from_numpy(env.get_screen()).permute(2, 0, 1)
  screen_cropped = screen[:, 300-(150+10):320, 200-(75+10):200+150+10]  # [3, 180, 245]
  win_id = vis.image(screen, win=win_id, opts={'title': 'screen'})
  vis.image(screen_cropped, win='cropped', opts={'title': 'cropped'})
  vis.image(tvt.functional.resize(screen_cropped, [180, 180]), win='cropped_resized', opts={'title': 'resized'})


def experiment():
  gt = GlobalTrainer({})
  gt.train()


def fake_press_a():
  print('will simulate pressing A key in 5s and every 2s')
  time.sleep(5)
  for _ in range(10):
    newevent = pygame.event.Event(pygame.KEYDOWN, unicode="a", key=pygame.K_a, mod=pygame.KMOD_NONE)
    pygame.event.post(newevent)
    time.sleep(2)


if __name__ == '__main__':
  argparser = argparse.ArgumentParser(prog='simple_goal_exps.py',
                                      description='Experiment on 2-DOF robot arm that learn to reach a target point')
  argparser.add_argument('--log_file', default='_tmp_simple_goal_exps_logs.txt', type=str)
  argparser.add_argument('--tensorboard_exp', default='simple_goal_experiment_1', type=str)
  args = argparser.parse_args()

  logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')
  
  vis = visdom.Visdom()  # https://github.com/fossasia/visdom#usage
  
  # env.joints_angle = [angle_joint1, angle_joint2]
  # joint1 -> torch.nn.Embedding(n_emb=91, emb_size=8)
  # joint2 -> torch.nn.Embedding(n_emb=181, emb_size=8)
  # env.get_screen() = np.ndarray = [400, 400, 3]

  # MAIN LOOP SIMULATION
  env = gym.make('gym_robot_arm:robot-arm-v0')
  env.render()
  env.reset()

  done = False
  act = False

  th = threading.Thread(target=experiment)
  # th = threading.Thread(target=fake_press_a)
  th.start()

  while not done:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        done = True
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_a:
          env.step(1)  # INC_J1
          act = True
        elif event.key == pygame.K_z:
          env.step(2)  # DEC_J1
          act = True
        elif event.key == pygame.K_o:
          env.step(3)  # INC_J2
          act = True
        elif event.key == pygame.K_p:
          env.step(4)  # DEC_J2
          act = True
    
    if act:
      print('Action taken!')
      stream_img(win_id='screen')
      act = False
    
    env.render()

  env.close()
  th.join()