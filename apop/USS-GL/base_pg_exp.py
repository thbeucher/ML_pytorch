import os
import ast
import gym
import sys
import torch
import random
import pygame
import visdom
import logging
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import deque

sys.path.append('../../../robot/')

MIN_ANGLE0, MAX_ANGLE0 = 0, 90  # joint1
MIN_ANGLE1, MAX_ANGLE1 = 0, 180  # joint2
MIN_X_TARGET, MAX_X_TARGET = 125, 350
MIN_Y_TARGET, MAX_Y_TARGET = 150, 300
MAX_MIN_ANGLE0 = MAX_ANGLE0 - MIN_ANGLE0
MAX_MIN_ANGLE1 = MAX_ANGLE1 - MIN_ANGLE1
MAX_MIN_X_TARGET = MAX_X_TARGET - MIN_X_TARGET
MAX_MIN_Y_TARGET = MAX_Y_TARGET - MIN_Y_TARGET
MAXS_MINS = torch.FloatTensor([MAX_MIN_ANGLE0, MAX_MIN_ANGLE1, MAX_MIN_X_TARGET, MAX_MIN_Y_TARGET])
MINS = torch.FloatTensor([MIN_ANGLE0, MIN_ANGLE1, MIN_X_TARGET, MIN_Y_TARGET])


class PolicyGradient(object):
  BASE_CONFIG = {'gamma': 0.99, 'model_folder': 'models/', 'model_fname': 'pg_base.pt', 'mtype': 'mlp',
                 'lr': 1e-3, 'update_step': 64, 'max_timesteps': 200}
  def __init__(self, config={}):
    self.config = {**PolicyGradient.BASE_CONFIG, **config}
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.instanciate_model()
    self.optimizer = torch.optim.AdamW(self.brain.parameters(), lr=self.config['lr'])
    self.saved_log_probs = []
    self.rewards = []
    self.gamma = self.config['gamma']
  
  def instanciate_model(self):
    self.brain = torch.nn.Sequential(torch.nn.Linear(4, 200),
                                     torch.nn.ReLU(inplace=True),
                                     torch.nn.Linear(200, 5),
                                     torch.nn.Softmax(-1)).to(self.device)
  
  def select_action(self, state, save_logprobs=True):
    probs = self.brain(state)
    distri = torch.distributions.Categorical(probs)
    action = distri.sample()
    if save_logprobs:
      self.saved_log_probs.append(distri.log_prob(action))
    return action.item()

  def update_brain(self, eps=1e-6):
    # compute returns -> R = r + gamma * R
    R = 0
    returns = []
    for r in self.rewards[::-1]:
      R = r + self.gamma * R
      returns.insert(0, R)
    # normalize returns
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps) if len(self.rewards) > 1 else returns
    # compute policy loss -> -log_prob * R
    policy_loss = []
    for log_prob, R in zip(self.saved_log_probs, returns):
      policy_loss.append(-log_prob * R)
    # zero_grad -> backward -> optimizer step
    self.optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    # vis.heatmap(X=self.brain[0].weight.grad, win='grad', opts={'colormap': 'seismic'})
    self.optimizer.step()
    # clear memory
    self.saved_log_probs.clear()
    self.rewards.clear()
  
  def save_brain(self):
    if not os.path.isdir(self.config['model_folder']):
      os.makedirs(self.config['model_folder'])
    torch.save(self.brain.state_dict(), os.path.join(self.config['model_folder'], self.config['model_fname']))

  def load_brain(self):
    save_name = os.path.join(self.config['model_folder'], self.config['model_fname'])
    if os.path.isfile(save_name):
      self.brain.load_state_dict(torch.load(save_name, map_location=self.device))
    else:
      print(f"File {save_name} doesn't exist")

  def train_loop(self, game_view=False, use_visdom=True, save_model=True):
    # env.target_pos | env.joints_angle
    env = gym.make('gym_robot_arm:robot-arm-v1')
    env.config['in_background'] = not game_view
    mode = 'human' if game_view else 'robot'
    env.reset()
    env.render(mode=mode)

    dist_eff_target = env.current_dist
    state = (torch.FloatTensor(env.joints_angle + list(env.target_pos)) - MINS) / MAXS_MINS

    i, j, k, m = 0, 0, 0, 0
    quit_game, time_to_target_ = False, deque(maxlen=100)
    while not quit_game:
      if game_view:
        for event in pygame.event.get():
          if event.type == pygame.QUIT:
            quit_game = True

      action = self.select_action(state)
      joints_angle, reward, target_reached, _ = env.step(action)
      i += 1
      j += 1

      env.render(mode=mode)

      self.rewards.append(reward)
      state = (torch.FloatTensor(joints_angle + list(env.target_pos)) - MINS) / MAXS_MINS

      # if i % self.config['update_step'] == 0:
      #   self.update_brain()
      #   if self.config['mtype'] == 'rnn':
      #     self.clone_state()
      
      if target_reached or j > self.config['max_timesteps']:
        if target_reached:
          self.update_brain()  # if reward received only at the end of the episode

          time_to_target = j / dist_eff_target
          # 2022-11-11 18:37:42,077 - INFO - Episode 3962 | dist=191.72 | n_steps=83 | ratio=2.310
          logging.info(f'Episode {k} | dist={dist_eff_target:.2f} | n_steps={j} | ratio={time_to_target:.3f}')
          time_to_target_.append(time_to_target)
          if use_visdom and k % 100 == 0:
            plot_metric(np.mean(time_to_target_), m)
            m += 1
          k += 1

        env.reset(to_reset='target')
        dist_eff_target = env.current_dist
        j = 0

        if self.config['mtype'] == 'rnn':
          self.reset_state()
      
      if save_model and k % 50 == 0:
        self.save_brain()
    
    env.close()


class RNNPolicyGradient(PolicyGradient):
  BASE_CONFIG = {'model_fname': 'pg_rnn.pt', 'mtype': 'rnn', 'lr': 1e-5, 'update_step': 16}
  def __init__(self, config={}):
    self.config = {**RNNPolicyGradient.BASE_CONFIG, **config}
    super().__init__(self.config)
    self.hidden_state = torch.zeros((1, 200))
    self.cell_state = torch.zeros((1, 200))
  
  def clone_state(self):
    self.hidden_state = self.hidden_state.detach()
    self.cell_state = self.cell_state.detach()
  
  def reset_state(self):
    self.hidden_state = torch.zeros((1, 200))
    self.cell_state = torch.zeros((1, 200))
  
  def instanciate_model(self):
    self.brain = torch.nn.LSTMCell(4, 200).to(self.device)
    self.brain_out = torch.nn.Sequential(torch.nn.Linear(200, 5), torch.nn.Softmax(-1)).to(self.device)
  
  def select_action(self, state, save_logprobs=True):
    self.hidden_state, self.cell_state = self.brain(state.reshape((1, -1)), (self.hidden_state, self.cell_state))
    probs = self.brain_out(self.hidden_state)
    distri = torch.distributions.Categorical(probs)
    action = distri.sample()
    if save_logprobs:
      self.saved_log_probs.append(distri.log_prob(action))
    return action.item()


def plot_metric(metric, iteration, win='time_to_target', title='Time-to-Target', ylabel='dist/time', xlabel='n_games'):
    if iteration == 0:
      vis.line(X=np.array([iteration, iteration]), Y=np.array([metric, metric]), win=win,
                                opts={'ylabel': ylabel, 'xlabel': xlabel, 'title': title})
    else:
      vis.line(X=np.array([iteration]), Y=np.array([metric]), win=win, update='append')


def train_net(game_view=False, use_visdom=True, save_model=True, net='pg', load_model=True):
  nets = {'pg': PolicyGradient, 'rnn_pg': RNNPolicyGradient}
  pg = nets[net]()
  if load_model:
    pg.load_brain()
  pg.train_loop(game_view=game_view, use_visdom=use_visdom, save_model=save_model)


def see_net(game_view=False):
  env = gym.make('gym_robot_arm:robot-arm-v1')
  env.config['in_background'] = not game_view
  mode = 'human' if game_view else 'robot'
  env.reset()
  env.render(mode=mode)

  pg = PolicyGradient()
  pg.load_brain()

  state = (torch.FloatTensor(env.joints_angle + list(env.target_pos)) - MINS) / MAXS_MINS

  quit_game, j = False, 0
  while not quit_game:
    if game_view:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          quit_game = True

    action = pg.select_action(state)
    joints_angle, reward, target_reached, _ = env.step(action)

    env.render(mode=mode)

    state = (torch.FloatTensor(joints_angle + list(env.target_pos)) - MINS) / MAXS_MINS

    if target_reached or j > 200:
      env.reset(to_reset='target')
      j = 0
  
  env.close()


def pg_exp():
  env = gym.make('gym_robot_arm:robot-arm-v1')
  env.reset()
  env.render()

  done = False
  action, target_reached = 0, False

  while not done:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        done = True
      
      ## MANUAL CONTROL ##
      # elif event.type == pygame.KEYDOWN:
      #   if event.key == pygame.K_a:
      #     action = 1
      #     body_info, _, target_reached, _ = env.step(action)  # INC_J1
      #   elif event.key == pygame.K_z:
      #     action = 2
      #     body_info, _, target_reached, _ = env.step(action)  # DEC_J1
      #   elif event.key == pygame.K_o:
      #     action = 3
      #     body_info, _, target_reached, _ = env.step(action)  # INC_J2
      #   elif event.key == pygame.K_p:
      #     action = 4
      #     body_info, _, target_reached, _ = env.step(action)  # DEC_J2
      ## MANUAL CONTROL ##

    action = random.randint(0, 4)
    joints_angle, reward, target_reached, _ = env.step(action)

    if target_reached:
      env.reset(to_reset='target')

    env.render()
  
  env.close()


def viz_vals(folder):
  all_data = {'n_episodes(x100)': [], 'n_steps/dist': []}
  for filename in os.listdir(folder):
    if 'logs_exp' in filename:
      with open(os.path.join(folder, filename), 'r') as f:
        data = f.read().splitlines()
      vals = [[int(l.split(' | ')[-4].split('Episode ')[-1]),\
               float(l.split(' | ')[-3].split('=')[-1]),\
               int(l.split(' | ')[-2].split('=')[-1]),\
               float(l.split(' | ')[-1].split('=')[-1])] for l in data]
      ratio = [el[-1] for el in vals]
      y = [np.mean(ratio[i:i+100]) for i in range(0, 30000, 100)]
      all_data['n_episodes(x100)'] += list(range(len(y)))
      all_data['n_steps/dist'] += y
  df = pd.DataFrame.from_dict(all_data)
  sns.lineplot(data=df, x='n_episodes(x100)', y='n_steps/dist')
  plt.show()
  return all_data


if __name__ == '__main__':
  argparser = argparse.ArgumentParser(prog='base_pg_exp.py',
                                      description='Experiment on 2-DOF robot arm that learn to reach a target point')
  argparser.add_argument('--log_file', default='_tmp_simple_pg_exp_logs.txt', type=str)
  argparser.add_argument('--use_visdom', default=True, type=ast.literal_eval)
  argparser.add_argument('--game_view', default=False, type=ast.literal_eval)
  argparser.add_argument('--save_model', default=True, type=ast.literal_eval)
  argparser.add_argument('--load_model', default=True, type=ast.literal_eval)
  argparser.add_argument('--net', default='pg', type=str)  # pg | rnn_pg
  args = argparser.parse_args()
  logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')

  if args.use_visdom:
    vis = visdom.Visdom()
  # ROADMAP
  # REINFORCE (https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py)
  # Hindsight goal generation
  # Bootstrap with IL/BC
  # pg_exp()

  rep = input('Start training? (y or n): ')
  if rep == 'y':
    train_net(game_view=args.game_view, use_visdom=args.use_visdom, save_model=args.save_model, net=args.net,
              load_model=args.load_model)
  
  rep = input('Start evaluation? (y or n): ')
  if rep == 'y':
    see_net(game_view=True)