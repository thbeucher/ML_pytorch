import os
import ast
import gym
import sys
import torch
import pygame
import logging
import argparse
import numpy as np

## VARIABLES for 2-DOF robot arm experiment #########################################################
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
#####################################################################################################

sys.path.append(os.path.abspath(__file__).replace('USS-GL/pg_exps.py', ''))
import utils as u
import pg_utils as pgu


def get_game_env(game_view=False):
  env = gym.make('gym_robot_arm:robot-arm-v1')
  env.config['in_background'] = not game_view
  render_mode = 'human' if game_view else 'robot'
  env.reset()
  env.render(mode=render_mode)
  return env, render_mode


class TOYModel(object):
  def __init__(self):
    super().__init__()
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = torch.nn.Sequential(torch.nn.Linear(4, 200),
                                     torch.nn.ReLU(inplace=True),
                                     torch.nn.Linear(200, 5),
                                     torch.nn.Softmax(-1)).to(self.device)
  
  def select_action(self, state):
    probs = self.model(state)
    distri = torch.distributions.Categorical(probs)
    action = distri.sample()
    return action.item(), distri.log_prob(action)


def train(game_view=False, lr=1e-3, max_game_timestep=200, n_game_scoring_average=100, use_visdom=False,
          load_model=True, save_model=True, save_name='models/toyModel.pt'):
  if use_visdom:
    vp = u.VisdomPlotter()
    plot_iter = 0

  env, render_mode = get_game_env(game_view=game_view)

  # Instanciate model and optimizer
  model = TOYModel()
  optimizer = torch.optim.AdamW(model.model.parameters(), lr=lr)

  if load_model:
    u.load_model(model.model, save_name)

  # Starting variables
  dist_eff_target = env.current_dist
  state = (torch.FloatTensor(env.joints_angle + list(env.target_pos)) - MINS) / MAXS_MINS

  # Storage variables
  rewards = []
  log_probs = []

  # Control variables
  quit_game = False
  current_game_timestep = 0
  time_to_target_memory = []

  while not quit_game:
    # If game window is open, catch closing event
    if game_view:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          quit_game = True
    
    # Get action from model then perform it
    action, log_prob = model.select_action(state)
    joints_angle, reward, target_reached, _ = env.step(action)

    current_game_timestep += 1
    env.render(mode=render_mode)

    # Store data and update state
    rewards.append(reward)
    log_probs.append(log_prob)
    state = (torch.FloatTensor(joints_angle + list(env.target_pos)) - MINS) / MAXS_MINS

    if target_reached or current_game_timestep > max_game_timestep:
      if target_reached:  # update policy only if the target is reached as there is no reward before that
        pgu.reinforce_update(rewards, log_probs, optimizer)

        time_to_target_memory.append(current_game_timestep / dist_eff_target)

        if use_visdom and len(time_to_target_memory) == n_game_scoring_average:
          average_ttt = np.mean(time_to_target_memory)
          vp.line_plot('Time to Target', 'Train', 'Policy Performance', plot_iter, average_ttt)
          logging.info(f'Episode {plot_iter}(x100) | Average time-to-target={average_ttt:.3f}')
          time_to_target_memory.clear()
          plot_iter += 1

          if save_model:
            u.save_checkpoint(model.model, None, save_name)

      # Reset game and related variables
      env.reset(to_reset='target')
      dist_eff_target = env.current_dist
      current_game_timestep = 0
  
  env.close()


if __name__ == '__main__':
  argparser = argparse.ArgumentParser(prog='pg_exps.py',
                                      description='Experiments on 2-DOF robot arm that learn to reach a target point')
  argparser.add_argument('--log_file', default='_tmp_pg_exps_logs.txt', type=str)
  argparser.add_argument('--use_visdom', default=True, type=ast.literal_eval)
  argparser.add_argument('--game_view', default=False, type=ast.literal_eval)
  argparser.add_argument('--save_model', default=True, type=ast.literal_eval)
  argparser.add_argument('--load_model', default=True, type=ast.literal_eval)
  args = argparser.parse_args()

  logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')
  
  rep = input('Start training? (y or n): ')
  if rep == 'y':
    train(game_view=args.game_view, use_visdom=args.use_visdom, load_model=args.load_model, save_model=args.save_model)