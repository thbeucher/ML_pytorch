import os
import ast
import gym
import sys
import time
import torch
import pygame
import random
import logging
import argparse
import numpy as np

from collections import deque
from datetime import timedelta
from scipy.stats import linregress

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
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = torch.nn.Sequential(torch.nn.Linear(4, 200),
                                     torch.nn.ReLU(inplace=True),
                                     torch.nn.Linear(200, 5),
                                     torch.nn.Softmax(-1)).to(self.device)
  
  def select_action(self, state):
    action_probs = self.model(state)
    distri = torch.distributions.Categorical(action_probs)
    action = distri.sample()
    action_logprob = distri.log_prob(action)
    return action.item(), action_logprob, distri.entropy()
  
  def act(self, state):
    action_probs = self.model(state)
    distri = torch.distributions.Categorical(action_probs)
    action = distri.sample()
    action_logprob = distri.log_prob(action)
    return action.detach(), action_logprob.detach()

  def evaluate(self, state, action):
    action_probs = self.model(state)
    distri = torch.distributions.Categorical(action_probs)
    action_logprob = distri.log_prob(action)
    return action_logprob, distri.entropy()


class TOYActorCritic(torch.nn.Module):
  BASE_CONFIG = {'state_size': 4, 'n_actions': 5, 'hidden_size': 200}
  def __init__(self, configuration={}):
    super().__init__()
    self.config = {**TOYActorCritic.BASE_CONFIG, **configuration}
    self.shared = torch.nn.Linear(self.config['state_size'], self.config['hidden_size'])
    self.actor = torch.nn.Linear(self.config['hidden_size'], self.config['n_actions'])
    self.critic = torch.nn.Linear(self.config['hidden_size'], 1)
  
  def forward(self, state, critic=False, keep_action_logits=False):
    out = torch.nn.functional.relu(self.shared(state))
    action_logits = self.actor(out)
    action_probs = action_logits if keep_action_logits else torch.nn.functional.softmax(action_logits, dim=-1)
    state_values = self.critic(out) if critic else None
    return action_probs, state_values


class TOYACModel(object):
  def __init__(self, model=TOYActorCritic, model_conf={}):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = model(model_conf).to(self.device)
  
  def act(self, state):  # [state_size]
    action_probs, _ = self.model(state)  # [n_actions]
    distri = torch.distributions.Categorical(action_probs)
    action = distri.sample()
    action_logprob = distri.log_prob(action)
    return action.detach(), action_logprob.detach()  # [], []

  def evaluate(self, state, action):  # [ep_len, state_size], [ep_len]
    action_probs, state_values = self.model(state, critic=True)  # [ep_len, n_actions], [ep_len, 1]
    distri = torch.distributions.Categorical(action_probs)
    action_logprob = distri.log_prob(action)  # [ep_len]
    return action_logprob, state_values, distri.entropy()
  
  def select_action(self, state):
    action_probs, state_values = self.model(state, critic=True)
    distri = torch.distributions.Categorical(action_probs)
    action = distri.sample()
    action_logprob = distri.log_prob(action)
    return action.item(), action_logprob, state_values, distri.entropy()


def pretrain_CE(dataset_folder='goal_reaching_dataset/', lr=1e-3, label_smoothing=0.1, episode_batch=False,
                coef_mse_critic=0.5, coef_CE_actor=1.0, n_epochs=50, batch_size=32, model=TOYActorCritic, model_conf={}):
  # filename = ep_X_len_Y.pt
  # content -> [states, body_infos, actions, rewards]
  # states/body_infos = list of FloatTensor | actions/rewards = list of int/float
  print(f'Start pretraining {model.__name__} model using cross-entropy loss...')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = model(model_conf).to(device)
  actor_criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
  critic_criterion = torch.nn.MSELoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

  # Preparing data
  all_states, all_actions, all_returns = [], [], []
  for fname in os.listdir(dataset_folder):
    states, _, actions, rewards = torch.load(os.path.join(dataset_folder, fname))
    returns = pgu.get_returns(rewards)

    if episode_batch:
      all_states.append(states)
      all_actions.append(actions)
      all_returns.append(returns)
    else:
      all_states += states
      all_actions += actions
      all_returns += returns
  
  # Training
  for epoch in range(n_epochs):
    tmp = list(zip(all_states, all_actions, all_returns))
    random.shuffle(tmp)
    all_states, all_actions, all_returns = zip(*tmp)

    batch_number = 0
    batch_size = 1 if episode_batch else batch_size
    for i in range(0, len(all_actions), batch_size):
      if episode_batch:
        batch_states = all_states[i]
        batch_actions = all_actions[i]
        batch_returns = all_returns[i]
      else:
        batch_states = all_states[i:i+batch_size]
        batch_actions = all_actions[i:i+batch_size]
        batch_returns = torch.stack(all_returns[i:i+batch_size])

      action_logits, state_values = model(torch.stack(batch_states), critic=True, keep_action_logits=True)

      actor_loss = coef_CE_actor * actor_criterion(action_logits, torch.LongTensor(batch_actions))
      critic_loss = coef_mse_critic * critic_criterion(state_values, batch_returns.unsqueeze(-1))
      loss = actor_loss + critic_loss

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      print(f'Epoch {epoch} | batch {batch_number} | loss={loss.item():.4f}', end='\r')
      batch_number += 1
  
  print(f'\nPretraining done.')
  return model.state_dict()


def train_reinforce(game_view=False, lr=1e-3, max_game_timestep=200, n_game_scoring_average=100, use_visdom=False,
                    load_model=True, save_model=True, save_name='models/toyModel.pt', AC=False, coef_entropy=0.01,
                    model=TOYActorCritic, model_conf={}, pretraining=False, episode_batch=False):
  if use_visdom:
    vp = u.VisdomPlotter()

  env, render_mode = get_game_env(game_view=game_view)

  # Instanciate model and optimizer
  policy = TOYACModel(model=model, model_conf=model_conf) if AC else TOYModel()
  optimizer = torch.optim.AdamW(policy.model.parameters(), lr=lr)

  if load_model:
    u.load_model(policy.model, save_name.replace('toy', 'toyAC') if AC else save_name)

  # Starting variables
  dist_eff_target = env.current_dist
  state = (torch.FloatTensor(env.joints_angle + list(env.target_pos)) - MINS) / MAXS_MINS

  # Storage variables
  rewards = []
  log_probs = []
  if AC:
    state_values = []

  # Control variables
  quit_game = False
  plot_iter = 0
  current_game_timestep = 0
  time_to_target_memory = []

  while not quit_game:  #TODO add stopping criteria if performance stop improving
    # If game window is open, catch closing event
    if game_view:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          quit_game = True
    
    # Get action from model then perform it
    if AC:
      action, log_prob, state_value, distri_entropy = policy.select_action(state)
    else:
      action, log_prob, distri_entropy = policy.select_action(state)

    joints_angle, reward, target_reached, _ = env.step(action)

    current_game_timestep += 1
    env.render(mode=render_mode)

    # Store data and update state
    rewards.append(reward)
    log_probs.append(log_prob)
    if AC:
      state_values.append(state_value)
    state = (torch.FloatTensor(joints_angle + list(env.target_pos)) - MINS) / MAXS_MINS

    if target_reached or current_game_timestep > max_game_timestep:
      if target_reached:  # update policy only if the target is reached as there is no reward before that
        pgu.reinforce_update(rewards, log_probs, distri_entropy, optimizer, clear_data=False, coef_entropy=coef_entropy,
                             state_values=state_values if AC else None)

        time_to_target_memory.append(current_game_timestep / dist_eff_target)

        if len(time_to_target_memory) == n_game_scoring_average:
          average_ttt = np.mean(time_to_target_memory)

          if use_visdom:
            vp.line_plot('Time to Target', 'Train', 'Policy Performance', plot_iter, average_ttt)

          logging.info(f'Episode {plot_iter+1}(x{n_game_scoring_average}) | Average time-to-target={average_ttt:.3f}')
          time_to_target_memory.clear()
          plot_iter += 1

          if save_model:
            u.save_checkpoint(policy.model, None, save_name.replace('toy', 'toyAC') if AC else save_name)

      # Reset game and related variables
      env.reset(to_reset='target')
      dist_eff_target = env.current_dist
      current_game_timestep = 0
      rewards.clear()
      log_probs.clear()
      if AC:
        state_values.clear()
  
  env.close()


@torch.no_grad()
def run_model(save_name='models/toyModel.pt', max_game_timestep=200, AC=False):
  env, render_mode = get_game_env(game_view=True)

  policy = TOYACModel() if AC else TOYModel()
  u.load_model(policy.model, save_name.replace('toy', 'toyAC') if AC else save_name)

  state = (torch.FloatTensor(env.joints_angle + list(env.target_pos)) - MINS) / MAXS_MINS

  quit_game = False
  current_game_timestep = 0

  while not quit_game:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        quit_game = True
    
    action, *_ = policy.select_action(state)
    joints_angle, _, target_reached, _ = env.step(action)

    current_game_timestep += 1
    env.render(mode=render_mode)

    state = (torch.FloatTensor(joints_angle + list(env.target_pos)) - MINS) / MAXS_MINS

    if target_reached or current_game_timestep > max_game_timestep:
      env.reset(to_reset='target')
      current_game_timestep = 0
  
  env.close()


def train_ppo(game_view=False, lr=1e-3, max_game_timestep=200, n_game_scoring_average=100, use_visdom=False,
              load_model=True, save_model=True, save_name='models/toyModel.pt', AC=False, coef_entropy=0.01,
              early_stopping_n_step_watcher=20, early_stopping_min_slope=0.001, pretraining=False, model=TOYActorCritic,
              model_conf={}, episode_batch=False):
  print(f'Start PPO training...')
  if use_visdom:
    vp = u.VisdomPlotter()

  env, render_mode = get_game_env(game_view=game_view)

  # Instanciate model and optimizer
  policy = TOYACModel(model=model, model_conf=model_conf) if AC else TOYModel()
  optimizer = torch.optim.AdamW(policy.model.parameters(), lr=lr)

  if pretraining:
    model_state_dict = pretrain_CE(model=model, model_conf=model_conf, episode_batch=episode_batch)
    policy.model.load_state_dict(model_state_dict)

  if load_model:
    u.load_model(policy.model, save_name.replace('toy', 'toyAC') if AC else save_name)
  
  old_policy = TOYACModel(model=model, model_conf=model_conf) if AC else TOYModel()
  old_policy.model.load_state_dict(policy.model.state_dict())

  # Starting variables
  dist_eff_target = env.current_dist
  state = (torch.FloatTensor(env.joints_angle + list(env.target_pos)) - MINS) / MAXS_MINS

  # Storage variables
  rewards = []
  log_probs = []
  actions = []
  states = []

  # Control variables
  quit_game = False
  plot_iter = 0
  current_game_timestep, n_total_game_timestep = 0, 0
  time_to_target_memory = []
  average_ttt_mem = deque(maxlen=early_stopping_n_step_watcher)
  x_linregress = list(range(early_stopping_n_step_watcher))
  start_time = time.time()

  while not quit_game:
    # If game window is open, catch closing event
    if game_view:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          quit_game = True
    
    # Get action from model then perform it
    action, log_prob = old_policy.act(state)

    joints_angle, reward, target_reached, _ = env.step(action.item())

    current_game_timestep += 1
    env.render(mode=render_mode)

    # Store data and update state
    rewards.append(reward)
    log_probs.append(log_prob)
    actions.append(action)
    states.append(state)

    state = (torch.FloatTensor(joints_angle + list(env.target_pos)) - MINS) / MAXS_MINS

    if target_reached or current_game_timestep > max_game_timestep:
      if target_reached:  # update policy only if the target is reached as there is no reward before that
        pgu.ppo_update(states, actions, log_probs, rewards, old_policy, policy, optimizer, AC=AC, coef_entropy=coef_entropy)

        time_to_target_memory.append(current_game_timestep / dist_eff_target)

        if len(time_to_target_memory) == n_game_scoring_average:
          average_ttt = np.mean(time_to_target_memory)
          average_ttt_mem.append(average_ttt)

          if use_visdom:
            vp.line_plot('Time to Target', 'Train', 'Policy Performance', plot_iter, average_ttt)

          logging.info(f'Episode {plot_iter+1}(x{n_game_scoring_average}) | Average time-to-target={average_ttt:.3f}')
          time_to_target_memory.clear()
          plot_iter += 1

          if save_model:
            u.save_checkpoint(policy.model, None, save_name.replace('toy', 'toyAC') if AC else save_name)
        
          # Early Stopping checks
          if len(average_ttt_mem) == early_stopping_n_step_watcher:
            slope, *_ = linregress(x_linregress, average_ttt_mem)
            logging.info(f'Time-to-Target slope = {slope:.4f}')
            if abs(slope) <= early_stopping_min_slope:
              quit_game = True

      # Reset game and related variables
      env.reset(to_reset='target')
      dist_eff_target = env.current_dist
      n_total_game_timestep += current_game_timestep
      current_game_timestep = 0
      rewards.clear()
      log_probs.clear()
      actions.clear()
      states.clear()
  
  logging.info(f'Performance achieved with {n_total_game_timestep:,} interaction with the environment')
  logging.info(f'Run done in {timedelta(seconds=int(time.time() - start_time))}')
  env.close()
  print('PPO training done.')


if __name__ == '__main__':
  # When adding gradient accumulation, it smooth the performance curve but slow down the learning
  # Convergence can be complicated to obtain and depend of the initialization
  # e.g. with seed=42 it seems hard to get convergence with PPO wo AC but easily obtained with seed=38
  argparser = argparse.ArgumentParser(prog='pg_exps.py',
                                      description='Experiments on 2-DOF robot arm that learn to reach a target point')
  argparser.add_argument('--log_file', default='_tmp_pg_exps_logs.txt', type=str)
  argparser.add_argument('--use_visdom', default=True, type=ast.literal_eval)
  argparser.add_argument('--game_view', default=False, type=ast.literal_eval)
  argparser.add_argument('--save_model', default=True, type=ast.literal_eval)
  argparser.add_argument('--load_model', default=True, type=ast.literal_eval)
  argparser.add_argument('--actor_critic', default=False, type=ast.literal_eval)
  argparser.add_argument('--save_name', default='models/toyModel.pt', type=str)
  argparser.add_argument('--coef_entropy', default=0.01, type=float)
  argparser.add_argument('--seed', default=42, type=int)
  argparser.add_argument('--algo', default='reinforce', type=str, choices=['reinforce', 'ppo'])
  argparser.add_argument('--pretraining', default=False, type=ast.literal_eval)
  argparser.add_argument('--force_training', default=False, type=ast.literal_eval)
  args = argparser.parse_args()

  logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')
  
  trainers = {'reinforce': train_reinforce, 'ppo': train_ppo}
  
  rep = input('Start training? (y or n): ') if not args.force_training else 'y'
  if rep == 'y':
    # seeding for reproducibility
    random.seed(args.seed * args.seed)
    torch.manual_seed(args.seed)

    u.dump_argparser_parameters(args)
    trainers[args.algo](game_view=args.game_view, use_visdom=args.use_visdom, load_model=args.load_model, save_model=args.save_model,
                        save_name=args.save_name, coef_entropy=args.coef_entropy, AC=args.actor_critic, pretraining=args.pretraining)
  
  rep = input('Run saved model? (y or n): ') if not args.force_training else 'n'
  if rep == 'y':
    run_model(save_name=args.save_name, AC=args.actor_critic)