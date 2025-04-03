import os
import ast
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

from pg_exps import train_ppo, train_reinforce, get_game_env, MINS, MAXS_MINS

sys.path.append(os.path.abspath(__file__).replace('USS-GL/pgmem_exps.py', ''))
import utils as u
import pg_utils as pgu


class Summarizer(torch.nn.Module):
  def __init__(self, vector_dim, hidden_size, n_output_tokens, method='MLP'):
    '''
    Summarize a sequence of p tokens of dim d to a sequence of k tokens
    '''
    super().__init__()
    self.method = method
    self.vector_dim = vector_dim
    if self.method == 'MLP':
      self.mlp = torch.nn.Sequential(
        torch.nn.Linear(vector_dim, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, n_output_tokens)
      )
    else:
      self.query_vector = torch.nn.Parameter(torch.randn(1, n_output_tokens, vector_dim))  # [1, k, d]
          
  def forward(self, x):  # [batch_size, n_input_tokens, vector_dim] = [b, p, d]
    if self.method == 'MLP':
      weight = torch.softmax(self.mlp(x), dim=-2)  # [b, p, d] -> [b, p, k]
      weight = weight.transpose(-1, -2)  # [b, p, k] -> [b, k, p]
    else:
      weight = torch.matmul(self.query_vector, x.transpose(-1, -2))  # [1, k, d] * [b, d, p] = [b, k, p]
      weight = torch.softmax(weight / (self.vector_dim ** 0.5), dim=-1)
    z = torch.matmul(weight, x)  # [b, k, p] * [b, p, d] = [b, k, d]
    return z


class ToyACmem(torch.nn.Module):
  BASE_CONFIG = {'concat': True, 'memory_size': 16, 'expand_memory_emb': True, 'expanded_memory_emb_size': 12,
                 'summarizer_hidden_size': 200, 'state_size': 4, 'shared_AC_hidden_size': 200,
                 'processor_hidden_size': 100}
  def __init__(self, configuration={}):
    super().__init__()
    self.config = {**ToyACmem.BASE_CONFIG, **configuration}
    memory_emb_size = self.config['expanded_memory_emb_size'] if self.config['expand_memory_emb'] else self.config['state_size']
    fstate_size = 2*memory_emb_size if self.config['concat'] else memory_emb_size

    self.shared = torch.nn.Linear(fstate_size, self.config['shared_AC_hidden_size'])
    self.actor = torch.nn.Linear(self.config['shared_AC_hidden_size'], 5)
    self.critic = torch.nn.Linear(self.config['shared_AC_hidden_size'], 1)

    self.register_buffer('memory', torch.FloatTensor(1, self.config['memory_size'], memory_emb_size).uniform_(0, 1))
    self.positional_encoder = u.PositionalEncoding(memory_emb_size)

    self.reader = Summarizer(memory_emb_size, self.config['summarizer_hidden_size'], 1)
    self.writer = Summarizer(memory_emb_size, self.config['summarizer_hidden_size'], self.config['memory_size'])

    if self.config['expand_memory_emb']:
      self.embedder = torch.nn.Linear(4, memory_emb_size)
      self.processor = torch.nn.Sequential(torch.nn.Linear(5+1+2*memory_emb_size, self.config['processor_hidden_size']),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(self.config['processor_hidden_size'], memory_emb_size))
  
  def forward(self, state, critic=False, keep_action_logits=False):  # [state_size] or [ep_len, state_size]
    if self.config['expand_memory_emb']:
      state = self.embedder(state)
    
    if len(state.size()) == 1:
      state = state.view(1, 1, -1)  # [emb_dim=d] -> [1=bs, 1=k, emb_dim=d]
    elif len(state.size()) == 2:
      state = state.unsqueeze(1)  # [bs, emb_dim] -> [bs, 1, emb_dim]
    
    bs, *_ = state.shape

    action_probs = torch.empty(bs, 1, 5)
    state_values = torch.empty(bs, 1, 1)
    new_memory = self.memory.clone()
    for i in range(bs):
      cstate = state[i:i+1]

      # [bs, 1, d] || [1, mem_size, d] = [bs, 1+mem_size, d] -> [bs, 1, d]
      mem_state = self.reader(torch.cat([cstate, self.positional_encoder(new_memory)], dim=-2))

      # [bs, 1, d] || [bs, 1, d] = [bs, 1, 2*d] if concat
      cstate = torch.cat([cstate, mem_state], dim=-1) if self.config['concat'] else mem_state

      out = torch.nn.functional.relu(self.shared(cstate))  # [bs, 1, d or 2d] -> [bs, 1, 200]

      action_logits = self.actor(out)  # [bs, 1, 200] -> [bs, 1, 5]
      caction_probs = action_logits if keep_action_logits else torch.softmax(action_logits, dim=-1)

      cstate_values = self.critic(out)  # [bs, 1, 200] -> [bs, 1, 1]

      if self.config['expand_memory_emb']:
        # [bs, 1, d] || [bs, 1, d] || [bs, 1, 5] || [bs, 1, 1] = [bs, 1, d+d+5+1] -> [bs, 1, d]
        mem_state = self.processor(torch.cat([state[i:i+1], mem_state, caction_probs, cstate_values], dim=-1))

      # [bs, 1, d] || [bs, 1, d] || [1, mem_size, d] = [bs, 1+1+mem_size, d] -> [bs, mem_size, d]
      new_memory = self.writer(torch.cat([state[i:i+1].detach(),
                                          mem_state.detach(),
                                          self.positional_encoder(new_memory.detach())], dim=-2))
      new_memory -= new_memory.min(-1, keepdim=True)[0]
      new_memory /= new_memory.max(-1, keepdim=True)[0]

      action_probs[i] = caction_probs[0]
      state_values[i] = cstate_values[0]
    
    self.memory = new_memory.detach().clone()

    return action_probs.squeeze(), state_values.squeeze(-1) if critic else None


class ContextPredictiveModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.context_encoder = torch.nn.Sequential(torch.nn.Linear(4, 12), torch.nn.ReLU(),
                                               torch.nn.TransformerEncoderLayer(12, 2, 48, batch_first=True))
    self.action_selector = torch.nn.Sequential(torch.nn.Linear(12, 32), torch.nn.ReLU(), 
                                               torch.nn.Linear(32, 5), torch.nn.Softmax(-1))
    self.reward_predictor = torch.nn.Sequential(torch.nn.Linear(12, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    self.next_state_predictor = torch.nn.Sequential(torch.nn.Linear(12 + 5, 64), torch.nn.ReLU(),
                                                    torch.nn.Linear(64, 4), torch.nn.Sigmoid())
  
  def forward(self, state, only_action=False):  # [ep_len, context_size, state_size]
    out = self.context_encoder(state)[:, -1:].squeeze(1)  # [ep_len, state_size]
    action_probs = self.action_selector(out)
    if only_action:
      return action_probs
    reward_pred = self.reward_predictor(out)
    action_onehot = torch.nn.functional.one_hot(action_probs.argmax(-1), num_classes=action_probs.shape[-1])
    next_state = self.next_state_predictor(torch.cat([out, action_onehot], dim=-1))
    return action_probs, reward_pred, next_state


class ActorEvaluatorWrapper(object):
  def __init__(self, device=None):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.model = ContextPredictiveModel().to(self.device)

  def act(self, state):  # [ep_len=1, context_size, state_size]
    action_probs = self.model(state, only_action=True).squeeze()
    distri = torch.distributions.Categorical(action_probs)
    action = distri.sample()
    action_logprob = distri.log_prob(action)
    return action.detach(), action_logprob.detach()  # [], []

  def evaluate(self, state, action):  # [ep_len, state_size], [ep_len]
    action_probs, reward_pred, next_state = self.model(state)
    distri = torch.distributions.Categorical(action_probs)
    action_logprob = distri.log_prob(action)  # [ep_len]
    return action_logprob, reward_pred, next_state, distri.entropy()  # [ep_len], [ep_len, 1], [ep_len, state_size]


def train_CPM(use_visdom=True, game_view=False, lr=1e-3, load_model=False, save_name='cpm_model.pt', context_size=8,
              max_game_timestep=100, n_epochs=5, eps_clip=0.2, coef_entropy=0.01, normalize_returns=False,
              n_game_scoring_average=100, early_stopping_n_step_watcher=20, save_model=False,
              early_stopping_min_slope=0.001):
  print('start CPM training...')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  norm_0_1 = lambda x: (x - x.min()) / (x.max() - x.min())

  if use_visdom:
    vp = u.VisdomPlotter()

  env, render_mode = get_game_env(game_view=game_view)

  acw = ActorEvaluatorWrapper(device=device)
  optimizer = torch.optim.AdamW(acw.model.parameters(), lr=lr)

  if load_model:
    u.load_model(acw.model, save_name)

  old_acw = ActorEvaluatorWrapper(device=device)
  old_acw.model.load_state_dict(acw.model.state_dict())
  
  # Starting variables
  dist_eff_target = env.current_dist
  state = (torch.FloatTensor(env.joints_angle + list(env.target_pos)) - MINS) / MAXS_MINS

  # Storage variables
  rewards = []
  log_probs = []
  actions = []
  context_states = deque([state] * context_size, maxlen=context_size)
  states = [torch.stack(list(context_states))]

  # Control variables
  quit_game = False
  plot_iter1, plot_iter2 = 0, 0
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
    
    current_context_state = torch.stack(list(context_states))
    action, log_prob = old_acw.act(current_context_state.unsqueeze(0).to(device))

    joints_angle, reward, target_reached, _ = env.step(action.item())

    current_game_timestep += 1
    env.render(mode=render_mode)

    # Store data and update state
    rewards.append(reward)
    log_probs.append(log_prob)
    actions.append(action)
    states.append(current_context_state)

    state = (torch.FloatTensor(joints_angle + list(env.target_pos)) - MINS) / MAXS_MINS
    context_states.append(state)

    if (target_reached or current_game_timestep > max_game_timestep) and len(rewards) > 1:
      old_states = torch.stack(states).to(device)
      old_actions = torch.stack(actions).to(device)
      old_log_probs = torch.stack(log_probs).to(device)

      for _ in range(n_epochs):
        # get reconstruction_rewards
        action_logprob, reward_pred, next_state, distri_entropy = acw.evaluate(old_states[:-1], old_actions)
        prediction_loss = torch.nn.functional.mse_loss(next_state, old_states[1:, -1], reduction='none').sum(-1)  # [ep_len]
        returns = prediction_loss.clone().detach()

        # if target_reached -> add discounted_rewards to reconstruction_rewards
        if target_reached:
          discounted_rewards = pgu.get_returns(rewards, normalize_returns=False).to(device)
          # reconstruction_rewards = norm_0_1(returns)
          # returns = discounted_rewards + reconstruction_rewards
          returns = returns + discounted_rewards
        
        if normalize_returns:
          returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        ratios = torch.exp(action_logprob - old_log_probs.detach())

        surr1 = ratios * returns
        surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * returns

        reward_pred_loss = torch.nn.functional.mse_loss(reward_pred.squeeze(), returns, reduction='none')

        loss = -torch.min(surr1, surr2) - coef_entropy * distri_entropy + prediction_loss + reward_pred_loss

        optimizer.zero_grad()
        # loss.mean().backward()
        loss.sum().backward()
        optimizer.step()
      
      old_acw.model.load_state_dict(acw.model.state_dict())

      if use_visdom:
        vp.line_plot('Next State Prediction loss', 'Train', 'Prediction loss', plot_iter1, prediction_loss.mean().item())
        vp.line_plot('Reward Prediction loss', 'Train', 'Reward loss', plot_iter1, reward_pred_loss.mean().item())
        vp.line_plot('action loss', 'Train', 'RL loss', plot_iter1, (-torch.min(surr1, surr2)).item())
        plot_iter1 += 1

      if target_reached:
        time_to_target_memory.append(current_game_timestep / dist_eff_target)

        if len(time_to_target_memory) == n_game_scoring_average:
          average_ttt = np.mean(time_to_target_memory)
          average_ttt_mem.append(average_ttt)

          if use_visdom:
            vp.line_plot('Time to Target', 'Train', 'Policy Performance', plot_iter2, average_ttt)

          logging.info(f'Episode {plot_iter2+1}(x{n_game_scoring_average}) | Average time-to-target={average_ttt:.3f}')
          time_to_target_memory.clear()
          plot_iter2 += 1

          if save_model:
            u.save_checkpoint(acw.model, None, save_name)
        
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

      state = (torch.FloatTensor(joints_angle + list(env.target_pos)) - MINS) / MAXS_MINS
      context_states.append(state)
      states.append(torch.stack(list(context_states)))
    
  logging.info(f'Performance achieved with {n_total_game_timestep:,} interaction with the environment')
  logging.info(f'Run done in {timedelta(seconds=int(time.time() - start_time))}')
  env.close()
  print('CPM training done.')


if __name__ == '__main__':
  argparser = argparse.ArgumentParser(prog='pgmem_exps.py', description='')
  argparser.add_argument('--log_file', default='_tmp_pgmem_exps_logs.txt', type=str)
  argparser.add_argument('--use_visdom', default=True, type=ast.literal_eval)
  argparser.add_argument('--game_view', default=False, type=ast.literal_eval)
  argparser.add_argument('--save_model', default=True, type=ast.literal_eval)
  argparser.add_argument('--load_model', default=False, type=ast.literal_eval)
  argparser.add_argument('--save_name', default='models/PGmemModel.pt', type=str)
  argparser.add_argument('--seed', default=42, type=int)
  argparser.add_argument('--pretraining', default=False, type=ast.literal_eval)
  argparser.add_argument('--concat', default=True, type=ast.literal_eval)
  argparser.add_argument('--expand_memory_emb', default=True, type=ast.literal_eval)
  argparser.add_argument('--memory_size', default=16, type=int)
  argparser.add_argument('--expanded_memory_emb_size', default=12, type=int)
  argparser.add_argument('--summarizer_hidden_size', default=100, type=int)
  argparser.add_argument('--state_size', default=4, type=int)
  argparser.add_argument('--shared_AC_hidden_size', default=200, type=int)
  argparser.add_argument('--processor_hidden_size', default=100, type=int)
  argparser.add_argument('--algo', default='ppo', type=str, choices=['reinforce', 'ppo'])
  argparser.add_argument('--force_training', default=False, type=ast.literal_eval)
  args = argparser.parse_args()

  logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')
  
  trainers = {'reinforce': train_reinforce, 'ppo': train_ppo}
  model_conf = {'concat': args.concat, 'memory_size': args.memory_size, 'expand_memory_emb': args.expand_memory_emb,
                'expanded_memory_emb_size': args.expanded_memory_emb_size, 'summarizer_hidden_size': args.summarizer_hidden_size,
                'state_size': args.state_size, 'shared_AC_hidden_size': args.shared_AC_hidden_size,
                'processor_hidden_size': args.processor_hidden_size}

  # seeding for reproducibility
  random.seed(args.seed * args.seed)
  torch.manual_seed(args.seed)
  
  rep = input('Start training? (y or n): ') if not args.force_training else 'y'
  if rep == 'y' or args.force_training:
    u.dump_argparser_parameters(args)
    trainers[args.algo](game_view=args.game_view, use_visdom=args.use_visdom, load_model=args.load_model, episode_batch=True,
                        save_model=args.save_model, save_name=args.save_name, AC=True, model=ToyACmem, pretraining=args.pretraining,
                        model_conf=model_conf)
  
  rep = input('Start Context-Predictive experiment? (y or n): ')
  if rep == 'y':
    train_CPM()