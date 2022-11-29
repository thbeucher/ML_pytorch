import os
import cma
import sys
import time
import torch
import visdom
import numpy as np
import multiprocessing

from tqdm import tqdm
from datetime import timedelta
from collections import OrderedDict

sys.path.append(os.path.abspath(__file__).replace('policy_learners.py', ''))

from arm_2dof_gym_env import RobotArmEnvVG
from utils import plot_metric, save_model, load_model


class REINFORCEmlpTrainer(object):
  '''It takes around 7000 training iterations (10min) to achieves 100% success rate'''
  BASE_CONFIG = {'models_folder': 'models/', 'model_name': 'reinforce_mlp_policy.pt', 'target_pos': [194, 165],
                 'use_visdom': True, 'n_validation_ep': 10, 'batch_size': 30, 'max_ep_len': 120, 'gamma': 0.99}
  def __init__(self, config={}):
    self.config = {**REINFORCEmlpTrainer.BASE_CONFIG, **config}
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if self.config['use_visdom']:
      self.vis = visdom.Visdom()
    
    self.instanciate_network()
    self.optimizer = torch.optim.AdamW(self.policy.parameters())

    ## currently, to use this code, you must remove ReLU layer from model and add bias=False for linear layers
    # if os.path.isfile('cma_np_policy.npz'):
    #   rep = input('Start with weights from CMA policy? (y or n):')
    #   if rep == 'y':
    #     arrs = np.load('cma_np_policy.npz')
    #     sd = OrderedDict([(k, torch.from_numpy(arr.T).float().to(self.device))\
    #                       for (k, v), arr in zip(self.policy.state_dict().items(), [arrs['input_layer'], arrs['output_layer']])])
    #     self.policy.load_state_dict(sd)
    
    save_name = os.path.join(self.config['models_folder'], self.config['model_name'])
    if os.path.isfile(save_name):
      rep = input(f'Load model from file {save_name}? (y or n): ')
      if rep == 'y':
        self.load_model()

  def instanciate_network(self):
    self.policy = torch.nn.Sequential(torch.nn.Linear(4, 64),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(64, 5),
                                      torch.nn.Softmax(dim=-1)).to(self.device)
  
  @staticmethod
  def collect_one_episode(policy, env=None, max_ep_len=120, device=None, target_pos=[194, 165]):
    env = RobotArmEnvVG({'in_background': True}) if env is None else env
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

    obs = torch.tensor(env.reset(to_reset='both', target_pos=target_pos), dtype=torch.float)

    actions, states, rewards = [], [], []

    for _ in range(max_ep_len):
      probs = policy(obs.to(device)).cpu()
      dist = torch.distributions.Categorical(probs=probs)
      action = dist.sample().item()
      obs_, reward, done, _ = env.step(action)

      actions.append(torch.tensor(action, dtype=torch.int))
      states.append(obs)
      rewards.append(reward)

      if done:
        break

      obs = torch.tensor(obs_, dtype=torch.float)
    
    return actions, states, rewards
  
  @staticmethod
  def rollout_episode(rewards, gamma=0.99):
    discounted_returns = []
    for i in range(len(rewards)):
      discounted_reward = 0.
      for j, r in enumerate(rewards[i:]):
        discounted_reward += (gamma**j) * r
      discounted_returns.append(torch.tensor(discounted_reward, dtype=torch.float))
    return discounted_returns
  
  def learn(self, env=None):
    actions, states, rewards = REINFORCEmlpTrainer.collect_one_episode(self.policy, env=env)
    discounted_rewards = REINFORCEmlpTrainer.rollout_episode(rewards)
    for state, action, d_reward in zip(states, actions, discounted_rewards):
      probs = self.policy(state.to(self.device)).cpu()
      dist = torch.distributions.Categorical(probs=probs)
      log_probs = dist.log_prob(action)

      loss = - log_probs * d_reward

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
  
  def learn_batch(self, env=None):
    actions, states, rewards = REINFORCEmlpTrainer.collect_one_episode(self.policy, env=env, max_ep_len=self.config['max_ep_len'],
                                                                       device=self.device, target_pos=self.config['target_pos'])
    discounted_rewards = REINFORCEmlpTrainer.rollout_episode(rewards, gamma=self.config['gamma'])
    for i in range(0, len(actions), self.config['batch_size']):
      probs = self.policy(torch.stack(states[i:i+self.config['batch_size']], 0).to(self.device)).cpu()
      dist = torch.distributions.Categorical(probs=probs)
      log_probs = dist.log_prob(torch.stack(actions[i:i+self.config['batch_size']], 0))

      dr_batch = torch.stack(discounted_rewards[i:i+self.config['batch_size']], 0)
      dr_batch /= (dr_batch.max() + 1e-9)  # without normalization it can not be able to converge

      loss = - torch.mean(log_probs * dr_batch)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
  
  @torch.no_grad()
  def evaluation(self, env=None, n_episodes=10, max_ep_len=120):
    env = RobotArmEnvVG() if env is None else env

    rewards = []
    n_goals_reached = 0
    for _ in range(n_episodes):
      obs = torch.tensor(env.reset(to_reset='both', target_pos=self.config['target_pos']), dtype=torch.float)
      env.render()

      cum_reward = 0
      for i in range(max_ep_len):
        probs = self.policy(obs.to(self.device)).cpu()
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample().item()

        obs_, reward, done, _ = env.step(action)
        env.render()

        cum_reward += reward

        if done:
          n_goals_reached += 1
          break

        obs = torch.tensor(obs_, dtype=torch.float)
      
      rewards.append(cum_reward / (i+1))
    
    return np.mean(rewards), n_goals_reached
  
  def train(self, n_iterations=100000, eval_step=100):
    t = time.time()
    env = RobotArmEnvVG()

    n_perfect_eval = 0
    for i in tqdm(range(n_iterations)):
      # self.learn(env=env)
      self.learn_batch(env=env)
      
      if i % eval_step == 0:
        mean_reward, n_goals_reached = self.evaluation(env=env, n_episodes=self.config['n_validation_ep'],
                                                       max_ep_len=self.config['max_ep_len'])
        plot_metric(self.vis, mean_reward.item(), i, win='mean_reward', title='Mean reward', ylabel='reward')
        plot_metric(self.vis, n_goals_reached, i, win='n_goals_reached', title='Number of reached goals', ylabel='n_goals_reached')

        if n_goals_reached == self.config['n_validation_ep']:
          self.save_model()
          n_perfect_eval += 1
          if n_perfect_eval >= 10:  # Early stopping
            print(f'Perfect behavior achieved in {i} timesteps ({timedelta(seconds=int(time.time() - t))})')
            break
        else:
          n_perfect_eval = 0
  
  def save_model(self, model_name=None):
    save_model(self.policy, models_folder=self.config['models_folder'],
               save_name=self.config['model_name'] if model_name is None else model_name)
  
  def load_model(self, model_name=None):
    load_model(self.policy, models_folder=self.config['models_folder'],
               save_name=self.config['model_name'] if model_name is None else model_name)


class MultiHeadModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.body = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.ReLU())
    self.action_head = torch.nn.Sequential(torch.nn.Linear(64, 5), torch.nn.Softmax(dim=-1))
    self.value_head = torch.nn.Linear(64, 1)
    self.reward_head = torch.nn.Sequential(torch.nn.Linear(64, 1), torch.nn.Sigmoid())
  
  def forward(self, state, return_all=False):
    out = self.body(state)
    action_probs = self.action_head(out)
    value_pred = self.value_head(out)
    reward_pred = self.reward_head(out)
    return (action_probs, value_pred, reward_pred) if return_all else action_probs


class REINFORCEmlpMHTrainer(REINFORCEmlpTrainer):
  BASE_CONFIG = {'model_name': 'reinforce_mlp_valueReward_policy.pt'}
  def __init__(self, config={}):
    self.config = {**REINFORCEmlpMHTrainer.BASE_CONFIG, **config}
    super().__init__(self.config)
    self.mse_criterion = torch.nn.MSELoss()
    self.bce_criterion = torch.nn.BCELoss()
  
  def instanciate_network(self):
    self.policy = MultiHeadModel().to(self.device)
  
  def learn_batch(self, env=None):
    actions, states, rewards = REINFORCEmlpTrainer.collect_one_episode(self.policy, env=env, max_ep_len=self.config['max_ep_len'],
                                                                       device=self.device, target_pos=self.config['target_pos'])
    discounted_rewards = REINFORCEmlpTrainer.rollout_episode(rewards, gamma=self.config['gamma'])
    target_values = torch.stack(discounted_rewards, 0).unsqueeze(-1)
    target_values /= (target_values.max() + 1e-9)
    states = torch.stack(states, 0).to(self.device)
    rewards[-1] = 1 if rewards[-1] == 10 else rewards[-1]
    target_rewards = torch.FloatTensor(rewards).unsqueeze(-1)

    for i in range(0, len(actions), self.config['batch_size']):
      action_probs, value_pred, reward_pred = self.policy(states[i:i+self.config['batch_size']], return_all=True)
      dist = torch.distributions.Categorical(probs=action_probs.cpu())
      log_probs = dist.log_prob(torch.stack(actions[i:i+self.config['batch_size']], 0))

      dr_batch = torch.stack(discounted_rewards[i:i+self.config['batch_size']], 0)
      dr_batch /= (dr_batch.max() + 1e-9)  # without normalization it can not be able to converge

      value_loss = self.mse_criterion(value_pred.cpu(), target_values[i:i+self.config['batch_size']])
      reward_loss = self.bce_criterion(reward_pred.cpu(), target_rewards[i:i+self.config['batch_size']])
      loss = - torch.mean(log_probs * dr_batch) + value_loss + reward_loss

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()


class NPPolicyModel(object):
  def __init__(self, add_act=False):
    self.add_act = add_act
    self.input_layer = np.random.randn(4, 64)
    self.output_layer = np.random.randn(64, 5)
  
  @staticmethod
  def ReLU(x):
    return x * (x > 0)
  
  def forward(self, x):
    out = x @ self.input_layer
    out = NPPolicyModel.ReLU(out) if self.add_act else out
    return out @ self.output_layer

  def get_weights(self):
    return np.concatenate([arr.flatten() for arr in [self.input_layer, self.output_layer]], axis=0)

  def set_weights(self, weights):
    self.input_layer = weights[:self.input_layer.size].reshape(self.input_layer.shape)
    self.output_layer = weights[self.input_layer.size:].reshape(self.output_layer.shape)
  
  def save_model(self, model_name='np_policy.npz'):
    np.savez(model_name, input_layer=self.input_layer, output_layer=self.output_layer)

  def load_model(self, model_name='np_policy.npz'):
    arrs = np.load(model_name)
    self.input_layer = arrs['input_layer']
    self.output_layer = arrs['output_layer']


def objective_fn(solution, n_episodes=100, max_ep_len=120, add_act=False, to_reset='arm'):
  model = NPPolicyModel(add_act=add_act)
  model.set_weights(solution)

  env = RobotArmEnvVG()
  env.reset(to_reset='both', target_pos=[194, 165])

  n_goal_reached = 0
  for _ in range(n_episodes):
    obs = env.reset(to_reset=to_reset)
    for _ in range(max_ep_len):
      action_logits = model.forward(obs)
      obs, _, done, _ = env.step(action_logits.argmax(-1))
      
      if done:
        n_goal_reached += 1
        break
  
  env.close()

  return -n_goal_reached


class CMATrainer(object):
  BASE_CONFIG = {'add_act': False, 'model_name': 'cma_np_policy.npz', 'sigma': 1., 'popsize': 64, 'maxiter': 30,
                 'max_ep_len': 120, 'to_reset': 'arm'}
  def __init__(self, config={}):
    self.config = {**CMATrainer.BASE_CONFIG, **config}
    self.np_policy = NPPolicyModel(add_act=self.config['add_act'])

    self.start_sol = self.np_policy.get_weights()

    if os.path.isfile(self.config['model_name']):
      rep = input('Use previous best solution? (y or n):')
      if rep == 'y':
        self.np_policy.load_model(model_name=self.config['model_name'])
        self.start_sol = self.np_policy.get_weights()
        print('start_sol=best_previous_sol')
  
  def train(self):
    print('Start CMA-ES policy exploration...')
    t = time.time()

    es = cma.CMAEvolutionStrategy(self.start_sol, self.config['sigma'],
                                  {'popsize': self.config['popsize'], 'maxiter': self.config['maxiter']})
    with cma.fitness_transformations.EvalParallel2(objective_fn, multiprocessing.cpu_count()) as eval_para:
      while not es.stop():
        X = es.ask()
        es.tell(X, eval_para(X))
        es.disp()
    
    print(f'Best number of goal reached = {-es.result[1]} | time_taken={timedelta(seconds=int(time.time() - t))}')
    self.np_policy.set_weights(es.result[0])
    self.np_policy.save_model(model_name=self.config['model_name'])

    self.evaluation()
  
  def evaluation(self, n_episodes=10):
    env = RobotArmEnvVG({'clock_tick': 30})
    env.reset(to_reset='both', target_pos=[194, 165])
    env.render()

    n_goal_reached = 0
    for _ in range(n_episodes):
      obs = env.reset(to_reset=self.config['to_reset'])
      for _ in range(self.config['max_ep_len']):
        action_logits = self.np_policy.forward(obs)
        obs, _, done, _ = env.step(action_logits.argmax(-1))
        env.render()

        if done:
          n_goal_reached += 1
          break
    
    print(f'Number of goal reached = {n_goal_reached}/{n_episodes}')


class ARPModel(torch.nn.Module):  # ARP = Auto-Regressive Policy
  BASE_CONFIG = {'input_size': 512, 'hidden_size': 256, 'num_layers': 2, 'bias': True, 'batch_first': True,
                 'dropout': 0., 'bidirectional': False}
  def __init__(self, config={}):
    super().__init__()
    self.config = {**ARPModel.BASE_CONFIG, **config}
    self.network = torch.nn.GRU(**self.config)
    self.predictor = torch.nn.Linear(self.config['hidden_size'], 5)
  
  def forward(self, x):
    out = self.network(x)
    return self.predictor(out)

  def save_model(self, save_path='models/arp_model.pt'):
    torch.save(self.state_dict(), save_path)

  def load_model(self, save_path='models/arp_model.pt', device=None):
    self.load_state_dict(torch.load(save_path, map_location=device))


class AutoRegressivePolicy(object):
  BASE_CONFIG = {}
  def __init__(self, config={}):
    self.config = {**AutoRegressivePolicy.BASE_CONFIG, **config}

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  def instanciate_models(self):
    pass

  def save_model(self):
    pass

  def load_model(self):
    pass

  def evaluation(self):
    pass

  def train(self):
    pass


if __name__ == '__main__':
  rep = input('Train mlp policy using REINFORCE? (y or n): ')
  if rep == 'y':
    trainer = REINFORCEmlpTrainer()
    trainer.train()
    # RUN 1 : Perfect behavior achieved in 6200 timesteps (0:09:46)
    # RUN 2 : Perfect behavior achieved in 6200 timesteps (0:08:03)
    # RUN 3 : Perfect behavior achieved in 15000 timesteps (0:18:26)
    # RUN 4 : Perfect behavior achieved in 7600 timesteps (0:11:27)
    # RUN 5 : Perfect behavior achieved in 8900 timesteps (0:12:34)
  
  rep = input('Train mlp MH policy using REINFORCE? (y or n): ')
  if rep == 'y':
    trainer = REINFORCEmlpMHTrainer()
    trainer.train()
    # RUN 1 : Perfect behavior achieved in 13700 timesteps (0:17:17)
    # RUN 2 : Perfect behavior achieved in 6500 timesteps (0:09:38)
    # RUN 3 : Perfect behavior achieved in 6400 timesteps (0:11:55)
    # RUN 4 : Perfect behavior achieved in 5100 timesteps (0:09:11)
    # RUN 5 : Perfect behavior achieved in 6100 timesteps (0:10:05)

  rep = input('Train policy using CMA? (y or n): ')
  if rep == 'y':
    trainer = CMATrainer()
    trainer.train()