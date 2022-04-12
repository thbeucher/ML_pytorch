from audioop import bias
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

from utils import plot_metric
from arm_2dof_gym_env import RobotArmEnvVG


class REINFORCETrainer(object):
  BASE_CONFIG = {'use_visdom': True}
  def __init__(self, config={}):
    self.config = {**REINFORCETrainer.BASE_CONFIG, **config}
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if self.config['use_visdom']:
      self.vis = visdom.Visdom()
    
    self.instanciate_network()
    self.optimizer = torch.optim.AdamW(self.policy.parameters())

    if os.path.isfile('cma_np_policy.npz'):
      rep = input('Start with weights from CMA policy? (y or n):')
      if rep == 'y':
        arrs = np.load('cma_np_policy.npz')
        sd = OrderedDict([(k, torch.from_numpy(arr.T).float().to(self.device))\
                          for (k, v), arr in zip(self.policy.state_dict().items(), [arrs['input_layer'], arrs['output_layer']])])
        self.policy.load_state_dict(sd)

  def instanciate_network(self):
    self.policy = torch.nn.Sequential(torch.nn.Linear(4, 64, bias=False),
                                      # torch.nn.ReLU(),
                                      torch.nn.Linear(64, 5, bias=False),
                                      torch.nn.Softmax(dim=-1)).to(self.device)
  
  @staticmethod
  def collect_one_episode(policy, env=None, max_ep_len=120, device=None):
    env = RobotArmEnvVG({'in_background': True}) if env is None else env
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

    obs = torch.tensor(env.reset(to_reset='both', target_pos=[194, 165]), dtype=torch.float)

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
    actions, states, rewards = REINFORCETrainer.collect_one_episode(self.policy, env=env)
    discounted_rewards = REINFORCETrainer.rollout_episode(rewards)
    for state, action, d_reward in zip(states, actions, discounted_rewards):
      probs = self.policy(state.to(self.device)).cpu()
      dist = torch.distributions.Categorical(probs=probs)
      log_probs = dist.log_prob(action)

      loss = - log_probs * d_reward

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
  
  def learn_batch(self, env=None):
    actions, states, rewards = REINFORCETrainer.collect_one_episode(self.policy, env=env)
    discounted_rewards = REINFORCETrainer.rollout_episode(rewards)
    for i in range(0, len(actions), 30):
      probs = self.policy(torch.stack(states[i:i+30], 0).to(self.device)).cpu()
      dist = torch.distributions.Categorical(probs=probs)
      log_probs = dist.log_prob(torch.stack(actions[i:i+30], 0))

      dr_batch = torch.stack(discounted_rewards[i:i+30], 0)
      dr_batch /= (dr_batch.max() + 1e-9)

      loss = - torch.mean(log_probs * dr_batch)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
  
  def evaluation(self, env=None, n_episodes=10, max_ep_len=120):
    env = RobotArmEnvVG() if env is None else env

    rewards = []
    for _ in range(n_episodes):
      obs = torch.tensor(env.reset(to_reset='both', target_pos=[194, 165]), dtype=torch.float)
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
          break

        obs = torch.tensor(obs_, dtype=torch.float)
      
      rewards.append(cum_reward / (i+1))
    
    return np.mean(rewards)
  
  def train(self, n_iterations=100000, eval_step=100):
    env = RobotArmEnvVG()

    for i in tqdm(range(n_iterations)):
      # self.learn(env=env)
      self.learn_batch(env=env)
      
      if i % eval_step == 0:
        mean_reward = self.evaluation(env=env)
        plot_metric(self.vis, mean_reward.item(), i, win='mean_reward', title='Mean reward', ylabel='reward')


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

if __name__ == '__main__':
  rep = input('Train policy using REINFORCE? (y or n): ')
  if rep == 'y':
    trainer = REINFORCETrainer()
    trainer.train()

  rep = input('Train policy using CMA? (y or n): ')
  if rep == 'y':
    trainer = CMATrainer()
    trainer.train()