import os
import ast
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
from torchvision.io import read_image
from torchvision.utils import make_grid, save_image

sys.path.append('../../../robot/')
sys.path.append(os.path.abspath(__file__).replace('USS-GL/simple_goal_exps2_env1.py', ''))

import models.gan_vae_divers as gvd


class NSP(torch.nn.Module):  # NSP=Next State Predictor
  def __init__(self):
    super().__init__()
    self.nsp_in_proj = torch.nn.Linear(32*10*10 + 4 + 8 + 8, 512)
    self.nsp = torch.nn.GRUCell(input_size=512, hidden_size=1024, bias=True)
    self.nsp_out_proj = torch.nn.Linear(1024, 32*10*10)

    self.action_embedder = torch.nn.Embedding(num_embeddings=5, embedding_dim=4)
    self.shoulder_embedder = torch.nn.Embedding(num_embeddings=91, embedding_dim=8)
    self.elbow_embedder = torch.nn.Embedding(num_embeddings=181, embedding_dim=8)

  def forward(self, quantized, action, shoulder_info, elbow_info, hidden=None):
    action_emb = self.action_embedder(action)  # [B] -> [B, 4]
    shoulder_emb = self.shoulder_embedder(shoulder_info)  # [B] -> [B, 8]
    elbow_emb = self.elbow_embedder(elbow_info)  # [B] -> [B, 8]
    # [B, 32*10*10 + 4 + 8 + 8] -> [B, 512]
    enriched_quantized = self.nsp_in_proj(torch.cat([quantized.view(action.size(0), -1),
                                                     action_emb, shoulder_emb, elbow_emb], 1))
    next_hidden = self.nsp(enriched_quantized, hidden)  # [B, 512] -> [B, 1024]
    next_quantized = self.nsp_out_proj(next_hidden)  # [B, 1024] -> [B, 32*10*10]
    return next_quantized.view(quantized.shape), next_hidden  # [B, 32*10*10] -> [B, 32, 10, 10]


class VFL(torch.nn.Module):  # VFL=Visual Feature Learner
  def __init__(self, config={}):
    super().__init__()
    self.encoder = gvd.VQVAE2Encoder(config.get('encoder_config', {
      'batch_norm': True,
      'down_convs_config': [[[3, 32, 4, 2, 1], torch.nn.ReLU],  # 3*180*180 -> 32*90*90
                            [[32, 64, 4, 2, 1], torch.nn.ReLU],  # 32*90*90 -> 64*45*45
                            [[64, 128, 5, 2, 1], torch.nn.ReLU],  # 64*45*45 -> 128*22*22
                            [[128, 128, 5, 2, 1], torch.nn.ReLU]],  # 128*22*22 -> 128*10*10
      'residual_convs_config': [{'convs_config': [[[128, 32, 3, 1, 1], torch.nn.ReLU], [[32, 128, 1, 1, 0], torch.nn.ReLU]]},
                                {'convs_config': [[[128, 32, 3, 1, 1], torch.nn.ReLU], [[32, 128, 1, 1, 0], torch.nn.ReLU]]}]}))
    self.pre_vq_conv = torch.nn.Conv2d(*config.get('pre_vq_conv_config', [128, 32, 1, 1, 0]))
    self.vq = gvd.VectorQuantizer(config.get('vq_config', {'n_embeddings': 32, 'embedding_dim': 32}))
    self.decoder = gvd.VQVAE2Decoder(config.get('decoder_config', {
      'batch_norm': False,
      'convs_config': [[32, 128, 3, 1, 1]],
      # 'residual_convs_config': [],
      'transpose_convs_config': [[[128, 128, 6, 2, 1], torch.nn.ReLU],
                                 [[128, 64, 5, 2, 1], torch.nn.ReLU],
                                 [[64, 32, 4, 2, 1], torch.nn.ReLU],
                                 [[32, 3, 4, 2, 1], None]]}))
  
  def forward(self, visual_input):  # [B, C, H, W]
    visual_latent = self.pre_vq_conv(self.encoder(visual_input))  # [B, 3, 180, 180] -> [B, 256, 10, 10] -> [B, 32, 10, 10]
    loss, quantized, perplexity, encodings = self.vq(visual_latent)
    next_state = self.decoder(quantized)  # [B, 32, 10, 10] -> [B, 3, 180, 180]
    return next_state, loss, quantized


class ActionPredictor(torch.nn.Module):
  # H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
  BASE_CONFIG = {'predictor_config': [
    {'type': torch.nn.Conv2d,  # 64*10*10 -> 128*5*5
    'params': {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'bias': False}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 128}},
    {'type': torch.nn.Conv2d,  # 128*5*5 -> 256*1*1
    'params': {'in_channels': 128, 'out_channels': 256, 'kernel_size': 5, 'stride': 1, 'padding': 0, 'bias': False}},
    {'type': torch.nn.ReLU, 'params': {'inplace': True}},
    {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 256}},
    {'type': torch.nn.Flatten, 'params': {}},
    {'type': torch.nn.Linear, 'params': {'in_features': 256, 'out_features': 5}}]}
  def __init__(self, config):
    super().__init__()
    self.config = {**ActionPredictor.BASE_CONFIG, **config}
    self.predictor = gvd.sequential_constructor(self.config['predictor_config'])
    
  
  def forward(self, state, next_state):
    return self.predictor(torch.cat([state, next_state], 1))


class ActorCriticBaseModel(torch.nn.Module):
  BASE_CONFIG = {'actor_config': [{'type': torch.nn.Linear,  'params': {'in_features': 3200, 'out_features': 2048}},
                                  {'type': torch.nn.ReLU,    'params': {}},
                                  {'type': torch.nn.Linear,  'params': {'in_features': 2048, 'out_features': 1024}},
                                  {'type': torch.nn.ReLU,    'params': {}},
                                  {'type': torch.nn.Linear,  'params': {'in_features': 1024, 'out_features': 5}},
                                  {'type': torch.nn.Softmax, 'params': {'dim': -1}}],
                 'critic_config': [{'type': torch.nn.Linear,  'params': {'in_features': 3200, 'out_features': 2048}},
                                   {'type': torch.nn.ReLU,    'params': {}},
                                   {'type': torch.nn.Linear,  'params': {'in_features': 2048, 'out_features': 1024}},
                                   {'type': torch.nn.ReLU,    'params': {}},
                                   {'type': torch.nn.Linear,  'params': {'in_features': 1024, 'out_features': 1}}],
                 'player': 'actor'}
  def __init__(self, config={}):
    super().__init__()
    self.config = {**ActorCriticBaseModel.BASE_CONFIG, **config}
    self.network = gvd.sequential_constructor(self.config[f"{self.config['player']}_config"])
  
  def forward(self, state):
    out = self.network(state)
    return torch.distributions.categorical.Categorical(out) if self.config['player'] == 'actor' else out


class ActorCritic(torch.nn.Module):
  def __init__(self, config={}):
    super().__init__()
    self.actor = ActorCriticBaseModel()
    self.critic = ActorCriticBaseModel({'player': 'critic'})
  
  def get_action(self, state):
    dist = self.actor(state)
    action = dist.sample()
    action_logprob = dist.log_prob(action)
    return action.detach(), action_logprob.detach()
  
  def evaluate(self, state, action):
    dist = self.actor(state)
    action_logprob = dist.log_prob(action)
    state_value = self.critic(state)
    return action_logprob, state_value, dist.entropy()


class ReplayMemory(object):
  def __init__(self):
    self.states = []
    self.actions = []
    self.logprobs = []
    self.rewards = []
    self.is_terminals = []
  
  def push(self, state, action, logprob, reward, is_terminal):
    self.states.append(state)
    self.actions.append(action)
    self.logprobs.append(logprob)
    self.rewards.append(reward)
    self.is_terminals.append(is_terminal)
  
  def clear(self):
    self.actions.clear()
    self.states.clear()
    self.logprobs.clear()
    self.rewards.clear()
    self.is_terminals.clear()


class PPOAgent(object):
  BASE_CONFIG = {'max_ep_len': 60, 'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 'lr_actor': 1e-4, 'lr_critic': 1e-3, 'gamma': 0.99, 'batch_size': 32,
                 'n_training_timestep': 12000, 'update_timestep': 16, 'n_epochs': 4, 'eps_clip': 0.2}
  def __init__(self, config={}):
    self.config = {**PPOAgent.BASE_CONFIG, **config}

    self.memory = ReplayMemory()

    self.policy = ActorCritic().to(self.config['device'])
    self.optimizer = torch.optim.AdamW([{'params': self.policy.actor.parameters(), 'lr': self.config['lr_actor']},
                                        {'params': self.policy.critic.parameters(), 'lr': self.config['lr_critic']}])

    self.old_policy = ActorCritic().to(self.config['device'])
    self.old_policy.load_state_dict(self.policy.state_dict())

    self.mse_criterion = torch.nn.MSELoss()
  
  @torch.no_grad()
  def get_action(self, state):
    action, action_logprob = self.old_policy.get_action(state)
    return action, action_logprob

  def learn(self):
    rewards, discounted_reward = [], 0
    for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
      if is_terminal:
        discounted_reward = 0
      discounted_reward = reward + self.config['gamma'] * discounted_reward
      rewards.insert(0, discounted_reward)
    
    rewards = torch.FloatTensor(rewards).to(self.config['device'])
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)  # Normalizing the rewards

    old_states = torch.cat(self.memory.states, dim=0).to(self.config['device'])  # list of [1, 3200] -> [batch_size, 3200]
    old_actions = torch.cat(self.memory.actions, dim=0).to(self.config['device'])  # [batch_size]
    old_logprobs = torch.cat(self.memory.logprobs, dim=0).to(self.config['device'])  # [batch_size]

    # Optimize policy for n epochs
    for _ in range(self.config['n_epochs']):
      # Evaluating old actions and values
      logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
      state_values = state_values.view(-1)
      # Finding policy ratio = policy/old_policy
      ratios = torch.exp(logprobs - old_logprobs)
      # Finding surrogate loss
      advantages = rewards - state_values
      surrogate1 = ratios * advantages
      surrogate2 = torch.clamp(ratios, 1 - self.config['eps_clip'], 1 + self.config['eps_clip']) * advantages
      # Final loss
      loss = -torch.min(surrogate1, surrogate2) + 0.5 * self.mse_criterion(state_values, rewards) - 0.01 * dist_entropy
      # Optimize
      self.optimizer.zero_grad()
      loss.mean().backward()
      self.optimizer.step()
    
    # Copy new weights into old policy network
    self.old_policy.load_state_dict(self.policy.state_dict())

    self.memory.clear()


class GlobalTrainer(object):
  BASE_CONFIG = {'batch_size': 32, 'n_actions': 5, 'memory_size': 6400,
                 'models_folder': 'models/', 'n_training_iterations': 3000}
  def __init__(self, config):
    self.config = {**GlobalTrainer.BASE_CONFIG, **config}

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.instanciate_model()
    self.instanciate_optimizers_n_criterions()

    self.memory = []
  
  def instanciate_model(self):
    self.vfl_learner = VFL().to(self.device)
    self.nsp_predictor = NSP().to(self.device)
    self.ppo_agent = PPOAgent()
  
  def instanciate_optimizers_n_criterions(self):
    self.mse_criterion = torch.nn.MSELoss()

    self.vfl_optimizer = torch.optim.AdamW(self.vfl_learner.parameters())
    self.nsp_optimizer = torch.optim.AdamW(self.nsp_predictor.parameters())

  def get_state(self):
    screen = tvt.functional.to_tensor(env.get_screen())
    screen_cropped = tvt.functional.crop(screen, 140, 115, 180, 245)
    screen_resized = tvt.functional.resize(screen_cropped, [180, 180])
    return screen_resized
  
  def plot_loss(self, loss, iteration, win='loss', title='loss evolution'):
    if iteration == 0:
      vis.line(X=np.array([iteration, iteration]), Y=np.array([loss, loss]), win=win,
                                opts={'ylabel': 'loss', 'xlabel': 'iteration', 'title': title})
    else:
      vis.line(X=np.array([iteration]), Y=np.array([loss]), win=win, update='append')
  
  def save_model(self, model, save_name='global_trainer_model.pt', put_in_models_folder=True):
    save_name = os.path.join(self.config['models_folder'], save_name) if put_in_models_folder else save_name
    if not os.path.isdir(os.path.dirname(save_name)):
      os.makedirs(os.path.dirname(save_name))
    torch.save({'model': model.state_dict()}, save_name)
  
  def load_model(self, model, map_location=None, save_name='global_trainer_model.pt', put_in_models_folder=True):
    save_name = os.path.join(self.config['models_folder'], save_name) if put_in_models_folder else save_name
    if os.path.isfile(save_name):
      data = torch.load(save_name, map_location=map_location)
      model.load_state_dict(data['model'])
    else:
      print(f"File {save_name} doesn't exist")
  
  def fill_vfl_memory(self):
    n_step = 0
    self.memory = []
    for _ in tqdm(range(self.config['memory_size'])):
      env.render()

      state = self.get_state()
      action = random.randint(0, 4)

      joints_angle, reward, done, _ = env.step(action)
      shoulder, elbow = map(lambda x: round(x), joints_angle)

      env.render()
      next_state = self.get_state()

      self.memory.append([state, next_state, action, shoulder, elbow])
      n_step += 1

      if done or n_step == 60:
        env.reset()
        n_step = 0

  def train_vfl_learner(self, n_iterations=3000, plot_progress=True, save_model=True):
    print('Fill vfl_memory...')
    self.fill_vfl_memory()

    print(f'Train vfl_learner for {n_iterations} iterations...')
    with tqdm(total=n_iterations) as t:
      for i in range(n_iterations):
        states, _, _, _, _ = zip(*random.sample(self.memory, self.config['batch_size']))
        states = torch.stack(states, 0).to(self.device)

        states_rec, vq_loss, _ = self.vfl_learner(states)

        self.vfl_optimizer.zero_grad()
        rec_loss = self.mse_criterion(states_rec, states)
        loss = vq_loss + rec_loss
        loss.backward()
        self.vfl_optimizer.step()

        if plot_progress and (i % 50 == 0 or i == (n_iterations-1)):
          vis.image(make_grid(states[:6].cpu(), nrow=3), win='state', opts={'title': 'state'})

          # Strangely, it plots black image so an hack is to save the image first then load it to pass it to visdom
          # vis.image(make_grid(state_rec[:6].cpu(), nrow=3), win='rec', opts={'title': 'reconstructed'})
          save_image(make_grid(states_rec[:6].cpu(), nrow=3), 'test_make_grid.png')
          vis.image(read_image('test_make_grid.png'), win='rec', opts={'title': 'reconstructed'})

          self.plot_loss(loss.item(), i, win='vfl_loss', title='VFL loss evolution')
        
        t.set_description(f'Loss={loss.item():.3f}')
        t.update(1)
    
    if save_model:
      self.save_model(self.vfl_learner, save_name='visual_feature_learner.pt')
  
  @torch.no_grad()
  def fill_nsp_memory(self):
    if len(self.memory) < self.config['memory_size']:
      self.fill_vfl_memory()

    for i in tqdm(range(0, len(self.memory), self.config['batch_size'])):
      states, next_states, actions, shoulders, elbows = zip(*self.memory[i:i+self.config['batch_size']])

      states = torch.stack(states, 0).to(self.device)
      next_states = torch.stack(next_states).to(self.device)

      actions = torch.LongTensor(actions)
      shoulders = torch.LongTensor(shoulders)
      elbows = torch.LongTensor(elbows)

      _, _, quantized = self.vfl_learner(states)
      _, _, next_quantized = self.vfl_learner(next_states)

      for j, (s, ns, q, nq, a, sh, e) in enumerate(zip(states.cpu(), next_states.cpu(), quantized.cpu(),
                                                       next_quantized.cpu(), actions, shoulders, elbows)):
        self.memory[i+j] = [s, ns, q, nq, a, sh, e]

  def train_nsp_predictor(self, n_iterations=3000, plot_progress=True, save_model=True):
    print('Fill nsp_memory...')
    self.fill_nsp_memory()

    print(f'Train nsp_predictor for {n_iterations} iterations...')
    i, hidden = 0, None
    with tqdm(total=n_iterations) as t:
      for _ in range(n_iterations):
        _, next_states, quantized, next_quantized, actions, shoulders, elbows = zip(*self.memory[i:i+self.config['batch_size']])

        next_states = torch.stack(next_states, 0)
        quantized = torch.stack(quantized, 0).to(self.device)
        next_quantized = torch.stack(next_quantized, 0).to(self.device)
        actions = torch.stack(actions, 0).to(self.device)
        shoulders = torch.stack(shoulders, 0).to(self.device)
        elbows = torch.stack(elbows, 0).to(self.device)

        pred_next_quantized, hidden = self.nsp_predictor(quantized, actions, shoulders, elbows, hidden=hidden)
        hidden = hidden.detach()

        self.nsp_optimizer.zero_grad()
        loss = self.mse_criterion(pred_next_quantized, next_quantized)
        loss.backward()
        self.nsp_optimizer.step()

        if plot_progress and (i % 50 == 0 or i == (n_iterations-1)):
          with torch.no_grad():
            next_state_rec = self.vfl_learner.decoder(next_quantized)
            pred_next_state_rec = self.vfl_learner.decoder(pred_next_quantized)

          vis.image(make_grid(next_states[:3], nrow=3), win='next_state', opts={'title': 'next_state'})

          save_image(make_grid(next_state_rec[:3].cpu(), nrow=3), 'test_make_grid1.png')
          vis.image(read_image('test_make_grid1.png'), win='next_state_rec', opts={'title': 'next_state_rec'})

          save_image(make_grid(pred_next_state_rec[:3].cpu(), nrow=3), 'test_make_grid2.png')
          vis.image(read_image('test_make_grid2.png'), win='pred_next_state_rec', opts={'title': 'pred_next_state_rec'})

          self.plot_loss(loss.item(), i, win='nsp_loss', title='NSP loss evolution')

        t.set_description(f'Loss={loss.item():.3f}')
        t.update(1)

        i += 1
        if i + self.config['batch_size'] >= len(self.memory):
          i = 0
    
    if save_model:
      self.save_model(self.nsp_predictor, save_name='next_state_predictor.pt')
  
  def train_policy_network(self, save_model=True):
    for i in tqdm(range(1, self.ppo_agent.config['n_training_timestep'] + 1)):
      env.reset(only_target=True)
      env.render()
      state = self.get_state().unsqueeze(0).to(self.device)  # [1, 3, 180, 180]

      cum_reward = 0
      for t in range(1, self.ppo_agent.config['max_ep_len'] + 1):
        with torch.no_grad():
          _, _, quantized = self.vfl_learner(state)  # [1, 32, 10, 10]
        quantized = quantized.view(1, -1)  # [1, 3200] -> todo=add body_info

        action, action_log_prob = self.ppo_agent.get_action(quantized)
        body_info, reward, done, _ = env.step(action.item())

        self.ppo_agent.memory.push(quantized.detach().cpu(), action.cpu(), action_log_prob.cpu(), reward, done)

        env.render()
        state = self.get_state().unsqueeze(0).to(self.device)

        cum_reward += reward

        if done:
          break
      
      self.plot_loss(cum_reward / t, i-1, win='ep_rewards', title='Episodic rewards')
      
      if i % self.ppo_agent.config['update_timestep'] == 0:
        self.ppo_agent.learn()
    
    if save_model:
      self.save_model(self.ppo_agent.policy, save_name='policy_network.pt')

  def train_routine(self, force_retrain, model_name, model_train_fn, model):
    if not os.path.isfile(os.path.join(self.config['models_folder'], model_name)):
      model_train_fn(n_iterations=self.config['n_training_iterations'])
    else:
      print(f'Loading {model_name} model...')
      self.load_model(model, save_name=model_name)

      if force_retrain:
        model_train_fn(n_iterations=self.config['n_training_iterations'])

  def train(self, force_retrain=False):
    self.train_routine(force_retrain, 'visual_feature_learner.pt', self.train_vfl_learner, self.vfl_learner)
    self.train_routine(force_retrain, 'next_state_predictor.pt', self.train_nsp_predictor, self.nsp_predictor)
    self.train_routine(force_retrain, 'policy_network.pt', self.train_policy_network, self.ppo_agent.policy)


def experiment(args):
  gt = GlobalTrainer({})
  print(f'vfl_learner -> n_parameters={sum(p.numel() for p in gt.vfl_learner.parameters() if p.requires_grad):,}')
  print(f'nsp_predictor -> n_parameters={sum(p.numel() for p in gt.nsp_predictor.parameters() if p.requires_grad):,}')
  gt.train(force_retrain=args.force_retrain, load_vfl_memory=args.load_vfl_memory)


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
  argparser.add_argument('--force_retrain', default=False, type=ast.literal_eval)
  args = argparser.parse_args()

  logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')
  
  vis = visdom.Visdom()  # https://github.com/fossasia/visdom#usage
  
  # env.joints_angle = [angle_joint1, angle_joint2]
  # joint1 -> torch.nn.Embedding(n_emb=91, emb_size=8)
  # joint2 -> torch.nn.Embedding(n_emb=181, emb_size=8)
  # env.get_screen() = np.ndarray = [400, 400, 3]

  # MAIN LOOP SIMULATION
  env = gym.make('gym_robot_arm:robot-arm-v1')
  env.reset()
  env.render()

  gt = GlobalTrainer({})

  print(f'vfl_learner -> n_parameters={sum(p.numel() for p in gt.vfl_learner.parameters() if p.requires_grad):,}')
  print(f'nsp_predictor -> n_parameters={sum(p.numel() for p in gt.nsp_predictor.parameters() if p.requires_grad):,}')

  gt.train(force_retrain=args.force_retrain)

  env.close()

  env = gym.make('gym_robot_arm:robot-arm-v0')
  env.render()
  env.reset()