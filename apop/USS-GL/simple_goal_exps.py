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
from PIL import Image
from itertools import islice
from torchvision.io import read_image
from collections import deque, namedtuple
from torchvision.utils import make_grid, save_image

sys.path.append('../../../robot/')
sys.path.append(os.path.abspath(__file__).replace('USS-GL/simple_goal_exps.py', ''))

import models.gan_vae_divers as gvd


def tensor_to_img(tensor):
  grid = make_grid(tensor)
  ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
  im = Image.fromarray(ndarr)
  return im


def state_to_label(tensor):
  return tensor.mul(255).add_(0.5).clamp_(0, 255)


class NextStatesPredictor(torch.nn.Module):
  # nspc = next_state_predictor_config
  BASE_CONFIG = {'vq_emb_size': 32,
                 'nspc': {'input_size': 32+5, 'hidden_size': 128, 'num_layers': 1, 'bias': True,
                          'batch_first': False, 'dropout': 0., 'bidirectional': True}}
  def __init__(self, config):
    super().__init__()
    self.config = {**NextStatesPredictor.BASE_CONFIG, **config}
    self.next_state_predictor = torch.nn.GRU(**self.config['nspc'])
    self.nsp_out = torch.nn.Linear(self.config['nspc']['hidden_size'] * 2, self.config['vq_emb_size'])
  
  def forward(self, states, actions, next_states=None):
    '''
    parameters:
      * states : torch.Tensor, shape=[B, C, H, W]
      * actions : torch.Tensor, shape=[B, n_step_ahead, n_actions]
      * next_states (optional, default=None) : torch.Tensor, shape=[n_step_ahead, B, C, H, W]
    '''
    B, C, H, W = states.shape
    states = states.permute(2, 3, 0, 1).contiguous().view(-1, B, C)  # [HW, B, C]

    predicted_next_states = []
    for i in range(actions.size(1)):
      action = actions[:, i].unsqueeze(0).repeat(states.size(0), 1, 1)
      predicted_next_state, _ = self.next_state_predictor(torch.cat([states, action], -1))  # [HW, B, hidden_size*2]
      predicted_next_state = self.nsp_out(predicted_next_state)  # [HW, B, C]
      predicted_next_states.append(predicted_next_state.permute(1, 2, 0).contiguous().view(B, C, H, W))

      if next_states is None:
        states = predicted_next_state
      else:
        states = next_states[i]
    
    return torch.stack(predicted_next_states)  # [n_step_ahead, B, C, H, W]


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


class ReplayMemory(object):
  def __init__(self, capacity, transition, save_folder='replay_memory/'):
    self.capacity = capacity
    self.memory = deque([], maxlen=capacity)
    self.transition = transition
    self.save_folder = save_folder

  def push(self, *args):
    self.memory.append(self.transition(*args))

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)
  
  def save(self):
    if not os.path.isdir(self.save_folder):
      os.makedirs(self.save_folder)

    ids_taken = set([int(fname.replace('.pt', '')) for fname in os.listdir(self.save_folder)])
    ids_to_save = []
    i = 0
    while len(ids_to_save) < len(self.memory):
      if i not in ids_taken and i not in ids_to_save:
        ids_to_save.append(i)
      else:
        i += 1
    
    for i, transition in zip(ids_to_save, self.memory):
      torch.save(list(transition), os.path.join(self.save_folder, f'{i}.pt'))

  def load(self):
    for fname in tqdm(random.sample(os.listdir(self.save_folder), self.capacity)):
      transition = torch.load(os.path.join(self.save_folder, fname))
      self.push(*transition)


class GlobalTrainer(object):
  BASE_CONFIG = {'batch_size': 32, 'n_actions': 5, 'vfl_memory_size': 12800, 'nsp_memory_size': 12800,
                 'models_folder': 'models/', 'n_training_iterations': 3000}
  def __init__(self, config):
    self.config = {**GlobalTrainer.BASE_CONFIG, **config}

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.instanciate_model()
    self.instanciate_optimizers_n_criterions()
  
  def instanciate_model(self):
    # vfl = visual_feature_learner
    # with batch normalization, it learns faster but creates some artefacts
    vfl_residualCNN_config = {'layers_config': [
      {'type': torch.nn.ReLU, 'params': {'inplace': True}},
      {'type': torch.nn.Conv2d, 'params': {'in_channels': 64, 'out_channels': 16, 'kernel_size': 3, 'stride': 1,
                                           'padding': 1, 'bias': False}},
      {'type': torch.nn.ReLU, 'params': {'inplace': True}},
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
      # 64*45*45 -> 64*22*22
      {'type': torch.nn.Conv2d, 'params': {'in_channels': 64, 'out_channels': 64, 'kernel_size': 5, 'stride': 2, 'padding': 1}},
      {'type': torch.nn.ReLU, 'params': {'inplace': True}},
      {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 64}},
      # 64*22*22 -> 64*10*10
      {'type': torch.nn.Conv2d, 'params': {'in_channels': 64, 'out_channels': 64, 'kernel_size': 5, 'stride': 2, 'padding': 1}},
      {'type': torch.nn.ReLU, 'params': {'inplace': True}},
      {'type': torch.nn.BatchNorm2d, 'params': {'num_features': 64}},
      # 64*10*10 -> 64*10*10
      {'type': torch.nn.Conv2d, 'params': {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}},
      # 2 Residual blocks
      {'type': gvd.ResidualCNN, 'params': {'config': vfl_residualCNN_config}},
      {'type': gvd.ResidualCNN, 'params': {'config': vfl_residualCNN_config}},
      {'type': torch.nn.ReLU, 'params': {'inplace': True}}]}
    vfl_pre_vq_conv_config = {'in_channels': 64, 'out_channels': 32, 'kernel_size': 1, 'stride': 1, 'padding': 0}
    vfl_decoder_config = {'layers_config': [
      # 32*10*10 -> 64*10*10
      {'type': torch.nn.Conv2d, 'params': {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}},
      # 2 Residual blocks
      {'type': gvd.ResidualCNN, 'params': {'config': vfl_residualCNN_config}},
      {'type': gvd.ResidualCNN, 'params': {'config': vfl_residualCNN_config}},
      # 64*10*10 -> 64*22*22
      {'type': torch.nn.ConvTranspose2d, 'params': {'in_channels': 64, 'out_channels': 64, 'kernel_size': 6,
                                                    'stride': 2, 'padding': 1}},
      {'type': torch.nn.ReLU, 'params': {'inplace': True}},
      # 64*22*22 -> 64*45*45
      {'type': torch.nn.ConvTranspose2d, 'params': {'in_channels': 64, 'out_channels': 64, 'kernel_size': 5,
                                                    'stride': 2, 'padding': 1}},
      {'type': torch.nn.ReLU, 'params': {'inplace': True}},
      # 64*45*45 -> 32*90*90
      {'type': torch.nn.ConvTranspose2d, 'params': {'in_channels': 64, 'out_channels': 32, 'kernel_size': 4,
                                                    'stride': 2, 'padding': 1}},
      {'type': torch.nn.ReLU, 'params': {'inplace': True}},
      # 32*90*90 -> 3*180*180
      {'type': torch.nn.ConvTranspose2d, 'params': {'in_channels': 32, 'out_channels': 3, 'kernel_size': 4,
                                                    'stride': 2, 'padding': 1}}]}
    self.visual_feature_learner = gvd.VQVAEModel({'encoder_config': vfl_encoder_config,
                                                  'pre_vq_conv_config': vfl_pre_vq_conv_config,
                                                  'vq_config': {'n_embeddings': 32, 'embedding_dim': 32},
                                                  'decoder_config': vfl_decoder_config}).to(self.device)
    self.next_states_predictor = NextStatesPredictor({}).to(self.device)
    self.action_predictor = ActionPredictor({}).to(self.device)
  
  def instanciate_optimizers_n_criterions(self):
    self.transition_vfl = namedtuple('Transition_vfl', ('state', 'action', 'next_state', 'reward'))
    self.transition_nsp = namedtuple('Transition_msap', ('quantized', 'action', 'next_quantized'))

    self.mse_criterion = torch.nn.MSELoss()
    self.ce_criterion = torch.nn.CrossEntropyLoss()

    self.vfl_optimizer = torch.optim.AdamW(self.visual_feature_learner.parameters())
    self.vfl_memory = ReplayMemory(self.config['vfl_memory_size'], self.transition_vfl)

    self.nsp_optimizer = torch.optim.AdamW(self.next_states_predictor.parameters())  # nsp = next_state_predictor
    self.nsp_memory = ReplayMemory(self.config['nsp_memory_size'], self.transition_nsp)

    self.ap_optimizer = torch.optim.AdamW(self.action_predictor.parameters())

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
  
  def fill_visual_feature_learner_memory(self, strategy='random', load=False, save=False):
    ori_screen = env.screen
    env.screen = pygame.Surface(env.config['window_size'])

    print('Fill Visual Feature Learner Memory...')
    if strategy == 'random':
      if load and os.path.isdir(self.vfl_memory.save_folder)\
              and len(os.listdir(self.vfl_memory.save_folder)) >= self.config['vfl_memory_size']:
        print(f'-> load transitions from folder {self.vfl_memory.save_folder}')
        self.vfl_memory.load()
      else:
        for _ in tqdm(range(self.config['vfl_memory_size'])):
          env.reset()
          env.render(mode='robot')

          state = self.get_state()
          action = random.randint(0, 4)

          env.step(action)
          env.render(mode='robot')

          next_state = self.get_state()
          self.vfl_memory.push(state, torch.LongTensor([action]), next_state, 0)
        
        if save:
          self.vfl_memory.save()

    env.screen = ori_screen
    global memory_filling_finished
    memory_filling_finished = True
  
  def train_visual_feature_learner(self, n_iterations=1000, plot_progress=True, save_model=True, load_memory=True):
    self.fill_visual_feature_learner_memory(load=load_memory)
    print(f'Training Visual Feature Learner for {n_iterations} iterations...')

    with tqdm(total=n_iterations) as t:
      for i in range(n_iterations):
        transitions = self.vfl_memory.sample(self.config['batch_size'])
        batch = self.transition_vfl(*zip(*transitions))
        states_batch = torch.stack(batch.state).to(self.device)
        vq_loss, state_rec, perplexity, encodings, quantized = self.visual_feature_learner(states_batch)

        self.vfl_optimizer.zero_grad()
        rec_loss = self.mse_criterion(state_rec, states_batch)
        loss = vq_loss + rec_loss
        loss.backward()
        self.vfl_optimizer.step()

        if plot_progress and (i % 50 == 0 or i == (n_iterations-1)):
          vis.image(make_grid(states_batch[:6].cpu(), nrow=3), win='state', opts={'title': 'state'})

          # Strangely, it plots black image so an hack is to save the image first then load it to pass it to visdom
          # vis.image(make_grid(state_rec[:6].cpu(), nrow=3), win='rec', opts={'title': 'reconstructed'})
          save_image(make_grid(state_rec[:6].cpu(), nrow=3), 'test_make_grid.png')
          vis.image(read_image('test_make_grid.png'), win='rec', opts={'title': 'reconstructed'})

          self.plot_loss(loss.item(), i)
        
        t.set_description(f'Loss={loss.item():.3f}')
        t.update(1)
    
    if save_model:
      self.save_model(self.visual_feature_learner, save_name='visual_feature_learner.pt')
  
  def online_visual_feature_learner_training(self, n_iterations=10000, plot_progress=True, save_model=True):
    ori_screen = env.screen
    env.screen = pygame.Surface(env.config['window_size'])

    print(f'Online Training of Visual Feature Learner for {n_iterations} iterations...')
    with tqdm(total=n_iterations) as t:
      for i in range(n_iterations):
        env.reset()
        env.render(mode='robot')

        state = self.get_state().unsqueeze(0).to(self.device)
        action = random.randint(0, 4)

        env.step(action)
        env.render(mode='robot')

        vq_loss, state_rec, perplexity, encodings, quantized = self.visual_feature_learner(state)

        self.vfl_optimizer.zero_grad()
        rec_loss = self.mse_criterion(state_rec, state)
        loss = vq_loss + rec_loss
        loss.backward()
        self.vfl_optimizer.step()

        if plot_progress and (i % 50 == 0 or i == (n_iterations-1)):
          vis.image(make_grid(state.cpu(), nrow=1), win='state', opts={'title': 'state'})

          # Strangely, it plots black image so an hack is to save the image first then load it to pass it to visdom
          # vis.image(make_grid(state_rec[:6].cpu(), nrow=3), win='rec', opts={'title': 'reconstructed'})
          save_image(make_grid(state_rec.cpu(), nrow=1), 'test_make_grid.png')
          vis.image(read_image('test_make_grid.png'), win='rec', opts={'title': 'reconstructed'})

          self.plot_loss(loss.item(), i)
        
        t.set_description(f'Loss={loss.item():.3f}')
        t.update(1)
    
    if save_model:
      self.save_model(self.visual_feature_learner, save_name='visual_feature_learner.pt')
    
    env.screen = ori_screen
  
  @torch.no_grad()
  def fill_nsp_memory(self, n_step_ahead=1):
    if n_step_ahead == 1:
      if len(self.vfl_memory) == 0:
        self.fill_visual_feature_learner_memory()
      print('Fill Next States Predictor Memory...')
      for _ in tqdm(range(self.config['nsp_memory_size']//self.config['batch_size'])):
        transitions = self.vfl_memory.sample(self.config['batch_size'])
        batch = self.transition_vfl(*zip(*transitions))

        states_batch = torch.stack(batch.state).to(self.device)  # [B, C=3, H=180, W=180]
        _, _, _, _, quantized = self.visual_feature_learner(states_batch)  # [B, C=32, H=10, W=10]

        next_states_batch = torch.stack(batch.next_state).to(self.device) # [B, C=3, H=180, W=180]
        _, _, _, _, next_quantized = self.visual_feature_learner(next_states_batch)  # [B, C=32, H=10, W=10]
        actions_batch = torch.nn.functional.one_hot(torch.cat(batch.action), self.config['n_actions']).unsqueeze(1)  # [B, 1, N_a]

        for q, a, nq in zip(quantized.cpu(), actions_batch, next_quantized.cpu()):
          self.nsp_memory.push(q, a, nq.unsqueeze(0))
    else:
      pass

  def train_next_states_predictor(self, n_iterations=1000, plot_progress=True, save_model=True, n_step_ahead=1):
    self.fill_nsp_memory(n_step_ahead=n_step_ahead)
    print(f'Training Next States Predictor for {n_iterations} iterations...')

    with tqdm(total=n_iterations) as t:
      for i in range(n_iterations):
        transitions = self.nsp_memory.sample(self.config['batch_size'])
        batch = self.transition_nsp(*zip(*transitions))
        quantized_batch = torch.stack(batch.quantized).to(self.device)  # [B, C=32, H=10, W=10]
        action_batch = torch.stack(batch.action).to(self.device)  # [B, n_step_ahead=1, n_actions=5]
        next_quantized_batch = torch.stack(batch.next_quantized, 1).to(self.device)  # [n_step_ahead=1, B, C=32, H=10, W=10]

        # [n_step_ahead=1, B, C=32, H=10, W=10]
        predicted_quantized = self.next_states_predictor(quantized_batch, action_batch, next_states=next_quantized_batch)
        
        self.nsp_optimizer.zero_grad()
        rec_loss = self.mse_criterion(predicted_quantized.view(-1, *quantized_batch.shape[1:]),
                                      next_quantized_batch.view(-1, *quantized_batch.shape[1:]))
        rec_loss.backward()
        self.nsp_optimizer.step()

        if plot_progress and (i % 50 == 0 or i == (n_iterations-1)):
          self.plot_loss(rec_loss.item(), i, win='rec_quantized_loss', title='rec_quantized_loss evolution')

        t.set_description(f'Loss={rec_loss.item():.3f}')
        t.update(1)
    
    if save_model:
      self.save_model(self.next_states_predictor, save_name='next_states_predictor.pt')
  
  @torch.no_grad()
  def eval_action_predictor(self, n_samples=40):
    preds, targets = [], []
    for i in range(0, self.config['batch_size']*n_samples, self.config['batch_size']):
      transitions = list(islice(self.nsp_memory.memory, i, i+self.config['batch_size']))
      batch = self.transition_nsp(*zip(*transitions))
      quantized_batch = torch.stack(batch.quantized).to(self.device)
      action_batch = torch.stack(batch.action).to(self.device)
      next_quantized_batch = torch.stack(batch.next_quantized, 1).to(self.device)
      predicted_action = self.action_predictor(quantized_batch, next_quantized_batch[0])

      preds += predicted_action.argmax(-1).cpu().tolist()
      targets += action_batch[:, 0].argmax(-1).cpu().tolist()
    # print(f'Accuracy = {sum([1 for p, t in zip(preds, targets) if p == t])/len(preds):.3f}')
    return sum([1 for p, t in zip(preds, targets) if p == t])/len(preds)

  def train_action_predictor(self, n_iterations=1000, plot_progress=True, save_model=True):
    if len(self.nsp_memory) < GlobalTrainer.BASE_CONFIG['nsp_memory_size']:
      self.fill_nsp_memory()
    
    transitions = self.vfl_memory.sample(self.config['batch_size'])
    batch = self.transition_vfl(*zip(*transitions))
    states_batch = torch.stack(batch.state).to(self.device)
    vq_loss, state_rec, perplexity, encodings, quantized = self.visual_feature_learner(states_batch)
    vis.image(make_grid(states_batch[:6].cpu(), nrow=3), win='state', opts={'title': 'state'})
    save_image(make_grid(state_rec[:6].cpu(), nrow=3), 'test_make_grid.png')
    vis.image(read_image('test_make_grid.png'), win='rec', opts={'title': 'reconstructed'})
    
    print(f'Training Action Predictor for {n_iterations} iterations...')
    with tqdm(total=n_iterations) as t:
      for i in range(n_iterations):
        transitions = self.nsp_memory.sample(self.config['batch_size'])
        batch = self.transition_nsp(*zip(*transitions))
        quantized_batch = torch.stack(batch.quantized).to(self.device)  # [B, C=32, H=10, W=10]
        action_batch = torch.stack(batch.action).to(self.device)  # [B, n_step_ahead=1, n_actions=5]
        next_quantized_batch = torch.stack(batch.next_quantized, 1).to(self.device)  # [n_step_ahead=1, B, C=32, H=10, W=10]

        predicted_action = self.action_predictor(quantized_batch, next_quantized_batch[0])

        self.ap_optimizer.zero_grad()
        loss = self.ce_criterion(predicted_action, action_batch[:, 0].float().argmax(-1))
        loss.backward()
        self.ap_optimizer.step()

        if plot_progress:
          if i % 50 == 0 or i == (n_iterations-1):
            self.plot_loss(loss.item(), i, win='ap_loss', title='action_predictor_loss evolution')

            acc = self.eval_action_predictor()
            self.plot_loss(acc, i, win='acc_action_predictor', title='action_predictor accuracy')

        t.set_description(f'Loss={loss.item():.3f}')
        t.update(1)
      
    if save_model:
      self.save_model(self.action_predictor, save_name='action_predictor.pt')
  
  def train(self, force_retrain=False, load_vfl_memory=True):
    if not os.path.isfile(os.path.join(self.config['models_folder'], 'visual_feature_learner.pt')):
      self.train_visual_feature_learner(n_iterations=self.config['n_training_iterations'], load_memory=load_vfl_memory)
    else:
      print('Loading Visual Feature Learner model...')
      self.load_model(self.visual_feature_learner, save_name='visual_feature_learner.pt')

      if force_retrain:
        self.train_visual_feature_learner(n_iterations=self.config['n_training_iterations'], load_memory=load_vfl_memory)
    
    if not os.path.isfile(os.path.join(self.config['models_folder'], 'next_states_predictor.pt')):
      self.train_next_states_predictor(n_iterations=self.config['n_training_iterations'])
    else:
      print('Loading Next States Predictor model...')
      self.load_model(self.next_states_predictor, save_name='next_states_predictor.pt')

      if force_retrain:
        self.train_next_states_predictor(n_iterations=self.config['n_training_iterations'])
    
    if not os.path.isfile(os.path.join(self.config['models_folder'], 'action_predictor.pt')):
      self.train_action_predictor(n_iterations=self.config['n_training_iterations'])
    else:
      print('Loading Next States Predictor model...')
      self.load_model(self.action_predictor, save_name='action_predictor.pt')

      if force_retrain:
        self.train_action_predictor(n_iterations=self.config['n_training_iterations'])


def experiment(args):
  gt = GlobalTrainer({})
  print(f'visual_feature_learner n_parameters={sum(p.numel() for p in gt.visual_feature_learner.parameters() if p.requires_grad):,}')
  print(f'next_states_predictor n_parameters={sum(p.numel() for p in gt.next_states_predictor.parameters() if p.requires_grad):,}')
  print(f'action_predictor n_parameters={sum(p.numel() for p in gt.action_predictor.parameters() if p.requires_grad):,}')
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
  argparser.add_argument('--load_vfl_memory', default=False, type=ast.literal_eval)
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
  memory_filling_finished = False

  th = threading.Thread(target=experiment, args=(args,))
  th.start()

  while not done:
    if memory_filling_finished:
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
        act = False
      
      env.render()

  env.close()
  th.join()


## LEGACY ##
# def joint_train_vfl_nsp(self, n_iterations=1000, plot_progress=True, save_model=True, load_memory=True, n_step_ahead=1):
#   self.fill_visual_feature_learner_memory(load=load_memory)
#   print(f'Training Visual Feature Learner for {n_iterations} iterations...')

#   with tqdm(total=n_iterations) as t:
#     for i in range(n_iterations):
#       transitions = self.vfl_memory.sample(self.config['batch_size'])
#       batch = self.transition_vfl(*zip(*transitions))
#       states_batch = torch.stack(batch.state).to(self.device)
#       vq_loss, state_rec, perplexity, encodings, quantized = self.visual_feature_learner(states_batch)

#       action_batch = torch.nn.functional.one_hot(torch.cat(batch.action), self.config['n_actions']).unsqueeze(1)
#       predicted_quantized = self.next_states_predictor(quantized, action_batch.to(self.device))
#       next_state_rec = self.visual_feature_learner.decoder(predicted_quantized.squeeze(0))

#       next_states_batch = torch.stack(batch.next_state).to(self.device)

#       self.vfl_optimizer.zero_grad()
#       self.nsp_optimizer.zero_grad()

#       state_rec_loss = self.mse_criterion(state_rec, states_batch)
#       next_state_rec_loss = self.mse_criterion(next_state_rec, next_states_batch)
#       loss = vq_loss + state_rec_loss + next_state_rec_loss
#       loss.backward()

#       self.vfl_optimizer.step()
#       self.nsp_optimizer.step()

#       if plot_progress and (i % 50 == 0 or i == (n_iterations-1)):
#         vis.image(make_grid(states_batch[:6].cpu(), nrow=3), win='state', opts={'title': 'state'})
#         save_image(make_grid(state_rec[:6].cpu(), nrow=3), 'test_make_grid.png')
#         vis.image(read_image('test_make_grid.png'), win='rec', opts={'title': 'reconstructed'})

#         self.plot_loss(loss.item(), i)
#         self.plot_loss(state_rec_loss.item(), i, win='s_rec_loss', title='state_rec_loss evolution')
#         self.plot_loss(next_state_rec_loss.item(), i, win='ns_rec_loss', title='next_state_rec_loss evolution')
      
#       t.set_description(f'Loss={loss.item():.3f}')
#       t.update(1)
  
#   if save_model:
#     self.save_model(self.visual_feature_learner, save_name='visual_feature_learner.pt')
#     self.save_model(self.next_states_predictor, save_name='next_states_predictor.pt')


# VQ-VAE-2
# self.visual_feature_learner = gvd.VQVAE2({
#       'encoder_bottom_config': {
#           'batch_norm': True,
#           'down_convs_config': [[[3, 32, 4, 2, 1], torch.nn.ReLU],  # 3*180*180 -> 32*90*90
#                                 [[32, 64, 4, 2, 1], torch.nn.ReLU],  # 32*90*90 -> 64*45*45
#                                 [[64, 128, 5, 2, 1], torch.nn.ReLU],  # 64*45*45 -> 128*22*22
#                                 [[128, 256, 5, 2, 1], torch.nn.ReLU]],  # 128*22*22 -> 256*10*10
#           'residual_convs_config': [{'convs_config': [[[256, 64, 3, 1, 1], torch.nn.ReLU], [[64, 256, 1, 1, 0], torch.nn.ReLU]]},
#                                     {'convs_config': [[[256, 64, 3, 1, 1], torch.nn.ReLU], [[64, 256, 1, 1, 0], torch.nn.ReLU]]}]},
#       'encoder_top_config': {
#           'batch_norm': True,
#           'down_convs_config': [[[256, 128, 3, 2, 1], torch.nn.ReLU],  # 256*10*10 -> 128*5*5
#                                 [[128, 256, 3, 1, 1], torch.nn.ReLU]],
#           'residual_convs_config': [{'convs_config': [[[256, 64, 3, 1, 1], torch.nn.ReLU], [[64, 256, 1, 1, 0], torch.nn.ReLU]]},
#                                     {'convs_config': [[[256, 64, 3, 1, 1], torch.nn.ReLU], [[64, 256, 1, 1, 0], torch.nn.ReLU]]}]},
#       'pre_vq_top_config': [256, 32, 1, 1, 0],
#       'pre_vq_bottom_config': [256+32, 32, 1, 1, 0],
#       'vq_top_config': {'n_embeddings': 64, 'embedding_dim': 32},
#       'vq_bottom_config': {'n_embeddings': 32, 'embedding_dim': 32},
#       'decoder_top_config': {'convs_config': [[32, 128, 3, 1, 1]],
#                             #  'residual_convs_config': [],
#                              'transpose_convs_config': [[[128, 32, 4, 2, 1], torch.nn.ReLU]]},
#       'decoder_bottom_config': {'convs_config': [[64, 128, 3, 1, 1]],
#                                 # 'residual_convs_config': [],
#                                 'transpose_convs_config': [[[128, 256, 6, 2, 1], torch.nn.ReLU],
#                                                            [[256, 128, 5, 2, 1], torch.nn.ReLU],
#                                                            [[128, 64, 4, 2, 1], torch.nn.ReLU],
#                                                            [[64, 3, 4, 2, 1], None]]},
#       'upsample_top_config': [32, 32, 4, 2, 1]
#       }).to(self.device)