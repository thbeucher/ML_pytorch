import os
import sys
import torch
import visdom
import random

from tqdm import tqdm

sys.path.append(os.path.abspath(__file__).replace('action_predictor.py', ''))
sys.path.append(os.path.abspath(__file__).replace('USS-GL/learners/action_predictor.py', ''))

import models.gan_vae_divers as gvd

from visual_feature_learner import VFLTrainer, VFL
from utils import load_model, save_model, plot_metric


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
  def __init__(self, config={}):
    super().__init__()
    self.config = {**ActionPredictor.BASE_CONFIG, **config}
    self.predictor = gvd.sequential_constructor(self.config['predictor_config'])
    
  def forward(self, state, next_state):
    return self.predictor(torch.cat([state, next_state], 1))

class APTrainer(object):
  BASE_CONFIG = {'batch_size': 30, 'use_visdom': True, 'memory_size': 7200, 'max_ep_len': 60,
                 'action_predictor_config': {}, 'save_name': 'action_predictor.pt', 'models_folder': 'models/',
                 'save_vfl_name': 'visual_feature_learner.pt', 'n_evaluated_samples': 720}
  def __init__(self, config={}):
    self.config = {**APTrainer.BASE_CONFIG, **config}

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.action_predictor = ActionPredictor(self.config['action_predictor_config'])
    self.ap_optimizer = torch.optim.AdamW(self.action_predictor.parameters())
    self.ce_criterion = torch.nn.CrossEntropyLoss()

    if self.config['use_visdom']:
      self.vis = visdom.Visdom()
    
    self.vfl_learner = VFL()
    load_model(self.vfl_learner, models_folder=self.config['models_folder'], save_name=self.config['save_vfl_name'],
               device=self.device)
  
  @torch.no_grad()
  def fill_ap_memory(self):
    self.memory = VFLTrainer.fill_memory(memory_size=self.config['memory_size'], max_ep_len=self.config['max_ep_len'])

    for i in tqdm(range(0, len(self.memory), self.config['batch_size'])):
      states, next_states, actions, shoulders, elbows, rewards, dones = zip(*self.memory[i:i+self.config['batch_size']])

      states = torch.stack(states, 0).to(self.device)
      next_states = torch.stack(next_states).to(self.device)

      _, _, quantized = self.vfl_learner(states)
      _, _, next_quantized = self.vfl_learner(next_states)

      for j, (s, ns, q, nq, a, sh, e, r, d) in enumerate(zip(states.cpu(), next_states.cpu(), quantized.cpu(),
                                                       next_quantized.cpu(), actions, shoulders, elbows,
                                                       rewards, dones)):
        self.memory[i+j] = [s, ns, q, nq, a, sh, e, r, d]

  @torch.no_grad()
  def evaluation(self):
    preds, targets = [], []
    for i in range(0, self.config['n_evaluated_samples'], self.config['batch_size']):
      _, _, quantized, next_quantized, actions, _, _, _, _ = zip(*self.memory[i:i+self.config['batch_size']])
      quantized = torch.stack(quantized, 0).to(self.device)
      next_quantized = torch.stack(next_quantized, 0).to(self.device)
      actions = torch.LongTensor(actions)

      predicted_action = self.action_predictor(quantized, next_quantized)

      preds += predicted_action.argmax(-1).cpu().tolist()
      targets += actions.tolist()
    # print(f'Accuracy = {sum([1 for p, t in zip(preds, targets) if p == t])/len(preds):.3f}')
    return sum([1 for p, t in zip(preds, targets) if p == t])/len(preds)

  def train(self, n_iterations=3000, plot_progress=True, save_model=True, fill_memory=True, **kwargs):
    if fill_memory:
      print('Fill training memory...')
      self.fill_ap_memory()
    
    print(f'Training Action Predictor for {n_iterations} iterations...')
    with tqdm(total=n_iterations) as t:
      for i in range(n_iterations):
        _, _, quantized, next_quantized, actions, _, _, _, _ = zip(*random.sample(self.memory, self.config['batch_size']))
        quantized = torch.stack(quantized, 0).to(self.device)
        next_quantized = torch.stack(next_quantized, 0).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)

        predicted_action = self.action_predictor(quantized, next_quantized)

        self.ap_optimizer.zero_grad()
        loss = self.ce_criterion(predicted_action, actions)
        loss.backward()
        self.ap_optimizer.step()

        if plot_progress:
          if i % 50 == 0 or i == (n_iterations-1):
            plot_metric(self.vis, loss.item(), i, win='ap_loss', title='action_predictor_loss evolution')

            acc = self.evaluation()
            plot_metric(self.vis, acc, i, win='acc_action_predictor', title='action_predictor accuracy')

        t.set_description(f'Loss={loss.item():.3f}')
        t.update(1)
      
    if save_model:
      self.save_model()
  
  def save_model(self, model_name=None):
    save_model(self.action_predictor, models_folder=self.config['models_folder'],
               save_name=self.config['save_name'] if model_name is None else model_name)
  
  def load_model(self, model_name=None):
    load_model(self.action_predictor, models_folder=self.config['models_folder'],
               save_name=self.config['save_name'] if model_name is None else model_name)