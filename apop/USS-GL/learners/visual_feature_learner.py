import os
import sys
import gym
import torch
import random
import visdom

from tqdm import tqdm
from torchvision.io import read_image
from torchvision.utils import make_grid, save_image

sys.path.append(os.path.abspath(__file__).replace('visual_feature_learner.py', ''))
sys.path.append(os.path.abspath(__file__).replace('USS-GL/learners/visual_feature_learner.py', ''))

import models.gan_vae_divers as gvd

from arm_2dof_gym_env import RobotArmEnvVG
from utils import get_state, load_model, save_model, plot_metric


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


class VFLTrainer(object):
  BASE_CONFIG = {'batch_size': 30, 'use_visdom': True, 'memory_size': 7200, 'max_ep_len': 60, 'vfl_config': {},
                 'save_name': 'visual_feature_learner.pt', 'models_folder': 'models/'}
  def __init__(self, config={}):
    self.config = {**VFLTrainer.BASE_CONFIG, **config}

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.vfl_learner = VFL(self.config['vfl_config'])
    self.vfl_optimizer = torch.optim.AdamW(self.vfl_learner.parameters())
    self.mse_criterion = torch.nn.MSELoss()

    if self.config['use_visdom']:
      self.vis = visdom.Visdom()

  @staticmethod
  def fill_memory(memory_size=7200, max_ep_len=60):
    env = RobotArmEnvVG({'in_background': True})
    env.reset(to_reset='both')

    n_step = 0
    memory = []
    for _ in tqdm(range(memory_size)):
      env.render(mode='background')

      state = get_state(env)
      action = random.randint(0, 4)

      obs, reward, done, _ = env.step(action)
      shoulder, elbow = map(lambda x: round(x), obs[2:])

      env.render(mode='background')
      next_state = get_state(env)

      memory.append([state, next_state, action, shoulder, elbow, reward, done])
      n_step += 1

      # As we don't use this filled memory for policy training (action_selector)
      # we don't need to respect terminal condition
      # if done or n_step == self.config['max_ep_len']:
      if n_step == max_ep_len:
        env.reset(to_reset='both')
        n_step = 0
    
    return memory

  def train(self, n_iterations=3000, plot_progress=True, save=True, **kwargs):
    print('Fill vfl_memory...')
    memory = VFLTrainer.fill_memory(memory_size=self.config['memory_size'], max_ep_len=self.config['max_ep_len'])

    print(f'Train vfl_learner for {n_iterations} iterations...')
    with tqdm(total=n_iterations) as t:
      for i in range(n_iterations):
        states, _, _, _, _, _, _ = zip(*random.sample(memory, self.config['batch_size']))
        states = torch.stack(states, 0).to(self.device)

        states_rec, vq_loss, _ = self.vfl_learner(states)

        self.vfl_optimizer.zero_grad()
        rec_loss = self.mse_criterion(states_rec, states)
        loss = vq_loss + rec_loss
        loss.backward()
        self.vfl_optimizer.step()

        if plot_progress and (i % 50 == 0 or i == (n_iterations-1)):
          self.vis.image(make_grid(states[:6].cpu(), nrow=3), win='state', opts={'title': 'state'})

          # Strangely, it plots black image so an hack is to save the image first then load it to pass it to visdom
          # vis.image(make_grid(state_rec[:6].cpu(), nrow=3), win='rec', opts={'title': 'reconstructed'})
          save_image(make_grid(states_rec[:6].cpu(), nrow=3), 'test_make_grid.png')
          self.vis.image(read_image('test_make_grid.png'), win='rec', opts={'title': 'reconstructed'})

          plot_metric(self.vis, loss.item(), i, win='vfl_loss', title='VFL loss evolution')
        
        t.set_description(f'Loss={loss.item():.3f}')
        t.update(1)
    
    if save:
      self.save_model()
  
  def save_model(self, model_name=None):
    save_model(self.vfl_learner, models_folder=self.config['models_folder'],
               save_name=self.config['save_name'] if model_name is None else model_name)
  
  def load_model(self, model_name=None):
    load_model(self.vfl_learner, models_folder=self.config['models_folder'],
               save_name=self.config['save_name'] if model_name is None else model_name)


if __name__ == '__main__':
  vfl_trainer = VFLTrainer({'save_name': 'test_vfl.pt', 'memory_size': 120})
  vfl_trainer.train(n_iterations=1)
  vfl_trainer.load_model()
  os.remove('models/test_vfl.pt')