import os
import sys
import torch
import random
import visdom

from tqdm import tqdm
from torchvision.io import read_image
from torchvision.utils import make_grid, save_image
from torchvision.transforms import RandomRotation, GaussianBlur

sys.path.append(os.path.abspath(__file__).replace('visual_feature_learner.py', ''))
sys.path.append(os.path.abspath(__file__).replace('USS-GL/learners/visual_feature_learner.py', ''))

import models.gan_vae_divers as gvd

from arm_2dof_gym_env import RobotArmEnvVG
from utils import get_state, load_model, save_model, plot_metric, msg_if_not_exist


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


class CustomDataset(torch.utils.data.Dataset):
  BASE_CONFIG = {'memory_size': 7200, 'reset_interval': 60, 'mask_size': 45}
  def __init__(self, config={}):
    super().__init__()
    self.config = {**CustomDataset.BASE_CONFIG, **config}

    self.memory = []
    self._fill_memory(memory_size=self.config['memory_size'], reset_interval=self.config['reset_interval'])

    self.rotater = RandomRotation(360)
    self.masks = self._create_masks()
    self.blurer = GaussianBlur(5, 3)  # kernel, sigma

    # 50% : doing nothing | 20% : random masking | 20% : random rotation | 5% : gaussian bluring | 5% : gaussian noise
    self.weights = [50, 20, 20, 5, 5]
    self.augments = ['nothing', 'masking', 'rotation', 'bluring', 'noise']
  
  def _create_masks(self):
    black_mask = torch.zeros(3, self.config['mask_size'], self.config['mask_size'])
    white_mask = torch.ones(3, self.config['mask_size'], self.config['mask_size'])
    red_mask = torch.zeros(3, self.config['mask_size'], self.config['mask_size'])
    red_mask[0] = 1
    green_mask = torch.zeros(3, self.config['mask_size'], self.config['mask_size'])
    green_mask[1] = 1
    blue_mask = torch.zeros(3, self.config['mask_size'], self.config['mask_size'])
    blue_mask[2] = 1
    return [black_mask, white_mask, red_mask, green_mask, blue_mask]
  
  def _fill_memory(self, memory_size=7200, reset_interval=60):
    env = RobotArmEnvVG({'in_background': True})

    env.reset(to_reset='both')
    env.render(mode='background')

    for i in tqdm(range(1, memory_size + 1)):
      state = get_state(env)
      action = random.randint(0, 4)

      obs, _, done, _ = env.step(action)
      shoulder, elbow = map(lambda x: round(x), obs[2:])

      env.render(mode='background')
      next_state = get_state(env)

      self.memory.append([state, action, shoulder, elbow, next_state, done])

      if i % reset_interval == 0:
        env.reset(to_reset='both')
        env.render(mode='background')

  def __len__(self):
    return len(self.memory)
  
  def __getitem__(self, idx):
    item = self.memory[idx]
    new_state = item[0].clone()

    augment = random.choices(self.augments, weights=self.weights)[0]
    if augment == 'masking':
      mask = random.choice(self.masks)
      start_x = random.randint(0, 180-self.config['mask_size'])
      start_y = random.randint(0, 180-self.config['mask_size'])
      new_state[:, start_x:start_x+self.config['mask_size'], start_y:start_y+self.config['mask_size']] = mask
    elif augment == 'rotation':
      new_state = self.rotater(new_state)
    elif augment == 'bluring':
      new_state = self.blurer(new_state)
    elif augment == 'noise':
      gaussian_noise = torch.randn(new_state.shape) * 0.01
      new_state = new_state + gaussian_noise

    return [new_state] + item[1:]


def custom_collate_fn(batch):
  states, actions, shoulders, elbows, next_states, dones = zip(*batch)
  states = torch.stack(states, 0)
  actions = torch.LongTensor(actions)
  shoulders = torch.LongTensor(shoulders)
  elbows = torch.LongTensor(elbows)
  next_states = torch.stack(next_states, 0)
  dones = torch.FloatTensor(dones)
  return states, actions, shoulders, elbows, next_states, dones


class VFLRModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 4, 2, 1), torch.nn.ReLU(),  # [B, 3, 180, 180] -> [B, 32, 90, 90]
                                       torch.nn.Conv2d(32, 64, 4, 2, 1), torch.nn.ReLU(),  # [B, 32, 90, 90] -> [B, 64, 45, 45]
                                       torch.nn.Conv2d(64, 128, 4, 2, 1), torch.nn.ReLU(),  # [B, 64, 45, 45] -> [B, 128, 22, 22]
                                       torch.nn.Conv2d(128, 256, 4, 2, 1), torch.nn.ReLU(),  # [B, 128, 22, 22] -> [B, 256, 11, 11]
                                       torch.nn.Conv2d(256, 512, 4, 2, 1), torch.nn.ReLU(),  # [B, 256, 11, 11] -> [B, 512, 5, 5]
                                       torch.nn.Conv2d(512, 1024, 5, 1, 0), torch.nn.ReLU(),  # [B, 512, 5, 5] -> [B, 1024, 1, 1]
                                       torch.nn.Flatten())
    self.action_embedder = torch.nn.Embedding(num_embeddings=5, embedding_dim=5)
    self.shoulder_embedder = torch.nn.Embedding(num_embeddings=91, embedding_dim=8)
    self.elbow_embedder = torch.nn.Embedding(num_embeddings=181, embedding_dim=8)
    self.proj = torch.nn.Sequential(torch.nn.Linear(1024 + 5 + 8 + 8, 2048), torch.nn.ReLU(), torch.nn.Linear(2048, 512))
  
  def forward(self, state, action, shoulder, elbow, mask_shoulder=False, mask_elbow=False):
    state_emb = self.encoder(state)
    action_emb = self.action_embedder(action)  # [B] -> [B, 5]
    shoulder_emb = torch.zeros(len(shoulder), 8, device=shoulder.device) if mask_shoulder else self.shoulder_embedder(shoulder)
    elbow_emb = torch.zeros(len(elbow), 8, device=elbow.device) if mask_elbow else self.elbow_embedder(elbow)  # [B] -> [B, 8]
    return self.proj(torch.cat([state_emb, action_emb, shoulder_emb, elbow_emb], 1))

  def save_model(self, save_path='models/vflr.pt'):
    torch.save(self.state_dict(), save_path)

  def load_model(self, save_path='models/vflr.pt', device=None):
    if msg_if_not_exist(save_path):
      self.load_state_dict(torch.load(save_path, map_location=device))


class VFLRecTrainer(object):
  BASE_CONFIG = {'model_folder': 'models/', 'n_epochs': 200, 'batch_size': 32, 'n_workers': 0,
                 'use_visdom': True, 'lr': 1e-4}
  def __init__(self, config={}):
    self.config = {**VFLRecTrainer.BASE_CONFIG, **config}
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.instanciate_models()

    self.optimizer = torch.optim.AdamW(list(self.vflr.parameters()) + list(self.rec_net.parameters()) +\
                                       list(self.rec_shoulder.parameters()) + list(self.rec_elbow.parameters()),
                                       lr=self.config['lr'])
    self.mse_criterion = torch.nn.MSELoss()
    self.ce_criterion = torch.nn.CrossEntropyLoss()

    if self.config['use_visdom']:
      self.vis = visdom.Visdom()

    if not os.path.isdir(self.config['model_folder']):
      os.makedirs(self.config['model_folder'])
    
    rep = input('Load models if exist? (y or n): ')
    if rep == 'y':
      self.load_model()
  
  def instanciate_models(self):
    self.vflr = VFLRModel().to(self.device)
    self.rec_net = torch.nn.Sequential(torch.nn.ConvTranspose2d(512, 512, 5, 1, 0), torch.nn.ReLU(),  # -> [B, 512, 5, 5]
                                       torch.nn.ConvTranspose2d(512, 256, 5, 2, 1), torch.nn.ReLU(),  # -> [B, 256, 11, 11]
                                       torch.nn.ConvTranspose2d(256, 128, 4, 2, 1), torch.nn.ReLU(),  # -> [B, 128, 22, 22]
                                       torch.nn.ConvTranspose2d(128, 64, 5, 2, 1), torch.nn.ReLU(),  # -> [B, 64, 45, 45]
                                       torch.nn.ConvTranspose2d(64, 32, 4, 2, 1), torch.nn.ReLU(),  # -> [B, 32, 90, 90]
                                       torch.nn.ConvTranspose2d(32, 3, 4, 2, 1)).to(self.device)  # -> [B, 3, 180, 180]
    self.rec_shoulder = torch.nn.Linear(512, 90).to(self.device)
    self.rec_elbow = torch.nn.Linear(512, 180).to(self.device)

  def train(self, plot_progress=True):
    dataset = CustomDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config['batch_size'], collate_fn=custom_collate_fn,
                                             num_workers=self.config['n_workers'], shuffle=True)
    
    with tqdm(total=self.config['n_epochs']) as t:
      for i in tqdm(range(self.config['n_epochs'])):
        for states, actions, shoulders, elbows, next_states, dones in tqdm(dataloader, leave=False):
          states, actions, shoulders = states.to(self.device), actions.to(self.device), shoulders.to(self.device)
          elbows, next_states = elbows.to(self.device), next_states.to(self.device)

          mask_shoulder = random.choice([True, False])
          mask_elbow = random.choice([True, False])

          emb = self.vflr(states, actions, shoulders, elbows, mask_shoulder=mask_shoulder, mask_elbow=mask_elbow)
          rec = self.rec_net(emb.unsqueeze(-1).unsqueeze(-1))

          rec_shoulder = self.rec_shoulder(emb)
          rec_elbow = self.rec_elbow(emb)

          self.optimizer.zero_grad()
          loss_state = self.mse_criterion(rec, next_states)
          loss_shoulder = self.ce_criterion(rec_shoulder, shoulders)
          loss_elbow = self.ce_criterion(rec_elbow, elbows)
          loss = loss_state + loss_shoulder + loss_elbow
          loss.backward()
          self.optimizer.step()

          # generator that try to create goal state

        if plot_progress:
          self.vis.image(make_grid(states[:3].cpu(), nrow=3), win='state', opts={'title': 'state'})
          self.vis.image(make_grid(next_states[:3].cpu(), nrow=3), win='next_state', opts={'title': 'next_state'})
          save_image(make_grid(rec[:3].cpu(), nrow=3), 'test_make_grid.png')
          self.vis.image(read_image('test_make_grid.png'), win='rec', opts={'title': 'reconstructed'})

          plot_metric(self.vis, loss.item(), i, win='vflr_loss', title='VFLR loss evolution')
          plot_metric(self.vis, loss_state.item(), i, win='vflr_loss_state', title='VFLR loss_state evolution')

          acc_shoulder = (rec_shoulder.argmax(-1) == shoulders).sum() / len(shoulders)
          plot_metric(self.vis, acc_shoulder.item(), i, win='vflr_acc_shoulder', title='VFLR acc_shoulder evolution')

          acc_elbow = (rec_elbow.argmax(-1) == elbows).sum() / len(elbows)
          plot_metric(self.vis, acc_elbow.item(), i, win='vflr_acc_elbow', title='VFLR acc_elbow evolution')

        t.set_description(f'Loss={loss.item():.3f}')
        t.update(1)
    
    self.save_model()
  
  def load_model(self, rec_net='models/rec_net.pt', rec_shoulder='models/rec_shoulder.pt', rec_elbow='models/rec_elbow.pt'):
    self.vflr.load_model()
    if msg_if_not_exist(rec_net):
      self.rec_net.load_state_dict(torch.load(rec_net, map_location=self.device))
    if msg_if_not_exist(rec_shoulder):
      self.rec_shoulder.load_state_dict(torch.load(rec_shoulder, map_location=self.device))
    if msg_if_not_exist(rec_elbow):
      self.rec_elbow.load_state_dict(torch.load(rec_elbow, map_location=self.device))

  def save_model(self, rec_net='models/rec_net.pt', rec_shoulder='models/rec_shoulder.pt', rec_elbow='models/rec_elbow.pt'):
    self.vflr.save_model()
    torch.save(self.rec_net.state_dict(), rec_net)
    torch.save(self.rec_shoulder.state_dict(), rec_shoulder)
    torch.save(self.rec_elbow.state_dict(), rec_elbow)



if __name__ == '__main__':
  # vfl_trainer = VFLTrainer({'save_name': 'test_vfl.pt', 'memory_size': 120})
  # vfl_trainer.train(n_iterations=1)
  # vfl_trainer.load_model()
  # os.remove('models/test_vfl.pt')

  vflr_trainer = VFLRecTrainer()
  vflr_trainer.train()
