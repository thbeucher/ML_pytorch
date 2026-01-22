import os
import sys
import json
import torch
import pygame
import random
import warnings
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import gymnasium as gym

from tqdm import tqdm
from collections import deque
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../../../robot/')

from encoding_exps import CNNAETrainer
from replay_buffer import ReplayBuffer
from cnn_layers import EnhancedResidualFullBlock
from flow_matching_exps import flow_matching_loss, euler_sampling, rk45_sampling, TimeEmbedding

warnings.filterwarnings("ignore")


class WorldModelFlowUnet(nn.Module):
  def __init__(self, img_chan=3, time_dim=64, x_start=False,
               add_action=True, n_actions=5, action_dim=8,
               add_is=False, is_n_values=36, is_dim=32,  # is = internal_state
               add_ds=False, ds_n_values=2, ds_dim=64,  # ds = done_signal
               ):
    super().__init__()
    self.time_emb = nn.Sequential(TimeEmbedding(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU())
    condition_dim = time_dim 

    self.add_action = add_action
    if add_action:
      self.action_emb = nn.Sequential(nn.Embedding(n_actions, action_dim), nn.Linear(action_dim, action_dim), nn.SiLU())
      condition_dim += action_dim

    self.add_is = add_is
    if add_is:
      self.is_emb = nn.Sequential(nn.Embedding(is_n_values, is_dim), nn.Linear(is_dim, is_dim), nn.SiLU())
      condition_dim += 2 * is_dim
    
    self.add_ds = add_ds
    if add_ds:
      self.ds_emb = nn.Sequential(nn.Embedding(ds_n_values, ds_dim), nn.Linear(ds_dim, ds_dim), nn.SiLU())
      condition_dim += ds_dim

    self.x_start = x_start
    start_dim = 2*img_chan if x_start else img_chan
    self.init_conv = nn.Conv2d(start_dim + condition_dim, 64, 3, 1, 1)

    self.down1 = EnhancedResidualFullBlock(64, 128, pooling=True, cond_emb=condition_dim)
    self.down2 = EnhancedResidualFullBlock(128, 256, pooling=True, cond_emb=condition_dim)
    self.down3 = EnhancedResidualFullBlock(256, 512, pooling=True, cond_emb=condition_dim, groups=1)

    self.up1 = EnhancedResidualFullBlock(512, 256, upscaling=True, cond_emb=condition_dim, groups=1)
    self.up2 = EnhancedResidualFullBlock(256, 128, upscaling=True, cond_emb=condition_dim)
    self.up3 = EnhancedResidualFullBlock(128, 64, upscaling=True, cond_emb=condition_dim)
    
    self.final_conv = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.SiLU(),
                                    nn.Conv2d(32, img_chan, 3, 1, 1))
  
  def forward(self, x, t, condition={}):  # image, time, action
    B, C, H, W = x.shape

    t_emb = self.time_emb(t.squeeze(-1))         # -> [B, time_dim]
    all_embed = [t_emb]

    if self.add_action:
      a_emb = self.action_emb(condition['action'].squeeze(-1))  # -> [B, action_dim]
      all_embed += [a_emb]
    
    if self.add_is:
      is_emb = self.is_emb(condition['internal_state'])  # [B, n_is=2] -> [B, n_is=2, is_dim=32]
      all_embed += [is_emb.flatten(1)]  # [B, 2*32]
    
    if self.add_ds:
      ds_emb = self.ds_emb(condition['done_signal'].squeeze(-1))  # [B, ds_dim=64]
      all_embed += [ds_emb]

    all_embed = torch.cat(all_embed, dim=1)  # -> [B, 104]

    all_cond = [all_embed[:, :, None, None].expand(-1, -1, H, W)]
    if self.x_start:
      all_cond = [condition['x_cond']] + all_cond
    
    x = torch.cat([x] + all_cond, dim=1)

    x = self.init_conv(x)                        # -> [B, 64, H, W]

    d1 = self.down1(x, c=all_embed)              # -> [B, 128, H/2, W/2]
    d2 = self.down2(d1, c=all_embed)             # -> [B, 256, H/4, W/4]
    d3 = self.down3(d2, c=all_embed)             # -> [B, 512, H/8, W/8]

    u1 = self.up1(d3, c=all_embed)               # -> [B, 256, H/4, W/4]
    u2 = self.up2(u1 + d2, c=all_embed)          # -> [B, 128, H/2, W/2]
    u3 = self.up3(u2 + d1, c=all_embed)          # -> [B, 64, H, W]

    velocity = self.final_conv(u3)               # -> [B, img_chan, H, W]
    return velocity


class NextStatePredictorTrainer:
  CONFIG = {'image_size':                        256,
            'resize_to':                         32,
            'image_chan':                        3,
            'normalize_image':                   True,
            'time_dim':                          64,
            'internal_state_dim':                2,
            'internal_state_n_values':           (90//5, 180//5),  # max_angle / angle_step
            'action_dim':                        1,
            'action_n_values':                   5,
            'action_emb':                        8,
            'replay_buffer_size':                10_000,
            'world_model_batch_size':            128,
            'learning_rate':                     2e-4,
            'replay_buffer_device':              'cpu',
            'render_mode':                       'rgb_array',
            'seed':                              42,
            'use_tf_logger':                     True,
            'world_model_max_train_steps':       10_000,
            'world_model_check_pred_loss_every': 100,
            'world_model_loss_history_size':     30,  # history size to check if loss still decrease
            'world_model_loss_history_eps':      5e-6,  # if loss fluctuate less than eps, stop training
            'train_on_delta':                    True,
            'start_on_noise_delta':              False,
            'weighted_time_sampling':            True,
            'noise_scale':                       1.0,
            'save_dir':                          'NSP_experiments/',
            'log_dir':                           'runs/',
            'exp_name':                          'nsp_base'}
  def __init__(self, config={}):
    self.config = {**NextStatePredictorTrainer.CONFIG, **config}

    self.device = torch.device('cuda' if torch.cuda.is_available() else
                               'mps' if torch.backends.mps.is_available() else
                               'cpu')
    print(f'Using device: {self.device}')

    if self.config['normalize_image']:
      self.clamp_denorm_fn = lambda x: (x.clamp(-1, 1) + 1) / 2
    self.get_train_params = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)

    # self.set_seed()

    self.env = gym.make("gymnasium_env:RobotArmEnv", render_mode=self.config['render_mode'], cropping=True)
    # self.env_config = self.env.unwrapped.config

    save_dir_run = os.path.join(self.config['save_dir'], self.config['exp_name'], self.config['log_dir'])
    self.tf_logger = SummaryWriter(save_dir_run) if self.config['use_tf_logger'] else None

    self.instanciate_model()
    self.set_training_utils()
    self.dump_config()
  
  def dump_config(self):
    with open(os.path.join(self.config['save_dir'],
                           self.config['exp_name'],
                           f"{self.config['exp_name']}_CONFIG.json"), 'w') as f:
      json.dump(self.config, f)
  
  def set_seed(self):
    # Set seeds for reproducibility
    torch.manual_seed(self.config['seed'])
    np.random.seed(self.config['seed'])
    random.seed(self.config['seed'])
    if self.device.type == 'cuda':
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

  def instanciate_model(self):
    self.world_model = WorldModelFlowUnet(img_chan=self.config['image_chan'],
                                          time_dim=self.config['time_dim'],
                                          n_actions=self.config['action_n_values'],
                                          action_dim=self.config['action_emb'],
                                          x_start=self.config['train_on_delta']).to(self.device)
    print(f'Instanciate Worl_Model (trainable parameters: {self.get_train_params(self.world_model):,})')
  
  def set_training_utils(self):
    resize_img = self.config['resize_to'] != self.config['image_size']
    self.replay_buffer = ReplayBuffer(self.config['internal_state_dim'],
                                      self.config['action_dim'],
                                      self.config['image_size'],
                                      resize_to=self.config['resize_to'] if resize_img else None,
                                      normalize_img=self.config['normalize_image'],
                                      capacity=self.config['replay_buffer_size'],
                                      device=self.config['replay_buffer_device'],
                                      target_device=self.device)
    self.wm_optimizer = torch.optim.AdamW(self.world_model.parameters(), lr=self.config['learning_rate'],
                                          weight_decay=1e-4, betas=(0.9, 0.999))

  def save_model(self, model_to_save, model_name):
    save_dir = os.path.join(self.config['save_dir'], self.config['exp_name'])
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}.pt")
    torch.save({model_name: model_to_save.state_dict()}, path)

  def load_model(self, model_to_load, model_name):
    path = os.path.join(self.config['save_dir'],
                        self.config['exp_name'],
                        f"{model_name}.pt")
    if os.path.isfile(path):
      model = torch.load(path, map_location=self.device)
      model_to_load.load_state_dict(model[model_name])
      print(f'Model loaded successfully from {path}...')
    else:
      print(f'File {path} not found... No loaded model.')

  def fill_memory(self, random_act=True, replay_buffer_size=None, max_episode_steps=50):
    print('Filling memory buffer...')
    obs, _ = self.env.reset()
    img = self.env.render()
    if replay_buffer_size is None:
      replay_buffer_size = self.config['replay_buffer_size']
    episode_step = 0
    for _ in tqdm(range(replay_buffer_size)):
      action = random.randint(0, 4) if random_act else 0  #TODO use policy
      next_obs, reward, terminated, truncated, info = self.env.step(action)
      next_img = self.env.render()
      episode_step += 1
      self.replay_buffer.add(obs//5,action, img, reward, terminated or episode_step > max_episode_steps,
                             next_obs//5, next_img)
      obs, img = next_obs, next_img

      if terminated or episode_step > max_episode_steps:
        obs, _ = self.env.reset()
        img = self.env.render()
        episode_step = 0
  
  @torch.no_grad()
  def autoplay(self):
    pygame.init()  # Initialize pygame for keyboard input
    running = True
    obs, _ = self.env.reset()
    while running:
      img = self.env.render()
      # Capture keyboard events
      for event in pygame.event.get():
        if event.type == pygame.QUIT:  # Close window
          running = False
        elif event.type == pygame.KEYDOWN:
          if event.key == pygame.K_q:  # Quit when 'Q' is pressed
            running = False
      action = random.randint(0, 4)  #TODO use policy
      obs, reward, terminated, truncated, info = self.env.step(action)
      if terminated or truncated:
        self.env.reset(options={'only_target': True})
    self.env.close()
    pygame.quit()
  
  def train_world_model(self):
    self.world_model.train()
    print('Training world model...')
    best_loss = torch.inf
    mean_loss = 0.0
    mean_loss_history = deque([False], maxlen=self.config['world_model_loss_history_size'])
    pbar = tqdm(range(self.config['world_model_max_train_steps']))
    for step in pbar:
      batch = self.replay_buffer.sample(self.config['world_model_batch_size'])
      x0 = batch['image']  # use image as starting distribution instead of gaussian noise
      x1 = batch['next_image']  # target distribution is the image to predict
      condition = {'action': batch['action']}

      if self.config['train_on_delta']:
        x1 = x1 - x0  # x1 = delta
        x1 = x1.clamp(-1, 1)

        if self.config['start_on_noise_delta']:
          x0 = torch.randn_like(x1) * self.config['noise_scale']
        else:
          x0 = torch.zeros_like(x1)

        condition['x_cond'] = batch['image']

      loss = flow_matching_loss(self.world_model, x1, x0=x0, condition=condition,
                                weighted_time_sampling=self.config['weighted_time_sampling'],
                                noise_scale=self.config['noise_scale'])

      self.wm_optimizer.zero_grad()
      loss.backward()
      self.wm_optimizer.step()

      mean_loss += (loss.item() - mean_loss) / (step + 1)

      mean_loss_history.append(abs(mean_loss - best_loss) < self.config['world_model_loss_history_eps'])

      if mean_loss < best_loss:
        self.save_model(self.world_model, 'world_model')
        best_loss = mean_loss

      if self.tf_logger:
        self.tf_logger.add_scalar('worlmodel_fm_loss', mean_loss, step)
        
        if step % self.config['world_model_check_pred_loss_every'] == 0:
          x1_pred = rk45_sampling(
            self.world_model,
            device=self.device,
            x=x0,
            # n_samples=x.shape[0],
            condition=condition,
            n_steps=10
          )

          x1_pred = x1_pred[-1]
          if self.config['train_on_delta']:
            x1_pred = batch['image'] + x1_pred
          x1_pred = x1_pred.clamp(-1, 1)

          self.tf_logger.add_scalar('world_model_pred_loss',
                                    torch.nn.functional.mse_loss(x1_pred, x1),
                                    step // self.config['world_model_check_pred_loss_every'])
          
          imagined_traj = [batch['image'][:1]]
          for action in torch.as_tensor([[[1]]]*12 + [[[3]]]*11, device=self.device, dtype=torch.long):
            if self.config['train_on_delta']:
              if self.config['start_on_noise_delta']:
                x = torch.randn_like(imagined_traj[-1]) * self.config['noise_scale']
              else:
                x = torch.zeros_like(imagined_traj[-1])
            else:
              x = imagined_traj[-1]

            condition = {'action': action}
            if self.config['train_on_delta']:
              condition['x_cond'] = imagined_traj[-1]

            x1_pred = rk45_sampling(self.world_model, device=self.device, x=x,
                                    n_samples=imagined_traj[-1].shape[0],
                                    condition=condition, n_steps=10)

            x1_pred = x1_pred[-1]
            if self.config['train_on_delta']:
              x1_pred = imagined_traj[-1] + x1_pred
            imagined_traj.append(x1_pred.clamp(-1, 1))

          self.tf_logger.add_images('generated_world_model_prediction',
                                    torch.cat(imagined_traj, dim=0),
                                    global_step=step)

      pbar.set_postfix(loss=f'{mean_loss:.6f}')

      # if all(mean_loss_history):
      #   break
    
    if self.tf_logger:
      x1_pred = rk45_sampling(self.world_model, device=self.device, x=x0[:8], condition=condition[:8], n_steps=4)
      x0, x1_pred, x1 = [self.clamp_denorm_fn(x) for x in [x0[:8], x1_pred[-1], x1[:8]]]
      img_pred_nextimg = torch.cat([x0, x1_pred, x1], dim=0)
      self.tf_logger.add_images('generated_world_model_prediction', img_pred_nextimg)
  
  def train(self):
    self.fill_memory()
    self.train_world_model()

  @torch.no_grad()
  def evaluate(self):
    pass


class SimpleImageDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_size=10_000, resize_to=32):
    super().__init__()
    print('Filling SimpleImageDataset...')
    transform = [torchvision.transforms.ToTensor(), torchvision.transforms.Resize(resize_to)]
    self.transform = torchvision.transforms.Compose(transform)
    env = gym.make("gymnasium_env:RobotArmEnv", render_mode='rgb_array', cropping=True)
    obs, _ = env.reset()
    img = env.render()
    self.images, self.actions = [], []
    for _ in tqdm(range(dataset_size)):
      action = random.randint(0, 4)
      self.images.append(self.transform(img))
      self.actions.append(action)
      next_obs, reward, terminated, truncated, info = env.step(action)
      img = env.render()
    env.close()
  
  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
    return self.images[index], self.actions[index]


class SimuCNNAETrainer(CNNAETrainer):
  CONFIG = {'dataset_size': 1_000,
            'exp_name':     'simu_ae_cnn',
            'save_dir':     'NSP_experiments/',
            'image_size':   32}
  def __init__(self, config={}):
    super().__init__(SimuCNNAETrainer.CONFIG)
    self.config = {**self.config, **config}

    train_dataset = SimpleImageDataset(self.config['dataset_size'])
    test_dataset = SimpleImageDataset(self.config['dataset_size']//10)
    self.set_dataloader(train_dataset=train_dataset, test_dataset=test_dataset, num_workers=0)


def get_args():
  parser = argparse.ArgumentParser(description='Next state predictor experiments')
  parser.add_argument('--trainer', '-t', type=str, default='nsp')
  parser.add_argument('--load_model', '-lm', action='store_true')
  parser.add_argument('--train_model', '-tm', action='store_true')
  parser.add_argument('--eval_model', '-em', action='store_true')
  parser.add_argument('--save_model', '-sm', action='store_true')
  parser.add_argument('--play_model', '-pm', action='store_true')
  parser.add_argument('--force_human_view', '-fhv', action='store_true')
  parser.add_argument('--experiment_name', '-en', type=str, default=None)
  return parser.parse_args()


if __name__ == '__main__':
  # env.unwrapped.get_screen(crop=crop)
  # plt.imshow(np_to_tensor(image).permute(1, 2, 0));pkt.tight_layout();plt.show()
  trainers = {'nsp': NextStatePredictorTrainer, 'simu_ae': SimuCNNAETrainer}
  args = get_args()

  config = {} if args.experiment_name is None else {'exp_name': args.experiment_name}
  config['render_mode'] = 'human' if args.play_model or args.force_human_view else 'rgb_array'

  print(f'Trainer: {args.trainer}')
  trainer = trainers[args.trainer](config)

  if args.load_model:
    trainer.load_model(trainer.world_model, 'world_model')

  if args.play_model:
    print('Start autoplay...')
    trainer.autoplay()
  
  if args.train_model:
    print('Start training...')
    trainer.train()
  
  # IDEA:
  # conv -> channel-wise attention reduction -> batch (sequence) treat as channel -> channel-wise attention
  #       I                  Z              SE(Z) = A            T              SE(T) = G
  # [B, 3, 32, 32]  -> [B, 64, 16, 16] -> [B, 1, 16, 16] -> [1, B, 16, 16] -> [1, 1, 16, 16]       1
  # [B, 64, 16, 16] -> [B, 128, 8, 8]  -> [B, 1, 8, 8]   -> [1, B, 8, 8]   -> [1, 1, 8, 8]         2
  # [B, 128, 8, 8]  -> [B, 256, 4, 4]  -> [B, 1, 4, 4]   -> [1, B, 4, 4]   -> [1, 1, 4, 4]         3
  # [B, 256, 4, 4]  -> [B, 512, 2, 2]  -> [B, 1, 2, 2]   -> [1, B, 2, 2]   -> [1, 1, 2, 2]         4
  # I -> I1 -> Z1 -> A1 | T1 -> G1 
  #
  # [1, 512, 2, 2]       || G4 -> [1, 256, 4, 4]
  # [1, 256, 4, 4]  + Z3 || G3 -> [1, 128, 8, 8]
  # [1, 128, 8, 8]  + Z2 || G2 -> [1, 64, 16, 16]
  # [1, 64, 16, 16] + Z1 || G1 -> [1, 3, 32, 32]