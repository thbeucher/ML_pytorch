import os
import sys
import copy
import json
import torch
import random
import warnings
import numpy as np
import torch.nn as nn
import gymnasium as gym

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from torch.utils.tensorboard import SummaryWriter

from vision_transformer.vit import *
from replay_buffer import ReplayBuffer, ReplayBufferDataset

sys.path.append('../../../robot/')
warnings.filterwarnings("ignore")


class AlteredPredictor(nn.Module):
    def __init__(
        self,
        *,
        image_size,   # int or tuple (H, W): spatial resolution of the input image
        patch_size,   # int or tuple (Ph, Pw): spatial size of each image patch
        dim,          # int: embedding dimension of each patch/token
        depth,        # int: number of Transformer encoder blocks
        heads,        # int: number of attention heads per Transformer block
        mlp_dim,      # int: hidden dimension of the feed-forward (MLP) layer
        channels=3,   # int: number of input image channels (3 for RGB)
        dim_head=64,  # int: dimension of each attention head
        dropout=0.,   # float: dropout rate inside attention and MLP layers
        emb_dropout=0.,# float: dropout rate applied to patch embeddings
        n_actions=5,
        action_dim=8,
      ):
        super().__init__()
        self.channels = channels
        self.image_height, self.image_width = pair(image_size)
        self.patch_height, self.patch_width = pair(patch_size)

        self.patch_dim = channels * self.patch_height * self.patch_width

        self.grid = [(self.image_height // self.patch_height), (self.image_width // self.patch_width)]
        self.n_patchs = self.grid[0] * self.grid[1]

        self.action_emb = nn.Sequential(nn.Embedding(n_actions, action_dim), nn.Linear(action_dim, dim), nn.SiLU())
        
        self.patchify = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width)
        self.to_patch_embedding = nn.Sequential(#nn.LayerNorm(self.patch_dim),
                                                nn.Linear(self.patch_dim, dim),
                                                nn.LayerNorm(dim))

        self.pos_embedding = posemb_sincos_2d(
          h = self.image_height // self.patch_height,
          w = self.image_width // self.patch_width,
          dim = dim,
        ) 
    
        self.dropout = nn.Dropout(emb_dropout)
    
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.find_changed_patch = nn.Sequential(nn.Linear(dim, 2*dim),
                                                nn.ReLU(True),
                                                nn.Linear(2*dim, 1),
                                                nn.Sigmoid())

        self.unpatchify = Rearrange(
          'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
          h=self.image_height // self.patch_height,
          w=self.image_width // self.patch_width,
          p1=self.patch_height,
          p2=self.patch_width,
          c=channels
        )

    def forward(self, patch, action):  # [B, n_patchs=64, N=48], [B, 1]
        patch = self.to_patch_embedding(patch)  # -> [B, n_patchs, dim]
    
        pos_emb = self.pos_embedding.to(patch.device, dtype=patch.dtype)
        patch = self.dropout(patch + pos_emb)

        action_emb = self.action_emb(action)  # [B, 1, dim]
        patch = torch.cat([patch, action_emb], dim=1)  # [B, n_patchs+1, dim]
    
        patch = self.transformer(patch)

        patchs_wo_action = patch[:, :-1]  # remove action token [B, n_patchs, dim]
        preds = self.find_changed_patch(patchs_wo_action)  # -> [B, n_patchs, 1]
    
        return preds


class AlterationPredictor(nn.Module):
  def __init__(self, patch_dim=48, dim=64, depth=4, heads=8, dim_head=32, mlp_dim=128, dropout=0.0,
               n_actions=5, action_dim=8):
    super().__init__()
    self.patch_embedder = nn.Sequential(nn.LayerNorm(patch_dim),
                                        nn.Linear(patch_dim, dim),
                                        nn.LayerNorm(dim))
    self.action_embedder = nn.Sequential(nn.Embedding(n_actions, action_dim), nn.Linear(action_dim, dim), nn.SiLU())
    self.pos_embedding = posemb_sincos_2d(h=8, w=8, dim=dim) 
    self.main = Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)
    self.to_patch_pixels = nn.Sequential(nn.Linear(dim, patch_dim), nn.Tanh())
  
  def forward(self, patch, pos_emb, action):  # [B, M<=n_patchs, N=48], [B, M, dim], [B, 1]
    patch_emb = self.patch_embedder(patch) + pos_emb
    action_emb = self.action_embedder(action)
    patch = torch.cat([patch_emb, action_emb], dim=1)  # [B, M+1, dim]

    patch = self.main(patch)

    patch_wo_action = patch[:, :-1]  # remove action token [B, M, dim]
    patch = self.to_patch_pixels(patch_wo_action)
    # [B, M, N=channels*patch_height*patch_width=3*4*4=48]
    return patch


class NSPTrainer:  # NextStatePredictor
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
            'replay_buffer_device':              'cpu',
            'render_mode':                       'rgb_array',
            'n_epochs':                          500,
            'batch_size':                        128,
            'use_tf_logger':                     True,
            'save_dir':                          'NSP_experiments/',
            'log_dir':                           'runs/',
            'exp_name':                          'nsp_twosteps_big2',
            'seed':                              42,
            }
  def __init__(self, config={}):
    self.config = {**NSPTrainer.CONFIG, **config}
    self.device = torch.device('cuda' if torch.cuda.is_available() else
                               'mps' if torch.backends.mps.is_available() else
                               'cpu')
    print(f'Using device: {self.device}')

    if self.config['normalize_image']:
      self.clamp_denorm_fn = lambda x: (x.clamp(-1, 1) + 1) / 2
    self.get_train_params = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)

    self.set_seed()

    self.env = gym.make("gymnasium_env:RobotArmEnv", render_mode=self.config['render_mode'], cropping=True)

    save_dir_run = os.path.join(self.config['save_dir'], self.config['exp_name'], self.config['log_dir'])
    self.tf_logger = SummaryWriter(save_dir_run) if self.config['use_tf_logger'] else None

    self.instanciate_model()
    self.set_trainer_utils()
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

  def instanciate_model(self):
    self.altered_predictor = AlteredPredictor(image_size=32,
                                              patch_size=4,
                                              dim=64,
                                              depth=2,
                                              heads=4,
                                              mlp_dim=128,
                                              dim_head=32,
                                              channels=3).to(self.device)
    print(f'Instanciate altered_predictor with {self.get_train_params(self.altered_predictor):,} params')
    self.alteration_predictor = AlterationPredictor(patch_dim=self.altered_predictor.patch_dim,
                                                    dim=256,
                                                    depth=8,
                                                    heads=12,
                                                    dim_head=64,
                                                    mlp_dim=1024,
                                                    dropout=0.0).to(self.device)
    print(f'Instanciate altered_predictor with {self.get_train_params(self.alteration_predictor):,} params')
  
  def set_trainer_utils(self):
    resize_img = True if self.config['resize_to'] != self.config['image_size'] else False
    self.replay_buffer = ReplayBuffer(self.config['internal_state_dim'],
                                      self.config['action_dim'],
                                      self.config['image_size'],
                                      resize_to=self.config['resize_to'] if resize_img else None,
                                      normalize_img=self.config['normalize_image'],
                                      capacity=self.config['replay_buffer_size'],
                                      device=self.config['replay_buffer_device'],
                                      target_device=self.device)
    self.altered_opt = torch.optim.AdamW(self.altered_predictor.parameters(), lr=1e-4,
                                         weight_decay=1e-4, betas=(0.9, 0.999))
    self.alteration_opt = torch.optim.AdamW(self.alteration_predictor.parameters(), lr=1e-4,
                                            weight_decay=1e-4, betas=(0.9, 0.999))
    self.bce_criterion = nn.BCELoss()
    self.mse_criterion = nn.MSELoss()
    
  def fill_memory(self, random_act=True):
    print('Filling memory buffer...')
    obs, _ = self.env.reset()
    img = self.env.render()
    for _ in tqdm(range(self.config['replay_buffer_size'])):
      action = random.randint(0, 4) if random_act else 0  #TODO use policy
      next_obs, reward, terminated, truncated, info = self.env.step(action)
      next_img = self.env.render()
      self.replay_buffer.add(obs//5, action, img, reward, terminated, next_obs//5, next_img)
      obs, img = next_obs, next_img
    
  def get_patch(self, patch, patch_mask):
    """
    patch: [B, Np, N]
    patch_mask: [B, Np] (bool)

    returns:
        patch_selected: [B, M, N]
        counts: [B]  (number of selected patches per batch)
    """
    B, Np, N = patch.shape

    counts = patch_mask.sum(dim=1)                  # [B]
    M = counts.max()

    # indices 0..(count-1) per batch
    idx = (torch.cumsum(patch_mask, dim=1) - 1).clamp(min=0)  # [B, Np]

    patch_selected = torch.zeros((B, M, N), device=patch.device, dtype=patch.dtype)

    patch_selected.scatter_(
      dim=1,
      index=idx.unsqueeze(-1).expand(-1, -1, N),
      src=patch * patch_mask.unsqueeze(-1)
    )

    return patch_selected, counts

  def get_next_image(self, patch, patch_predicted, patch_mask):
    """
    patch:           [B, Np, N]      original patches
    patch_predicted: [B, M, N]       predicted patches (only for selected ones)
    patch_mask:      [B, Np] (bool)  which patches were selected

    returns:
        patch_next:  [B, Np, N]
    """
    B, Np, N = patch.shape

    # Start from original patches
    patch_next = patch.clone()

    # Compute index of each selected patch (same logic as get_patch)
    idx = (torch.cumsum(patch_mask, dim=1) - 1)  # [B, Np]
    idx = idx.clamp(min=0)

    # Mask out invalid positions
    valid = patch_mask.unsqueeze(-1)  # [B, Np, 1]

    # Gather predicted patches corresponding to each masked position
    predicted_full = torch.gather(
        patch_predicted,
        dim=1,
        index=idx.unsqueeze(-1).expand(-1, -1, N)
    )

    # Replace only masked patches
    patch_next = torch.where(valid, predicted_full, patch_next)

    return patch_next

  def add_highlight(self, mask, img):
    # --- Convert patch mask → pixel mask --- #
    # M: [B, 64, 1] → [B, 64]
    patch_mask = mask.squeeze(-1).bool()
    # reshape to patch grid
    grid_height, grid_width = self.altered_predictor.grid
    patch_mask = patch_mask.view(-1, grid_height, grid_width)  # [B, 8, 8]
    # expand each patch to pixels
    pixel_mask = patch_mask.repeat_interleave(self.alter_predictor.patch_size, dim=1)\
                            .repeat_interleave(self.alter_predictor.patch_size, dim=2)
    # pixel_mask: [B, 32, 32]

    # --- Create green overlay --- #
    img_vis = img.clone()
    green = torch.zeros_like(img_vis)
    green[:, 1, :, :] = 1.0   # pure green channel

    # --- Blend image + green filter --- #
    alpha = 0.5  # strength of highlight
    mask = pixel_mask.unsqueeze(1)  # [B, 1, 32, 32]
    img_vis = torch.where(
      mask,
      (1 - alpha) * img_vis + alpha * green,
      img_vis
    )
    return img_vis
  
  def train_alter_predictor(self):
    self.altered_predictor.train()
    self.alteration_predictor.train()

    # dataset = ReplayBufferDataset(self.replay_buffer)
    # self.dataloader = DataLoader(dataset, batch_size=self.config['batch_size'],
    #                              shuffle=True, num_workers=4, pin_memory=False)

    mean_accuracy = 0.0
    mean_altered_loss = 0.0
    mean_alteration_loss = 0.0
    mean_n_patch_to_modify = 0.0
    best_loss = torch.inf

    pbar = tqdm(range(self.config['n_epochs']))
    for epoch in pbar:
      for i in tqdm(range(len(self.replay_buffer)), leave=False):
        batch = self.replay_buffer.sample(self.config['batch_size'])

        patch = self.altered_predictor.patchify(batch['image'])  # [B, n_patchs=64, N=48]
        patch_next_image = self.altered_predictor.patchify(batch['next_image'])  # [B, n_patchs, N]

        ###################################
        # --- TRAIN altered predictor --- #
        ###################################
        # Predicting patchs that will effectively change
        patch_change_pred = self.altered_predictor(patch, batch['action'])  # [B, n_patchs, 1]
        patch_change_pred_mask = (patch_change_pred > 0.5)  # [B, n_patchs, 1]

        # Construct changed patch target
        patch_diff = patch != patch_next_image # [B, n_patchs, N]
        altered_patch_target = patch_diff.sum(dim=2, keepdim=True) > 0  # [B, n_patchs, 1]

        altered_loss = self.bce_criterion(patch_change_pred, altered_patch_target)

        self.altered_opt.zero_grad()
        altered_loss.backward()
        self.altered_opt.step()

        ######################################
        # --- TRAIN alteration predictor --- #
        ######################################
        patch_change_gt_mask = altered_patch_target.squeeze(-1)
        # Get patchs to predict based on selected patchs by the altered network
        patch_to_pred, counts = self.get_patch(patch, patch_change_gt_mask)  # [B, M, N], M
        pos_emb = self.alteration_predictor.pos_embedding.to(patch.device, dtype=patch.dtype)  # [n_patchs, dim]
        patch_to_pred_pos_emb, _ = self.get_patch(pos_emb.unsqueeze(0).repeat(patch.shape[0], 1, 1),
                                                  patch_change_gt_mask)
        # -> [B, M, dim]

        if counts.max() > 0:  # if there is some patch to change
          patch_predicted = self.alteration_predictor(patch_to_pred.detach(),
                                                      patch_to_pred_pos_emb.detach(),
                                                      batch['action'].detach())
          patch_next_image_predicted = self.get_next_image(patch,
                                                           patch_predicted,
                                                           patch_change_gt_mask)
          # -> [B, n_patchs, N]

          # Get patch to reconstruct
          patch_target, _ = self.get_patch(patch_next_image, patch_change_gt_mask)

          alteration_loss = self.mse_criterion(patch_predicted, patch_target)

          self.alteration_opt.zero_grad()
          alteration_loss.backward()
          self.alteration_opt.step()
        else:
          patch_next_image_predicted = patch
          alteration_loss = None

        # --- Compute metrics --- #
        altered_accuracy = (patch_change_pred_mask == altered_patch_target).float().mean()

        mean_altered_loss += (altered_loss.item() - mean_altered_loss) / (i + 1)
        if alteration_loss:
          mean_alteration_loss += (alteration_loss.item() - mean_alteration_loss) / (i + 1)
        mean_accuracy += (altered_accuracy.item() - mean_accuracy) / (i + 1)
        if counts.max() > 0:
          mean_n_patch_to_modify += (counts.float().mean().item() - mean_n_patch_to_modify) / (i + 1)

      # --- Log metrics --- #
      self.tf_logger.add_scalar('altered_loss', mean_altered_loss, epoch)
      self.tf_logger.add_scalar('alteration_loss', mean_alteration_loss, epoch)
      self.tf_logger.add_scalar('altered_accuracy', mean_accuracy, epoch)
      self.tf_logger.add_scalar('mean_n_patch_to_modify', int(mean_n_patch_to_modify), epoch)
      pbar.set_postfix(mean_accuracy=f'{mean_accuracy:.3f}')

      if mean_alteration_loss < best_loss:
        self.save_model(self.altered_predictor, 'altered_predictor')
        self.save_model(self.alteration_predictor, 'alteration_predictor')
        best_loss = mean_alteration_loss

      # --- Creates imaginary trajectory --- #
      self.altered_predictor.eval()
      self.alteration_predictor.eval()
      with torch.no_grad():
        imagined_traj = [patch[:1]]
        for action in torch.as_tensor([[[1]]]*12 + [[[3]]]*11, device=self.device, dtype=torch.long):
          # get patchs that are expected to change
          patch_change_pred = self.altered_predictor(imagined_traj[-1], action)
          # get mask to select only predicted changed patchs
          patch_change_pred_mask = (patch_change_pred > 0.5).squeeze(-1)
          # select only predicted changed patchs
          patch_to_pred, counts = self.get_patch(imagined_traj[-1], patch_change_pred_mask)
          pos_emb = self.alteration_predictor.pos_embedding.to(patch.device, dtype=patch.dtype)
          patch_to_pred_pos_emb, _ = self.get_patch(pos_emb.unsqueeze(0), patch_change_pred_mask)
          # make prediction on next patchs if there is some to predict
          if counts.max() > 0:  # if there is some patch to change
            patch_predicted = self.alteration_predictor(patch_to_pred,
                                                        patch_to_pred_pos_emb,
                                                        action)
            patch_next_image_predicted = self.get_next_image(imagined_traj[-1],
                                                             patch_predicted,
                                                             patch_change_pred_mask)
          else:  # if no patchs to change, next image is the original one
            patch_next_image_predicted = imagined_traj[-1]
          imagined_traj.append(patch_next_image_predicted)
        # unpatchify image
        predicted_traj = [self.altered_predictor.unpatchify(p) for p in imagined_traj]
      self.tf_logger.add_images('generated_world_model_prediction',
                                torch.cat(predicted_traj, dim=0),
                                global_step=epoch)
      self.altered_predictor.train()
      self.alteration_predictor.train()
  
  def train(self):
    self.fill_memory()
    self.train_alter_predictor()


if __name__ == '__main__':
  trainer = NSPTrainer()
  trainer.load_model(trainer.altered_predictor, 'altered_predictor')
  trainer.load_model(trainer.alteration_predictor, 'alteration_predictor')
  trainer.train()