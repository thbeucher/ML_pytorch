import torch
import argparse
import torch.nn as nn

from tqdm import tqdm
from pytorch_msssim import SSIM

from encoding_exps import CNNAE
from flow_matching_exps import flow_matching_loss, rk45_sampling
from next_state_predictor_exps import NextStatePredictorTrainer, WorldModelFlowUnet


class NISPredictor(nn.Module):
  def __init__(self, n_actions=5, action_dim=8, is1_n_values=19, is2_n_values=37, is_dim=32, mlp_dim=64):
    super().__init__()
    self.action_emb = nn.Sequential(nn.Embedding(n_actions, action_dim),
                                    nn.Linear(action_dim, action_dim),
                                    nn.SiLU(True))
    self.is_emb = nn.Sequential(nn.Embedding(max(is1_n_values, is2_n_values), is_dim),
                                nn.Linear(is_dim, is_dim),
                                nn.SiLU(True))

    self.main = nn.Sequential(nn.Linear(action_dim + 2*is_dim, mlp_dim), nn.SiLU(True),
                              nn.Linear(mlp_dim, mlp_dim), nn.SiLU(True))

    self.predictor1 = nn.Linear(mlp_dim, is1_n_values)
    self.predictor2 = nn.Linear(mlp_dim, is2_n_values)
  
  def forward(self, action, internal_state):
    action_emb = self.action_emb(action.squeeze(-1))          # -> [B, 8]
    is_emb = self.is_emb(internal_state).flatten(1)           # -> [B, 2, 32] -> [B, 64]
    main = self.main(torch.cat([action_emb, is_emb], dim=1))  # [B, 72] -> [B, 64]
    is1 = self.predictor1(main)                               # -> [B, 18]
    is2 = self.predictor2(main)                               # -> [B, 36]
    return is1, is2


class ImageToTwoHeads(nn.Module):
  def __init__(self, is1_n_values=18, is2_n_values=36, hidden_dim=512):
      super().__init__()

      self.encoder = nn.Sequential(
          # [B, 3, 32, 32]
          nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, 16, 16]
          nn.BatchNorm2d(64),
          nn.SiLU(),

          nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 128, 8, 8]
          nn.BatchNorm2d(128),
          nn.SiLU(),

          nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # [B, 256, 4, 4]
          nn.BatchNorm2d(256),
          nn.SiLU(),

          nn.Conv2d(256, hidden_dim, kernel_size=3, stride=1, padding=1),  # [B, 512, 4, 4]
          nn.BatchNorm2d(hidden_dim),
          nn.SiLU(),
      )

      # Global Average Pooling → [B, hidden_dim]
      self.pool = nn.AdaptiveAvgPool2d(1)

      # Two prediction heads
      self.head1 = nn.Linear(hidden_dim, is1_n_values)
      self.head2 = nn.Linear(hidden_dim, is2_n_values)

  def forward(self, x):
    """
    x: [B, 3, 32, 32]
    """
    h = self.encoder(x)
    h = self.pool(h).flatten(1)

    out1 = self.head1(h)  # [B, 18]
    out2 = self.head2(h)  # [B, 36]

    return out1, out2


class NSPTrainer(NextStatePredictorTrainer):
  CONFIG = {'exp_name':                             'threesteps_nsp',
            'replay_buffer_size':                   1_000,
            'internal_state_dim':                   2,
            'internal_state_n_values':              (90//5+1, 180//5+1),  # max_angle / angle_step
            'internal_state_emb':                   32,
            'internal_world_model_max_train_steps': 10_000,
            'internal_world_model_batch_size':      128,
            'image_cleaner_max_train_steps':        300,
            'image_cleaner_batch_size':             128,
            'lambda_mse':                           0.8,
            'lambda_ssim':                          0.2,
            'world_model_max_train_steps':          3_000,}
  def __init__(self, config={}):
    super().__init__(NSPTrainer.CONFIG)
    self.config = {**self.config, **config}
  
  def instanciate_model(self):
    self.world_model = WorldModelFlowUnet(img_chan=self.config['image_chan'],
                                          time_dim=self.config['time_dim'],
                                          add_action=False,
                                          add_is=True,
                                          is_n_values=max(self.config['internal_state_n_values']),
                                          is_dim=self.config['internal_state_emb']).to(self.device)
    print(f'Instanciate Worl_Model (trainable parameters: {self.get_train_params(self.world_model):,})')
    self.internal_world_model = NISPredictor(n_actions=self.config['action_n_values'],
                                            action_dim=self.config['action_emb'],
                                            is1_n_values=self.config['internal_state_n_values'][0],
                                            is2_n_values=self.config['internal_state_n_values'][1],
                                            is_dim=self.config['internal_state_emb']).to(self.device)
    print(f'Instanciate Internal_Worl_Model (trainable parameters: {self.get_train_params(
      self.internal_world_model):,})')
    self.image_cleaner = CNNAE({'encoder_archi': 'BigCNNEncoder'}).to(self.device)
    print(f'Instanciate ImageCleaner (trainable parameters: {self.get_train_params(self.image_cleaner):,})')
  
  def train_image_cleaner(self):
    print('Training ImageCleaner...')
    self.image_cleaner.train()
    self.world_model.eval()

    self.ic_opt = torch.optim.AdamW(self.image_cleaner.parameters(), lr=1e-3)
    self.mse_criterion = nn.MSELoss()
    self.ssim_criterion = SSIM(data_range=1, size_average=True, channel=3)

    best_loss = torch.inf
    mean_rec_loss = 0.0
    mean_mse_loss = 0.0
    mean_ssim_loss = 0.0
    pbar = tqdm(range(self.config['image_cleaner_max_train_steps']))
    for step in pbar:
      batch = self.replay_buffer.sample(self.config['image_cleaner_batch_size'])
      
      with torch.no_grad():
        condition = {'internal_state': batch['internal_state']}
        x1_pred = rk45_sampling(
          self.world_model,
          device=self.device,
          n_samples=batch['image'].shape[0],
          condition=condition,
          n_steps=4
        )
        x1_pred = self.clamp_denorm_fn(x1_pred[-1].detach())

      rec = self.image_cleaner(x1_pred)
      target = self.clamp_denorm_fn(batch['image'])
      mse_loss = self.mse_criterion(rec, target)
      ssim_loss = 1 - self.ssim_criterion(rec, target)
      rec_loss = self.config['lambda_mse'] * mse_loss + self.config['lambda_ssim'] * ssim_loss

      self.ic_opt.zero_grad()
      rec_loss.backward()
      self.ic_opt.step()

      mean_mse_loss += (mse_loss.item() - mean_mse_loss) / (step + 1)
      mean_ssim_loss += (ssim_loss.item() - mean_ssim_loss) / (step + 1)
      mean_rec_loss += (rec_loss.item() - mean_rec_loss) / (step + 1)

      if self.tf_logger is not None:
        self.tf_logger.add_scalar('ic_reconstruction_loss', mean_rec_loss, step)
        self.tf_logger.add_scalar('ic_mse_loss', mean_mse_loss, step)
        self.tf_logger.add_scalar('ic_ssim_loss', mean_ssim_loss, step)

        if step % 10 == 0:
          ori_fm_rec = torch.cat([batch['image'][:8], x1_pred[:8], rec[:8]], dim=0)
          self.tf_logger.add_images(f'image_cleaned', ori_fm_rec, global_step=step//10)
      
      if mean_rec_loss < best_loss:
        best_loss = mean_rec_loss
        self.save_model(self.image_cleaner, 'image_cleaner')

      pbar.set_postfix(loss=f'{mean_rec_loss:.4f}')
  
  def train_world_model(self):
    print('Training world model...')
    self.world_model.train()

    best_loss = torch.inf
    mean_loss = 0.0

    pbar = tqdm(range(self.config['world_model_max_train_steps']))
    for step in pbar:
      batch = self.replay_buffer.sample(self.config['world_model_batch_size'])
      x1 = batch['next_image']  # target distribution is the image to predict
      condition = {'internal_state': batch['internal_state']}

      loss = flow_matching_loss(self.world_model, x1, condition=condition,
                                weighted_time_sampling=self.config['weighted_time_sampling'],
                                noise_scale=self.config['noise_scale'])

      self.wm_optimizer.zero_grad()
      loss.backward()
      self.wm_optimizer.step()

      mean_loss += (loss.item() - mean_loss) / (step + 1)

      if mean_loss < best_loss:
        self.save_model(self.world_model, 'world_model')
        best_loss = mean_loss
      
      if self.tf_logger:
        self.tf_logger.add_scalar('worlmodel_projection_fm_loss', mean_loss, step)
        
        if step % self.config['world_model_check_pred_loss_every'] == 0:
          x1_pred = rk45_sampling(
            self.world_model,
            device=self.device,
            n_samples=x1.shape[0],
            condition=condition,
            n_steps=4
          )
          x1_pred = x1_pred[-1].clamp(-1, 1)

          self.tf_logger.add_scalar('world_model_projection_pred_loss',
                                    torch.nn.functional.mse_loss(x1_pred, x1),
                                    step // self.config['world_model_check_pred_loss_every'])
          self.tf_logger.add_images('generated_world_model_projection_prediction',
                                    torch.cat([x1[:8], x1_pred[:8]], dim=0),
                                    global_step=step)

      pbar.set_postfix(loss=f'{mean_loss:.6f}')
  
  def train_internal_world_model(self):
    print('Training Internal World Model...')
    self.internal_world_model.train()

    self.iwm_opt = torch.optim.AdamW(self.internal_world_model.parameters(), lr=1e-2)
    self.iwm_criterion = nn.CrossEntropyLoss()

    best_loss = torch.inf
    mean_loss = 0.0
    mean_acc1, mean_acc2 = 0.0, 0.0

    pbar = tqdm(range(self.config['internal_world_model_max_train_steps']))
    for step in pbar:
      batch = self.replay_buffer.sample(self.config['internal_world_model_batch_size'])
      nis1_pred, nis2_pred = self.internal_world_model(batch['action'], batch['internal_state'])  # [B, 18], [B, 36]

      target = batch['next_internal_state']
      is1_target, is2_target = target[:, 0], target[:, 1]

      loss1 = self.iwm_criterion(nis1_pred, is1_target)
      loss2 = self.iwm_criterion(nis2_pred, is2_target)
      loss = loss1 + loss2

      self.iwm_opt.zero_grad()
      loss.backward()
      self.iwm_opt.step()

      mean_loss += (loss.item() - mean_loss) / (step + 1)
      # Accuracies
      pred1 = nis1_pred.argmax(dim=1)  # [B]
      pred2 = nis2_pred.argmax(dim=1)  # [B]
      acc1 = (pred1 == is1_target).float().mean()
      acc2 = (pred2 == is2_target).float().mean()
      mean_acc1 += (acc1.item() - mean_acc1) / (step + 1)
      mean_acc2 += (acc2.item() - mean_acc2) / (step + 1)

      if mean_loss < best_loss:
        self.save_model(self.internal_world_model, 'internal_world_model')
        best_loss = mean_loss
      
      if self.tf_logger:
        self.tf_logger.add_scalar('iwm_loss', mean_loss, step)
        self.tf_logger.add_scalar('iwm_accuracy1', mean_acc1, step)
        self.tf_logger.add_scalar('iwm_accuracy2', mean_acc2, step)
      
      pbar.set_postfix(loss=f'{mean_loss:.6f}')
  
  def train(self):
    self.fill_memory()
    self.train_internal_world_model()
    self.train_world_model()
    self.train_image_cleaner()
  
  @torch.no_grad()
  def show_imagined_trajectory(self):
    self.fill_memory(replay_buffer_size=500)
    self.internal_world_model.eval()
    self.world_model.eval()
    self.image_cleaner.eval()

    batch = self.replay_buffer.sample(1)

    imagined_traj = [batch['image'][:1]]
    internal_state = batch['internal_state'][:1]  # [1, 2], action = [1, 1]
    for action in torch.as_tensor([[[1]]]*12 + [[[3]]]*11, device=self.device, dtype=torch.long):
      # ----------------------------------------------------------------- #
      # STEP1: from internal_state and action -> find next_internal_state #
      # ----------------------------------------------------------------- #
      nis1_pred, nis2_pred = self.internal_world_model(action, internal_state)  # [B, 18], [B, 36]
      pred1 = nis1_pred.argmax(dim=1)
      pred2 = nis2_pred.argmax(dim=1)
      internal_state = torch.stack([pred1, pred2], dim=1)

      # ----------------------------------------------------------------- #
      # STEP2: from internal_state -> get image from world model          #
      # ----------------------------------------------------------------- #
      x1_pred = rk45_sampling(
        self.world_model,
        device=self.device,
        n_samples=internal_state.shape[0],
        condition={'internal_state': internal_state},
        n_steps=10
      )
      x1_pred = self.clamp_denorm_fn(x1_pred[-1])

      # ------------------------------------------------------------------------- #
      # STEP3: from imagined image -> clean world model image to fit with reality #
      # ------------------------------------------------------------------------- #
      rec = self.image_cleaner(x1_pred)

      imagined_traj.append(rec)

    self.tf_logger.add_images('generated_traj_threesteps_nsp', torch.cat(imagined_traj, dim=0))


def get_args():
  parser = argparse.ArgumentParser(description='Next state predictor experiments')
  parser.add_argument('--trainer', '-t', type=str, default='nsp')
  parser.add_argument('--load_model', '-lm', action='store_true')
  parser.add_argument('--train_model', '-tm', action='store_true')
  parser.add_argument('--eval_model', '-em', action='store_true')
  parser.add_argument('--save_model', '-sm', action='store_true')
  parser.add_argument('--play_model', '-pm', action='store_true')
  parser.add_argument('--show_imagined_trajectory', '-sit', action='store_true')
  parser.add_argument('--force_human_view', '-fhv', action='store_true')
  parser.add_argument('--experiment_name', '-en', type=str, default=None)
  return parser.parse_args()


if __name__ == '__main__':
  trainers = {'nsp': NSPTrainer}
  args = get_args()

  config = {} if args.experiment_name is None else {'exp_name': args.experiment_name}
  config['render_mode'] = 'human' if args.play_model or args.force_human_view else 'rgb_array'

  print(f'Trainer: {args.trainer}')
  trainer = trainers[args.trainer](config)

  if args.load_model:
    trainer.load_model(trainer.world_model, 'world_model')
    trainer.load_model(trainer.internal_world_model, 'internal_world_model')
    trainer.load_model(trainer.image_cleaner, 'image_cleaner')

  if args.play_model:
    print('Start autoplay...')
    trainer.autoplay()
  
  if args.train_model:
    print('Start training...')
    trainer.train()
  
  if args.show_imagined_trajectory:
    print('Show Imagined Trajectory...')
    trainer.show_imagined_trajectory()