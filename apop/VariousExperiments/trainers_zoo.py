import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from collections import deque
from pytorch_msssim import SSIM

from models_zoo import CNNAE, WGANGP, WorldModelFlowUnet, ISPredictor
from helpers_zoo import gradient_penalty, flow_matching_loss, rk45_sampling


class BaseTrainer:
  CONFIG = {
    'experiment_name':     'base_trainer',
    'model_save_dir':      'models/',
    'experiment_save_dir': 'experiments/',
    'model_name':          'Identity',
    'lr':                  1e-4,
    }
  def __init__(self, config={}):
    self.config = {**BaseTrainer.CONFIG, **config}

    self.device = torch.device('cuda' if torch.cuda.is_available() else
                               'mps' if torch.backends.mps.is_available() else
                               'cpu')
    print(f'Using device: {self.device}')

    self.get_train_params = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)

    self.instanciate_model()
    self.set_training_utils()

    print(f"Instanciate {self.config['model_name']} with n_params: {self.get_train_params(self.model):,}")

    self.save_dir = os.path.join(self.config['model_save_dir'], self.config['experiment_name'])
    os.makedirs(self.save_dir, exist_ok=True)
    print(f'Model will be saved in: {self.save_dir}.pt')
  
  def instanciate_model(self):
    self.model = torch.nn.Identity()
  
  def set_training_utils(self):
    self.mdl_opt = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'])

  def train(self):
    pass

  def evaluate(self):
    pass

  def save(self):
    path = os.path.join(self.save_dir, f"{self.config['model_name']}.pt")
    torch.save({'model': self.model.state_dict()}, path)

  def load(self):
    path = os.path.join(self.save_dir, f"{self.config['model_name']}.pt")
    if os.path.isfile(path):
      data = torch.load(path, map_location=self.device)
      self.model.load_state_dict(data['model'])
      print(f'Model loaded successfully from {path}...')
    else:
      print(f'File {path} not found... No loaded model.')
  
  def save_optimizer(self, optimizer=None, optimizer_name=None):
    path = os.path.join(self.save_dir, f'{optimizer_name}.pt' if optimizer_name else 'mdl_opt.pt')
    torch.save({'optimizer': optimizer.state_dict() if optimizer else self.mdl_opt.state_dict()}, path)

  def load_optimizer(self, optimizer=None, optimizer_name=None):
    path = os.path.join(self.save_dir, f'{optimizer_name}.pt' if optimizer_name else 'mdl_opt.pt')
    if os.path.isfile(path):
      data = torch.load(path, map_location=self.device)
      if optimizer:
        optimizer.load_state_dict(data['optimizer'])
      else:
        self.mdl_opt.load_state_dict(data['optimizer'])
      print(f'Optimizer loaded successfully from {path}...')
    else:
      print(f'File {path} not found... No optimizer loaded.')


class ImageCleaner(BaseTrainer):
  '''Takes an altered Image and try to recover the original Image'''
  CONFIG = {
    'experiment_name':  'image_cleaner',
    'model_name':       'CNNAE',
    'lambda_mse':       0.8,
    'lambda_ssim':      0.2,
    }
  def __init__(self, config={}):
    super().__init__(config={**ImageCleaner.CONFIG, **config})
  
  def instanciate_model(self):
    self.model = CNNAE({'encoder_archi': 'BigCNNEncoder'}).to(self.device)
  
  def set_training_utils(self):
    super().set_training_utils()
    self.ssim_loss = SSIM(data_range=1, size_average=True, channel=3)
  
  def train(self, get_data_fn, n_max_steps=300, batch_size=128, image_key='image', target_image_key='target_image'):
    '''get_data_fn: function that take batch_size and return a dict with keys=(image_key, target_image_key)'''
    self.model.train()

    mean_rec_loss = 0.0
    mean_mse_loss = 0.0
    mean_ssim_loss = 0.0

    pbar = tqdm(range(n_max_steps))
    for step in pbar:
      batch = get_data_fn(batch_size)
      rec = self.model(batch[image_key])
      target = batch[target_image_key]

      mse_loss = F.mse_loss(rec, target)
      ssim_loss = 1 - self.ssim_loss(rec, target)
      rec_loss = self.config['lambda_mse'] * mse_loss + self.config['lambda_ssim'] * ssim_loss

      self.mdl_opt.zero_grad()
      rec_loss.backward()
      self.mdl_opt.step()

      mean_mse_loss += (mse_loss.item() - mean_mse_loss) / (step + 1)
      mean_ssim_loss += (ssim_loss.item() - mean_ssim_loss) / (step + 1)
      mean_rec_loss += (rec_loss.item() - mean_rec_loss) / (step + 1)

      pbar.set_description(f'Loss: {mean_rec_loss:.4f}')
    
    return mean_mse_loss, mean_ssim_loss, mean_rec_loss
  
  @torch.no_grad()
  def evaluate(self, images, target_images, batch_size=128):
    '''Direct access to images/target_images list|tensor'''
    self.model.eval()

    n_data = len(images)

    running_rec_loss = 0.0
    running_mse_loss = 0.0
    running_ssim_loss = 0.0

    pbar = tqdm(range(0, n_data, batch_size))
    for i in pbar:
      img_batch = images[i:i+batch_size].to(self.device)
      target_img_batch = target_images[i:i+batch_size].to(self.device)

      rec = self.model(img_batch)

      mse_loss = F.mse_loss(rec, target_img_batch)
      ssim_loss = 1 - self.ssim_loss(rec, target_img_batch)
      rec_loss = self.config['lambda_mse'] * mse_loss + self.config['lambda_ssim'] * ssim_loss

      running_mse_loss += mse_loss.item()
      running_ssim_loss += ssim_loss.item()
      running_rec_loss += rec_loss.item()

    return running_mse_loss/n_data, running_ssim_loss/n_data, running_rec_loss/n_data


class GANGoalImagePredictorTrainer(BaseTrainer):
  CONFIG = {
    'experiment_name':         'gan_goal_image_predictor',
    'model_name':              'WGANGP',
    'lambda_mse':              0.8,
    'lambda_ssim':             0.2,
    'rec_loss_lambda':         10.0,
    'gradient_penalty_lambda': 10.0,
  }
  def __init__(self, config={}):
    super().__init__(config={**GANGoalImagePredictorTrainer.CONFIG, **config})
  
  def instanciate_model(self):
    self.model = WGANGP().to(self.device)
  
  def set_training_utils(self):
    self.ssim_loss = SSIM(data_range=1, size_average=True, channel=3)
    self.ae_opt = torch.optim.AdamW(self.model.auto_encoder.parameters(), lr=self.config['lr'])
    self.critic_opt = torch.optim.AdamW(self.model.critic.parameters(), lr=self.config['lr'])
  
  def save_optimizer(self):
    super().save_optimizer(self.ae_opt, 'gan_gip_ae_opt')
    super().save_optimizer(self.critic_opt, 'gan_gip_ae_opt')
  
  def load_optimizer(self):
    super().load_optimizer(self.ae_opt, 'gan_gip_ae_opt')
    super().load_optimizer(self.critic_opt, 'gan_gip_ae_opt')
  
  def train(self, get_data_fn, n_max_steps=200, batch_size=128, n_critic_steps=5,
            goal_img_key='goal_image', start_img_key='image', tf_logger=None):
    '''Generator creates goal image starting from start_image'''
    print('GANGoalImagePredictor training pass...')
    self.model.train()

    mean_gp = 0.0
    mean_critic_loss = 0.0

    mean_gen_loss = 0.0
    mean_mse_loss = 0.0
    mean_ssim_loss = 0.0
    mean_rec_loss = 0.0

    pbar = tqdm(range(n_max_steps))
    for step in pbar:
      batch = get_data_fn(batch_size)
      real_imgs = batch[goal_img_key]
      start_imgs = batch[start_img_key]

      # -----------------------------
      # Train Critic (multiple steps)
      # -----------------------------
      fake_imgs = self.model.auto_encoder(start_imgs).detach()
      running_gp, running_critic_loss = 0.0, 0.0
      for _ in range(n_critic_steps):
        real_scores = self.model.critic(real_imgs)
        fake_scores = self.model.critic(fake_imgs)

        # WGAN-GP critic loss: E[fake] - E[real] + GP
        gp = gradient_penalty(self.model.critic, real_imgs, fake_imgs, self.device,
                              gp_lambda=self.config['gradient_penalty_lambda'])
        loss_critic = fake_scores.mean() - real_scores.mean() + gp

        self.critic_opt.zero_grad()
        loss_critic.backward()
        self.critic_opt.step()

        running_gp += gp.item()
        running_critic_loss += loss_critic.item()
      
      mean_gp += (running_gp/n_critic_steps - mean_gp) / (step + 1)
      mean_critic_loss += (running_critic_loss/n_critic_steps - mean_critic_loss) / (step + 1)
      
      # -----------------------------------
      # Train Generator (Encoder + Decoder)
      # -----------------------------------
      fake_imgs = self.model.auto_encoder(start_imgs)
      # Generator tries to minimize -E[critic(fake)] (i.e. maximize critic score)
      gen_adv = -self.model.critic(fake_imgs).mean()

      mse_loss = F.mse_loss(fake_imgs, real_imgs)
      ssim_loss = 1 - self.ssim_loss(fake_imgs, real_imgs)
      rec_loss = self.config['lambda_mse'] * mse_loss + self.config['lambda_ssim'] * ssim_loss

      gen_loss = gen_adv + self.config['rec_loss_lambda'] * rec_loss

      self.ae_opt.zero_grad()
      gen_loss.backward()
      self.ae_opt.step()

      mean_gen_loss += (gen_adv.item() - mean_gen_loss) / (step + 1)
      mean_rec_loss += (rec_loss.item() - mean_rec_loss) / (step + 1)
      mean_mse_loss += (mse_loss.item() - mean_mse_loss) / (step + 1)
      mean_ssim_loss += (ssim_loss.item() - mean_ssim_loss) / (step + 1)

      if tf_logger:
        tf_logger.add_scalar('gan_goal_image_predictor_rec_loss', mean_rec_loss, step)

      pbar.set_description(f'Loss: {mean_rec_loss:.4f}')
    
    if tf_logger:
      tf_logger.add_images('generated_gan_goal_image_train',
                           torch.cat([start_imgs[:8], real_imgs[:8], fake_imgs[:8]], dim=0),
                           global_step=1)
    return mean_gp, mean_critic_loss, mean_gen_loss, mean_mse_loss, mean_ssim_loss, mean_rec_loss

  @torch.no_grad()
  def evaluate(self, get_data_fn, n_steps=2, batch_size=32, tf_logger=None,
               goal_img_key='goal_image', start_img_key='image'):
    self.model.eval()
    running_loss = 0.
    pbar = tqdm(range(n_steps))
    for i in pbar:
      batch = get_data_fn(batch_size)
      goal_images = batch[goal_img_key]
      start_images = batch[start_img_key]

      rec = self.model.auto_encoder(start_images[i:i+batch_size].to(self.device))

      mse_loss = F.mse_loss(rec, goal_images[i:i+batch_size])
      ssim_loss = 1 - self.ssim_loss(rec, goal_images[i:i+batch_size])
      loss = self.config['lambda_mse'] * mse_loss + self.config['lambda_ssim'] * ssim_loss

      running_loss += loss.item()
    if tf_logger:
      tf_logger.add_images('generated_gan_goal_image_test',
                           torch.cat([start_images[:8], goal_images[:8], rec[:8]], dim=0),
                           global_step=1)
    return running_loss/n_steps
  
  @torch.no_grad()
  def infer(self, start_images):
    self.model.eval()
    return self.model.auto_encoder(start_images)


class FlowGoalImagePredictorTrainer(BaseTrainer):
  CONFIG = {
    'experiment_name':         'flow_goal_image_predictor',
    'model_name':              'WorldModelFlowUnet',
    'time_dim':                64,
  }
  def __init__(self, config={}):
    super().__init__(config={**FlowGoalImagePredictorTrainer.CONFIG, **config})
  
  def instanciate_model(self):
    self.model = WorldModelFlowUnet(img_chan=3,
                                    time_dim=self.config['time_dim'],
                                    add_action=False,
                                    add_is=False,
                                    add_ds=False,
                                    x_start=True).to(self.device)
  
  def train(self, get_data_fn, n_max_steps=500, batch_size=128, tf_logger=None,
            img_cleaner=None, img_normalizer=None,
            goal_img_key='goal_image', start_img_key='image'):
    '''Generator creates goal image starting from start_image and gaussian noise'''
    print('FlowGoalImagePredictor training pass...')
    self.model.train()

    mean_loss = 0.0

    pbar = tqdm(range(n_max_steps))
    for step in pbar:
      batch = get_data_fn(batch_size)
      x1 = batch[goal_img_key]  # target distribution is the image to predict
      condition = {'x_cond': batch[start_img_key]}

      # ------------------------------------------------------------------------------------------ #
      # STEP1: Train the world_model to produce a corresponding image with provided internal_state #
      # ------------------------------------------------------------------------------------------ #
      # Moving random (gaussian) distribution to x1 distribution based on provided internal state  #
      loss = flow_matching_loss(self.model, x1, condition=condition)

      self.mdl_opt.zero_grad()
      loss.backward()
      self.mdl_opt.step()

      mean_loss += (loss.item() - mean_loss) / (step + 1)
      
      if tf_logger:
        tf_logger.add_scalar('flow_goal_image_predictor_loss', mean_loss, step)
        
        if step % 20 == 0:
          x1_pred = rk45_sampling(
            self.model,
            device=self.device,
            n_samples=x1.shape[0],
            condition=condition,
            n_steps=2
          )
          x1_pred = x1_pred[-1]

          if img_cleaner and img_normalizer:
            with torch.no_grad():
              x1_pred_cleaned = img_cleaner(img_normalizer(x1_pred))
            img_comparison = torch.cat([img_normalizer(batch[start_img_key][:8]),
                                        img_normalizer(x1[:8]),
                                        img_normalizer(x1_pred[:8]),
                                        x1_pred_cleaned[:8]], dim=0)
          else:
            img_comparison = torch.cat([batch[start_img_key][:8], x1[:8], x1_pred[:8]], dim=0)

          tf_logger.add_scalar('flow_goal_image_predictor_rec_loss', F.mse_loss(x1_pred, x1), step)
          tf_logger.add_images('generated_gan_goal_image_train', img_comparison, global_step=step)

      pbar.set_description(f'Loss: {mean_loss:.6f}')
    return mean_loss
  
  @torch.no_grad()
  def evaluate(self, get_data_fn, n_steps=1, batch_size=32, tf_logger=None,
               img_cleaner=None, img_normalizer=None,
               goal_img_key='goal_image', start_img_key='image'):
    self.model.eval()
    pbar = tqdm(range(n_steps))
    for i in pbar:
      batch = get_data_fn(batch_size)
      x1 = batch[goal_img_key]
      condition = {'x_cond': batch[start_img_key]}

      x1_pred = rk45_sampling(
            self.model,
            device=self.device,
            n_samples=x1.shape[0],
            condition=condition,
            n_steps=4
          )
      x1_pred = x1_pred[-1]

      if img_cleaner and img_normalizer:
        with torch.no_grad():
          x1_pred_cleaned = img_cleaner(img_normalizer(x1_pred))
        img_comparison = torch.cat([img_normalizer(batch[start_img_key][:8]),
                                    img_normalizer(x1[:8]),
                                    img_normalizer(x1_pred[:8]),
                                    x1_pred_cleaned[:8]], dim=0)
      else:
        img_comparison = torch.cat([batch[start_img_key][:8], x1[:8], x1_pred[:8]], dim=0)

      if tf_logger:
        tf_logger.add_images('generated_gan_goal_image_test', img_comparison, global_step=1)
    return F.mse_loss(x1_pred, x1)

  @torch.no_grad()
  def infer(self, image, img_cleaner=None, img_normalizer=None):
    self.model.eval()

    condition = {'x_cond': image}

    x1_pred = rk45_sampling(
          self.model,
          device=self.device,
          n_samples=image.shape[0],
          condition=condition,
          n_steps=4
        )
    x1_pred = x1_pred[-1]

    x1_pred_cleaned = x1_pred.clone()
    if img_cleaner and img_normalizer:
      with torch.no_grad():
        x1_pred_cleaned = img_cleaner(img_normalizer(x1_pred))

    return x1_pred_cleaned


class ISPredictorTrainer(BaseTrainer):
  CONFIG = {
    'experiment_name':         'internal_state_predictor',
    'model_name':              'ISPredictor',
  }
  def __init__(self, config={}):
    super().__init__(config={**ISPredictorTrainer.CONFIG, **config})
  
  def instanciate_model(self):
    self.model = ISPredictor().to(self.device)
  
  def train(self, get_data_fn, n_max_steps=10_000, batch_size=128, tf_logger=None,
            internal_state_key='goal_internal_state', image_key='goal_image_generated'):
    print('Internal State Predictor training pass...')
    self.model.train()

    mean_loss = 0.0
    acc1_window, acc2_window = deque([0.0], maxlen=50), deque([0.0], maxlen=50)

    pbar = tqdm(range(n_max_steps))
    for step in pbar:
      batch = get_data_fn(batch_size)

      # Infer internal_state from replay_buffer images
      g1_logits, g2_logits = self.model(batch[image_key])

      loss = (
        F.cross_entropy(g1_logits, batch[internal_state_key][:, 0]) +
        F.cross_entropy(g2_logits, batch[internal_state_key][:, 1])
      )

      self.mdl_opt.zero_grad()
      loss.backward()
      self.mdl_opt.step()

      mean_loss += (loss.item() - mean_loss) / (step + 1)

      # Accuracies
      acc1_window.append((g1_logits.argmax(dim=1) == batch[internal_state_key][:, 0]).float().mean().item())
      acc2_window.append((g2_logits.argmax(dim=1) == batch[internal_state_key][:, 1]).float().mean().item())
      mean_acc1 = sum(acc1_window) / len(acc1_window)
      mean_acc2 = sum(acc2_window) / len(acc2_window)
      
      if tf_logger:
        tf_logger.add_scalar('isp_loss', mean_loss, step)
        tf_logger.add_scalar('isp_accuracy1', mean_acc1, step)
        tf_logger.add_scalar('isp_accuracy2', mean_acc2, step)
      
      mean_acc = (mean_acc1 + mean_acc2) / 2
      pbar.set_description(f'Loss: {mean_loss:.6f} | Mean_acc: {mean_acc:.2f}')
    
    return mean_loss, mean_acc
  
  @torch.no_grad()
  def evaluate(self, get_data_fn, n_steps=1, batch_size=32, tf_logger=None,
               internal_state_key='goal_internal_state', image_key='goal_image_generated'):
    self.model.eval()
    pbar = tqdm(range(n_steps))
    for i in pbar:
      batch = get_data_fn(batch_size)

      g1_logits, g2_logits = self.model(batch[image_key])

      acc1 = (g1_logits.argmax(dim=1) == batch[internal_state_key][:, 0]).float().mean().item()
      acc2 = (g2_logits.argmax(dim=1) == batch[internal_state_key][:, 1]).float().mean().item()
    return acc1, acc2


if __name__ == '__main__':
  ggp = GANGoalImagePredictorTrainer()
  # --- Dummy test --- #
  # from replay_buffer import ReplayBuffer
  # ic = ImageCleaner()

  # rb = ReplayBuffer(internal_state_dim=2, action_dim=1, image_size=256, image_chan=3, resize_to=32,
  #                   normalize_img=True, capacity=10, device='cpu', target_device=ic.device)
  # for _ in range(10):
  #   rb.add([0, 0], [0], torch.rand(3, 32, 32), 0, False, [0, 0], torch.rand(3, 32, 32))
  
  # losses = ic.train(get_data_fn=lambda x: rb.sample(x), n_max_steps=2, batch_size=3, target_image_key='next_image')
  # print(losses)

  # ic.evaluate(rb.image, rb.next_image, batch_size=3)