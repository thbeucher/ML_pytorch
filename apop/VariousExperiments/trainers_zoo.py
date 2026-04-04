import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from tqdm import tqdm
from collections import deque
from pytorch_msssim import SSIM

from helpers_zoo import gradient_penalty, flow_matching_loss, rk45_sampling
from models_zoo import CNNAE, WGANGP, WorldModelFlowUnet, ISPredictor, CNNAENSP, MemoryBankFuturStatePredictor


class BaseTrainer:
  CONFIG = {
    'experiment_name':     'base_trainer',
    'model_save_dir':      'models/',
    'experiment_save_dir': 'experiments/',
    'model_name':          'Identity',
    'lr':                  1e-4,
    'weight_decay':        1e-2,
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
    print(f'Model will be saved in: {self.save_dir}')

    self.augmentations = [
      T.RandomHorizontalFlip(),
      T.RandomVerticalFlip(),
      T.RandomGrayscale(),
      T.RandomPerspective(),
      T.RandomRotation(degrees=45),
      T.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
      T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.1)]),
      T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    ]
  
  def instanciate_model(self):
    self.model = torch.nn.Identity()
  
  def set_training_utils(self):
    self.mdl_opt = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'],
                                     weight_decay=self.config['weight_decay'])

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
      return True
    else:
      print(f'File {path} not found... No loaded model.')
      return False
  
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


class CNNAETrainer(BaseTrainer):
  '''Can be used as ImageCleaner -> Takes an altered Image and try to recover the original Image'''
  CONFIG = {
    'experiment_name':  'image_cleaner',
    'model_name':       'CNNAE',
    'lambda_mse':       0.8,
    'lambda_ssim':      0.2,
    'model_config':     {'encoder_archi': 'BigCNNEncoder'}
    }
  def __init__(self, config={}):
    super().__init__(config={**CNNAETrainer.CONFIG, **config})
  
  def instanciate_model(self):
    self.model = CNNAE(self.config['model_config']).to(self.device)
  
  def set_training_utils(self):
    super().set_training_utils()
    self.ssim_loss = SSIM(data_range=1, size_average=True, channel=3)
  
  def train(self, get_data_fn, n_max_steps=300, batch_size=128, tf_logger=None,
            image_key='image', target_image_key='target_image', augment=False):
    '''get_data_fn: function that take batch_size and return a dict with keys=(image_key, target_image_key)'''
    print('CNNAE training pass...')
    self.model.train()

    mean_rec_loss = 0.0
    mean_mse_loss = 0.0
    mean_ssim_loss = 0.0

    pbar = tqdm(range(n_max_steps))
    for step in pbar:
      batch = get_data_fn(batch_size)

      if augment:
        # aug_batch = (batch[image_key].clone() + 1) / 2  # Convert from [-1, 1] to [0, 1]
        aug_batch = batch[image_key].clone()
        part_size = batch_size // len(self.augmentations)
        i = 0
        for aug_fn in self.augmentations:
          aug_batch[i:i+part_size] = aug_fn(aug_batch[i:i+part_size])
          i += part_size
        # aug_batch = aug_batch * 2 - 1  # Convert back to [-1, 1]
        batch[image_key] = torch.cat([batch[image_key], aug_batch], dim=0)
        batch[target_image_key] = torch.cat([batch[target_image_key], batch[target_image_key]], dim=0)

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
      if tf_logger is not None:
        tf_logger.add_scalar('cnnae_rec_loss', mean_rec_loss, step)
    
    if tf_logger is not None:
      tf_logger.add_images('cnnae_reconstructed_images', torch.cat([batch[image_key][:8], rec[:8]], dim=0))
    
    return mean_mse_loss, mean_ssim_loss, mean_rec_loss
  
  @torch.no_grad()
  def evaluate(self, get_data_fn, n_max_steps=1, batch_size=8, tf_logger=None,
               image_key='image', target_image_key='target_image'):
    '''Direct access to images/target_images list|tensor'''
    self.model.eval()

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

      mean_mse_loss += (mse_loss.item() - mean_mse_loss) / (step + 1)
      mean_ssim_loss += (ssim_loss.item() - mean_ssim_loss) / (step + 1)
      mean_rec_loss += (rec_loss.item() - mean_rec_loss) / (step + 1)

    if tf_logger is not None:
      tf_logger.add_images('cnnae_reconstructed_images_eval', torch.cat([batch[image_key][:8],
                                                                         rec[:8], target[:8]], dim=0))
    
    return mean_mse_loss, mean_ssim_loss, mean_rec_loss


class CNNAEEmbNSPTrainer(BaseTrainer):
  '''Train an AutoEncoder with skip connections to reconstruct input image
     and predict the next state in the embedding space'''
  CONFIG = {
    'experiment_name':  'ae_embedding_nsp',  # CNN AutoEncoder Embedding Next State Prediction
    'model_name':       'CNNAENSP',
    'lambda_mse':       0.8,
    'lambda_ssim':      0.2,
    'model_config':     {'ae_config': {'encoder_archi': 'BigCNNEncoder',
                                       'skip_connection': True,
                                       'linear_bottleneck': True,
                                       'latent_dim': 128}}
    }
  def __init__(self, config={}):
    super().__init__(config={**CNNAEEmbNSPTrainer.CONFIG, **config})
  
  def instanciate_model(self):
    self.model = CNNAENSP(self.config['model_config']).to(self.device)
  
  def set_training_utils(self):
    super().set_training_utils()
    self.ssim_loss = SSIM(data_range=1, size_average=True, channel=3)
  
  def train(self, get_data_fn, n_max_steps=300, batch_size=128, tf_logger=None, augment=False,
            image_key='image', target_image_key='target_image', start_step=0):
    '''get_data_fn: function that take batch_size and return a dict with keys=(image_key, target_image_key)'''
    print('CNNAE training pass...')
    self.model.train()

    mean_rec_loss = 0.0
    mean_mse_loss = 0.0
    mean_ssim_loss = 0.0
    mean_nsp_loss = 0.0
    mean_loss = 0.0

    pbar = tqdm(range(n_max_steps))
    for step in pbar:
      batch = get_data_fn(batch_size)

      if augment:
        # aug_batch = (batch[image_key].clone() + 1) / 2  # Convert from [-1, 1] to [0, 1]
        aug_batch = batch[image_key].clone()
        part_size = batch_size // len(self.augmentations)
        i = 0
        for aug_fn in self.augmentations:
          aug_batch[i:i+part_size] = aug_fn(aug_batch[i:i+part_size])
          i += part_size
        # aug_batch = aug_batch * 2 - 1  # Convert back to [-1, 1]
        batch[image_key] = torch.cat([batch[image_key], aug_batch], dim=0)
        batch[target_image_key] = torch.cat([batch[target_image_key], batch[target_image_key]], dim=0)

      rec, pred_next_latent = self.model(batch[image_key],
                                         condition={'action': batch['action'],
                                                    'internal_state': batch['internal_state']})
      target_next_latent = self.model.get_embedding(batch[target_image_key]).detach()
      target_rec = batch[image_key]

      # Reconstruction loss
      mse_loss = F.mse_loss(rec, target_rec)
      ssim_loss = 1 - self.ssim_loss(rec, target_rec)
      rec_loss = self.config['lambda_mse'] * mse_loss + self.config['lambda_ssim'] * ssim_loss

      # Next state prediction loss (MSE in latent space)
      nsp_loss = F.mse_loss(pred_next_latent, target_next_latent)

      loss = rec_loss + nsp_loss

      self.mdl_opt.zero_grad()
      loss.backward()
      self.mdl_opt.step()

      mean_mse_loss += (mse_loss.item() - mean_mse_loss) / (step + 1)
      mean_ssim_loss += (ssim_loss.item() - mean_ssim_loss) / (step + 1)
      mean_rec_loss += (rec_loss.item() - mean_rec_loss) / (step + 1)
      mean_nsp_loss += (nsp_loss.item() - mean_nsp_loss) / (step + 1)
      mean_loss += (loss.item() - mean_loss) / (step + 1)

      pbar.set_description(f'Loss: {mean_loss:.4f}')
      if tf_logger is not None:
        tf_logger.add_scalar('cnnae_rec_loss', mean_rec_loss, start_step + step)
        tf_logger.add_scalar('cnnae_nsp_loss', mean_nsp_loss, start_step + step)

    if tf_logger is not None:
      tf_logger.add_images('cnnae_reconstructed_images', torch.cat([batch[image_key][:8], rec[:8]], dim=0))
    
    return mean_mse_loss, mean_ssim_loss, mean_rec_loss, mean_nsp_loss, mean_loss
  
  @torch.no_grad()
  def evaluate(self, get_data_fn, n_max_steps=1, batch_size=8, tf_logger=None,
               image_key='image', target_image_key='target_image', step=0):
    '''Direct access to images/target_images list|tensor'''
    self.model.eval()

    mean_rec_loss = 0.0
    mean_mse_loss = 0.0
    mean_ssim_loss = 0.0
    mean_nsp_loss = 0.0

    pbar = tqdm(range(n_max_steps))
    for step in pbar:
      batch = get_data_fn(batch_size)
      rec, pred_next_latent = self.model(batch[image_key],
                                         condition={'action': batch['action'],
                                                    'internal_state': batch['internal_state']})
      target = batch[image_key]
      target_next_latent = self.model.get_embedding(batch[target_image_key])

      mse_loss = F.mse_loss(rec, target)
      ssim_loss = 1 - self.ssim_loss(rec, target)
      rec_loss = self.config['lambda_mse'] * mse_loss + self.config['lambda_ssim'] * ssim_loss

      nsp_loss = F.mse_loss(pred_next_latent, target_next_latent)

      mean_mse_loss += (mse_loss.item() - mean_mse_loss) / (step + 1)
      mean_ssim_loss += (ssim_loss.item() - mean_ssim_loss) / (step + 1)
      mean_rec_loss += (rec_loss.item() - mean_rec_loss) / (step + 1)
      mean_nsp_loss += (nsp_loss.item() - mean_nsp_loss) / (step + 1)

    if tf_logger is not None:
      current_emb = self.model.get_embedding(batch[image_key][:8])
      partial_rec = self.model.ae.decode(current_emb)
      tf_logger.add_images('cnnae_reconstructed_images_eval', torch.cat([batch[image_key][:8],
                                                                         rec[:8],
                                                                         partial_rec,
                                                                         target[:8]], dim=0))
      tf_logger.add_scalar('cnnae_rec_eval_loss', mean_rec_loss, step)
      tf_logger.add_scalar('cnnae_nsp_eval_loss', mean_nsp_loss, step)
    
    return mean_mse_loss, mean_ssim_loss, mean_rec_loss, mean_nsp_loss


class MemoryBankGoalEmbeddingPredictorTrainer(BaseTrainer):
  CONFIG = {
    'experiment_name': 'memory_bank_goal_predictor',
    'model_name':      'MemoryBankGoalPredictor',
    'model_config':    {'embed_dim': 128,
                        'num_heads': 8,
                        'num_layers': 2,
                        'hidden_dim': 256,
                        'max_memory_size': 5}
  }
  def __init__(self, config={}):
    super().__init__(config={**MemoryBankGoalEmbeddingPredictorTrainer.CONFIG, **config})
  
  def instanciate_model(self):
    self.model = MemoryBankFuturStatePredictor(**self.config['model_config']).to(self.device)
  
  def train(self, buffer, memory_bank, n_max_steps=100, batch_size=32, tf_logger=None, memory_size=5,
            start_step=0):
    self.model.train()

    mean_loss = 0.0

    BM = batch_size + memory_size
    memory_ep = memory_bank.sample_episode_batch(memory_size, 60, random_window=False, success_reward=10)
    ep_sizes = memory_ep['episode_size']
    memory = memory_ep['image_embedding'][:, :ep_sizes.max()].unsqueeze(0).repeat(BM, 1, 1, 1)
    memory_ep_len = ep_sizes.unsqueeze(0).repeat(BM, 1)  # [1, M]
    memory_goal = memory_ep['image_embedding'][torch.arange(memory_size), ep_sizes-1]  # [M, E]

    pbar = tqdm(range(n_max_steps))
    for step in pbar:
      batch = buffer.sample_image_is_goal_batch(batch_size)
      memory_batch = memory_bank.sample_image_is_goal_batch(memory_size)

      current_state = torch.cat([batch['image_embedding'], memory_batch['image_embedding']], dim=0)  # [B+M, E]

      predicted_goal = self.model(current_state, memory, memory_ep_len)  # [B+M, E]

      batch_goal = buffer.other_stored_obj['image_embedding'][batch['goal_idx']].to(self.device)  # [B, E]
      target_goal = torch.cat([batch_goal, memory_goal], dim=0)  # [B+M, E]

      loss = F.mse_loss(predicted_goal, target_goal)
      self.mdl_opt.zero_grad()
      loss.backward()
      self.mdl_opt.step()

      mean_loss += (loss.item() - mean_loss) / (step + 1)

      if tf_logger:
        tf_logger.add_scalar('memory_bank_goal_predictor_loss', mean_loss, start_step + step)
      
      pbar.set_description(f'Loss: {mean_loss:.4f}')
    
    return mean_loss

  def train_goal_prediction(self, buffer, memory_bank, n_training_episodes=100, batch_size=32, tf_logger=None,
                            memory_size=5, start_step=0):
    self.model.train()
    
    # --- 1. Prepare the FIXED CONTEXT (The Demonstrations) ---
    # These are the M episodes the Transformer will attend to for every prediction.
    memory_ep = memory_bank.sample_episode_batch(memory_size, 60, random_window=False, success_reward=10)
    mem_imgs = memory_ep['image_embedding'] # [M, E, D]
    mem_lens = memory_ep['episode_size']    # [M]

    # --- 2. Prepare TRAINING DATA (The 20/80 Split) ---
    # Generalization: Brand new episodes from the buffer (n_training_episodes)
    gen_ep = buffer.sample_episode_batch(n_training_episodes, 60, random_window=False, success_reward=10)
    
    # Retrieval: The EXACT same episodes used in the memory context (memory_size)
    # This allows the model to learn perfect retrieval for known context.
    ret_ep = memory_ep 
    
    all_states = []
    all_goals = []
    
    # Process both sets into a flat list of (state, goal) pairs
    for ep_set in [gen_ep, ret_ep]:
      imgs = ep_set['image_embedding'] # [N_eps, Max_T, D]
      lens = ep_set['episode_size']    # [N_eps]
      
      for i in range(imgs.shape[0]):
        length = lens[i].item()
        # Goal is the final state of the trajectory
        goal = imgs[i, length - 1] 
        # States are all valid frames in this trajectory
        states = imgs[i, :length] 
        
        all_states.append(states)
        # Repeat goal to match the number of states in this episode
        all_goals.append(goal.unsqueeze(0).repeat(length, 1))

    # Flatten into tensors for the "shuffled deck"
    train_x = torch.cat(all_states, dim=0)
    train_y = torch.cat(all_goals, dim=0)
    
    num_total_states = train_x.size(0)
    indices = torch.randperm(num_total_states)
    num_batches = num_total_states // batch_size
    
    mean_loss = 0.0

    # --- 3. Training Loop ---
    pbar = tqdm(range(num_batches), leave=False)
    for i in pbar:
      batch_idx = indices[i * batch_size : (i + 1) * batch_size]
      curr_states = train_x[batch_idx]   # [B, D]
      target_goals = train_y[batch_idx]  # [B, D]
      
      # Expand the fixed context to match the batch size
      curr_batch_size = curr_states.size(0)
      # Using .expand() is O(1) memory - it just creates new strides for the same data
      expanded_mem = mem_imgs.unsqueeze(0).expand(curr_batch_size, -1, -1, -1)
      expanded_mem_lens = mem_lens.unsqueeze(0).expand(curr_batch_size, -1)

      # Predict the goal embedding
      # Transformer attends to expanded_mem based on the query curr_states
      predicted_goals = self.model(curr_states, expanded_mem, expanded_mem_lens)
      
      # Mean Squared Error Loss in embedding space
      loss = F.mse_loss(predicted_goals, target_goals)
      
      self.mdl_opt.zero_grad()
      loss.backward()
      self.mdl_opt.step()

      # Statistics
      mean_loss += (loss.item() - mean_loss) / (i + 1)
      if tf_logger:
        tf_logger.add_scalar('mbgep_train_loss', mean_loss, start_step + i)
      
      pbar.set_description(f'Loss: {mean_loss:.4f} | Total States: {num_total_states}')

    return mean_loss
  
  @torch.no_grad()
  def evaluate(self, buffer, memory_bank, batch_size=10, tf_logger=None, memory_size=5, step=0):
    self.model.eval()

    memory_ep = memory_bank.sample_episode_batch(memory_size, 60, random_window=False, success_reward=10)
    ep_sizes = memory_ep['episode_size']
    memory = memory_ep['image_embedding'][:, :ep_sizes.max()].unsqueeze(0).repeat(batch_size, 1, 1, 1)
    memory_ep_len = ep_sizes.unsqueeze(0).repeat(batch_size, 1)  # [1, M]
    
    batch = buffer.sample_image_is_goal_batch(batch_size)

    predicted_goal = self.model(batch['image_embedding'], memory, memory_ep_len)

    target_goal = buffer.other_stored_obj['image_embedding'][batch['goal_idx']].to(self.device)

    loss = F.mse_loss(predicted_goal, target_goal)

    if tf_logger:
      tf_logger.add_scalar('memory_bank_goal_predictor_eval_loss', loss.item(), step)
    
    return loss.item()
  
  @torch.no_grad()
  def evaluate_goal_prediction(self, buffer, memory_bank, n_trajectories=10, memory_size=5,
                               step=0, tf_logger=None):
    self.model.eval()

    # --- 1. Prepare the Fixed Memory (The Task Examples) ---
    # These 5 episodes are the "context" for all predictions in this eval call.
    memory_ep = memory_bank.sample_episode_batch(memory_size, 60, random_window=False, success_reward=10)
    mem_imgs = memory_ep['image_embedding']  # [M, E, D]
    mem_lens = memory_ep['episode_size']     # [M]
    # Goals of these memory episodes (for the retrieval test later)
    assert (mem_lens > 0).all(), "Empty episodes found in memory bank"
    mem_goals = mem_imgs[torch.arange(memory_size), mem_lens - 1]

    # --- 2. Evaluate GENERALIZATION (80% case) ---
    # Sample full trajectories from the test buffer (episodes the model hasn't seen)
    eval_ep = buffer.sample_episode_batch(n_trajectories, 60, random_window=False, success_reward=10)
    traj_states = eval_ep['image_embedding'] # [B, T, D]
    traj_lens = eval_ep['episode_size']      # [B]
    B, T, D = traj_states.shape
    
    # The true goal of each test trajectory
    test_goals = traj_states[torch.arange(B), traj_lens - 1] # [B, D]

    # Prepare inputs: Repeat the 5 memory episodes for every step in the test trajectories
    # flat_states: [B * T, D]
    flat_states = traj_states.reshape(B * T, D)
    # expanded_mem: [B * T, M, E, D]
    expanded_mem = mem_imgs.unsqueeze(0).repeat(B * T, 1, 1, 1)
    expanded_lens = mem_lens.unsqueeze(0).repeat(B * T, 1)

    preds_gen = self.model(flat_states, expanded_mem, expanded_lens).view(B, T, D)

    # --- 3. Evaluate RETRIEVAL (20% case) ---
    # Here, the current_states come from the memory_bank itself.
    # We'll just pass all states from the 5 memory episodes through the model.
    M, ME, _ = mem_imgs.shape
    flat_mem_states = mem_imgs.reshape(M * ME, D)
    # Context is still the same 5 episodes
    context_mem = mem_imgs.unsqueeze(0).repeat(M * ME, 1, 1, 1)
    context_lens = mem_lens.unsqueeze(0).repeat(M * ME, 1)
    
    preds_ret = self.model(flat_mem_states, context_mem, context_lens).view(M, ME, D)

    # --- 4. Calculate Advantage & Error Metrics ---
    
    # A. Generalization Metrics (Buffer)
    test_goals_exp = test_goals.unsqueeze(1).expand(-1, T, -1)
    mask_gen = torch.arange(T, device=self.device).unsqueeze(0) < traj_lens.unsqueeze(1)
    
    dist_pred_gen = torch.norm(preds_gen - test_goals_exp, p=2, dim=-1)[mask_gen]
    dist_state_gen = torch.norm(traj_states - test_goals_exp, p=2, dim=-1)[mask_gen]
    
    err_gen = dist_pred_gen.mean().item()
    adv_gen = (dist_state_gen - dist_pred_gen).mean().item()

    # B. Retrieval Metrics (Memory)
    mem_goals_exp = mem_goals.unsqueeze(1).expand(-1, ME, -1)
    mask_ret = torch.arange(ME, device=self.device).unsqueeze(0) < mem_lens.unsqueeze(1)
    
    dist_pred_ret = torch.norm(preds_ret - mem_goals_exp, p=2, dim=-1)[mask_ret]
    dist_state_ret = torch.norm(mem_imgs - mem_goals_exp, p=2, dim=-1)[mask_ret]
    
    err_ret = dist_pred_ret.mean().item()
    adv_ret = (dist_state_ret - dist_pred_ret).mean().item()

    # --- 5. Logging ---
    if tf_logger:
      tf_logger.add_scalar('mbgep_eval_generalization_error', err_gen, step)
      tf_logger.add_scalar('mbgep_eval_generalization_advantage', adv_gen, step)
      tf_logger.add_scalar('mbgep_eval_retrieval_error', err_ret, step)
      tf_logger.add_scalar('mbgep_eval_retrieval_advantage', adv_ret, step)
    
    return err_gen, adv_gen, err_ret, adv_ret


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


class FlowImagePredictorTrainer(BaseTrainer):
  CONFIG = {
    'experiment_name': 'flow_image_predictor',
    'model_name':      'WorldModelFlowUnet',
    'model_config': {
      'img_chan':   3,
      'time_dim':   64,
      'add_action': False
    },
  }
  def __init__(self, config={}):
    super().__init__(config={**FlowImagePredictorTrainer.CONFIG, **config})
  
  def instanciate_model(self):
    self.model = WorldModelFlowUnet(**self.config['model_config']).to(self.device)
  
  def train(self, get_data_fn, n_max_steps=500, batch_size=128, tf_logger=None,
            target_img_key='target_image', condition_key=None, generate_every=20, n_gen_steps=2):
    print('FlowImagePredictor training pass...')
    self.model.train()

    mean_loss = 0.0

    pbar = tqdm(range(n_max_steps))
    for step in pbar:
      batch = get_data_fn(batch_size)
      x1 = batch[target_img_key]  # target distribution is the image to predict
      if condition_key is not None:
        condition = {condition_key: batch[condition_key]}

      loss = flow_matching_loss(self.model, x1, condition=condition)

      self.mdl_opt.zero_grad()
      loss.backward()
      self.mdl_opt.step()

      mean_loss += (loss.item() - mean_loss) / (step + 1)
      
      if tf_logger:
        tf_logger.add_scalar('flow_image_predictor_loss', mean_loss, step)
        
        if step % generate_every == 0:
          x1_pred = rk45_sampling(
            self.model,
            device=self.device,
            n_samples=x1.shape[0],
            condition=condition,
            n_steps=n_gen_steps
          )
          x1_pred = x1_pred[-1]

          img_comparison = torch.cat([x1[:8], x1_pred[:8]], dim=0)

          tf_logger.add_scalar('flow_image_predictor_rec_loss', F.mse_loss(x1_pred, x1), step)
          tf_logger.add_images('generated_flow_image_train', img_comparison, global_step=step)

      pbar.set_description(f'Loss: {mean_loss:.6f}')
    return mean_loss
  

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
          tf_logger.add_images('generated_flow_goal_image_train', img_comparison, global_step=step)

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
    self.model = ISPredictor(layer_norm_predictor=True).to(self.device)
  
  def train(self, get_data_fn, n_max_steps=20_000, batch_size=128, tf_logger=None,
            internal_state_key='goal_internal_state', image_key='goal_image_generated',
            margin=1.0, hinge_weight=0.0, augment=False, label_smoothing=0.0):
    print('Internal State Predictor training pass...')
    self.model.train()

    if augment:
      train_aug = [
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomGrayscale(),
        T.RandomPerspective(),
        T.RandomRotation(degrees=45),
        T.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.1)]),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
      ]

    mean_loss = 0.0
    acc1_window, acc2_window = deque([0.0], maxlen=50), deque([0.0], maxlen=50)

    pbar = tqdm(range(n_max_steps))
    for step in pbar:
      batch = get_data_fn(batch_size)

      if augment:
        aug_batch = (batch[image_key].clone() + 1) / 2  # Convert from [-1, 1] to [0, 1]
        part_size = batch_size // len(train_aug)
        i = 0
        for aug_fn in train_aug:
          aug_batch[i:i+part_size] = aug_fn(aug_batch[i:i+part_size])
          i += part_size
        aug_batch = aug_batch * 2 - 1  # Convert back to [-1, 1]
        batch[image_key] = torch.cat([batch[image_key], aug_batch], dim=0)
        batch[internal_state_key] = torch.cat([batch[internal_state_key], batch[internal_state_key]], dim=0)

      # Infer internal_state from replay_buffer images
      g1_logits, g2_logits = self.model(batch[image_key])

      ce_loss = (
        F.cross_entropy(g1_logits, batch[internal_state_key][:, 0], label_smoothing=label_smoothing) +
        F.cross_entropy(g2_logits, batch[internal_state_key][:, 1], label_smoothing=label_smoothing)
      )

      # ----------------------------------
      # 2) Hinge Loss (Contrastive in-batch)
      # ----------------------------------
      hinge_loss = 0.0
      if hinge_weight != 0.0:
        pred = torch.cat([g1_logits, g2_logits], dim=1)  # [B, 56]
        # Create one-hot targets
        toh1 = F.one_hot(batch[internal_state_key][:, 0], num_classes=g1_logits.shape[1]).float()  # [B, 19]
        toh2 = F.one_hot(batch[internal_state_key][:, 1], num_classes=g2_logits.shape[1]).float()  # [B, 37]
        targets = torch.cat([toh1, toh2], dim=1)  # [B, 56]
        # Similarity matrix
        logits_sim = torch.matmul(pred, targets.T)
        # Positive scores (diagonal)
        pos_scores = torch.diag(logits_sim)
        # Negative scores (off-diagonal)
        mask = torch.eye(logits_sim.size(0), device=self.device).bool()
        neg_scores = logits_sim[~mask].view(logits_sim.size(0), -1)
        # Hinge loss
        hinge_loss = torch.relu(margin - (pos_scores.unsqueeze(1) - neg_scores)).mean()

      # Total loss (CE + Hinge) and backward
      loss = ce_loss + hinge_weight * hinge_loss
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


class EmbeddingOptimizerTrainer(BaseTrainer):
  '''Optimizes a model that takes encoder embeddings and produces diverse goal embeddings.
  
  The model takes detached encoder embeddings as input and outputs n diverse embeddings.
  Each output embedding reconstructs well through the frozen decoder, but all n embeddings
  for a given input are maximally diverse.
  '''
  CONFIG = {
    'experiment_name':      'embedding_optimizer',
    'model_name':           'EmbeddingGenerator',
    'lr':                   1e-3,
    'weight_decay':         0.0,
    'latent_dim':           128,
    'n_goal_embeddings':    8,        # Number of goal embeddings to generate per input
    'lambda_rec':           1.0,      # Reconstruction loss weight
    'lambda_diversity':     0.1,      # Diversity loss weight
    'lambda_ssim':          0.0,      # SSIM loss weight (0 to disable)
    'hidden_dim':           256,      # Hidden dimension for the generator model
  }
  def __init__(self, autoencoder, config={}):
    '''
    Args:
      autoencoder: trained CNNAE model (encoder will be used for input, decoder frozen)
      config: configuration dict
    '''
    self.autoencoder = autoencoder.eval()
    # Freeze decoder weights (encoder will be used but gradients detached)
    for param in self.autoencoder.parameters():
      param.requires_grad = False
    
    super().__init__(config={**EmbeddingOptimizerTrainer.CONFIG, **config})
  
  def instanciate_model(self):
    # Model that takes encoder embedding and outputs n goal embeddings
    self.model = nn.Sequential(
      nn.Linear(self.config['latent_dim'], self.config['hidden_dim']),
      nn.ReLU(),
      nn.Linear(self.config['hidden_dim'], self.config['hidden_dim']),
      nn.ReLU(),
      nn.Linear(self.config['hidden_dim'], self.config['n_goal_embeddings'] * self.config['latent_dim']),
      nn.Unflatten(-1, (self.config['n_goal_embeddings'], self.config['latent_dim']))
    ).to(self.device)
  
  def _compute_cosine_diversity_loss(self, embeddings):
    '''Compute loss that maximizes diversity between embeddings.
    
    Args:
      embeddings: [batch_size, n_embeddings, latent_dim]
    
    Returns:
      loss: scalar, lower is more diverse
    '''
    # Normalize embeddings
    emb_normalized = F.normalize(embeddings, dim=-1)  # [B, n, d]
    
    # Compute pairwise cosine similarity for each batch element
    sim_matrices = torch.matmul(emb_normalized, emb_normalized.transpose(-2, -1))  # [B, n, n]
    
    # Zero out diagonal (self-similarity)
    batch_size, n_emb = embeddings.shape[0], embeddings.shape[1]
    eye = torch.eye(n_emb, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
    sim_matrices = sim_matrices * (1 - eye)
    
    # Return mean of absolute similarity values
    loss = torch.abs(sim_matrices).mean()
    
    return loss
  
  def _reconstruct(self, embeddings):
    '''Reconstruct images from embeddings using frozen decoder.
    
    Args:
      embeddings: [batch_size * n_embeddings, latent_dim]
    
    Returns:
      reconstructed: [batch_size * n_embeddings, 3, 32, 32]
    '''
    with torch.no_grad():
      batch_size_n = embeddings.shape[0]
      
      if self.autoencoder.config['linear_bottleneck']:
        # Need to reshape embeddings back to spatial form
        latent_spatial = self.autoencoder.fc_dec(embeddings)  # -> [B*n, 256*4*4]
        latent_spatial = latent_spatial.view(batch_size_n, 256, 4, 4)
      else:
        latent_spatial = embeddings.view(batch_size_n, 256, 4, 4)
      
      # Pass through decoder
      reconstructed = self.autoencoder.up(
        latent_spatial,
        None,  # No skip connection for now
        None
      )
    
    return reconstructed
  
  def train(self, get_data_fn, n_max_steps=500, batch_size=32, tf_logger=None,
            image_key='image', target_image_key='goal_image'):
    '''Train the embedding generator model.
    
    Args:
      get_data_fn: function that returns dict with image_key and target_image_key
      n_max_steps: number of training steps
      batch_size: batch size
      tf_logger: optional tensorboard logger
      image_key: key for current images in batch
      target_image_key: key for target images in batch
    '''
    print('Embedding Generator training pass...')
    self.model.train()
    
    if self.config['lambda_ssim'] > 0:
      ssim_loss_fn = SSIM(data_range=1, size_average=True, channel=3)
    
    mean_rec_loss = 0.0
    mean_diversity_loss = 0.0
    mean_total_loss = 0.0
    
    pbar = tqdm(range(n_max_steps))
    for step in pbar:
      batch = get_data_fn(batch_size)
      current_images = batch[image_key]  # [B, 3, H, W]
      goal_images = batch[target_image_key]  # [B, 3, H, W]
      
      # Get encoder embeddings (detach to prevent gradient flow to encoder)
      with torch.no_grad():
        if self.autoencoder.config['linear_bottleneck']:
          encoder_embeddings = self.autoencoder.down(current_images)  # Get d3
          encoder_embeddings = encoder_embeddings[-1]  # Last output is d3
          encoder_embeddings = self.autoencoder.embedder(encoder_embeddings.flatten(1))  # [B, latent_dim]
        else:
          _, _, d3 = self.autoencoder.down(current_images)
          encoder_embeddings = d3.flatten(1)  # [B, 256*4*4], but we need to handle this
      
      # Generate goal embeddings
      goal_embeddings = self.model(encoder_embeddings.detach())  # [B, n_goal_embeddings, latent_dim]
      
      # Flatten for reconstruction
      goal_embeddings_flat = goal_embeddings.view(-1, self.config['latent_dim'])  # [B*n, latent_dim]
      
      # Reconstruct images from goal embeddings
      reconstructed = self._reconstruct(goal_embeddings_flat)  # [B*n, 3, H, W]
      
      # Tile goal images to match reconstructed batch
      goal_images_tiled = goal_images.unsqueeze(1).expand(-1, self.config['n_goal_embeddings'], -1, -1, -1)
      goal_images_tiled = goal_images_tiled.reshape(-1, *goal_images.shape[1:])  # [B*n, 3, H, W]
      
      # Reconstruction loss
      rec_loss = F.mse_loss(reconstructed, goal_images_tiled)
      if self.config['lambda_ssim'] > 0:
        ssim_loss_val = 1 - ssim_loss_fn(reconstructed, goal_images_tiled)
        rec_loss = rec_loss + self.config['lambda_ssim'] * ssim_loss_val
      
      # Diversity loss (goal embeddings should be diverse for each input)
      div_loss = self._compute_cosine_diversity_loss(goal_embeddings)
      
      # Total loss
      total_loss = (self.config['lambda_rec'] * rec_loss + 
                    self.config['lambda_diversity'] * div_loss)
      
      # Optimization step
      self.mdl_opt.zero_grad()
      total_loss.backward()
      self.mdl_opt.step()
      
      mean_rec_loss += (rec_loss.item() - mean_rec_loss) / (step + 1)
      mean_diversity_loss += (div_loss.item() - mean_diversity_loss) / (step + 1)
      mean_total_loss += (total_loss.item() - mean_total_loss) / (step + 1)
      
      if tf_logger:
        tf_logger.add_scalar('emb_gen_rec_loss', rec_loss.item(), step)
        tf_logger.add_scalar('emb_gen_diversity_loss', div_loss.item(), step)
        tf_logger.add_scalar('emb_gen_total_loss', total_loss.item(), step)
      
      pbar.set_description(f'Loss: {mean_total_loss:.4f}')
    
    return mean_rec_loss, mean_diversity_loss, mean_total_loss
  
  @torch.no_grad()
  def evaluate(self, get_data_fn, n_max_steps=1, batch_size=8, tf_logger=None,
               image_key='image', target_image_key='goal_image'):
    '''Evaluate the embedding generator on a dataset.
    
    Args:
      get_data_fn: function that returns dict with image_key and target_image_key
      n_max_steps: number of evaluation steps
      batch_size: batch size
      tf_logger: optional tensorboard logger
      image_key: key for current images in batch
      target_image_key: key for target images in batch
    
    Returns:
      mean_rec_loss, mean_diversity_loss, mean_total_loss
    '''
    self.model.eval()
    self.autoencoder.eval()

    if self.config['lambda_ssim'] > 0:
      ssim_loss_fn = SSIM(data_range=1, size_average=True, channel=3)
    
    mean_rec_loss = 0.0
    mean_diversity_loss = 0.0
    mean_total_loss = 0.0

    pbar = tqdm(range(n_max_steps))
    for step in pbar:
      batch = get_data_fn(batch_size)
      current_images = batch[image_key]  # [B, 3, H, W]
      goal_images = batch[target_image_key]  # [B, 3, H, W]
      
      # Get encoder embeddings (detached)
      with torch.no_grad():
        if self.autoencoder.config['linear_bottleneck']:
          encoder_embeddings = self.autoencoder.down(current_images)
          encoder_embeddings = encoder_embeddings[-1]  # d3
          encoder_embeddings = self.autoencoder.embedder(encoder_embeddings.flatten(1))
        else:
          _, _, d3 = self.autoencoder.down(current_images)
          encoder_embeddings = d3.flatten(1)
      
      # Generate goal embeddings
      goal_embeddings = self.model(encoder_embeddings)  # [B, n_goal_embeddings, latent_dim]
      
      # Flatten for reconstruction
      goal_embeddings_flat = goal_embeddings.view(-1, self.config['latent_dim'])
      
      # Reconstruct images from goal embeddings
      reconstructed = self._reconstruct(goal_embeddings_flat)  # [B*n, 3, H, W]
      
      # Tile goal images to match reconstructed batch
      goal_images_tiled = goal_images.unsqueeze(1).expand(-1, self.config['n_goal_embeddings'], -1, -1, -1)
      goal_images_tiled = goal_images_tiled.reshape(-1, *goal_images.shape[1:])
      
      # Reconstruction loss
      rec_loss = F.mse_loss(reconstructed, goal_images_tiled)
      if self.config['lambda_ssim'] > 0:
        ssim_loss_val = 1 - ssim_loss_fn(reconstructed, goal_images_tiled)
        rec_loss = rec_loss + self.config['lambda_ssim'] * ssim_loss_val
      
      # Diversity loss
      div_loss = self._compute_cosine_diversity_loss(goal_embeddings)
      
      # Total loss
      total_loss = (self.config['lambda_rec'] * rec_loss + 
                    self.config['lambda_diversity'] * div_loss)
      
      mean_rec_loss += (rec_loss.item() - mean_rec_loss) / (step + 1)
      mean_diversity_loss += (div_loss.item() - mean_diversity_loss) / (step + 1)
      mean_total_loss += (total_loss.item() - mean_total_loss) / (step + 1)

    if tf_logger is not None:
      # Log some example reconstructions
      tf_logger.add_images('emb_gen_reconstructed_images_eval', 
                          torch.cat([current_images[:8], goal_images[:8], reconstructed[:8]], dim=0))

    return mean_rec_loss, mean_diversity_loss, mean_total_loss
  
  @torch.no_grad()
  def infer(self, current_images):
    '''Generate diverse goal embeddings from current images.
    
    Args:
      current_images: [batch_size, 3, H, W] current state images
    
    Returns:
      goal_embeddings: [batch_size, n_goal_embeddings, latent_dim]
      reconstructed_goals: [batch_size, n_goal_embeddings, 3, H, W]
    '''
    self.model.eval()
    self.autoencoder.eval()
    
    # Get encoder embeddings
    with torch.no_grad():
      if self.autoencoder.config['linear_bottleneck']:
        encoder_embeddings = self.autoencoder.down(current_images)
        encoder_embeddings = encoder_embeddings[-1]  # d3
        encoder_embeddings = self.autoencoder.embedder(encoder_embeddings.flatten(1))
      else:
        _, _, d3 = self.autoencoder.down(current_images)
        encoder_embeddings = d3.flatten(1)
    
    # Generate goal embeddings
    goal_embeddings = self.model(encoder_embeddings)
    
    # Reconstruct goal images
    goal_embeddings_flat = goal_embeddings.view(-1, self.config['latent_dim'])
    reconstructed_goals_flat = self._reconstruct(goal_embeddings_flat)
    reconstructed_goals = reconstructed_goals_flat.view(
      current_images.shape[0], self.config['n_goal_embeddings'], *reconstructed_goals_flat.shape[1:]
    )
    
    return goal_embeddings, reconstructed_goals


if __name__ == '__main__':
  ggp = GANGoalImagePredictorTrainer()
  # --- Dummy test --- #
  # from replay_buffer import ReplayBuffer
  # ic = CNNAETrainer()

  # rb = ReplayBuffer(internal_state_dim=2, action_dim=1, image_size=256, image_chan=3, resize_to=32,
  #                   normalize_img=True, capacity=10, device='cpu', target_device=ic.device)
  # for _ in range(10):
  #   rb.add([0, 0], [0], torch.rand(3, 32, 32), 0, False, [0, 0], torch.rand(3, 32, 32))
  
  # losses = ic.train(get_data_fn=lambda x: rb.sample(x), n_max_steps=2, batch_size=3,
  #                   target_image_key='next_image')
  # print(losses)

  # ic.evaluate(rb.image, rb.next_image, batch_size=3)