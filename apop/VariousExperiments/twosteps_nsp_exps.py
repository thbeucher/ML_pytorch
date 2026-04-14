"""
Two-Step Next State Predictor Experiments Script

This script implements a hierarchical two-step approach for predicting next states in a reinforcement learning environment.
It uses two specialized models to efficiently model state transitions:

1. AlteredPredictor: A Vision Transformer (ViT) that predicts which image patches are likely to change
   given the current image and a sequence of actions. It outputs a binary mask indicating altered patches.

2. AlterationPredictor: A Transformer model that predicts the exact pixel-level changes (alterations)
   for the patches identified as changing by the AlteredPredictor.

The system is trained end-to-end using a replay buffer filled with transitions from a RobotArmEnv.
The two-step process allows for efficient prediction by focusing computation on relevant image regions.

Key Features:
- Patch-based processing with ViT for spatial understanding.
- Sequence-aware action conditioning.
- Training on image deltas (changes) for stability.
- Evaluation through imagined trajectory generation.
- TensorBoard logging for monitoring training progress.

Environment: Custom RobotArmEnv from gymnasium.
Training: Supervised learning on collected transitions, with separate optimizers for each predictor.
"""
import os
import sys
import json
import torch
import random
import logging
import warnings
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm
from typing import Callable, Optional, Tuple
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from helpers_zoo import create_gif_from_images
from models_zoo import ISPredictorFromPatchIndex, ObjectPredictor, AlteredPredictor, AlterationPredictor

sys.path.append('../../../robot/')
from gymnasium_env.envs.robot_arm import forward_kinematics
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Object Finding Helper ---
def find_object_center(
    frame: np.ndarray,
    color_condition: Callable[[np.ndarray], np.ndarray]) -> Optional[Tuple[float, float]]:
  """
  Finds the geometric center (mean) of pixels that match a given color condition.
  """
  pixels = np.where(color_condition(frame))
  if pixels[0].size > 0:
    y_coords, x_coords = pixels
    center_y, center_x = np.mean(y_coords), np.mean(x_coords)
    return center_x, center_y
  return None

def find_patches_with_color(
  patches: np.ndarray,
  color: str,
) -> list[int]:
  """
  Takes patches and a color, and returns all patch indices containing the specified color.
  """
  hand_condition = lambda frame: (frame[:, :, 2] > frame[:, :, 0]) & (frame[:, :, 2] > frame[:, :, 1]) & (frame[:, :, 2] > 0.1)
  target_condition = lambda frame: (frame[:, :, 0] > frame[:, :, 1]) & (frame[:, :, 0] > frame[:, :, 2]) & (frame[:, :, 0] > 0.1)
  if color == 'blue':
    color_condition = hand_condition
  elif color == 'red':
    color_condition = target_condition
  else:
    return []

  present_patches_indices = []
  for i, patch in enumerate(patches):
    if color_condition(patch).any():
      present_patches_indices.append(i)
  return present_patches_indices


class NSPTrainer:  # NextStatePredictor
  """
  NSPTrainer: Trainer class for the Two-Step Next State Predictor system.

  This class manages the training and evaluation of the AlteredPredictor and AlterationPredictor models.
  It handles environment interaction, replay buffer management, model instantiation, optimization,
  and logging. The trainer fills a replay buffer with environment transitions, trains the models
  to predict next states hierarchically, and evaluates performance through imagined trajectory generation.
  Used for experiments in model-based reinforcement learning with efficient state prediction.

  TRAINING PIPELINE OVERVIEW:
  1. Initialize: Create models, optimizers, replay buffer
  2. Fill Memory: Collect random rollouts from environment
  3. Training Loop:
      - Sample batch of episodes from replay buffer
      - For each time step in episode:
        a) Pass current image to AlteredPredictor → get which patches will change
        b) Pass changed patches to AlterationPredictor → get pixel reconstructions
        c) Compute losses for both models
        d) Update model weights
      - Log metrics and generate evaluation trajectories
  4. Evaluation: Generate imagined trajectories to visualize predictions

  KEY FEATURES:
  - Two-model hierarchy for efficiency
  - Sequence conditioning: action sequences of varying lengths
  - Internal state tracking: robot maintains state across steps
  - Multi-objective loss: combination of reconstruction and state prediction
  - TensorBoard logging: monitor training progress
  """
  CONFIG = {'image_size':                        256,     # Original environment image size
            'resize_to':                         32,      # Resize to for memory efficiency (256 → 32)
            'image_chan':                        3,       # RGB
            'normalize_image':                   True,    # Normalize to [-1, 1]
            'time_dim':                          64,
            'internal_state_dim':                2,       # Robot state: [angle_joint1, angle_joint2]
            'internal_state_n_values':           (90//5+1, 180//5+1),  # max_angle / angle_step +1 for inclusive
            'action_dim':                        1,       # Single action per step
            'action_n_values':                   5,       # 5 possible actions (0-4)
            'action_emb':                        8,       # Action embedding dimension
            'n_train_episodes':                  128,
            'train_buffer_size':                 100*60,
            'n_test_episodes':                   10,
            'test_buffer_size':                  10*60,
            'replay_buffer_device':              'cpu',   # Store on CPU to save GPU memory
            'render_mode':                       'rgb_array',
            'n_epochs':                          100,
            'batch_size':                        128,
            'use_tf_logger':                     True,
            'save_dir':                          'NSP_experiments/',
            'log_dir':                           'runs/',
            'exp_name':                          'nsp_twosteps_big_rollout_actionIS_newPE_multiheadOP_wholeobject',
            'seed':                              42,
            'rollout_size':                      30,    # Max episode length for sampling
            'use_internal_state':                True,  # Include internal state in training
            'internal_state_emb_dim':            16,    # Internal state embedding dimension
            'add_hand_and_target_patch_indices': True, # Add hand and target patch indices to the replay buffer
            'patch_mask_ratio':                  0.,  # Ratio of patches to mask during training
            }
  def __init__(self, config={}):
    """
    Initialize the NSPTrainer.
    
    Args:
      config: dict of hyperparameters to override defaults
    """
    # Merge provided config with defaults
    self.config = {**NSPTrainer.CONFIG, **config}
    # === DEVICE SETUP ===
    # Try GPU first, then M1 Mac (MPS), fallback to CPU
    self.device = torch.device('cuda' if torch.cuda.is_available() else
                               'mps' if torch.backends.mps.is_available() else
                               'cpu')
    logger.info(f'Using device: {self.device}')

    # === IMAGE PROCESSING ===
    # Clamp and normalize: transforms [-1, 1] → [0, 1]
    if self.config['normalize_image']:
      self.clamp_denorm_fn = lambda x: (x.clamp(-1, 1) + 1) / 2
    
    # === UTILITY FUNCTIONS ===
    # Count trainable parameters in a model
    self.get_train_params = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)

     # === REPRODUCIBILITY ===
    self.set_seed()

    # === ENVIRONMENT ===
    # Create RobotArmEnv from gymnasium
    self.env = gym.make("gymnasium_env:RobotArmEnv", render_mode=self.config['render_mode'], cropping=True)

    # === TENSORBOARD LOGGING ===
    # Setup TensorBoard logger for monitoring
    save_dir_run = os.path.join(self.config['save_dir'], self.config['exp_name'], self.config['log_dir'])
    self.tf_logger = SummaryWriter(save_dir_run) if self.config['use_tf_logger'] else None

    self.instanciate_model()  # === MODEL SETUP ===
    self.set_trainer_utils()  # === OPTIMIZER & BUFFER SETUP ===
    self.dump_config()        # === SAVE CONFIG ===
  
  def dump_config(self):
    """Save configuration to JSON file for reproducibility."""
    with open(os.path.join(self.config['save_dir'],
                           self.config['exp_name'],
                           f"{self.config['exp_name']}_CONFIG.json"), 'w') as f:
      json.dump(self.config, f)
  
  def set_seed(self):
    """Set random seeds for reproducibility across PyTorch, NumPy, and Python."""
    torch.manual_seed(self.config['seed'])
    np.random.seed(self.config['seed'])
    random.seed(self.config['seed'])
    # For deterministic behavior on GPU (may be slower)
    if self.device.type == 'cuda':
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  
  def save_model(self, model_to_save, model_name):
    """
    Save model checkpoint to disk.
    
    Args:
      model_to_save: PyTorch model to save
      model_name: name for the checkpoint file
    """
    save_dir = os.path.join(self.config['save_dir'], self.config['exp_name'])
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}.pt")
    # Save as dictionary with model name as key
    torch.save({model_name: model_to_save.state_dict()}, path)

  def load_model(self, model_to_load, model_name):
    """
    Load model checkpoint from disk.
    
    Args:
      model_to_load: PyTorch model to load weights into
      model_name: name of checkpoint file
    """
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
    """
    Instantiate both AlteredPredictor and AlterationPredictor models.
    
    ARCHITECTURE CHOICES:
    - AlteredPredictor: Smaller (2 layers, 64 dim) for efficiency
      - Fast patch-level filtering
      - Lightweight computation
    
    - AlterationPredictor: Larger (8 layers, 256 dim) for quality
      - Detailed pixel reconstruction
      - Can afford to be expensive since only processes changed patches
    """
    # === ALTERED PREDICTOR (Step 1) ===
    # Lightweight model for identifying changed patches
    self.altered_predictor = AlteredPredictor(
      image_size=32,          # Input: 32x32 (resized from 256x256)
      patch_size=4,           # Patches: 4x4 → 8x8 grid = 64 patches
      dim=64,                 # Embedding dimension
      depth=2,                # 2 Transformer blocks
      heads=4,                # 4 attention heads
      mlp_dim=128,            # MLP dimension
      dim_head=32,            # Dimension per head
      channels=3,             # RGB
      is1_n_values=self.config['internal_state_n_values'][0],      # 19
      is2_n_values=self.config['internal_state_n_values'][1],      # 37
      is_emb_dim=self.config['internal_state_emb_dim']             # 16
    ).to(self.device)
    print(f'Instanciate altered_predictor with {self.get_train_params(self.altered_predictor):,} params')
    # === ALTERATION PREDICTOR (Step 2) ===
    # Heavyweight model for detailed pixel reconstruction
    self.alteration_predictor = AlterationPredictor(
      patch_dim=self.altered_predictor.patch_dim,    # 48 pixels
      dim=256,                # Larger embedding dimension
      depth=8,                # 8 Transformer blocks
      heads=12,               # 12 attention heads
      dim_head=64,            # Larger per-head dimension
      mlp_dim=1024,           # Larger MLP
      dropout=0.0,            # No dropout (small dataset)
      is1_n_values=self.config['internal_state_n_values'][0],      # 19
      is2_n_values=self.config['internal_state_n_values'][1],      # 37
      is_emb_dim=self.config['internal_state_emb_dim'],            # 16
      grid_h=self.altered_predictor.grid[0],  # 8 (from AlteredPredictor)
      grid_w=self.altered_predictor.grid[1]   # 8 (from AlteredPredictor)
    ).to(self.device)
    print(f'Instanciate alteration_predictor with {self.get_train_params(self.alteration_predictor):,} params')
    # === OBJECT PREDICTOR ===
    self.object_predictor = ObjectPredictor(
        image_size=32,
        patch_size=2,
        dim=128,
        depth=4,
        heads=8,
        mlp_dim=256,
        dim_head=64,
        channels=3,
        is1_n_values=self.config['internal_state_n_values'][0],
        is2_n_values=self.config['internal_state_n_values'][1],
        is_emb_dim=self.config['internal_state_emb_dim'],
        patch_mask_ratio=self.config['patch_mask_ratio'],
    ).to(self.device)
    print(f'Instanciate object_predictor with {self.get_train_params(self.object_predictor):,} params')
    self.is_predictor_from_patch_index = ISPredictorFromPatchIndex(
        patch_index_n_values=self.object_predictor.n_patchs,
        is1_n_values=self.config['internal_state_n_values'][0],
        is2_n_values=self.config['internal_state_n_values'][1],
    ).to(self.device)
    print(f'Instanciate is_predictor_from_patch_index with {self.get_train_params(self.is_predictor_from_patch_index):,} params')
  
  def set_trainer_utils(self):
    """Initialize replay buffer, optimizers, and loss functions."""
    # === REPLAY BUFFER ===
    # Stores environment transitions for training
    resize_img = True if self.config['resize_to'] != self.config['image_size'] else False
    self.train_buffer = ReplayBuffer(self.config['internal_state_dim'],
                                       self.config['action_dim'],
                                       self.config['image_size'],
                                       resize_to=self.config['resize_to'] if resize_img else None,
                                       normalize_img=self.config['normalize_image'],
                                       capacity=self.config['train_buffer_size'],
                                       device=self.config['replay_buffer_device'],
                                       target_device=self.device)
    self.test_buffer = ReplayBuffer(self.config['internal_state_dim'],
                                      self.config['action_dim'],
                                      self.config['image_size'],
                                      resize_to=self.config['resize_to'] if resize_img else None,
                                      normalize_img=self.config['normalize_image'],
                                      capacity=self.config['test_buffer_size'],
                                      device=self.config['replay_buffer_device'],
                                      target_device=self.device)
    # === OPTIMIZERS ===
    # Separate optimizers for each model
    self.altered_opt = torch.optim.AdamW(self.altered_predictor.parameters(), lr=1e-4,
                                         weight_decay=1e-4, betas=(0.9, 0.999))
    self.alteration_opt = torch.optim.AdamW(self.alteration_predictor.parameters(), lr=1e-4,
                                            weight_decay=1e-4, betas=(0.9, 0.999))
    self.object_predictor_opt = torch.optim.AdamW(self.object_predictor.parameters(), lr=1e-4,
                                                  weight_decay=1e-4, betas=(0.9, 0.999))
    self.is_predictor_from_patch_index_opt = torch.optim.AdamW(self.is_predictor_from_patch_index.parameters(), lr=1e-4,
                                                               weight_decay=1e-4, betas=(0.9, 0.999))
    
    # Robot arm constants for forward kinematics
    self.robot_arm_ori = tuple(self.env.unwrapped.config['arm_ori'])
    self.robot_link_size = self.env.unwrapped.config['link_size']
    self.robot_crop_box = self.env.unwrapped.crop_box

    # === LOSS FUNCTIONS ===
    self.bce_criterion = nn.BCELoss()  # Binary Cross-Entropy for change prediction (0/1 classification)
    self.mse_criterion = nn.MSELoss()  # Mean Squared Error for pixel reconstruction
    self.bce_with_logits_criterion = nn.BCEWithLogitsLoss()
    self.ce_criterion = nn.CrossEntropyLoss()
    self.kl_criterion = nn.KLDivLoss(reduction='batchmean')
      
  def fill_memory(self, replay_buffer, act='random', n_episodes=128, max_episode_steps=60):
    """
    Collects trajectories from the environment and populates the given replay buffer.

    This function drives the data collection process by interacting with the
    environment based on a specified action strategy. It records the transitions
    (state, action, reward, etc.) and stores them for later use in training.

    Args:
        replay_buffer: The replay buffer to fill.
        act (str): The action strategy to use. Can be:
                   'random' - Selects actions uniformly at random.
                   'best' - Follows a hardcoded "optimal" policy.
                   'policy' - (Placeholder) Uses a learned policy.
        n_episodes (int): The number of episodes to collect.
        max_episode_steps (int): The maximum number of steps per episode before
                                 a reset is forced.
    """
    print(f'Filling memory buffer... ({act=})')
    obs, _ = self.env.reset()
    img = self.env.render()

    for _ in tqdm(range(n_episodes)):
      episode_step = 0
      for _ in range(max_episode_steps):
        if act == 'policy':
          action = random.randint(0, 4)
        elif act == 'best':
          action = self.env.unwrapped.get_best_action()
        else:
          action = random.randint(0, 4)

        next_obs, reward, terminated, truncated, info = self.env.step(action)
        next_img = self.env.render()

        episode_step += 1

        replay_buffer.add(obs//5, action, img, reward, terminated or episode_step >= max_episode_steps,
                          next_obs//5, next_img)
        obs, img = next_obs, next_img

        if terminated or episode_step >= max_episode_steps:
          obs, _ = self.env.reset()
          img = self.env.render()
          episode_step = 0
          break
    
  def get_patch(self, patch, mask, return_indices=False):
    """
    Extract selected patches based on a boolean mask.

    FUNCTIONALITY:
    - Selects patches where mask[b, p] = True
    - Pads to maximum number of selected patches in batch
    - Optionally returns spatial indices of selected patches
    
    EXAMPLE:
    Input:  patch[B=2, Np=64, N=48], mask[B=2, Np=64]
    Batch 0: mask has 10 True values → 10 patches selected
    Batch 1: mask has 15 True values → 15 patches selected
    Output: padded[B=2, M=15, N=48] where M=max(10, 15)=15
    Batch 0 patches are at positions 0-9, positions 10-14 are zero-padded
    
    KEY ENHANCEMENT:
    When return_indices=True, also returns the spatial indices of selected patches.
    This allows AlterationPredictor to compute correct positional embeddings.

    Args:
      patch: [B, Np, N] - patches to select from
      mask:  [B, Np] (bool) - which patches to select
      return_indices: if True, also return the spatial indices of selected patches

    returns:
        padded: [B, M, N] where M is max number of selected patches
        counts: [B] (number of selected patches per batch)
        indices: [B, M] (spatial indices of selected patches, only if return_indices=True)
    """
    B, Np, N = patch.shape

    mask = mask.bool()        # Convert to boolean mask
    counts = mask.sum(dim=1)  # [B], count of selected patches per batch
    M = counts.max().item()   # Maximum number of selected patches across batch

    # Early exit if no patches are selected
    if M == 0:
      if return_indices:
        return patch.new_zeros((B, 0, N)), counts, patch.new_zeros((B, 0), dtype=torch.long)
      else:
        return patch.new_zeros((B, 0, N)), counts

    # === COMPUTE TARGET POSITIONS ===
    # For each selected patch, compute its position in output
    # Example: if patches [2, 5, 15] are selected, they get positions [0, 1, 2]
    # ranks: cumulative sum of mask - 1
    # [B, Np] - for each element, position in output array
    ranks = torch.cumsum(mask.to(torch.int64), dim=1) - 1  # [B, Np]

    # === FIND SELECTED PATCH INDICES ===
    # Where nonzero: get (batch_idx, patch_idx) of True values
    batch_idx, patch_idx = mask.nonzero(as_tuple=True)  # [total_selected]
    # === MAP TO TARGET POSITIONS ===
    # For each selected patch, where should it go in output
    # Map of selected patches to their target position in the output
    target_pos = ranks[batch_idx, patch_idx]          # [total_selected]

    # === EXTRACT SELECTED PATCHES ===
    # Get the actual patch values
    selected = patch[batch_idx, patch_idx]            # [total_selected, N]

    # === CREATE PADDED OUTPUT ===
    # Initialize zero-filled output
    padded = patch.new_zeros((B, M, N))
    # Place selected patches at target positions
    padded[batch_idx, target_pos] = selected

    # === OPTIONAL: RETURN INDICES ===
    if return_indices:
      # Create indices tensor and fill it with selected patch indices
      indices = patch.new_zeros((B, M), dtype=torch.long)
      indices[batch_idx, target_pos] = patch_idx
      return padded, counts, indices
    else:
      return padded, counts
  
  def add_hand_and_target_patch_indices_to_memory(self, buffer, batch_size=128):
    '''Add a variable to the replay buffer by generating it with the provided generator and
    adding it in batches.'''
    print(f'Add hand_patch_index and target_patch_index to buffer...')
    
    hand_patch_index = torch.zeros(buffer.capacity, self.object_predictor.n_patchs, dtype=torch.long)
    target_patch_index = torch.zeros(buffer.capacity, self.object_predictor.n_patchs, dtype=torch.long)

    for i in tqdm(range(0, buffer.size, batch_size)):
      images = buffer.image[i:i+batch_size]
      for j, img in enumerate(images):
        # The image is normalized, so we need to denormalize it
        img = (img + 1) / 2
        img_np = img.permute(1, 2, 0).cpu().numpy()

        patch_size = self.object_predictor.patch_height
        image_size = self.config['resize_to']
        num_patches_per_row = image_size // patch_size
        
        patches_np = []
        for r in range(num_patches_per_row):
            for c in range(num_patches_per_row):
                patch = img_np[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size, :]
                patches_np.append(patch)
        patches_np = np.array(patches_np)

        # --- Hand detection (Multi-label) ---
        hand_patches_indices = find_patches_with_color(patches_np, 'blue')
        if hand_patches_indices:
            multi_hot_hand = torch.zeros(self.object_predictor.n_patchs)
            multi_hot_hand[hand_patches_indices] = 1
            hand_patch_index[i+j] = multi_hot_hand

        # --- Target detection (Multi-label) ---
        target_patches_indices = find_patches_with_color(patches_np, 'red')
        if target_patches_indices:
            multi_hot_target = torch.zeros(self.object_predictor.n_patchs)
            
            # Exclude hand patches from target patches
            if hand_patches_indices:
                hand_patches_set = set(hand_patches_indices)
                target_patches_indices = [p for p in target_patches_indices if p not in hand_patches_set]

            multi_hot_target[target_patches_indices] = 1
            target_patch_index[i+j] = multi_hot_target
    
    buffer.add_variable(hand_patch_index, 'hand_patch_index')
    buffer.add_variable(target_patch_index, 'target_patch_index')

  def get_next_image(self, patch, patch_predicted, patch_mask):
    """
    Reconstruct full image by applying predicted changes to original patches.
        
    LOGIC:
    - Start with original patches
    - For positions where mask is True, replace with predicted patches
    - Result: image with changed regions updated, unchanged regions original
    
    EXAMPLE:
    Original image has patches [p0, p1, p2, ..., p63]
    Mask selects patches [2, 15, 42] (3 patches changed)
    Predictions provide 3 new patch values [p2', p15', p42']
    Result: [p0, p1, p2', ..., p42', ...]

    patch:           [B, Np, N]      original patches
    patch_predicted: [B, M, N]       predicted patches (only for selected ones)
    patch_mask:      [B, Np] (bool)  which patches were selected

    returns:
        patch_next:  [B, Np, N]
    """
    patch_mask = patch_mask.bool()

    # If no patches are predicted to change, return original image patches.
    if not patch_mask.any():
      return patch.clone()

    # === COMPUTE TARGET POSITIONS ===
    # Same logic as get_patch: which position in predicted array does each patch map to
    # [B, Np] - position in output array
    ranks = torch.cumsum(patch_mask.to(torch.int64), dim=1) - 1  # [B, Np]
    # Get indices of selected patches
    batch_idx, patch_idx = patch_mask.nonzero(as_tuple=True)
    target_pos = ranks[batch_idx, patch_idx]

    # === RECONSTRUCT IMAGE ===
    # Start with original patches
    patch_next = patch.clone()

    # Replace changed patches with predictions
    # patch_idx: where to put in full image
    # target_pos: where to get from predictions
    patch_next[batch_idx, patch_idx] = patch_predicted[batch_idx, target_pos]
    return patch_next
  
  def compute_altered_loss(self, patch, action, patch_next_image, internal_state=None, next_internal_state=None):
    """
    Compute loss for the AlteredPredictor (Step 1 model).
    
    LOSS COMPONENTS:
    1. Binary Cross-Entropy: predicted change probability vs. ground truth
    2. (Optional) Cross-Entropy: predicted internal states vs. ground truth
    
    GROUND TRUTH GENERATION:
    - Compare original and next patches
    - If L1 difference > threshold: patch changed
    - This creates binary labels for each patch
    
    Args:
      patch: [B, n_patchs, patch_dim] - original patches
      action: [B, n_actions_seq] - action sequence
      patch_next_image: [B, n_patchs, patch_dim] - next image patches
      internal_state: [B, 2] - current internal state
      next_internal_state: [B, 2] - next internal state
    
    Returns:
      altered_loss: scalar - total loss
      patch_change_pred_mask: [B, n_patchs, 1] bool - hard predictions (>0.5)
      altered_patch_target: [B, n_patchs, 1] bool - ground truth changed patches
    """
    # === FORWARD PASS ===
    # Get predictions from AlteredPredictor
    # patch_change_pred: [B, n_patchs, 1] - probability [0,1]
    # next_is*_logits: [B, is*_n_values] - internal state predictions
    patch_change_pred, next_is1_logits, next_is2_logits = self.altered_predictor(patch, action, internal_state)
    
    # === HARD THRESHOLDING ===
    # Convert soft probabilities to binary predictions
    patch_change_pred_mask = (patch_change_pred > 0.5)  # [B, n_patchs, 1]

    # === GROUND TRUTH GENERATION ===
    # Compute L1 norm of difference between patches
    # If difference > threshold, patch changed
    patch_diff = (patch - patch_next_image).abs().sum(dim=2, keepdim=True)
    altered_patch_target = patch_diff > 1e-3  # [B, n_patchs, 1]

    # === BCE LOSS ===
    # Binary cross-entropy between predictions and targets
    altered_loss = self.bce_criterion(patch_change_pred, altered_patch_target)
    
    # === OPTIONAL: INTERNAL STATE PREDICTION LOSS ===
    # If training with internal state, also predict next state
    if next_internal_state is not None:
      next_is1_loss = F.cross_entropy(next_is1_logits, next_internal_state[:, 0].long())
      next_is2_loss = F.cross_entropy(next_is2_logits, next_internal_state[:, 1].long())
      altered_loss += next_is1_loss + next_is2_loss
    
    return altered_loss, patch_change_pred_mask, altered_patch_target

  def compute_alteration_loss(self, patch, patch_next_image, patch_change_gt_mask, action, internal_state=None, next_internal_state=None):
    """
    Compute loss for the AlterationPredictor (Step 2 model).
    
    STRATEGY:
    1. Select only patches marked as changed by gt_mask
    2. Get spatial indices of selected patches
    3. Forward through AlterationPredictor with indices
    4. Compare predictions to ground truth changed patches
    5. Compute MSE loss for pixel reconstruction
    
    EFFICIENCY:
    - Only processes changed patches (subset of 64)
    - Computation proportional to number of changes, not total patches
    
    Args:
      patch: [B, n_patchs, patch_dim] - original patches
      patch_next_image: [B, n_patchs, patch_dim] - next image patches (ground truth)
      patch_change_gt_mask: [B, n_patchs] bool - which patches changed (from AlteredPredictor target)
      action: [B, n_actions_seq] - action sequence
      internal_state: [B, 2] - current internal state
      next_internal_state: [B, 2] - next internal state
    
    Returns:
      alteration_loss: scalar - total loss (0 if no patches changed)
      patch_next_image_predicted: [B, n_patchs, patch_dim] - reconstructed next image
      counts: [B] - number of changed patches per batch
    """
    # === SELECT CHANGED PATCHES ===
    # Use ground truth mask (from training labels) to select patches
    # Returns selected patches and their spatial indices
    # [B, M, patch_dim], [B], [B, M]
    patch_to_pred, counts, patch_indices = self.get_patch(patch, patch_change_gt_mask, return_indices=True)

    # === FORWARD PASS ===
    if counts.max() > 0:  # Only if there are patches to predict
      # Forward AlterationPredictor with spatial indices
      # patch_indices ensures correct positional embeddings
      patch_predicted, next_is1_logits, next_is2_logits = self.alteration_predictor(patch_to_pred.detach(),
                                                                                    patch_indices.detach(),
                                                                                    action.detach(),
                                                                                    internal_state.detach() if internal_state is not None else None)
      # === RECONSTRUCT FULL IMAGE ===
      # Put predictions back into full image grid
      # [B, n_patchs, patch_dim]
      patch_next_image_predicted = self.get_next_image(patch,
                                                       patch_predicted,
                                                       patch_change_gt_mask)

      # === EXTRACT GROUND TRUTH FOR CHANGED PATCHES ===
      # Get ground truth values for changed patches (for comparison)
      # [B, M, patch_dim]
      patch_target, _, _ = self.get_patch(patch_next_image, patch_change_gt_mask, return_indices=True)

      # === MSE LOSS ===
      # Compare predicted to ground truth pixels
      # Handle padding: only compare up to actual number of changed patches
      alteration_loss = self.mse_criterion(patch_predicted[:, :patch_target.size(1)], patch_target)
      
      # === OPTIONAL: INTERNAL STATE LOSS ===
      if next_internal_state is not None:
        next_is1_loss = F.cross_entropy(next_is1_logits, next_internal_state[:, 0].long())
        next_is2_loss = F.cross_entropy(next_is2_logits, next_internal_state[:, 1].long())
        alteration_loss += next_is1_loss + next_is2_loss
    else:
      # No patches changed: trivial case
      patch_next_image_predicted = patch
      alteration_loss = 0.0

    return alteration_loss, patch_next_image_predicted, counts

  def update_models(self, altered_loss, alteration_loss):
    """
    Update both models using their respective optimizers.

    Args:
        altered_loss: scalar - loss for AlteredPredictor
        alteration_loss: scalar - loss for AlterationPredictor (can be 0)
    """
    # === UPDATE ALTERED PREDICTOR ===
    self.altered_opt.zero_grad()  # Clear old gradients
    altered_loss.backward()       # Compute gradients
    self.altered_opt.step()       # Update weights

    # === UPDATE ALTERATION PREDICTOR ===
    self.alteration_opt.zero_grad()
    if alteration_loss:  # Only if loss is non-zero
      alteration_loss.backward()
    self.alteration_opt.step()

  def compute_metrics(self, patch_change_pred_mask, altered_patch_target, alteration_loss, counts):
    """
    Compute training metrics for logging.
    
    Args:
      patch_change_pred_mask: [B, n_patchs, 1] - predicted changes
      altered_patch_target: [B, n_patchs, 1] - ground truth changes
      alteration_loss: scalar - alteration loss
      counts: [B] - number of changed patches per batch
    
    Returns:
      altered_accuracy: float - accuracy of change prediction
      alteration_loss_val: float - alteration loss value
      n_patch_val: float - average number of changed patches
    """
    # === ACCURACY ===
    # Fraction of patches correctly predicted as changed/unchanged
    altered_accuracy = (patch_change_pred_mask == altered_patch_target).float().mean()
    # === LOSS VALUES ===
    # Extract scalar values for logging
    alteration_loss_val = alteration_loss.item() if alteration_loss else 0.0
    # === AVERAGE PATCHES CHANGED ===
    # Mean count of changed patches across batch
    n_patch_val = counts.float().mean().item() if counts.max() > 0 else 0.0
    
    return altered_accuracy, alteration_loss_val, n_patch_val

  def log_epoch(self, epoch, mean_altered_loss, mean_alteration_loss, mean_accuracy, mean_n_patch_to_modify, image, next_image, patch_next_image_predicted):
    """
    Log epoch metrics and reconstructed images to TensorBoard.
    
    Args:
      epoch: int - epoch number
      mean_altered_loss: float - average BCE loss
      mean_alteration_loss: float - average MSE loss
      mean_accuracy: float - average accuracy
      mean_n_patch_to_modify: float - average patches changed
      image: [B, C, H, W] - original images
      next_image: [B, C, H, W] - ground truth next images
      patch_next_image_predicted: [B, n_patchs, patch_dim] - predicted patches
    """
    # === SCALAR LOGGING ===
    # Log losses and metrics
    self.tf_logger.add_scalar('altered_loss', mean_altered_loss, epoch)
    self.tf_logger.add_scalar('alteration_loss', mean_alteration_loss, epoch)
    self.tf_logger.add_scalar('altered_accuracy', mean_accuracy, epoch)
    self.tf_logger.add_scalar('mean_n_patch_to_modify', int(mean_n_patch_to_modify), epoch)

    # === IMAGE LOGGING ===
    # Visualize: [original | ground truth | predicted]
    # First 8 images from batch
    self.tf_logger.add_images(
      'alterationNet_reconstructed_image',
      torch.cat([image[:8],
                 next_image[:8],
                 self.altered_predictor.unpatchify(patch_next_image_predicted[:8])], dim=0),
      global_step=epoch
    )

  def save_best_models(self, mean_accuracy, best_accuracy, mean_alteration_loss, best_loss):
    """
    Save models if they achieve better performance.
    
    STRATEGY:
    - Save AlteredPredictor if accuracy improves
    - Save AlterationPredictor if MSE loss improves
    
    Args:
      mean_accuracy: current epoch accuracy
      best_accuracy: best seen so far
      mean_alteration_loss: current epoch alteration loss
      best_loss: best loss seen so far
    
    Returns:
      updated_accuracy: max of current and best
      updated_loss: min of current and best
    """
    if mean_accuracy > best_accuracy:
      self.save_model(self.altered_predictor, 'altered_predictor')
    if mean_alteration_loss < best_loss:
      self.save_model(self.alteration_predictor, 'alteration_predictor')
    return max(mean_accuracy, best_accuracy), min(mean_alteration_loss, best_loss)

  def generate_imagined_trajectory(self, image, epoch, initial_internal_state=None):
    """
    Generate dual trajectories: 1-step ahead and 5-step ahead predictions.
    
    EVALUATION STRATEGY:
    - Generate TWO independent trajectories using different prediction horizons
    - 1-step trajectory: Accumulate actions one at a time (current method)
    - 5-step trajectory: Jump ahead by predicting 5 steps at once, then continue
    
    PURPOSE:
    - Compare single-step vs multi-step prediction quality
    - Validate if multi-step training helps long-horizon prediction
    - Identify which horizon generalizes better
    - See if 5-step predictions are more stable or diverge earlier
    
    Args:
      image: [B, C, H, W] - initial image
      epoch: int - epoch number for logging
      initial_internal_state: [B, 2] - initial internal state corresponding to image
    """
    self.altered_predictor.eval()
    self.alteration_predictor.eval()

    # === Generate 1-step ahead trajectory ===
    traj_1step = self.generate_trajectory_with_horizon(
      image=image,
      initial_internal_state=initial_internal_state,
      horizon=1,
    )

    # === Generate 5-step ahead trajectory ===
    traj_5step = self.generate_trajectory_with_horizon(
      image=image,
      initial_internal_state=initial_internal_state,
      horizon=5,
    )

    # === Generate trajectory from initial state (no horizon, direct prediction) ===
    traj_direct = self.generate_trajectory_from_initial_state(
      image=image,
      initial_internal_state=initial_internal_state
    )

    # === LOG BOTH TRAJECTORIES TO TENSORBOARD ===
    # 1-step trajectory (fine-grained predictions)
    self.tf_logger.add_images(
      'generated_world_model_prediction_1step',
      torch.cat(traj_1step, dim=0),
      global_step=epoch
    )
    
    # 5-step trajectory (multi-step predictions)
    self.tf_logger.add_images(
      'generated_world_model_prediction_5step',
      torch.cat(traj_5step, dim=0),
      global_step=epoch
    )

    # Direct prediction trajectory (from initial state)
    self.tf_logger.add_images(
      'generated_world_model_prediction_direct',
      torch.cat(traj_direct, dim=0),
      global_step=epoch
    )

    # === SAVE TRAJECTORIES AS GIFS ===
    # gif_dir = os.path.join(self.config['save_dir'], self.config['exp_name'], 'gifs')
    # os.makedirs(gif_dir, exist_ok=True)
    
    # create_gif_from_images(traj_1step, os.path.join(gif_dir, f'traj_1step_epoch_{epoch}.gif'))
    # create_gif_from_images(traj_5step, os.path.join(gif_dir, f'traj_5step_epoch_{epoch}.gif'))
    # create_gif_from_images(traj_direct, os.path.join(gif_dir, f'traj_direct_epoch_{epoch}.gif'))

    self.altered_predictor.train()
    self.alteration_predictor.train()

  @torch.no_grad()
  def generate_trajectory_with_horizon(self, image, initial_internal_state, horizon):
    """
    Generate trajectory using specified prediction horizon.
    
    When horizon=1: Predict 1 step at a time (fine-grained)
    When horizon=5: Jump 5 steps ahead each time (coarse-grained)
    
    EXAMPLE with horizon=5:
    Step 0: Image starts at time 0
    Step 1: Predict 5 steps ahead → Image at time 5
    Step 2: Use predictions to advance by 5 more → Image at time 10
    ...
    Total steps in trajectory = ceil(23 / 5) = 5 images
    
    vs horizon=1:
    Each step advances by 1 → 23 images total
    
    Args:
      image: [B, C, H, W] - initial image
      initial_internal_state: [B, 2] - starting internal state
      horizon: int - how many steps to predict ahead at once (1 or 5)
    
    Returns:
      trajectory: list of [1, C, H, W] images
    """
    # Initialize trajectory with first image
    imagined_traj = [image[:1]]
    
    # Initialize internal state
    current_internal_state = initial_internal_state[:1] if initial_internal_state is not None else None
    
    # === ACTION SEQUENCE ===
    # 12 steps of action 1, then 11 steps of action 3 (total 23 steps)
    all_actions = torch.as_tensor([[[1]]]*12 + [[[3]]]*11, device=self.device, dtype=torch.long)
    
    # Process actions in chunks of size 'horizon'
    fake_actions = []
    step = 0
    
    for idx, fake_action in enumerate(all_actions):
      fake_actions.append(fake_action)
      
      # Only make a prediction when we've accumulated 'horizon' actions
      # or when we're at the last action
      if len(fake_actions) == horizon or idx == len(all_actions) - 1:
        # Flatten action sequence: [[1], [1], ...] → [1, 1, ...]
        accumulated_actions = torch.cat(fake_actions, dim=-1)
        
        # === STEP 1: ALTERED PREDICTOR ===
        # Get current image and convert to patches
        patch = self.altered_predictor.patchify(imagined_traj[-1])
        
        # Predict which patches will change (for the accumulated action sequence)
        patch_change_pred, next_is1_logits, next_is2_logits = self.altered_predictor(
          patch, 
          accumulated_actions, 
          current_internal_state
        )
        
        # Update internal state for next prediction
        # This is the state after taking 'horizon' steps
        current_internal_state = torch.stack([
          torch.argmax(next_is1_logits, dim=1), 
          torch.argmax(next_is2_logits, dim=1)
        ], dim=1)
        
        # === STEP 2: ALTERATION PREDICTOR ===
        # Convert predictions to binary mask (>0.5)
        patch_change_pred_mask = (patch_change_pred > 0.5).squeeze(-1)
        
        # Select only predicted changed patches and get their spatial indices
        patch_to_pred, counts, patch_indices = self.get_patch(
          patch, 
          patch_change_pred_mask, 
          return_indices=True
        )
        
        # Predict pixel values for changed patches if any exist
        if counts.max() > 0:
          patch_predicted, _, _ = self.alteration_predictor(
            patch_to_pred,
            patch_indices,
            accumulated_actions,
            current_internal_state
          )
          # Reconstruct full image with predictions
          patch_next_image_predicted = self.get_next_image(
            patch,
            patch_predicted,
            patch_change_pred_mask
          )
        else:  # if no patches to change, next image is the original one
          patch_next_image_predicted = patch
        
        # === RECONSTRUCT AND SAVE ===
        # Convert patches back to image
        cleaned_img = self.altered_predictor.unpatchify(patch_next_image_predicted)
        imagined_traj.append(cleaned_img)
        
        # Reset action accumulator for next prediction
        fake_actions = []
        step += 1
    
    return imagined_traj

  @torch.no_grad()
  def generate_trajectory_from_initial_state(self, image, initial_internal_state=None):
    """
    Generate trajectory by predicting each step directly from the initial state.
    
    Each step n is predicted using:
    - Starting image: initial image (constant)
    - Actions: [1]*12 + [3]*11 (fixed sequence, 23 total steps)
    - Starting state: initial_internal_state (constant)
    
    Returns:
      trajectory: list of [1, C, H, W] images (1 initial + 23 predictions)
    """
    imagined_traj = [image[:1]]
    
    # Precompute constants
    initial_patch = self.altered_predictor.patchify(image[:1])
    starting_internal_state = initial_internal_state[:1] if initial_internal_state is not None else None
    
    # Fixed action sequence: 12x action 1, then 11x action 3
    all_actions = torch.tensor([[[1]]]*12 + [[[3]]]*11, device=self.device, dtype=torch.long)
    
    for step_idx in range(1, len(all_actions) + 1):
      accumulated_actions = all_actions[:step_idx].T.reshape(1, -1)
      
      # Altered predictor: which patches change?
      patch_change_pred, next_is1_logits, next_is2_logits = self.altered_predictor(
        initial_patch, accumulated_actions, starting_internal_state
      )
      
      # Get next internal state
      next_internal_state = torch.stack([
        torch.argmax(next_is1_logits, dim=1),
        torch.argmax(next_is2_logits, dim=1)
      ], dim=1)
      
      # Alteration predictor: predict pixel changes for changed patches
      patch_change_pred_mask = patch_change_pred.squeeze(-1) > 0.5
      patch_to_pred, counts, patch_indices = self.get_patch(
        initial_patch, patch_change_pred_mask, return_indices=True
      )
      
      if counts.max() > 0:
        patch_predicted, _, _ = self.alteration_predictor(
          patch_to_pred, patch_indices, accumulated_actions, next_internal_state
        )
        patch_result = self.get_next_image(initial_patch, patch_predicted, patch_change_pred_mask)
      else:
        patch_result = initial_patch
      
      imagined_traj.append(self.altered_predictor.unpatchify(patch_result))
  
    return imagined_traj

  def train_alter_predictor(self):
    """
    Main training loop for both AlteredPredictor and AlterationPredictor.
    
    TRAINING SCHEDULE:
    - For each epoch:
      - Sample 10 batches of episodes
      - For each batch:
        - Sample random time steps (1 to rollout_size)
        - Compute losses for both models
        - Update model weights
      - Log metrics and images
      - Generate evaluation trajectory
      - Save best models
    
    MULTI-STEP TRAINING:
    - Each training example uses different action sequence lengths
    - Teaches models to handle varying action horizons
    - Improves robustness
    """
    self.altered_predictor.train()
    self.alteration_predictor.train()

    # === TRACKING BEST MODELS ===
    best_loss = torch.inf
    best_accuracy = 0.0

    # === MAIN LOOP ===
    pbar = tqdm(range(self.config['n_epochs']))
    for epoch in pbar:
      # === EPOCH STATISTICS ===
      mean_accuracy = 0.0
      mean_altered_loss = 0.0
      mean_alteration_loss = 0.0
      mean_n_patch_to_modify = 0.0

      # === MINI-BATCH LOOP ===
      # Process 10 batches per epoch
      for i in tqdm(range(10), leave=False):
        episode = self.train_buffer.sample_episode_batch(self.config['batch_size'], self.config['rollout_size'])

        # === GET INITIAL IMAGE ===
        image = episode['image'][:, 0]  # [B, C, H, W]
        # === VARIABLE-LENGTH SEQUENCES ===
        # Sample 5 different time steps for diversity
        # Each time step requires predicting further into future
        # Teaches model to handle different action sequence lengths
        # for t in torch.randint(1, self.config['rollout_size'], (5,)).tolist():
        # Train on ALL timesteps from 1 to rollout_size
        for t in range(1, self.config['rollout_size'] + 1):
          # Get state at time t
          next_image = episode['next_image'][:, t-1]    # Image at step t
          action = episode['action'][:, :t].flatten(1)  # Actions 0:t
          internal_state = episode['internal_state'][:, t-1] if self.config['use_internal_state'] else None
          next_internal_state = episode['next_internal_state'][:, t-1] if self.config['use_internal_state'] else None

          # === PATCHIFY ===
          # Convert images to patches
          # [B, C, H, W] → [B, n_patchs, patch_dim]
          patch = self.altered_predictor.patchify(image)  # [B, n_patchs=64, N=48]
          patch_next_image = self.altered_predictor.patchify(next_image)  # [B, n_patchs, N]

          # === COMPUTE LOSSES ===
          # Step 1: AlteredPredictor loss
          altered_loss, patch_change_pred_mask, altered_patch_target = self.compute_altered_loss(patch,
                                                                                                 action,
                                                                                                 patch_next_image,
                                                                                                 internal_state if self.config['use_internal_state'] else None,
                                                                                                 next_internal_state if self.config['use_internal_state'] else None)
          # Step 2: AlterationPredictor loss (uses ground truth mask)
          patch_change_gt_mask = altered_patch_target.squeeze(-1)
          alteration_loss, patch_next_image_predicted, counts = self.compute_alteration_loss(patch,
                                                                                             patch_next_image,
                                                                                             patch_change_gt_mask,
                                                                                             action,
                                                                                             internal_state if self.config['use_internal_state'] else None,
                                                                                             next_internal_state if self.config['use_internal_state'] else None)

          # === UPDATE MODELS ===
          self.update_models(altered_loss, alteration_loss)

        # === COMPUTE METRICS ===
        altered_accuracy, alt_loss_val, n_patch_val = self.compute_metrics(patch_change_pred_mask, altered_patch_target, alteration_loss, counts)

        # === RUNNING AVERAGES ===
        # Use running mean to avoid storing all values
        mean_altered_loss += (altered_loss.item() - mean_altered_loss) / (i + 1)
        if alteration_loss:
          mean_alteration_loss += (alt_loss_val - mean_alteration_loss) / (i + 1)
        mean_accuracy += (altered_accuracy.item() - mean_accuracy) / (i + 1)
        if counts.max() > 0:
          mean_n_patch_to_modify += (n_patch_val - mean_n_patch_to_modify) / (i + 1)

      # === LOG EPOCH ===
      self.log_epoch(epoch, mean_altered_loss, mean_alteration_loss, mean_accuracy, mean_n_patch_to_modify,
                     image, next_image, patch_next_image_predicted)

      # === SAVE BEST MODELS ===
      best_accuracy, best_loss = self.save_best_models(mean_accuracy, best_accuracy, mean_alteration_loss, best_loss)

      # === GENERATE IMAGINED TRAJECTORY ===
      internal_state = episode['internal_state'][:, 0] if self.config['use_internal_state'] else None
      self.generate_imagined_trajectory(image, epoch, internal_state)
      # === UPDATE PROGRESS BAR ===
      pbar.set_description(f'Mean_acc: {mean_accuracy:.3f}')
  
  def train_object_predictor(self, n_epochs=100, n_steps_per_epoch=4, batch_size=32):
    """
    Train the ObjectPredictor model.
    """
    self.object_predictor.train()
    pbar = tqdm(range(n_epochs), desc='Training Object Predictor')
    for epoch in pbar:
      mean_hand_loss = 0.0
      mean_target_loss = 0.0
      mean_hand_acc = 0.0
      mean_target_acc = 0.0

      mean_hand_precision, mean_hand_recall = 0.0, 0.0
      mean_target_precision, mean_target_recall = 0.0, 0.0
      for i in tqdm(range(n_steps_per_epoch), leave=False):
        batch = self.train_buffer.sample(batch_size, distinct_episodes=True)
        
        image = batch['image']
        internal_state = batch['internal_state']
        hand_patch_index = batch['hand_patch_index']
        target_patch_index = batch['target_patch_index']

        patch = self.object_predictor.patchify(image)

        # Get simultaneous predictions for hand and target
        hand_preds, target_preds, patch_embeddings = self.object_predictor(patch, internal_state)
        
        # Calculate losses for both heads
        loss_hand = self.bce_with_logits_criterion(hand_preds, hand_patch_index.float())
        loss_target = self.bce_with_logits_criterion(target_preds, target_patch_index.float())

        loss = loss_hand + loss_target
        
        self.object_predictor_opt.zero_grad()
        loss.backward()
        self.object_predictor_opt.step()
        
        # --- Compute Accuracy ---
        with torch.no_grad():
          hand_preds_binary = (torch.sigmoid(hand_preds) > 0.5).float()
          hand_acc = (hand_preds_binary == hand_patch_index).float().mean().item()
          
          target_preds_binary = (torch.sigmoid(target_preds) > 0.5).float()
          target_acc = (target_preds_binary == target_patch_index).float().mean().item()

          # --- Precision/Recall for Hand ---
          tp_hand = ((hand_preds_binary == 1) & (hand_patch_index == 1)).sum().item()
          fp_hand = ((hand_preds_binary == 1) & (hand_patch_index == 0)).sum().item()
          fn_hand = ((hand_preds_binary == 0) & (hand_patch_index == 1)).sum().item()
          
          precision_hand = tp_hand / (tp_hand + fp_hand) if (tp_hand + fp_hand) > 0 else 0.0
          recall_hand = tp_hand / (tp_hand + fn_hand) if (tp_hand + fn_hand) > 0 else 0.0
          
          # --- Precision/Recall for Target ---
          tp_target = ((target_preds_binary == 1) & (target_patch_index == 1)).sum().item()
          fp_target = ((target_preds_binary == 1) & (target_patch_index == 0)).sum().item()
          fn_target = ((target_preds_binary == 0) & (target_patch_index == 1)).sum().item()

          precision_target = tp_target / (tp_target + fp_target) if (tp_target + fp_target) > 0 else 0.0
          recall_target = tp_target / (tp_target + fn_target) if (tp_target + fn_target) > 0 else 0.0

        mean_hand_loss += loss_hand.item()
        mean_target_loss += loss_target.item()
        mean_hand_acc += hand_acc
        mean_target_acc += target_acc
        mean_hand_precision += precision_hand
        mean_hand_recall += recall_hand
        mean_target_precision += precision_target
        mean_target_recall += recall_target

      mean_hand_loss /= n_steps_per_epoch
      mean_target_loss /= n_steps_per_epoch
      mean_hand_acc /= n_steps_per_epoch
      mean_target_acc /= n_steps_per_epoch
      mean_hand_precision /= n_steps_per_epoch
      mean_hand_recall /= n_steps_per_epoch
      mean_target_precision /= n_steps_per_epoch
      mean_target_recall /= n_steps_per_epoch


      if self.tf_logger:
        self.tf_logger.add_scalar('object_predictor_hand_loss', mean_hand_loss, epoch)
        self.tf_logger.add_scalar('object_predictor_target_loss', mean_target_loss, epoch)
        self.tf_logger.add_scalar('object_predictor_hand_acc', mean_hand_acc, epoch)
        self.tf_logger.add_scalar('object_predictor_target_acc', mean_target_acc, epoch)
        self.tf_logger.add_scalar('object_predictor_hand_precision', mean_hand_precision, epoch)
        self.tf_logger.add_scalar('object_predictor_hand_recall', mean_hand_recall, epoch)
        self.tf_logger.add_scalar('object_predictor_target_precision', mean_target_precision, epoch)
        self.tf_logger.add_scalar('object_predictor_target_recall', mean_target_recall, epoch)

      pbar.set_description(f'Object Predictor - Hand Acc: {mean_hand_acc:.2f}, Target Acc: {mean_target_acc:.2f}')

      self.evaluate_object_predictor(epoch)
    
    self.save_model(self.object_predictor, 'object_predictor')

  @torch.no_grad()
  def evaluate_object_predictor(self, epoch, batch_size=10):
    self.object_predictor.eval()
    
    mean_hand_loss = 0.0
    mean_target_loss = 0.0
    mean_hand_acc = 0.0
    mean_target_acc = 0.0
    
    mean_hand_precision, mean_hand_recall = 0.0, 0.0
    mean_target_precision, mean_target_recall = 0.0, 0.0
    
    n_steps = self.test_buffer.size // batch_size
    print(f'n_steps for evaluation: {n_steps} | {batch_size=} | {self.test_buffer.size=}')
    for step in range(n_steps):
      batch = self.test_buffer.sample(batch_size, distinct_episodes=True)
      
      image = batch['image']
      internal_state = batch['internal_state']
      hand_patch_index = batch['hand_patch_index']
      target_patch_index = batch['target_patch_index']

      patch = self.object_predictor.patchify(image)

      # Get simultaneous predictions
      hand_preds, target_preds, _ = self.object_predictor(patch, internal_state)

      # Calculate losses
      loss_hand = self.bce_with_logits_criterion(hand_preds, hand_patch_index.float())
      loss_target = self.bce_with_logits_criterion(target_preds, target_patch_index.float())
      
      # --- Compute Accuracy ---
      hand_preds_binary = (torch.sigmoid(hand_preds) > 0.5).float()
      hand_acc = (hand_preds_binary == hand_patch_index).float().mean().item()

      target_preds_binary = (torch.sigmoid(target_preds) > 0.5).float()
      target_acc = (target_preds_binary == target_patch_index).float().mean().item()

      # --- Precision/Recall for Hand ---
      tp_hand = ((hand_preds_binary == 1) & (hand_patch_index == 1)).sum().item()
      fp_hand = ((hand_preds_binary == 1) & (hand_patch_index == 0)).sum().item()
      fn_hand = ((hand_preds_binary == 0) & (hand_patch_index == 1)).sum().item()
      
      precision_hand = tp_hand / (tp_hand + fp_hand) if (tp_hand + fp_hand) > 0 else 0.0
      recall_hand = tp_hand / (tp_hand + fn_hand) if (tp_hand + fn_hand) > 0 else 0.0
      
      # --- Precision/Recall for Target ---
      tp_target = ((target_preds_binary == 1) & (target_patch_index == 1)).sum().item()
      fp_target = ((target_preds_binary == 1) & (target_patch_index == 0)).sum().item()
      fn_target = ((target_preds_binary == 0) & (target_patch_index == 1)).sum().item()

      precision_target = tp_target / (tp_target + fp_target) if (tp_target + fp_target) > 0 else 0.0
      recall_target = tp_target / (tp_target + fn_target) if (tp_target + fn_target) > 0 else 0.0

      mean_hand_loss += loss_hand.item()
      mean_target_loss += loss_target.item()
      mean_hand_acc += hand_acc
      mean_target_acc += target_acc
      mean_hand_precision += precision_hand
      mean_hand_recall += recall_hand
      mean_target_precision += precision_target
      mean_target_recall += recall_target


    mean_hand_loss /= n_steps
    mean_target_loss /= n_steps
    mean_hand_acc /= n_steps
    mean_target_acc /= n_steps
    mean_hand_precision /= n_steps
    mean_hand_recall /= n_steps
    mean_target_precision /= n_steps
    mean_target_recall /= n_steps

    if self.tf_logger:
      self.tf_logger.add_scalar('test_object_predictor_hand_loss', mean_hand_loss, epoch)
      self.tf_logger.add_scalar('test_object_predictor_target_loss', mean_target_loss, epoch)
      self.tf_logger.add_scalar('test_object_predictor_hand_acc', mean_hand_acc, epoch)
      self.tf_logger.add_scalar('test_object_predictor_target_acc', mean_target_acc, epoch)
      self.tf_logger.add_scalar('test_object_predictor_hand_precision', mean_hand_precision, epoch)
      self.tf_logger.add_scalar('test_object_predictor_hand_recall', mean_hand_recall, epoch)
      self.tf_logger.add_scalar('test_object_predictor_target_precision', mean_target_precision, epoch)
      self.tf_logger.add_scalar('test_object_predictor_target_recall', mean_target_recall, epoch)

    self.object_predictor.train()

  def train_is_predictor_from_patch_index(self, n_epochs=100, n_steps_per_epoch=60, batch_size=128):
    """
    Train the ISPredictorFromPatchIndex model.
    """
    self.is_predictor_from_patch_index.train()
    pbar = tqdm(range(n_epochs), desc='Training ISPredictorFromPatchIndex')
    for epoch in pbar:
      mean_is1_loss = 0.0
      mean_is2_loss = 0.0
      mean_patch_acc = 0.0
      mean_patch_acc_from_true_is = 0.0
      for i in tqdm(range(n_steps_per_epoch), leave=False):
        batch = self.train_buffer.sample(batch_size)
        
        internal_state = batch['internal_state']
        hand_patch_index = batch['hand_patch_index'].squeeze(-1)

        is1_preds, is2_preds = self.is_predictor_from_patch_index(hand_patch_index)
        
        loss_is1 = self.ce_criterion(is1_preds, internal_state[:, 0])
        loss_is2 = self.ce_criterion(is2_preds, internal_state[:, 1])
        
        loss = loss_is1 + loss_is2
        
        self.is_predictor_from_patch_index_opt.zero_grad()
        loss.backward()
        self.is_predictor_from_patch_index_opt.step()
        
        with torch.no_grad():
          # --- Accuracy from PREDICTED internal state ---
          is1_pred_idx = torch.argmax(is1_preds, dim=1)
          is2_pred_idx = torch.argmax(is2_preds, dim=1)
          pred_angles1 = is1_pred_idx * 5
          pred_angles2 = is2_pred_idx * 5

          # --- Accuracy from TRUE internal state (for debugging the calculation) ---
          true_angles1 = internal_state[:, 0] * 5
          true_angles2 = internal_state[:, 1] * 5

          patch_size = self.object_predictor.patch_height
          num_patches_per_row = self.config['resize_to'] // patch_size
          
          predicted_patch_indices = []
          true_patch_indices_for_debug = []

          for i in range(len(pred_angles1)):
            # --- Calculation from PREDICTION ---
            _, _, x_eff_pred, y_eff_pred = forward_kinematics(
              (pred_angles1[i].item(), pred_angles2[i].item()),
              self.robot_arm_ori, self.robot_link_size
            )
            x_scaled_pred = (x_eff_pred - self.robot_crop_box[0]) * (self.config['resize_to'] / (self.robot_crop_box[1] - self.robot_crop_box[0]))
            y_scaled_pred = (y_eff_pred - self.robot_crop_box[2]) * (self.config['resize_to'] / (self.robot_crop_box[3] - self.robot_crop_box[2]))
            patch_col_pred = int(x_scaled_pred // patch_size)
            patch_row_pred = int(y_scaled_pred // patch_size)
            predicted_patch_indices.append(patch_row_pred * num_patches_per_row + patch_col_pred)

            # --- Calculation from TRUE state (for debugging) ---
            _, _, x_eff_true, y_eff_true = forward_kinematics(
              (true_angles1[i].item(), true_angles2[i].item()),
              self.robot_arm_ori, self.robot_link_size
            )
            x_scaled_true = (x_eff_true - self.robot_crop_box[0]) * (self.config['resize_to'] / (self.robot_crop_box[1] - self.robot_crop_box[0]))
            y_scaled_true = (y_eff_true - self.robot_crop_box[2]) * (self.config['resize_to'] / (self.robot_crop_box[3] - self.robot_crop_box[2]))
            patch_col_true = int(x_scaled_true // patch_size)
            patch_row_true = int(y_scaled_true // patch_size)
            true_patch_indices_for_debug.append(patch_row_true * num_patches_per_row + patch_col_true)

          predicted_patch_indices = torch.tensor(predicted_patch_indices, device=self.device)
          true_patch_indices_for_debug = torch.tensor(true_patch_indices_for_debug, device=self.device)

          patch_acc_from_pred = (predicted_patch_indices == hand_patch_index).float().mean().item()
          patch_acc_from_true = (true_patch_indices_for_debug == hand_patch_index).float().mean().item()

        mean_is1_loss += loss_is1.item()
        mean_is2_loss += loss_is2.item()
        mean_patch_acc += patch_acc_from_pred
        mean_patch_acc_from_true_is += patch_acc_from_true

      mean_is1_loss /= n_steps_per_epoch
      mean_is2_loss /= n_steps_per_epoch
      mean_patch_acc /= n_steps_per_epoch
      mean_patch_acc_from_true_is /= n_steps_per_epoch

      if self.tf_logger:
        self.tf_logger.add_scalar('is_predictor_from_patch_index_is1_loss', mean_is1_loss, epoch)
        self.tf_logger.add_scalar('is_predictor_from_patch_index_is2_loss', mean_is2_loss, epoch)
        self.tf_logger.add_scalar('is_predictor_patch_acc_from_pred', mean_patch_acc, epoch)
        self.tf_logger.add_scalar('is_predictor_patch_acc_from_true_is (DEBUG)', mean_patch_acc_from_true_is, epoch)

      pbar.set_description(f'ISPredictor - Patch Acc (Pred): {mean_patch_acc:.2f}, Patch Acc (True IS): {mean_patch_acc_from_true_is:.2f}')
    
    self.save_model(self.is_predictor_from_patch_index, 'is_predictor_from_patch_index')
        
  def train(self):
    # Step 1: Collect environment transitions
    self.fill_memory(self.train_buffer, n_episodes=self.config['n_train_episodes'])
    self.fill_memory(self.test_buffer, n_episodes=self.config['n_test_episodes'], act='best')

    # Step 2: Train Altered & Alteration models
    # print('Training AlteredPredictor and AlterationPredictor...')
    # self.train_alter_predictor()

    # Step 3: Optionally, add hand and target patch indices to the replay buffer
    if self.config['add_hand_and_target_patch_indices']:
      self.add_hand_and_target_patch_indices_to_memory(self.train_buffer)
      self.add_hand_and_target_patch_indices_to_memory(self.test_buffer)
      print('Training ObjectPredictor...')
      self.train_object_predictor()
      # print('Training ISPredictorFromPatchIndex...')
      # self.train_is_predictor_from_patch_index()
    
    # Step 4: train an RL policy using the ObjectPredictor for distance to goal computation


if __name__ == '__main__':
  # === INITIALIZE TRAINER ===
  trainer = NSPTrainer()
  # === LOAD PRE-TRAINED MODELS (if available) ===
  trainer.load_model(trainer.altered_predictor, 'altered_predictor')
  trainer.load_model(trainer.alteration_predictor, 'alteration_predictor')
  # === START TRAINING ===
  trainer.train()

  # NOTES:
  # When I make the hand position prediction harder 
  # (by not offsetting the color centroid to match the real centroid and using patch scoring for hand)
  # I observe that the test target position prediction get way higher
  # To mimic the result but using centroid version for the hand and correct offset
  # I tried several things but none of them reproduce the result:
  #   - regularization on target network (or on hand network) like l1(sparsity)/l2/dropout/reducing_layer_size
  #   - weighting more the hand loss.
  #     The weighting do the job at the beginning, almost matching the results but as
  #     the training continue, the gap is widening
  #   - random patch masking or hand patch masking
  #   - hand label jittering
  #   - spatial label smoothing