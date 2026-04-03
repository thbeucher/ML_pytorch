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

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from vision_transformer.vit import *
from replay_buffer import ReplayBuffer

sys.path.append('../../../robot/')
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AlteredPredictor(nn.Module):
  """
  AlteredPredictor: A Vision Transformer (ViT) model that predicts which image patches are likely to change
  given the current image patches and a sequence of actions.

  This model processes image patches through a transformer encoder, incorporates action embeddings,
  and outputs a probability for each patch indicating whether it will be altered in the next state.
  Used as the first step in the two-step next state prediction pipeline to identify regions of change.

  ARCHITECTURE OVERVIEW:
  1. Patchify: Splits input image into non-overlapping patches (e.g., 32x32 image → 8x8=64 patches of 4x4 pixels)
  2. Patch Embedding: Projects each patch from pixel space to embedding space
  3. Positional Embedding: Adds 2D sinusoidal positional encoding to preserve spatial information
  4. Action Embedding: Encodes the action sequence into a single embedding token
  5. Internal State Embedding: Optionally encodes the robot's internal state (angle, position)
  6. Transformer Encoder: Processes all tokens (patches + action + state) through multi-head attention
  7. Change Prediction Head: Binary classification (Sigmoid) to predict if each patch will change
  8. Internal State Prediction Heads: Predicts the next internal state values

  INPUT SHAPES:
  - patch: [B, n_patchs=64, N=48] - flattened image patches (B=batch_size, N=patch_dim=channels*ph*pw)
  - action: [B, 1] - discrete action indices (one action per forward pass)
  - internal_state: [B, 2] - two internal state values, angle joint1 and angle joint2, discretized into bins

  OUTPUT SHAPES:
  - preds: [B, n_patchs=64, 1] - probability [0,1] that each patch will change
  - next_is1_logits: [B, is1_n_values=19] - logits for angle prediction (19 discrete bins)
  - next_is2_logits: [B, is2_n_values=37] - logits for position prediction (37 discrete bins)

  PURPOSE IN TWO-STEP PIPELINE:
  This model acts as a filter that identifies which patches are relevant for detailed prediction.
  By predicting changes efficiently at the patch level, it reduces computational load for the
  more expensive AlterationPredictor which only processes changed patches.
  """
  def __init__(
    self,
    *,
    image_size,       # int or tuple (H, W): spatial resolution of the input image (e.g., 32x32)
    patch_size,       # int or tuple (Ph, Pw): spatial size of each image patch (e.g., 4x4)
    dim,              # int: embedding dimension of each patch/token (e.g., 64)
    depth,            # int: number of Transformer encoder blocks (e.g., 2)
    heads,            # int: number of attention heads per Transformer block (e.g., 4)
    mlp_dim,          # int: hidden dimension of the feed-forward (MLP) layer (e.g., 128)
    channels=3,       # int: number of input image channels (3 for RGB)
    dim_head=64,      # int: dimension of each attention head (e.g., 64)
    dropout=0.,       # float: dropout rate inside attention and MLP layers
    emb_dropout=0.,   # float: dropout rate applied to patch embeddings
    n_actions=5,      # int: number of discrete actions
    action_dim=8,     # int: intermediate embedding dimension for actions
    is1_n_values=19,  # number of discrete values for internal state 1 (angle joint1)
    is2_n_values=37,  # number of discrete values for internal state 2 (angle joint2)
    is_emb_dim=16     # int: intermediate embedding dimension for internal states
  ):
    super().__init__()
    # === SPATIAL SETUP ===
    # Store image and patch dimensions
    self.channels = channels
    self.image_height, self.image_width = pair(image_size)
    self.patch_height, self.patch_width = pair(patch_size)

    # Calculate dimensions derived from image and patch size
    self.patch_dim = channels * self.patch_height * self.patch_width

    # Calculate grid dimensions: how many patches along each axis
    self.grid = [(self.image_height // self.patch_height), (self.image_width // self.patch_width)]
    self.n_patchs = self.grid[0] * self.grid[1]  # Total patches: e.g., 64

    # === ACTION EMBEDDING ===
    # Converts discrete action ID (0-4) → action_dim → dim embedding
    # Discrete action becomes continuous representation that can be added to other embeddings
    self.action_emb = nn.Sequential(
      nn.Embedding(n_actions, action_dim),    # [B, 1] → [B, 1, action_dim]
      nn.Linear(action_dim, dim), nn.SiLU())  # [B, 1, action_dim] → [B, 1, dim]

    # === INTERNAL STATE EMBEDDING ===
    # Embed internal state (joint angles): converts discrete bin index to continuous
    self.is1_emb = nn.Sequential(
      nn.Embedding(is1_n_values, is_emb_dim),  # is1_n_values = 19 (0-18 inclusive)
      nn.Linear(is_emb_dim, is_emb_dim),
      nn.SiLU()
    )
    self.is2_emb = nn.Sequential(
      nn.Embedding(is2_n_values, is_emb_dim),  # is2_n_values = 37 (0-36 inclusive)
      nn.Linear(is_emb_dim, is_emb_dim),
      nn.SiLU()
    )
    # Project concatenated internal state embeddings to model dimension
    # Concatenates is1_emb + is2_emb (is_emb_dim*2) and projects to dim
    self.is_proj = nn.Linear(2 * is_emb_dim, dim)
    
    # === INTERNAL STATE PREDICTION HEADS ===
    # Predict next internal state values
    self.predict_next_is1 = nn.Linear(dim, is1_n_values)
    self.predict_next_is2 = nn.Linear(dim, is2_n_values)
    
    # === PATCH EMBEDDING ===
    # Rearrange: Convert image [B, C, H, W] → patches [B, n_patchs, patch_dim]
    # Example: [B, 3, 32, 32] → [B, 64, 48] (64 patches of 3*4*4=48 pixels)
    self.patchify = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width)
    # Project patch pixels to embedding dimension
    self.to_patch_embedding = nn.Sequential(#nn.LayerNorm(self.patch_dim),
                                            nn.Linear(self.patch_dim, dim),  # [B, n_patchs, patch_dim] → [B, n_patchs, dim]
                                            nn.LayerNorm(dim))               # Normalize for training stability

    # === POSITIONAL EMBEDDING ===
    # Pre-compute 2D sinusoidal positional embeddings for the entire grid
    # This encodes "which patch is where" using sine/cosine functions
    # Result: [n_patchs=64, dim] tensor where each row is the embedding for patch at (h, w)
    self.pos_embedding = posemb_sincos_2d(
      h = self.image_height // self.patch_height,
      w = self.image_width // self.patch_width,
      dim = dim,
    ) 

    # Dropout applied after adding positional embeddings
    self.dropout = nn.Dropout(emb_dropout)

    # === TRANSFORMER ENCODER ===
    # Multi-head self-attention blocks that allow patches to exchange information
    # Processes: [B, n_patchs+1(+1), dim] → [B, n_patchs+1(+1), dim]
    self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    # === CHANGE PREDICTION HEAD ===
    # Binary classification head: for each patch, output probability [0, 1] of change
    # Architecture: dim → 2*dim (expansion) → 1 (binary) with Sigmoid activation
    self.find_changed_patch = nn.Sequential(nn.Linear(dim, 2*dim),
                                            nn.ReLU(True),
                                            nn.Linear(2*dim, 1),
                                            nn.Sigmoid())

    # === UNPATCHIFY ===
    # Reconstruct image from patches for visualization
    # [B, n_patchs, patch_dim] → [B, C, H, W]
    self.unpatchify = Rearrange(
      'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
      h=self.image_height // self.patch_height,
      w=self.image_width // self.patch_width,
      p1=self.patch_height,
      p2=self.patch_width,
      c=channels
    )

  def forward(self, patch, action, internal_state=None):
    """
    COMPUTATION FLOW:
    1. Embed patches to embedding dimension
    2. Add positional information
    3. Embed and concatenate action
    4. (Optional) Embed and concatenate internal state
    5. Process through Transformer
    6. Extract patch tokens and predict change probability
    7. Predict next internal state
    
    Args:
      patch: [B, n_patchs=64, patch_dim=48] - flattened image patches
      action: [B, 1] - discrete action indices
      internal_state: [B, 2] - (optional) internal state bins
    
    Returns:
      preds: [B, n_patchs=64, 1] - change probabilities for each patch
      next_is1_logits: [B, is1_n_values] - next joint1 angle logits
      next_is2_logits: [B, is2_n_values] - next joint2 angle logits
    """
    # Step 1: Project patches from pixel space to embedding space
    # [B, n_patchs, patch_dim] → [B, n_patchs, dim]
    patch = self.to_patch_embedding(patch)

    # Step 2: Add positional embeddings to preserve spatial information
    # Positional encoding tells each patch "where are you in the grid"
    pos_emb = self.pos_embedding.to(patch.device, dtype=patch.dtype)
    # Apply dropout after adding positional info for regularization
    # [B, n_patchs, dim] + [n_patchs, dim] = [B, n_patchs, dim]
    patch = self.dropout(patch + pos_emb)

    # Step 3: Embed and process action
    # [B, 1] → [B, 1, dim]
    action_emb = self.action_emb(action)
    # Collect all tokens to be processed by Transformer
    tokens = [patch, action_emb]

    # Step 4: (Optional) Embed internal state
    if internal_state is not None:
      # [B, 1] → [B, 1, is_emb_dim] each, then concatenate → [B, 1, 2*is_emb_dim]
      internal_emb = torch.cat([self.is1_emb(internal_state[:, 0]), self.is2_emb(internal_state[:, 1])], dim=-1)
      # Project to model dimension: [B, 1, 2*is_emb_dim] → [B, 1, dim]
      internal_emb = self.is_proj(internal_emb).unsqueeze(1)  # [B, 1, dim]
      tokens.append(internal_emb)

    # Step 5: Concatenate all tokens
    # Result: [B, n_patchs+1] or [B, n_patchs+2] depending on whether internal_state provided
    # Each token is [B, dim]
    patch = torch.cat(tokens, dim=1)  # [B, n_patchs+1(+1), dim]

    # Step 6: Process through Transformer
    # Self-attention allows patches to communicate with each other and with action/state
    # [B, n_patchs+1(+1), dim] → [B, n_patchs+1(+1), dim]
    patch = self.transformer(patch)

    # Step 7: Extract patch tokens (remove action/internal state tokens)
    # Keep only the first n_patchs tokens for patch-level predictions
    # [B, n_patchs, dim]
    patchs_wo_extra = patch[:, :self.n_patchs]  # Exclude action/internal tokens

    # Step 8: Predict which patches will change
    # For each patch embedding, predict if it will change: [B, n_patchs, dim] → [B, n_patchs, 1]
    # Output is probability [0, 1] due to Sigmoid
    preds = self.find_changed_patch(patchs_wo_extra)  # -> [B, n_patchs, 1]

    # Step 9: Predict next internal state
    # Use the internal state token if available (richer information), else use action token
    if internal_state is not None:
        # Use the internal state token (after Transformer processing)
        next_is_token = internal_emb.squeeze(1)  # [B, 1, dim] → [B, dim]
    else:
        # Fallback to action token if no internal state provided
        next_is_token = action_emb.squeeze(1)  # [B, 1, dim] → [B, dim]
    
    # Predict next internal state values as logits (unnormalized probabilities)
    # [B, dim] → [B, is1_n_values] and [B, is2_n_values]
    next_is1_logits = self.predict_next_is1(next_is_token)  # [B, is1_n_values]
    next_is2_logits = self.predict_next_is2(next_is_token)  # [B, is2_n_values]

    return preds, next_is1_logits, next_is2_logits


class AlterationPredictor(nn.Module):
  """
  AlterationPredictor: A Transformer model that predicts the exact pixel-level changes (alterations)
  for image patches identified as changing by the AlteredPredictor.

  This model takes selected patches, their positional embeddings, and actions as input,
  processes them through a transformer, and outputs the predicted altered patch pixels.
  Used as the second step in the two-step pipeline to generate precise next-state predictions.

  ARCHITECTURE OVERVIEW:
  1. Patch Embedding: Projects selected patches to embedding space
  2. Positional Embedding (from indices): Encodes spatial location of selected patches in original grid
  3. Action Embedding: Encodes action sequence
  4. Internal State Embedding: Optionally encodes robot state
  5. Transformer: Processes only the relevant patches + action + state
  6. Pixel Reconstruction: Reconstructs pixel values for altered patches
  7. Internal State Prediction: Predicts next internal state

  KEY DIFFERENCE FROM AlteredPredictor:
  - Works on a SUBSET of patches (only those marked as changed)
  - Receives SPATIAL INDICES of selected patches to compute correct positional embeddings
  - More expressive (larger network) for detailed pixel-level prediction
  - Focuses computation only where needed (computational efficiency)

  INPUT SHAPES:
  - patch: [B, M<=n_patchs, patch_dim=48] - only selected patches (M varies per batch)
  - patch_indices: [B, M] - spatial indices of selected patches in the 8x8 grid (0-63)
  - action: [B, n_actions_seq] - flattened sequence of actions
  - internal_state: [B, 2] - internal state values

  OUTPUT SHAPES:
  - patch: [B, M, patch_dim=48] - reconstructed pixel values for selected patches
  - next_is1_logits: [B, is1_n_values] - logits for next internal state 1
  - next_is2_logits: [B, is2_n_values] - logits for next internal state 2

  PURPOSE IN TWO-STEP PIPELINE:
  This model generates fine-grained predictions for patches identified as changed.
  By only processing changed patches, it's much more efficient than processing all 64 patches.
  """
  def __init__(self, patch_dim=48, dim=64, depth=4, heads=8, dim_head=32, mlp_dim=128, dropout=0.0,
               n_actions=5, action_dim=8, is1_n_values=19, is2_n_values=37, is_emb_dim=16,
               grid_h=8, grid_w=8):
    super().__init__()
    # === PATCH EMBEDDING ===
    # Projects pixel patch to embedding space
    self.patch_embedder = nn.Sequential(nn.LayerNorm(patch_dim),
                                        nn.Linear(patch_dim, dim),
                                        nn.LayerNorm(dim))
    # === ACTION EMBEDDING ===
    # Same as AlteredPredictor
    self.action_embedder = nn.Sequential(nn.Embedding(n_actions, action_dim), nn.Linear(action_dim, dim), nn.SiLU())
    
    # === INTERNAL STATE EMBEDDING ===
        # Same as AlteredPredictor
    self.is1_emb = nn.Sequential(
      nn.Embedding(is1_n_values, is_emb_dim),  # is1_n_values = 19 (0-18 inclusive)
      nn.Linear(is_emb_dim, is_emb_dim),
      nn.SiLU()
    )
    self.is2_emb = nn.Sequential(
      nn.Embedding(is2_n_values, is_emb_dim),  # is2_n_values = 37 (0-36 inclusive)
      nn.Linear(is_emb_dim, is_emb_dim),
      nn.SiLU()
    )
    self.is_proj = nn.Linear(2 * is_emb_dim, dim)

    # === GRID STORAGE FOR POSITIONAL ENCODING ===
    # Store grid dimensions to compute positional embeddings from patch indices
    self.grid_h = grid_h
    self.grid_w = grid_w
    self.dim = dim

    # === TRANSFORMER ===
    # Process selected patches + action + state through Transformer
    # Note: Unlike AlteredPredictor, we don't have a static positional embedding here
    # Positional info is provided dynamically based on selected patch indices
    self.main = Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)
    
    # === PIXEL RECONSTRUCTION ===
    # Project embedding back to pixel space to reconstruct altered patches
    # [B, M, dim] → [B, M, patch_dim=48]
    self.to_patch_pixels = nn.Sequential(nn.Linear(dim, patch_dim))#, nn.Tanh())
    
    # === INTERNAL STATE PREDICTION ===
    # Predict next internal state
    self.predict_next_is1 = nn.Linear(dim, is1_n_values)
    self.predict_next_is2 = nn.Linear(dim, is2_n_values)
  
  def compute_pos_emb_from_indices(self, indices, dtype):
    """
    Compute positional embeddings from patch spatial indices.

    Instead of using pre-computed embeddings for all patches and then selecting them,
    we compute embeddings ONLY for the selected patches based on their actual grid positions.
    This ensures the model knows the true spatial location of each patch.

    EXAMPLE:
    If patches [5, 15, 42] are selected from 64 patches in an 8x8 grid:
    - Patch 5 is at grid position (0, 5) → gets positional embedding for (0, 5)
    - Patch 15 is at grid position (1, 7) → gets positional embedding for (1, 7)
    - Patch 42 is at grid position (5, 2) → gets positional embedding for (5, 2)
    
    The model then sees these three patches with correct positional information,
    even though they appear at positions [0, 1, 2] in the input sequence.
    
    Args:
      indices: [B, M] tensor of patch indices
      dtype: data type for embeddings
      
    Returns:
      pos_emb: [B, M, dim] positional embeddings
    """
    return posemb_sincos_2d_from_indices(
      indices,
      grid_h=self.grid_h,
      grid_w=self.grid_w,
      dim=self.dim,
      dtype=dtype
    )
  
  def forward(self, patch, patch_indices, action, internal_state=None):
    """
    COMPUTATION FLOW:
    1. Compute positional embeddings from patch spatial indices
    2. Embed selected patches
    3. Add positional information to patch embeddings
    4. Embed action sequence
    5. (Optional) Embed internal state
    6. Process through Transformer
    7. Reconstruct pixel values
    8. Predict next internal state
    
    Args:
        patch: [B, M<=n_patchs, patch_dim=48] - selected patches only
        patch_indices: [B, M] - spatial indices in original grid
        action: [B, n_actions_seq] - action sequence
        internal_state: [B, 2] - (optional) internal state
    
    Returns:
        patch: [B, M, patch_dim=48] - reconstructed altered pixels
        next_is1_logits: [B, is1_n_values] - next state 1 logits
        next_is2_logits: [B, is2_n_values] - next state 2 logits
    """
    # Step 1: Compute positional embeddings from spatial indices and add them
    # Get correct positional embeddings based on actual grid positions
    # [B, M] → [B, M, dim] (positional embedding for each selected patch)
    pos_emb = self.compute_pos_emb_from_indices(patch_indices, patch.dtype)

    # Step 2-3: Embed patches to embedding space
    # Compute positional embeddings based on actual spatial indices
    # [B, M, patch_dim] → [B, M, dim]
    patch_emb = self.patch_embedder(patch) + pos_emb  # [B, M, dim]

    # Step 4: Embed action
    # [B, n_actions_seq] → [B, n_actions_seq, dim]
    # Note: action is a sequence of actions, not just one action
    action_emb = self.action_embedder(action)
    # Collect tokens for Transformer
    tokens = [patch_emb, action_emb]
    
    # Step 5: (Optional) Embed internal state
    if internal_state is not None:
      # Embed and concatenate internal state components
      internal_emb = torch.cat([self.is1_emb(internal_state[:, 0]), self.is2_emb(internal_state[:, 1])], dim=-1)
      # [B, 1, 2*is_emb_dim] → [B, 1, dim]
      internal_emb = self.is_proj(internal_emb).unsqueeze(1)
      tokens.append(internal_emb)

    # Step 6: Concatenate all tokens
    # Result: [B, M+n_actions_seq(+1), dim]
    patch = torch.cat(tokens, dim=1)

    # Step 7: Process through Transformer
    # [B, M+n_actions_seq(+1), dim] → [B, M+n_actions_seq(+1), dim]
    patch = self.main(patch)

    # Step 8: Extract patch tokens (remove action/state tokens)
    # Keep only the first M tokens for patches
    # [B, M, dim]
    patch_wo_extra = patch[:, :patch_emb.size(1)]  # Exclude action/internal tokens

    # Step 9: Reconstruct pixel values from embeddings
    # [B, M, dim] → [B, M, patch_dim=48]
    patch = self.to_patch_pixels(patch_wo_extra)  # [B, M, N=channels*patch_height*patch_width=3*4*4=48]
    
    # Step 10: Predict next internal state
    # Predict next internal state using the internal token if available, else action token
    if internal_state is not None:
        # Use the internal state token
        # [B, 1, dim] → [B, dim]
        next_is_token = internal_emb.squeeze(1)
    else:
        # Fallback to action token
        # Handle both single action and multiple actions
        next_is_token = action_emb[:, 0] if action_emb.size(1) == 1 else action_emb.mean(dim=1)  # [B, dim]
    
    # [B, dim] → [B, is1_n_values] and [B, is2_n_values]
    next_is1_logits = self.predict_next_is1(next_is_token)  # [B, is1_n_values]
    next_is2_logits = self.predict_next_is2(next_is_token)  # [B, is2_n_values]
    
    return patch, next_is1_logits, next_is2_logits


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
            'replay_buffer_size':                10_000,  # Number of transitions to collect
            'replay_buffer_device':              'cpu',   # Store on CPU to save GPU memory
            'render_mode':                       'rgb_array',
            'n_epochs':                          100,
            'batch_size':                        128,
            'use_tf_logger':                     True,
            'save_dir':                          'NSP_experiments/',
            'log_dir':                           'runs/',
            'exp_name':                          'nsp_twosteps_big_rollout_actionIS_newPE',
            'seed':                              42,
            'rollout_size':                      30,    # Max episode length for sampling
            'use_internal_state':                True,  # Include internal state in training
            'internal_state_emb_dim':            16,    # Internal state embedding dimension
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
  
  def set_trainer_utils(self):
    """Initialize replay buffer, optimizers, and loss functions."""
    # === REPLAY BUFFER ===
    # Stores environment transitions for training
    resize_img = True if self.config['resize_to'] != self.config['image_size'] else False
    self.replay_buffer = ReplayBuffer(self.config['internal_state_dim'],
                                      self.config['action_dim'],
                                      self.config['image_size'],
                                      resize_to=self.config['resize_to'] if resize_img else None,
                                      normalize_img=self.config['normalize_image'],
                                      capacity=self.config['replay_buffer_size'],
                                      device=self.config['replay_buffer_device'],
                                      target_device=self.device)
    # === OPTIMIZERS ===
    # Separate optimizers for each model
    self.altered_opt = torch.optim.AdamW(self.altered_predictor.parameters(), lr=1e-4,
                                         weight_decay=1e-4, betas=(0.9, 0.999))
    self.alteration_opt = torch.optim.AdamW(self.alteration_predictor.parameters(), lr=1e-4,
                                            weight_decay=1e-4, betas=(0.9, 0.999))
    # === LOSS FUNCTIONS ===
    self.bce_criterion = nn.BCELoss()  # Binary Cross-Entropy for change prediction (0/1 classification)
    self.mse_criterion = nn.MSELoss()  # Mean Squared Error for pixel reconstruction
    
  def fill_memory(self, random_act=True, max_episode_steps=60):
    """
    Collect random environment trajectories and fill the replay buffer.
    
    COLLECTION STRATEGY:
    - Random actions: each step selects uniformly random action
    - Episodes: reset after terminal state or max_episode_steps
    - Stores: (state, action, image, reward, done, next_state, next_image)
    
    Args:
        random_act: if True, use random actions; else use policy (TODO)
        max_episode_steps: max steps per episode
    """
    print('Filling memory buffer...')
    # Initialize environment
    obs, _ = self.env.reset()
    img = self.env.render()
    episode_step = 0
    # Collect transitions
    for _ in tqdm(range(self.config['replay_buffer_size'])):
      # === SELECT ACTION ===
      action = random.randint(0, 4) if random_act else 0  #TODO use policy
      # === STEP ENVIRONMENT ===
      # Get next observation, reward, done flag
      next_obs, reward, terminated, truncated, info = self.env.step(action)
      next_img = self.env.render()
      episode_step += 1
      # === STORE TRANSITION ===
      # obs//5: quantize state
      # img: RGB observation
      self.replay_buffer.add(obs//5,action, img, reward, terminated or episode_step > max_episode_steps,
                             next_obs//5, next_img)
      # Update state for next iteration
      obs, img = next_obs, next_img

      # === EPISODE RESET ===
      # Start new episode if done
      if terminated or episode_step > max_episode_steps:
        obs, _ = self.env.reset()
        img = self.env.render()
        episode_step = 0
    
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
  
  def add_highlight(self, mask, img):
    """
    Add green highlight to visualize which patches are predicted to change.
    
    VISUALIZATION PIPELINE:
    1. Convert patch mask [B, 64] to pixel mask [B, 32, 32]
    2. Create green overlay
    3. Blend with original image where mask is True
    
    Args:
        mask: [B, n_patchs, 1] - patch change predictions
        img: [B, C, H, W] - image to highlight
    
    Returns:
        img_vis: [B, C, H, W] - image with green highlights
    """
    # === CONVERT PATCH MASK TO PIXEL MASK ===
    # M: [B, 64, 1] → [B, 64]
    patch_mask = mask.squeeze(-1).bool()
    # Reshape to grid: [B, 64] → [B, 8, 8]
    grid_height, grid_width = self.altered_predictor.grid
    patch_mask = patch_mask.view(-1, grid_height, grid_width)  # [B, 8, 8]
    # Expand each patch to pixels: [B, 8, 8] → [B, 32, 32]
    # Each patch (4x4 pixels) is repeated
    pixel_mask = patch_mask.repeat_interleave(self.altered_predictor.patch_height, dim=1)\
                            .repeat_interleave(self.altered_predictor.patch_width, dim=2)
    # pixel_mask: [B, 32, 32]

    # --- Create green overlay --- #
    img_vis = img.clone()
    green = torch.zeros_like(img_vis)
    green[:, 1, :, :] = 1.0   # pure green channel

    # === BLEND ===
    # Where mask is True: blend original with green
    # Where mask is False: keep original
    alpha = 0.5  # strength of highlight
    mask = pixel_mask.unsqueeze(1)  # [B, 1, 32, 32]
    img_vis = torch.where(
      mask,
      (1 - alpha) * img_vis + alpha * green,
      img_vis
    )
    return img_vis
  
  def sample_training_batch(self):
    """Sample a batch of episodes for training."""
    return self.replay_buffer.sample_episode_batch(self.config['batch_size'], self.config['rollout_size'])

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
    Generate trajectory by repeatedly applying the two-step predictor.
    
    EVALUATION STRATEGY:
    - Use predefined action sequence: 12 steps of action 1, then 11 steps of action 3
    - For each step:
      a) Predict which patches will change
      b) Predict new pixel values for changed patches
      c) Reconstruct image
      d) Save to trajectory
    - Visualize as image grid: [step0, step1, ..., step23]
    
    PURPOSE:
    - Visualize long-horizon prediction quality
    - Detect if model "hallucinate" or diverges over time
    
    Args:
      image: [B, C, H, W] - initial image
      epoch: int - epoch number for logging
      initial_internal_state: [B, 2] - initial internal state corresponding to image
    """
    self.altered_predictor.eval()
    self.alteration_predictor.eval()

    with torch.no_grad():
      # Initialize trajectory with first image
      imagined_traj = [image[:1]]
      fake_actions = []
      # Initialize internal state
      current_internal_state = initial_internal_state[:1] if initial_internal_state is not None else None

      # === ACTION SEQUENCE ===
      # 12 steps of action 1
      # 11 steps of action 3
      for fake_action in torch.as_tensor([[[1]]]*12 + [[[3]]]*11, device=self.device, dtype=torch.long):
        fake_actions.append(fake_action)

        # Flatten action sequence: [[1], [1], ...] → [1, 1, ...]
        fake_action = torch.cat(fake_actions, dim=-1)

        # === STEP 1: ALTERED PREDICTOR ===
        # Get current image and convert to patches
        patch = self.altered_predictor.patchify(imagined_traj[-1])

        # get patches that are expected to change
        patch_change_pred, next_is1_logits, next_is2_logits = self.altered_predictor(patch, fake_action, current_internal_state)

        # Update internal state for next step
        current_internal_state = torch.stack([torch.argmax(next_is1_logits, dim=1), torch.argmax(next_is2_logits, dim=1)], dim=1)

        # === STEP 2: ALTERATION PREDICTOR ===
        # Convert predictions to binary mask (>0.5)
        # get mask to select only predicted changed patches
        patch_change_pred_mask = (patch_change_pred > 0.5).squeeze(-1)

        # select only predicted changed patches and get their spatial indices
        patch_to_pred, counts, patch_indices = self.get_patch(patch, patch_change_pred_mask, return_indices=True)

        # Predict pixel values for changed patches if any exist
        if counts.max() > 0:
          patch_predicted, _, _ = self.alteration_predictor(patch_to_pred,
                                                            patch_indices,
                                                            fake_action,
                                                            current_internal_state)
          # Reconstruct full image with predictions
          patch_next_image_predicted = self.get_next_image(patch,
                                                           patch_predicted,
                                                           patch_change_pred_mask)
        else:  # if no patches to change, next image is the original one
          patch_next_image_predicted = patch
        # === RECONSTRUCT AND SAVE ===
        # Convert patches back to image
        cleaned_img = self.altered_predictor.unpatchify(patch_next_image_predicted)
        imagined_traj.append(cleaned_img)

    # === LOG TO TENSORBOARD ===
    # Visualize full trajectory as image grid
    self.tf_logger.add_images('generated_world_model_prediction',
                              torch.cat(imagined_traj, dim=0),
                              global_step=epoch)
    self.altered_predictor.train()
    self.alteration_predictor.train()

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
        episode = self.sample_training_batch()

        # === GET INITIAL IMAGE ===
        image = episode['image'][:, 0]  # [B, C, H, W]
        # === VARIABLE-LENGTH SEQUENCES ===
        # Sample 5 different time steps for diversity
        # Each time step requires predicting further into future
        # Teaches model to handle different action sequence lengths
        for t in torch.randint(1, self.config['rollout_size'], (5,)).tolist():
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
  
  def train(self):
    """
    Main entry point: fill buffer then train.
    """
    # Step 1: Collect environment transitions
    self.fill_memory()
    # Step 2: Train models
    self.train_alter_predictor()


if __name__ == '__main__':
  # === INITIALIZE TRAINER ===
  trainer = NSPTrainer()
  # === LOAD PRE-TRAINED MODELS (if available) ===
  trainer.load_model(trainer.altered_predictor, 'altered_predictor')
  trainer.load_model(trainer.alteration_predictor, 'alteration_predictor')
  # === START TRAINING ===
  trainer.train()