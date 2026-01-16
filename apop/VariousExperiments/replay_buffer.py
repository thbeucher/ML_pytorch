import torch
import random
from collections import deque
from torchvision import transforms
from torch.utils.data import Dataset


class ReplayBuffer:
  def __init__(self, internal_state_dim, action_dim, image_size, image_chan=3, resize_to=None,
               normalize_img=False, capacity=10_000, device='cpu', target_device=None):
    self.capacity = capacity

    self.device = torch.device(device)
    self.target_device = target_device if target_device is not None else self.device

    # --- image transformations ---
    img_transforms = [transforms.ToTensor()]  # Maps to [0, 1]
    if normalize_img:
      img_transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # Maps to [-1, 1]
    if resize_to is not None:
      img_transforms.append(transforms.Resize(resize_to))
      image_size = resize_to
    self.image_transforms = transforms.Compose(img_transforms)

    # --- storage ---
    self.internal_state = torch.zeros((capacity, internal_state_dim), device=self.device, dtype=torch.long)
    self.next_internal_state = torch.zeros((capacity, internal_state_dim), device=self.device, dtype=torch.long)
    self.action = torch.zeros((capacity, action_dim), device=self.device, dtype=torch.long)
    self.image = torch.zeros((capacity, image_chan, image_size, image_size), device=self.device, dtype=torch.float32)
    self.next_image = torch.zeros((capacity, image_chan, image_size, image_size),
                                  device=self.device, dtype=torch.float32)
    self.reward = torch.zeros((capacity, 1), device=self.device, dtype=torch.float32)
    self.done = torch.zeros((capacity, 1), device=self.device, dtype=torch.long)

    self.episode_id = torch.zeros((capacity,), dtype=torch.long, device=self.device)
    self.current_episode_id = 0

    self.ptr = 0
    self.size = 0
  
  def __len__(self):
    return self.size
  
  def prepare_data(self, internal_state, action, image, reward, done, next_internal_state, next_image):
    if internal_state is not None and not torch.is_tensor(internal_state):
      internal_state = torch.as_tensor(internal_state, device=self.device, dtype=torch.long)
    if action is not None and not torch.is_tensor(action):
      action = torch.as_tensor(action, device=self.device, dtype=torch.long)
    if image is not None and not torch.is_tensor(image):
      image = self.image_transforms(image)
    if next_internal_state is not None and not torch.is_tensor(next_internal_state):
      next_internal_state = torch.as_tensor(next_internal_state, device=self.device, dtype=torch.long)
    if next_image is not None and not torch.is_tensor(next_image):
      next_image = self.image_transforms(next_image)
    return internal_state, action, image, reward, done, next_internal_state, next_image

  @torch.no_grad()
  def add(self, internal_state, action, image, reward, done, next_internal_state, next_image):
    state = self.prepare_data(internal_state, action, image, reward, done, next_internal_state, next_image)
    internal_state, action, image, reward, done, next_internal_state, next_image = state

    self.internal_state[self.ptr].copy_(internal_state.to(self.device))
    self.action[self.ptr].copy_(action.to(self.device))
    self.image[self.ptr].copy_(image.to(self.device))
    self.reward[self.ptr] = reward
    self.done[self.ptr] = done
    self.next_internal_state[self.ptr].copy_(next_internal_state.to(self.device))
    self.next_image[self.ptr].copy_(next_image.to(self.device))

    self.episode_id[self.ptr] = self.current_episode_id
    # ---- advance episode counter ----
    if done:
      self.current_episode_id += 1

    self.ptr = (self.ptr + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)

  def sample(self, batch_size):
    idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
    batch = {
      "internal_state": self.internal_state[idxs].to(self.target_device),
      "action": self.action[idxs].to(self.target_device),
      "image": self.image[idxs].to(self.target_device),
      "reward": self.reward[idxs].to(self.target_device),
      "done": self.done[idxs].to(self.target_device),
      "next_internal_state": self.next_internal_state[idxs].to(self.target_device),
      "next_image": self.next_image[idxs].to(self.target_device),
    }
    return batch

  def sample_image_is_goal_batch(self, batch_size):
    """
    Sample (image_t, internal_state_t) -> internal_state_goal
    Goal is the terminal state of the same episode.
    """
    assert self.size > 0

    # ---- sample random transitions ----
    done_mask = self.done[:self.size].squeeze(-1) == 1
    done_idxs = torch.where(done_mask)[0]
    # Only sample transitions BEFORE the last done
    max_idx = done_idxs[-1].item() + 1  # include terminal state
    idxs = torch.randint(0, max_idx, (batch_size,), device=self.device)

    images = self.image[idxs]
    states = self.internal_state[idxs]
    eps_ids = self.episode_id[idxs]

    goals = []

    for eid in eps_ids:
      # ---- terminal state for that episode ----
      ep_mask = (self.episode_id[:self.size] == eid)
      done_mask = self.done[:self.size].squeeze(-1) == 1
      terminal_idxs = torch.where(ep_mask & done_mask)[0]

      # should be exactly one, but safe anyway
      goal_idx = terminal_idxs[-1]
      goals.append(self.internal_state[goal_idx])

    batch = {
      "image": images.to(self.target_device),
      "internal_state": states.to(self.target_device),
      "goal_internal_state": torch.stack(goals).to(self.target_device),
    }

    return batch


class ReplayBufferDataset(Dataset):
  def __init__(self, replay_buffer):
    self.buffer = replay_buffer

  def __len__(self):
    return len(self.buffer)

  def __getitem__(self, idx):
    return {
      "internal_state": self.buffer.internal_state[idx],
      "action": self.buffer.action[idx],
      "image": self.buffer.image[idx],
      "reward": self.buffer.reward[idx],
      "done": self.buffer.done[idx],
      "next_internal_state": self.buffer.next_internal_state[idx],
      "next_image": self.buffer.next_image[idx],
    }