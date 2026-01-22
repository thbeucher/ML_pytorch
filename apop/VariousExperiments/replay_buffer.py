import torch
from torchvision import transforms
from torch.utils.data import Dataset


class ReplayBuffer:
  def __init__(self,
               internal_state_dim: int,
               action_dim: int,
               image_size: int,
               image_chan: int = 3,
               resize_to: int | None = None,
               normalize_img: bool = False,
               capacity: int = 10_000,
               device: str = 'cpu',
               target_device: torch.device | None = None):
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

    # --- priority / loss storage ---
    self.loss = torch.zeros((capacity,), device=self.device, dtype=torch.float32)

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
    if done:
      self.current_episode_id += 1

    self.ptr = (self.ptr + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)
  
  @torch.no_grad()
  def add_prioritize(self, internal_state, action, image, reward, done,
                     next_internal_state, next_image, loss: float):
    state = self.prepare_data(internal_state, action, image, reward, done,
                              next_internal_state, next_image)
    internal_state, action, image, reward, done, next_internal_state, next_image = state

    # ---- choose index ----
    if self.size < self.capacity:
        idx = self.ptr
        self.ptr = (self.ptr + 1) % self.capacity
        self.size += 1
    else:
        # replace the smallest-loss transition
        idx = torch.argmin(self.loss[:self.size]).item()

    # ---- store transition ----
    self.internal_state[idx].copy_(internal_state.to(self.device))
    self.action[idx].copy_(action.to(self.device))
    self.image[idx].copy_(image.to(self.device))
    self.reward[idx] = reward
    self.done[idx] = done
    self.next_internal_state[idx].copy_(next_internal_state.to(self.device))
    self.next_image[idx].copy_(next_image.to(self.device))

    # ---- store loss / priority ----
    self.loss[idx] = float(loss)

    # ---- episode bookkeeping ----
    self.episode_id[idx] = self.current_episode_id
    if done:
      self.current_episode_id += 1

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
  
  def sample_prioritized(self, batch_size, alpha=1.0, eps=1e-6):
    """
    Biased sampling: probability ∝ (loss + eps)^alpha
    alpha = 0   -> uniform
    alpha = 1   -> linear priority
    alpha > 1   -> stronger bias toward high loss
    """
    assert self.size > 0

    # ---- compute priorities ----
    # alpha = 0 -> no bias, uniform sampling | = 0.5 soft | 2 = very strong
    losses = self.loss[:self.size]
    priorities = (losses + eps) ** alpha
    probs = priorities / priorities.sum()

    # ---- sample indices ----
    idxs = torch.multinomial(probs, batch_size, replacement=True)

    batch = {
        "internal_state": self.internal_state[idxs].to(self.target_device),
        "action": self.action[idxs].to(self.target_device),
        "image": self.image[idxs].to(self.target_device),
        "reward": self.reward[idxs].to(self.target_device),
        "done": self.done[idxs].to(self.target_device),
        "next_internal_state": self.next_internal_state[idxs].to(self.target_device),
        "next_image": self.next_image[idxs].to(self.target_device),
        "loss": self.loss[idxs].to(self.target_device),  # optional, useful for updates
    }

    return batch
  
  def find_finished_episode(self, success_reward=10):
    # ---- find episodes whose final transition has reward == success_reward ----
    valid_episode_ids = []

    for eid in range(self.current_episode_id):
      ep_mask = (self.episode_id[:self.size] == eid)
      ep_idxs = torch.where(ep_mask)[0]
      if len(ep_idxs) == 0:
        continue

      last_idx = ep_idxs[-1]
      if self.reward[last_idx].item() == success_reward:
        valid_episode_ids.append(eid)

    self.n_finished_episode = len(valid_episode_ids)
    assert len(valid_episode_ids) > 0, "No episodes with final reward == success_reward found!"

    valid_episode_ids = torch.tensor(valid_episode_ids, device=self.device)
    return valid_episode_ids
  
  def sample_image_is_goal_batch(self, batch_size, n_fake_goals=4, success_reward=10):
    """
    Sample (image_t, internal_state_t) -> goal
    Goal is the final state of episodes whose LAST reward == success_reward.
    Only such episodes are sampled.
    """
    assert self.size > 0

    B = batch_size

    images = torch.zeros((B, *self.image.shape[1:]),
                         device=self.target_device,
                         dtype=self.image.dtype)
    states = torch.zeros((B, self.internal_state.shape[-1]),
                         device=self.target_device,
                         dtype=self.internal_state.dtype)
    goal_states = torch.zeros_like(states)
    goal_images = torch.zeros_like(images)
    fake_goal_states = torch.zeros((B, n_fake_goals, self.internal_state.shape[-1]),
                                    device=self.target_device,
                                    dtype=self.internal_state.dtype)

    # ---- find episodes whose final transition has reward == success_reward ----
    valid_episode_ids = self.find_finished_episode(success_reward=success_reward)

    # ---- sample only valid episodes ----
    sampled_eids = valid_episode_ids[
      torch.randint(0, len(valid_episode_ids), (B,), device=self.device)
    ]

    for b, eid in enumerate(sampled_eids):
      ep_mask = (self.episode_id[:self.size] == eid)
      ep_idxs = torch.where(ep_mask)[0]

      if len(ep_idxs) == 0:
        continue

      # ---- goal is final state of the episode ----
      goal_idx = ep_idxs[-1]

      # ---- sample a random timestep from the episode ----
      t_idx = ep_idxs[torch.randint(0, len(ep_idxs), (1,), device=self.device).item()]

      # ---- fill sample ----
      images[b] = self.image[t_idx].to(self.target_device)
      states[b] = self.internal_state[t_idx].to(self.target_device)

      goal_states[b] = self.internal_state[goal_idx].to(self.target_device)
      goal_images[b] = self.image[goal_idx].to(self.target_device)

      # ---- fake goals: any non-final state from same episode ----
      non_final_idxs = ep_idxs[:-1]

      if len(non_final_idxs) == 0:
        fake_idxs = goal_idx.repeat(n_fake_goals)
      else:
        rand_idxs = torch.randint(0, len(non_final_idxs), (n_fake_goals,), device=self.device)
        fake_idxs = non_final_idxs[rand_idxs]

      fake_goal_states[b] = self.internal_state[fake_idxs].to(self.target_device)

    batch = {
      "image": images,
      "internal_state": states,
      "goal_internal_state": goal_states,
      "goal_image": goal_images,
      "fake_goal_internal_state": fake_goal_states,  # [B, K, D]
    }

    return batch
  
  def sample_episode_batch(self, batch_size, episode_length, random_window=True, success_reward=None):
    B, T = batch_size, episode_length

    batch = {
      "internal_state": torch.zeros(
        (B, T, self.internal_state.shape[-1]), device=self.target_device, dtype=self.internal_state.dtype
      ),
      "action": torch.zeros(
        (B, T, self.action.shape[-1]), device=self.target_device, dtype=self.action.dtype
      ),
      "image": torch.zeros(
        (B, T, *self.image.shape[1:]), device=self.target_device, dtype=self.image.dtype
      ),
      "reward": torch.zeros(
        (B, T, 1), device=self.target_device, dtype=self.reward.dtype
      ),
      "done": torch.zeros(
        (B, T, 1), device=self.target_device, dtype=self.done.dtype
      ),
      "next_internal_state": torch.zeros(
        (B, T, self.next_internal_state.shape[-1]),
        device=self.target_device,
        dtype=self.next_internal_state.dtype,
      ),
      "next_image": torch.zeros(
        (B, T, *self.next_image.shape[1:]),
        device=self.target_device,
        dtype=self.next_image.dtype,
      ),
    }

    if success_reward is None:
      # ---- sample random episode ids ----
      episode_ids = torch.randint(0, self.current_episode_id, (batch_size,), device=self.device)
    else:
      # ---- find episodes whose final transition has reward == success_reward ----
      valid_episode_ids = self.find_finished_episode(success_reward=success_reward)

      # ---- sample only valid episodes ----
      episode_ids = valid_episode_ids[torch.randint(0, len(valid_episode_ids), (B,), device=self.device)]

    for b, eid in enumerate(episode_ids):
      ep_mask = (self.episode_id[:self.size] == eid)
      ep_idxs = torch.where(ep_mask)[0]

      if len(ep_idxs) == 0:
        continue

      ep_len = len(ep_idxs)

      # ---- choose random window ----
      if ep_len >= T:
        if random_window:
          start = torch.randint(0, ep_len - T + 1, (1,)).item()
          window_idxs = ep_idxs[start:start + T]
        else:
          window_idxs = ep_idxs[-T:]
        L = T
      else:
        # take whole episode and pad later
        window_idxs = ep_idxs
        L = ep_len

      # ---- copy real data ----
      batch["internal_state"][b, :L] = self.internal_state[window_idxs].to(self.target_device)
      batch["action"][b, :L] = self.action[window_idxs].to(self.target_device)
      batch["image"][b, :L] = self.image[window_idxs].to(self.target_device)
      batch["reward"][b, :L] = self.reward[window_idxs].to(self.target_device)
      batch["done"][b, :L] = self.done[window_idxs].to(self.target_device)
      batch["next_internal_state"][b, :L] = self.next_internal_state[window_idxs].to(self.target_device)
      batch["next_image"][b, :L] = self.next_image[window_idxs].to(self.target_device)

      # ---- pad if shorter ----
      if L < T:
        last_is = batch["internal_state"][b, L - 1]
        last_img = batch["image"][b, L - 1]
        last_next_is = batch["next_internal_state"][b, L - 1]
        last_next_img = batch["next_image"][b, L - 1]

        batch["internal_state"][b, L:] = last_is.unsqueeze(0).expand(T - L, -1)
        batch["image"][b, L:] = last_img.unsqueeze(0).expand(T - L, -1, -1, -1)
        batch["next_internal_state"][b, L:] = last_next_is.unsqueeze(0).expand(T - L, -1)
        batch["next_image"][b, L:] = last_next_img.unsqueeze(0).expand(T - L, -1, -1, -1)
        # action/reward/done already zero

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