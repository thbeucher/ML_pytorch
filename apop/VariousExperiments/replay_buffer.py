#
# | Class/Function                  | Description                                                                                             |
# |---------------------------------|---------------------------------------------------------------------------------------------------------|
# | `ReplayBuffer`                  | A class to store and manage transitions for reinforcement learning.                                     |
# | `__init__(...)`                 | Initializes the replay buffer with given dimensions, capacity, and other settings.                      |
# | `__len__()`                     | Returns the number of transitions currently stored in the buffer.                                       |
# | `set_hand_condition(...)`       | Sets a hand condition used for calculating target patches.                                              |
# | `prepare_data(...)`             | Converts input data (states, actions, etc.) to the correct tensor format and device.                    |
# | `add(...)`                      | Adds a single transition (state, action, reward, done, next_state) to the buffer.                       |
# | `add_prioritize(...)`           | Adds a transition to the buffer, replacing the one with the smallest loss if the buffer is full.        |
# | `add_variable(...)`             | Stores an additional tensor variable alongside the standard transition data.                            |
# | `get_batch(...)`                | Retrieves a batch of transitions from the buffer given a set of indices.                                |
# | `get_episodes_sizes()`          | Returns the number of transitions in each episode stored in the buffer.                                 |
# | `get_first_states()`            | Retrieves the first transition from every episode in the buffer.                                        |
# | `get_sampling_indices(...)`     | Generates indices for sampling a batch, with an option for distinct episodes.                           |
# | `sample(...)`                   | Samples a random batch of transitions from the buffer.                                                  |
# | `sample_prioritized(...)`       | Samples a batch of transitions using a priority distribution based on stored losses.                    |
# | `sample_from_successful_episodes(...)` | Samples transitions exclusively from episodes that were marked as successful.                    |
# | `sample_image_is_goal_batch(...)` | Samples a batch for goal-conditioned tasks, providing (state, image) and corresponding goal states.   |
# | `sample_episode_batch(...)`     | Samples a batch of entire episodes (or fixed-length windows from them).                                 |
#
import torch
from torchvision import transforms
from torch.utils.data import Dataset

import helpers_zoo as hz


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
               target_device: torch.device | None = None,
               internal_state_dtype=torch.long,
               action_dtype=torch.long):
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
    self.internal_state = torch.zeros((capacity, internal_state_dim), device=self.device, dtype=internal_state_dtype)
    self.next_internal_state = torch.zeros((capacity, internal_state_dim), device=self.device, dtype=internal_state_dtype)
    self.action = torch.zeros((capacity, action_dim), device=self.device, dtype=action_dtype)
    self.image = torch.zeros((capacity, image_chan, image_size, image_size), device=self.device, dtype=torch.float32)
    self.next_image = torch.zeros((capacity, image_chan, image_size, image_size),
                                  device=self.device, dtype=torch.float32)
    self.reward = torch.zeros((capacity, 1), device=self.device, dtype=torch.float32)
    self.done = torch.zeros((capacity, 1), device=self.device, dtype=torch.long)
    self.target_patch_gt = torch.full((capacity,), -1, dtype=torch.long, device=self.device)

    self.successful_episodes = []

    # --- priority / loss storage ---
    self.loss = torch.zeros((capacity,), device=self.device, dtype=torch.float32)

    self.other_stored_obj = {}

    self.episode_id = torch.zeros((capacity,), dtype=torch.long, device=self.device)
    self.current_episode_id = 0
    self.hand_condition = None

    self.ptr = 0
    self.size = 0
  
  def __len__(self):
    return self.size
  
  def set_hand_condition(self, hand_condition):
    self.hand_condition = hand_condition

  def _update_target_for_episode(self, episode_id, next_image_of_rewarded_state):
    if self.hand_condition is None:
      return

    # find_object_center expects a batch.
    pos = hz.find_object_center(next_image_of_rewarded_state.unsqueeze(0), self.hand_condition)

    if not torch.isnan(pos).any():
      # The image is 32x32, and the patch grid is 16x16. Patch size is 2x2.
      patch_x = (pos[0, 0] / 2).long()
      patch_y = (pos[0, 1] / 2).long()
      patch_idx = patch_y * 16 + patch_x

      # Update all transitions of this episode
      episode_indices = torch.where(self.episode_id == episode_id)[0]
      if episode_indices.numel() > 0:
        self.target_patch_gt[episode_indices] = patch_idx.item()

  def prepare_data(self, internal_state=None, action=None, image=None, reward=None, done=None,
                   next_internal_state=None, next_image=None):
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
  def add(self, internal_state, action, image, reward, done, next_internal_state, next_image, sucess_reward=10):
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
      if reward == sucess_reward:
        self.successful_episodes.append(self.current_episode_id)
        self._update_target_for_episode(self.current_episode_id, next_image)
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

  def add_variable(self, variable, name):
    self.other_stored_obj[name] = variable.to(self.device)
  
  def get_batch(self, idxs):
    batch = {
      "internal_state": self.internal_state[idxs].to(self.target_device),
      "action": self.action[idxs].to(self.target_device),
      "image": self.image[idxs].to(self.target_device),
      "reward": self.reward[idxs].to(self.target_device),
      "done": self.done[idxs].to(self.target_device),
      "next_internal_state": self.next_internal_state[idxs].to(self.target_device),
      "next_image": self.next_image[idxs].to(self.target_device),
      "target_patch_gt": self.target_patch_gt[idxs].to(self.target_device),
      "loss": self.loss[idxs].to(self.target_device),
    }
    other_vars = {k: v[idxs].to(self.target_device) for k, v in self.other_stored_obj.items()}
    return {**batch, **other_vars}
  
  def get_episodes_sizes(self):
    """
    Returns the size of each episode in the buffer.
    """
    if self.size == 0:
      return torch.tensor([], dtype=torch.long, device=self.device), torch.tensor([], dtype=torch.long, device=self.device)

    all_eids = self.episode_id[:self.size]
    sorted_eids, _ = torch.sort(all_eids)
    
    unique_eids, counts = torch.unique_consecutive(sorted_eids, return_counts=True)
    
    return unique_eids, counts

  def get_first_states(self, return_last_states=False):
    """
    Provides the first and optionally the last state of all available episodes.
    The batch size will correspond to the number of episodes.
    """
    if self.size == 0:
      return {}

    all_eids = self.episode_id[:self.size]
    
    # Sort episode IDs to group them
    sorted_eids, sort_perm = torch.sort(all_eids)
    
    # Original buffer indices, but sorted by episode ID
    sorted_indices = torch.arange(self.size, device=self.device)[sort_perm]
    
    # Find the boundaries of each episode's segment
    group_eids, counts = torch.unique_consecutive(sorted_eids, return_counts=True)
    
    # Calculate the starting position of each group in the sorted tensor
    segment_starts = torch.cat([torch.tensor([0], device=self.device), torch.cumsum(counts, 0)[:-1]])
    
    # Get the indices within the `sorted_indices` tensor that correspond to the first state of each episode
    # Retrieve the original buffer indices
    first_state_idxs = sorted_indices[segment_starts]

    if return_last_states:
        # Calculate the ending position of each group
        segment_ends = torch.cumsum(counts, 0) - 1
        last_state_idxs = sorted_indices[segment_ends]
        return self.get_batch(first_state_idxs), self.get_batch(last_state_idxs)
    
    return self.get_batch(first_state_idxs)
  
  def get_sampling_indices(self, batch_size, distinct_episodes=False):
    if distinct_episodes:
      # Get all unique episode IDs present in the buffer
      unique_episode_ids = self.episode_id[:self.size].unique()

      # Check if we have enough unique episodes to sample from
      if len(unique_episode_ids) < batch_size:
        raise ValueError(f"Cannot sample {batch_size} distinct episodes. "
                          f"Only {len(unique_episode_ids)} unique episodes available.")

      # 1. Sample 'batch_size' episode IDs without replacement
      perm = torch.randperm(len(unique_episode_ids), device=self.device)
      sampled_eids = unique_episode_ids[perm[:batch_size]]

      # --- Vectorized Sampling from Groups ---
      # The goal is to select one random transition from each of the `sampled_eids`.
      # This is a vectorized implementation of "group by episode_id and sample 1".

      # 2. Create a mask to find all transitions belonging to the sampled episodes
      all_eids = self.episode_id[:self.size]
      mask = torch.isin(all_eids, sampled_eids)

      # 3. Get the buffer indices and episode IDs of the relevant transitions
      all_buffer_indices = torch.arange(self.size, device=self.device)
      filtered_indices = all_buffer_indices[mask]
      filtered_eids = all_eids[mask]

      # 4. Sort the filtered transitions by episode ID to group them together
      # This is the key step that enables vectorized processing of the groups.
      sorted_eids, sort_perm = torch.sort(filtered_eids)
      sorted_indices = filtered_indices[sort_perm]

      # 5. Find the boundaries of each episode's segment in the sorted tensor
      # `unique_consecutive` is efficient on sorted data. It gives us the unique
      # episode IDs that were actually found, and the number of transitions for each.
      group_eids, counts = torch.unique_consecutive(sorted_eids, return_counts=True)
      
      if len(group_eids) < batch_size:
        # Not all sampled episodes were found in the buffer.
        # The resulting batch will be smaller than requested.
        print(f"Warning: Only {len(group_eids)} of the {batch_size} sampled episodes were found in the buffer.")

      # 6. Calculate the starting position of each group in the sorted tensor
      segment_starts = torch.cat([torch.tensor([0], device=self.device), torch.cumsum(counts, 0)[:-1]])

      # 7. Generate a random offset for each group to pick one random transition
      rand_per_group = torch.rand(len(counts), device=self.device)
      random_offsets = (rand_per_group * counts).to(torch.long)

      # 8. Get the indices within the `sorted_indices` tensor by adding the offsets
      indices_into_sorted = segment_starts + random_offsets
      
      # 9. Finally, retrieve the original buffer indices
      idxs = sorted_indices[indices_into_sorted]
    else:
      # Standard uniform sampling
      idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
    return idxs

  def sample(self, batch_size, distinct_episodes=False):
    idxs = self.get_sampling_indices(batch_size, distinct_episodes=distinct_episodes)
    return self.get_batch(idxs)
  
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
    return self.get_batch(idxs)
  
  def sample_from_successful_episodes(self, batch_size, distinct_episodes=False):
    """
    Samples a batch of transitions exclusively from episodes that were successful.
    If distinct_episodes is True, each transition in the batch comes from a different episode.
    """
    if not self.successful_episodes:
      return None

    successful_eids = torch.tensor(list(set(self.successful_episodes)), device=self.device)

    if distinct_episodes:
      if len(successful_eids) < batch_size:
        # print(f"Warning: Requested batch size {batch_size} is larger than the number of successful episodes "
        #       f"{len(successful_eids)}. Returning a smaller batch.")
        batch_size = len(successful_eids)

      # Sample episode IDs without replacement
      perm = torch.randperm(len(successful_eids), device=self.device)
      sampled_eids = successful_eids[perm[:batch_size]]

      idxs = []
      for eid in sampled_eids:
        # Find all transitions for the current episode
        episode_indices = torch.where((self.episode_id[:self.size] == eid) &
                                      torch.isin(self.episode_id[:self.size], successful_eids))[0]
        if len(episode_indices) > 0:
          # Sample one transition from this episode
          sample_idx = torch.randint(0, len(episode_indices), (1,), device=self.device).item()
          idxs.append(episode_indices[sample_idx])
      
      if not idxs:
        return None
      idxs = torch.tensor(idxs, device=self.device)
    else:
      # Original behavior: sample from all successful transitions
      mask = torch.isin(self.episode_id[:self.size], successful_eids)
      valid_indices = torch.where(mask)[0]

      if len(valid_indices) == 0:
        return None

      # Sample with replacement from the valid indices
      sample_perms = torch.randint(0, len(valid_indices), (batch_size,), device=self.device)
      idxs = valid_indices[sample_perms]

    return self.get_batch(idxs)
  
  def sample_image_is_goal_batch(self, batch_size, n_fake_goals=4):
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
    valid_episode_ids = torch.tensor(self.successful_episodes, device=self.device)

    # ---- sample only valid episodes ----
    sampled_eids = valid_episode_ids[
      torch.randint(0, len(valid_episode_ids), (B,), device=self.device)
    ]

    idxs, goal_idxs = [], []
    for b, eid in enumerate(sampled_eids):
      ep_mask = (self.episode_id[:self.size] == eid)
      ep_idxs = torch.where(ep_mask)[0]

      if len(ep_idxs) == 0:
        continue

      # ---- goal is final state of the episode ----
      goal_idx = ep_idxs[-1]
      goal_idxs.append(goal_idx.item())

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

      idxs.append(b)

    batch = {
      "image": images,
      "internal_state": states,
      "goal_internal_state": goal_states,
      "goal_image": goal_images,
      "fake_goal_internal_state": fake_goal_states,  # [B, K, D]
      "goal_idx": goal_idxs
    }
    other_vars = {k: v[idxs].to(self.target_device) for k, v in self.other_stored_obj.items()}
    return {**batch, **other_vars}
  
  def sample_episode_batch(self, batch_size, episode_length, random_window=True, success_reward=None, episode_ids=None):
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
      "episode_size": torch.zeros((B,), device=self.target_device, dtype=torch.long),
    }
    batch = {**batch, **{k: torch.zeros((B, T, *v.shape[1:]),
                                        device=self.target_device,
                                        dtype=v.dtype) for k, v in self.other_stored_obj.items()}}
    
    if episode_ids is None:
      if success_reward is None:
        # ---- sample random episode ids ----
        episode_ids = torch.randint(0, self.current_episode_id, (batch_size,), device=self.device)
      else:
        # ---- find episodes whose final transition has reward == success_reward ----
        valid_episode_ids = torch.tensor(self.successful_episodes, device=self.device)

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

      batch["episode_size"][b] = L

      for k, v in self.other_stored_obj.items():
        batch[k][b, :L] = v[window_idxs].to(self.target_device)

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