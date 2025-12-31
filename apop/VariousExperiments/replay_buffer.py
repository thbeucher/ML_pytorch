import torch
from torchvision import transforms


class ReplayBuffer:
  def __init__(self, internal_state_dim, action_dim, image_size, image_chan=3, resize_to=None,
               normalize_img=False, capacity=10_000, device='cpu', target_device=None):
    self.capacity = capacity

    self.device = torch.device(device)
    self.target_device = target_device if target_device is not None else self.device

    img_transforms = [transforms.ToTensor()]  # Maps to [0, 1]
    if normalize_img:
      img_transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # Maps to [-1, 1]
    if resize_to is not None:
      img_transforms.append(transforms.Resize(resize_to))
      image_size = resize_to
    self.image_transforms = transforms.Compose(img_transforms)

    self.internal_state = torch.zeros((capacity, internal_state_dim), device=self.device, dtype=torch.long)
    self.next_internal_state = torch.zeros((capacity, internal_state_dim), device=self.device, dtype=torch.long)
    self.action = torch.zeros((capacity, action_dim), device=self.device, dtype=torch.long)
    self.image = torch.zeros((capacity, image_chan, image_size, image_size), device=self.device, dtype=torch.float32)
    self.next_image = torch.zeros((capacity, image_chan, image_size, image_size),
                                  device=self.device, dtype=torch.float32)
    self.reward = torch.zeros((capacity, 1), device=self.device, dtype=torch.float32)
    self.done = torch.zeros((capacity, 1), device=self.device, dtype=torch.long)

    self.ptr = 0
    self.size = 0

  @torch.no_grad()
  def add(self, internal_state, action, image, reward, done, next_internal_state, next_image):
    if not torch.is_tensor(internal_state):
      internal_state = torch.as_tensor(internal_state, device=self.device, dtype=torch.long)
    if not torch.is_tensor(action):
      action = torch.as_tensor(action, device=self.device, dtype=torch.long)
    if not torch.is_tensor(image):
      image = self.image_transforms(image)
    if not torch.is_tensor(next_internal_state):
      next_internal_state = torch.as_tensor(next_internal_state, device=self.device, dtype=torch.long)
    if not torch.is_tensor(next_image):
      next_image = self.image_transforms(next_image)

    self.internal_state[self.ptr].copy_(internal_state.to(self.device))
    self.action[self.ptr].copy_(action.to(self.device))
    self.image[self.ptr].copy_(image.to(self.device))
    self.reward[self.ptr] = reward
    self.done[self.ptr] = done
    self.next_internal_state[self.ptr].copy_(next_internal_state.to(self.device))
    self.next_image[self.ptr].copy_(next_image.to(self.device))

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

  def __len__(self):
    return self.size