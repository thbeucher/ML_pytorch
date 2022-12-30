import torch


def compute_discounted_reward_MCE(rewards, discount_factor=0.99):
  '''
  Args:
    * rewards : list of int
  '''
  discounted_reward = 0
  returns = []
  for reward in rewards[::-1]:
    discounted_reward = reward + discount_factor * discounted_reward
    returns.insert(0, discounted_reward)
  return returns


def get_returns(rewards, normalize_returns=True, discount_factor=0.99, eps=1e-8):
  returns = compute_discounted_reward_MCE(rewards, discount_factor=discount_factor)

  returns = torch.tensor(returns)
  if normalize_returns and len(rewards) > 1:
    returns = (returns - returns.mean()) / (returns.std() + eps)
  
  return returns


def reinforce_update(rewards, log_probs, distri_entropy, optimizer, state_values=None, normalize_returns=True,
                     discount_factor=0.99, eps=1e-8, clear_data=True, coef_mse_critic=0.5, coef_entropy=0.01):
  '''
  Args:
    * rewards : list of int
    * log_probs : list of torch.Tensor
    * optimizer : torch.optim.Optimizer object
    * state_values (optional) : list of torch.Tensor, compute advantage and critic loss if provided
  '''
  returns = get_returns(rewards, normalize_returns=normalize_returns, discount_factor=discount_factor, eps=eps)

  if state_values is not None:
    state_values = torch.stack(state_values).squeeze(dim=1)
    advantages = returns - state_values.detach()

    critic_loss = coef_mse_critic * torch.nn.functional.mse_loss(state_values, returns)
    loss = -torch.stack(log_probs) * advantages + critic_loss - coef_entropy * distri_entropy
  else:
    loss = -torch.stack(log_probs) * returns - coef_entropy * distri_entropy

  # Perform optimization step
  optimizer.zero_grad()
  loss.sum().backward()
  optimizer.step()

  if clear_data:
    rewards.clear()
    log_probs.clear()


def ppo_update():
  '''
  '''
  pass