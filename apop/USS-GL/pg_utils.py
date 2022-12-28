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


def reinforce_update(rewards, log_probs, optimizer, normalize_returns=True, discount_factor=0.99, eps=1e-8, clear_data=True):
  '''
  Args:
    * rewards : list of int
    * log_probs : list of torch.Tensor
    * optimizer : torch.optim.Optimizer object
  '''
  returns = compute_discounted_reward_MCE(rewards, discount_factor=discount_factor)

  returns = torch.tensor(returns)
  if normalize_returns and len(rewards) > 1:
    returns = (returns - returns.mean()) / (returns.std() + eps)
  
  policy_loss = -torch.stack(log_probs) * returns

  # Perform optimization step
  optimizer.zero_grad()
  policy_loss = policy_loss.sum()
  policy_loss.backward()
  optimizer.step()

  if clear_data:
    rewards.clear()
    log_probs.clear()


def ppo_update():
  '''
  '''
  pass