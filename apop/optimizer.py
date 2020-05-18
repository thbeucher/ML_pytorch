import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

############################################################################################################################
### from https://github.com/kaituoxu/Speech-Transformer/blob/master/src/transformer/optimizer.py                         ###
############################################################################################################################
class TransformerOptimizer(object):
  """A simple wrapper class for Noam learning rate scheduling"""
  def __init__(self, optimizer, k, d_model, warmup_steps=4000):
    self.optimizer = optimizer
    self.k = k
    self.init_lr = d_model ** (-0.5)
    self.warmup_steps = warmup_steps
    self.step_num = 0
    self.param_groups = [{'lr': self.init_lr}]

  def zero_grad(self):
    self.optimizer.zero_grad()

  def step(self):
    self._update_lr()
    self.optimizer.step()

  def _update_lr(self):
    self.step_num += 1
    lr = self.k * self.init_lr * min(self.step_num ** (-0.5), self.step_num * (self.warmup_steps ** (-1.5)))
    self.param_groups[0]['lr'] = lr
    for param_group in self.optimizer.param_groups:
      param_group['lr'] = lr

  def load_state_dict(self, state_dict):
    self.optimizer.load_state_dict(state_dict)

  def state_dict(self):
    return self.optimizer.state_dict()

  def set_k(self, k):
    self.k = k


class GoogleLR(_LRScheduler):
	def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1):
		self.current_step = 0
		self.k = d_model ** (-0.5)
		self.ws = warmup_steps ** (-1.5)
		super(GoogleLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		self.current_step += 1
		current_lr = self.k * min(self.current_step ** (-0.5), self.current_step * self.ws)
		return [current_lr for i in range(len(self.base_lrs))]


class LinearLR(_LRScheduler):
  def __init__(self, optimizer, start_lr, end_lr, num_iter, last_epoch=-1):
    self.step_lr = (end_lr - start_lr) / num_iter
    super(LinearLR, self).__init__(optimizer, last_epoch)

  def get_lr(self):
    return [base_lr + self.last_epoch * self.step_lr for base_lr in self.base_lrs]


class LinearLR2(_LRScheduler):
  def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
    self.end_lr = end_lr
    self.num_iter = num_iter
    super(LinearLR2, self).__init__(optimizer, last_epoch)

  def get_lr(self):
    curr_iter = self.last_epoch + 1
    r = curr_iter / self.num_iter
    return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]

############################################################################################################################
### from https://github.com/egg-west/AdamW-pytorch/blob/master/adamW.py                                                  ###
############################################################################################################################
class AdamW(Optimizer):
  """Implements Adam algorithm.
  It has been proposed in `Adam: A Method for Stochastic Optimization`_.
  Arguments:
      params (iterable): iterable of parameters to optimize or dicts defining
          parameter groups
      lr (float, optional): learning rate (default: 1e-3)
      betas (Tuple[float, float], optional): coefficients used for computing
          running averages of gradient and its square (default: (0.9, 0.999))
      eps (float, optional): term added to the denominator to improve
          numerical stability (default: 1e-8)
      weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
      amsgrad (boolean, optional): whether to use the AMSGrad variant of this
          algorithm from the paper `On the Convergence of Adam and Beyond`_
  .. _Adam\: A Method for Stochastic Optimization:
      https://arxiv.org/abs/1412.6980
  .. _On the Convergence of Adam and Beyond:
      https://openreview.net/forum?id=ryQu7f-RZ
  """

  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
    if not 0.0 <= lr:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if not 0.0 <= eps:
      raise ValueError("Invalid epsilon value: {}".format(eps))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    super(AdamW, self).__init__(params, defaults)

  def __setstate__(self, state):
    super(AdamW, self).__setstate__(state)
    for group in self.param_groups:
      group.setdefault('amsgrad', False)

  def step(self, closure=None):
    """Performs a single optimization step.
    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data
        if grad.is_sparse:
          raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
        amsgrad = group['amsgrad']

        state = self.state[p]

        # State initialization
        if len(state) == 0:
          state['step'] = 0
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(p.data)
          # Exponential moving average of squared gradient values
          state['exp_avg_sq'] = torch.zeros_like(p.data)
          if amsgrad:
            # Maintains max of all exp. moving avg. of sq. grad. values
            state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
          max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        # if group['weight_decay'] != 0:
        #     grad = grad.add(group['weight_decay'], p.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
          # Maintains the maximum of all 2nd moment running avg. till now
          torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
          # Use the max. for normalizing running avg. of gradient
          denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
          denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

        # p.data.addcdiv_(-step_size, exp_avg, denom)
        p.data.add_(-step_size,  torch.mul(p.data, group['weight_decay']).addcdiv_(1, exp_avg, denom) )

    return loss


############################################################################################################################
### from https://github.com/jinserk/pytorch-asr/blob/master/asr/utils/lr_scheduler.py                                    ###
############################################################################################################################
class CosineAnnealingWithRestartsLR(_LRScheduler):
  """Set the learning rate of each parameter group using a cosine annealing
  schedule, where :math:`\eta_{max}` is set to the initial lr and
  :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
      \cos(\frac{T_{cur}}{T_{max}}\pi))
    When last_epoch=-1, sets initial lr as lr.
    It has been proposed in
  `SGDR: Stochastic Gradient Descent with Warm Restarts`_. This implements
  the cosine annealing part of SGDR, the restarts and number of iterations multiplier.
    Args:
      optimizer (Optimizer): Wrapped optimizer.
      T_max (int): Maximum number of iterations.
      T_mult (float): Multiply T_max by this number after each restart. Default: 1.
      eta_min (float): Minimum learning rate. Default: 0.
      last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
      https://arxiv.org/abs/1608.03983
  """
  def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, T_mult=1):
    self.T_max = T_max
    self.T_mult = T_mult
    self.restart_every = T_max
    self.eta_min = eta_min
    self.restarts = 0
    self.restarted_at = 0
    super().__init__(optimizer, last_epoch)

  def restart(self):
    self.restart_every *= self.T_mult
    self.restarted_at = self.last_epoch

  def cosine(self, base_lr):
    return self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.step_n / self.restart_every)) / 2

  @property
  def step_n(self):
    return self.last_epoch - self.restarted_at

  def get_lr(self):
    if self.step_n >= self.restart_every:
      self.restart()
    return [self.cosine(base_lr) for base_lr in self.base_lrs]


############################################################################################################################
### from https://github.com/mpyrozhok/adamwr/blob/master/adamw.py                                                        ###
############################################################################################################################
class AdamW2(Optimizer):
  """Implements Adam algorithm.
  Arguments:
      params (iterable): iterable of parameters to optimize or dicts defining
          parameter groups
      lr (float, optional): learning rate (default: 1e-3)
      betas (Tuple[float, float], optional): coefficients used for computing
          running averages of gradient and its square (default: (0.9, 0.999))
      eps (float, optional): term added to the denominator to improve
          numerical stability (default: 1e-8)
      weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
      amsgrad (boolean, optional): whether to use the AMSGrad variant of this
          algorithm from the paper `On the Convergence of Adam and Beyond`_
  """

  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    #super(AdamW, self).__init__(params, defaults)
    super().__init__(params, defaults)

  def step(self, closure=None):
    """Performs a single optimization step.
    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data
        if grad.is_sparse:
          raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
        amsgrad = group['amsgrad']

        state = self.state[p]

        # State initialization
        if len(state) == 0:
          state['step'] = 0
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(p.data)
          # Exponential moving average of squared gradient values
          state['exp_avg_sq'] = torch.zeros_like(p.data)
          if amsgrad:
            # Maintains max of all exp. moving avg. of sq. grad. values
            state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
          max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
          # Maintains the maximum of all 2nd moment running avg. till now
          torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
          # Use the max. for normalizing running avg. of gradient
          denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
          denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

        if group['weight_decay'] != 0:
          decayed_weights = torch.mul(p.data, group['weight_decay'])
          p.data.addcdiv_(-step_size, exp_avg, denom)
          p.data.sub_(decayed_weights)
        else:
          p.data.addcdiv_(-step_size, exp_avg, denom)

    return loss


class ReduceMaxLROnRestart:
  def __init__(self, ratio=0.75):
    self.ratio = ratio

  def __call__(self, eta_min, eta_max):
    return eta_min, eta_max * self.ratio


class ExpReduceMaxLROnIteration:
  def __init__(self, gamma=1):
    self.gamma = gamma

  def __call__(self, eta_min, eta_max, iterations):
    return eta_min, eta_max * self.gamma ** iterations


class CosinePolicy:
  def __call__(self, t_cur, restart_period):
    return 0.5 * (1. + math.cos(math.pi * (t_cur / restart_period)))


class ArccosinePolicy:
  def __call__(self, t_cur, restart_period):
    return (math.acos(max(-1, min(1, 2 * t_cur / restart_period - 1))) / math.pi)


class TriangularPolicy:
  def __init__(self, triangular_step=0.5):
    self.triangular_step = triangular_step

  def __call__(self, t_cur, restart_period):
    inflection_point = self.triangular_step * restart_period
    point_of_triangle = (t_cur / inflection_point
                          if t_cur < inflection_point
                          else 1.0 - (t_cur - inflection_point)
                          / (restart_period - inflection_point))
    return point_of_triangle


############################################################################################################################
### from https://github.com/mpyrozhok/adamwr/blob/master/cyclic_scheduler.py                                             ###
############################################################################################################################
class CyclicLRWithRestarts(_LRScheduler):
  """Decays learning rate with cosine annealing, normalizes weight decay
  hyperparameter value, implements restarts.
  https://arxiv.org/abs/1711.05101
  Args:
      optimizer (Optimizer): Wrapped optimizer.
      batch_size: minibatch size
      epoch_size: training samples per epoch
      restart_period: epoch count in the first restart period
      t_mult: multiplication factor by which the next restart period will expand/shrink
      policy: ["cosine", "arccosine", "triangular", "triangular2", "exp_range"]
      min_lr: minimum allowed learning rate
      verbose: print a message on every restart
      gamma: exponent used in "exp_range" policy
      eta_on_restart_cb: callback executed on every restart, adjusts max or min lr
      eta_on_iteration_cb: callback executed on every iteration, adjusts max or min lr
      triangular_step: adjusts ratio of increasing/decreasing phases for triangular policy
  Example:
      >>> scheduler = CyclicLRWithRestarts(optimizer, 32, 1024, restart_period=5, t_mult=1.2)
      >>> for epoch in range(100):
      >>>     scheduler.step()
      >>>     train(...)
      >>>         ...
      >>>         optimizer.zero_grad()
      >>>         loss.backward()
      >>>         optimizer.step()
      >>>         scheduler.batch_step()
      >>>     validate(...)
  """
  def __init__(self, optimizer, batch_size, epoch_size, restart_period=100,
                t_mult=2, last_epoch=-1, verbose=False,
                policy="cosine", policy_fn=None, min_lr=1e-7,
                eta_on_restart_cb=None, eta_on_iteration_cb=None,
                gamma=1.0, triangular_step=0.5):

    if not isinstance(optimizer, Optimizer):
      raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))

    self.optimizer = optimizer

    if last_epoch == -1:
      for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])
        group.setdefault('minimum_lr', min_lr)
    else:
      for i, group in enumerate(optimizer.param_groups):
        if 'initial_lr' not in group:
          raise KeyError("param 'initial_lr' is not specified "
                          "in param_groups[{}] when resuming an"
                          " optimizer".format(i))

    self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]

    self.min_lrs = [group['minimum_lr'] for group in optimizer.param_groups]

    self.base_weight_decays = [group['weight_decay'] for group in optimizer.param_groups]

    self.policy = policy
    self.eta_on_restart_cb = eta_on_restart_cb
    self.eta_on_iteration_cb = eta_on_iteration_cb
    if policy_fn is not None:
      self.policy_fn = policy_fn
    elif self.policy == "cosine":
      self.policy_fn = CosinePolicy()
    elif self.policy == "arccosine":
      self.policy_fn = ArccosinePolicy()
    elif self.policy == "triangular":
      self.policy_fn = TriangularPolicy(triangular_step=triangular_step)
    elif self.policy == "triangular2":
      self.policy_fn = TriangularPolicy(triangular_step=triangular_step)
      self.eta_on_restart_cb = ReduceMaxLROnRestart(ratio=0.5)
    elif self.policy == "exp_range":
      self.policy_fn = TriangularPolicy(triangular_step=triangular_step)
      self.eta_on_iteration_cb = ExpReduceMaxLROnIteration(gamma=gamma)

    self.last_epoch = last_epoch
    self.batch_size = batch_size
    self.epoch_size = epoch_size

    self.iteration = 0
    self.total_iterations = 0

    self.t_mult = t_mult
    self.verbose = verbose
    self.restart_period = math.ceil(restart_period)
    self.restarts = 0
    self.t_epoch = -1
    self.epoch = -1

    self.eta_min = 0
    self.eta_max = 1

    self.end_of_period = False
    self.batch_increments = []
    self._set_batch_increment()

  def _on_restart(self):
    if self.eta_on_restart_cb is not None:
      self.eta_min, self.eta_max = self.eta_on_restart_cb(self.eta_min, self.eta_max)

  def _on_iteration(self):
    if self.eta_on_iteration_cb is not None:
      self.eta_min, self.eta_max = self.eta_on_iteration_cb(self.eta_min,
                                                            self.eta_max,
                                                            self.total_iterations)

  def get_lr(self, t_cur):
    eta_t = (self.eta_min + (self.eta_max - self.eta_min) * self.policy_fn(t_cur, self.restart_period))

    weight_decay_norm_multi = math.sqrt(self.batch_size / (self.epoch_size * self.restart_period))

    lrs = [min_lr + (base_lr - min_lr) * eta_t for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)]
    weight_decays = [base_weight_decay * eta_t * weight_decay_norm_multi for base_weight_decay in self.base_weight_decays]

    if (self.t_epoch + 1) % self.restart_period < self.t_epoch:
      self.end_of_period = True

    if self.t_epoch % self.restart_period < self.t_epoch:
      if self.verbose:
        print("Restart {} at epoch {}".format(self.restarts + 1, self.last_epoch))
      self.restart_period = math.ceil(self.restart_period * self.t_mult)
      self.restarts += 1
      self.t_epoch = 0
      self._on_restart()
      self.end_of_period = False

    return zip(lrs, weight_decays)

  def _set_batch_increment(self):
    d, r = divmod(self.epoch_size, self.batch_size)
    batches_in_epoch = d + 2 if r > 0 else d + 1
    self.iteration = 0
    self.batch_increments = torch.linspace(0, 1, batches_in_epoch).tolist()

  def step(self):
    self.last_epoch += 1
    self.t_epoch += 1
    self._set_batch_increment()
    self.batch_step()

  def batch_step(self):
    try:
      t_cur = self.t_epoch + self.batch_increments[self.iteration]
      self._on_iteration()
      self.iteration += 1
      self.total_iterations += 1
    except (IndexError):
      raise StopIteration("Epoch size and batch size used in the "
                          "training loop and while initializing "
                          "scheduler should be the same.")

    for param_group, (lr, weight_decay) in zip(self.optimizer.param_groups, self.get_lr(t_cur)):
      param_group['lr'] = lr
      param_group['weight_decay'] = weight_decay


def update_lr(optimizer, lr):
  for g in optimizer.param_groups:
    g['lr'] = lr


def update_mom(optimizer, mom):
  for g in optimizer.param_groups:
    g['momentum'] = mom


############################################################################################################################
### from https://github.com/nachiket273/One_Cycle_Policy/blob/master/OneCycle.py                                         ###
############################################################################################################################
class OneCycle(object):
  """
  In paper (https://arxiv.org/pdf/1803.09820.pdf), author suggests to do one cycle during 
  whole run with 2 steps of equal length. During first step, increase the learning rate 
  from lower learning rate to higher learning rate. And in second step, decrease it from 
  higher to lower learning rate. This is Cyclic learning rate policy. Author suggests one 
  addition to this. - During last few hundred/thousand iterations of cycle reduce the 
  learning rate to 1/100th or 1/1000th of the lower learning rate.
  Also, Author suggests that reducing momentum when learning rate is increasing. So, we make 
  one cycle of momentum also with learning rate - Decrease momentum when learning rate is 
  increasing and increase momentum when learning rate is decreasing.
  Args:
      nb              Total number of iterations including all epochs
      max_lr          The optimum learning rate. This learning rate will be used as highest 
                      learning rate. The learning rate will fluctuate between max_lr to
                      max_lr/div and then (max_lr/div)/div.
      momentum_vals   The maximum and minimum momentum values between which momentum will
                      fluctuate during cycle.
                      Default values are (0.95, 0.85)
      prcnt           The percentage of cycle length for which we annihilate learning rate
                      way below the lower learnig rate.
                      The default value is 10
      div             The division factor used to get lower boundary of learning rate. This
                      will be used with max_lr value to decide lower learning rate boundary.
                      This value is also used to decide how much we annihilate the learning 
                      rate below lower learning rate.
                      The default value is 10.
  
  Usage:
    >>> lr, mom = onecycle.calc()
    >>> update_lr(optimizer, lr)
    >>> update_mom(optimizer, mom)
    
    >>> output = model(var_ip)
    >>> loss = criterion(output, var_tg)
    ...
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
  """
  def __init__(self, nb, max_lr, momentum_vals=(0.95, 0.85), prcnt= 10 , div=10):
    self.nb = nb
    self.div = div
    self.step_len =  int(self.nb * (1- prcnt/100)/2)
    self.high_lr = max_lr
    self.low_mom = momentum_vals[1]
    self.high_mom = momentum_vals[0]
    self.prcnt = prcnt
    self.iteration = 0
    self.lrs = []
    self.moms = []
      
  def calc(self):
    self.iteration += 1
    lr = self.calc_lr()
    mom = self.calc_mom()
    return (lr, mom)
      
  def calc_lr(self):
    if self.iteration==self.nb:
      self.iteration = 0
      self.lrs.append(self.high_lr/self.div)
      return self.high_lr/self.div
    if self.iteration > 2 * self.step_len:
      ratio = (self.iteration - 2 * self.step_len) / (self.nb - 2 * self.step_len)
      lr = self.high_lr * ( 1 - 0.99 * ratio)/self.div
    elif self.iteration > self.step_len:
      ratio = 1- (self.iteration -self.step_len)/self.step_len
      lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
    else :
      ratio = self.iteration/self.step_len
      lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
    self.lrs.append(lr)
    return lr
  
  def calc_mom(self):
    if self.iteration==self.nb:
      self.iteration = 0
      self.moms.append(self.high_mom)
      return self.high_mom
    if self.iteration > 2 * self.step_len:
      mom = self.high_mom
    elif self.iteration > self.step_len:
      ratio = (self.iteration -self.step_len)/self.step_len
      mom = self.low_mom + ratio * (self.high_mom - self.low_mom)
    else :
      ratio = self.iteration/self.step_len
      mom = self.high_mom - ratio * (self.high_mom - self.low_mom)
    self.moms.append(mom)
    return mom
  
  def step(self, whatever):
    pass


############################################################################################################################
### from https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py                                      ###
############################################################################################################################
class CyclicLR(_LRScheduler):
  def __init__(self,
                optimizer,
                base_lr,
                max_lr,
                step_size_up=2000,
                step_size_down=None,
                mode='triangular',
                gamma=1.,
                scale_fn=None,
                scale_mode='cycle',
                cycle_momentum=True,
                base_momentum=0.8,
                max_momentum=0.9,
                last_epoch=-1):

    if not isinstance(optimizer, Optimizer):
      raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
    self.optimizer = optimizer

    base_lrs = self._format_param('base_lr', optimizer, base_lr)
    if last_epoch == -1:
      for lr, group in zip(base_lrs, optimizer.param_groups):
        group['lr'] = lr

    self.max_lrs = self._format_param('max_lr', optimizer, max_lr)

    step_size_up = float(step_size_up)
    step_size_down = float(step_size_down) if step_size_down is not None else step_size_up
    self.total_size = step_size_up + step_size_down
    self.step_ratio = step_size_up / self.total_size

    if mode not in ['triangular', 'triangular2', 'exp_range'] and scale_fn is None:
      raise ValueError('mode is invalid and scale_fn is None')

    self.mode = mode
    self.gamma = gamma

    if scale_fn is None:
      if self.mode == 'triangular':
        self.scale_fn = self._triangular_scale_fn
        self.scale_mode = 'cycle'
      elif self.mode == 'triangular2':
        self.scale_fn = self._triangular2_scale_fn
        self.scale_mode = 'cycle'
      elif self.mode == 'exp_range':
        self.scale_fn = self._exp_range_scale_fn
        self.scale_mode = 'iterations'
    else:
      self.scale_fn = scale_fn
      self.scale_mode = scale_mode

    self.cycle_momentum = cycle_momentum
    if cycle_momentum:
      if 'momentum' not in optimizer.defaults:
        raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')

      base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
      if last_epoch == -1:
        for momentum, group in zip(base_momentums, optimizer.param_groups):
          group['momentum'] = momentum
      self.base_momentums = list(map(lambda group: group['momentum'], optimizer.param_groups))
      self.max_momentums = self._format_param('max_momentum', optimizer, max_momentum)

    super(CyclicLR, self).__init__(optimizer, last_epoch)

  def _format_param(self, name, optimizer, param):
    """Return correctly formatted lr/momentum for each param group."""
    if isinstance(param, (list, tuple)):
      if len(param) != len(optimizer.param_groups):
        raise ValueError("expected {} values for {}, got {}".format(
          len(optimizer.param_groups), name, len(param)))
      return param
    else:
      return [param] * len(optimizer.param_groups)

  def _triangular_scale_fn(self, x):
    return 1.

  def _triangular2_scale_fn(self, x):
    return 1 / (2. ** (x - 1))

  def _exp_range_scale_fn(self, x):
    return self.gamma**(x)

  def get_lr(self):
    """Calculates the learning rate at batch index. This function treats
    `self.last_epoch` as the last batch index.

    If `self.cycle_momentum` is ``True``, this function has a side effect of
    updating the optimizer's momentum.
    """
    cycle = math.floor(1 + self.last_epoch / self.total_size)
    x = 1. + self.last_epoch / self.total_size - cycle
    if x <= self.step_ratio:
      scale_factor = x / self.step_ratio
    else:
      scale_factor = (x - 1) / (self.step_ratio - 1)

    lrs = []
    for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
      base_height = (max_lr - base_lr) * scale_factor
      if self.scale_mode == 'cycle':
        lr = base_lr + base_height * self.scale_fn(cycle)
      else:
        lr = base_lr + base_height * self.scale_fn(self.last_epoch)
      lrs.append(lr)

    if self.cycle_momentum:
      momentums = []
      for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
        base_height = (max_momentum - base_momentum) * scale_factor
        if self.scale_mode == 'cycle':
          momentum = max_momentum - base_height * self.scale_fn(cycle)
        else:
          momentum = max_momentum - base_height * self.scale_fn(self.last_epoch)
        momentums.append(momentum)
      for param_group, momentum in zip(self.optimizer.param_groups, momentums):
        param_group['momentum'] = momentum

    return lrs


############################################################################################################################
### from https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py                                                    ###
############################################################################################################################
class RAdam(Optimizer):
  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    self.buffer = [[None, None, None] for ind in range(10)]
    super(RAdam, self).__init__(params, defaults)

  def __setstate__(self, state):
    super(RAdam, self).__setstate__(state)

  def step(self, closure=None):
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data.float()
        if grad.is_sparse:
          raise RuntimeError('RAdam does not support sparse gradients')

        p_data_fp32 = p.data.float()

        state = self.state[p]

        if len(state) == 0:
          state['step'] = 0
          state['exp_avg'] = torch.zeros_like(p_data_fp32)
          state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
        else:
          state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
          state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        exp_avg.mul_(beta1).add_(1 - beta1, grad)

        state['step'] += 1
        buffered = self.buffer[int(state['step'] % 10)]
        if state['step'] == buffered[0]:
          N_sma, step_size = buffered[1], buffered[2]
        else:
          buffered[0] = state['step']
          beta2_t = beta2 ** state['step']
          N_sma_max = 2 / (1 - beta2) - 1
          N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
          buffered[1] = N_sma

          # more conservative since it's an approximated value
          if N_sma >= 5:
            step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
          else:
            step_size = 1.0 / (1 - beta1 ** state['step'])
          buffered[2] = step_size

        if group['weight_decay'] != 0:
          p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

        # more conservative since it's an approximated value
        if N_sma >= 5:            
          denom = exp_avg_sq.sqrt().add_(group['eps'])
          p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
        else:
          p_data_fp32.add_(-step_size * group['lr'], exp_avg)

        p.data.copy_(p_data_fp32)

    return loss


class PlainRAdam(Optimizer):
  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    super(PlainRAdam, self).__init__(params, defaults)

  def __setstate__(self, state):
    super(PlainRAdam, self).__setstate__(state)

  def step(self, closure=None):
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data.float()
        if grad.is_sparse:
          raise RuntimeError('RAdam does not support sparse gradients')

        p_data_fp32 = p.data.float()

        state = self.state[p]

        if len(state) == 0:
          state['step'] = 0
          state['exp_avg'] = torch.zeros_like(p_data_fp32)
          state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
        else:
          state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
          state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        exp_avg.mul_(beta1).add_(1 - beta1, grad)

        state['step'] += 1
        beta2_t = beta2 ** state['step']
        N_sma_max = 2 / (1 - beta2) - 1
        N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

        if group['weight_decay'] != 0:
          p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

        # more conservative since it's an approximated value
        if N_sma >= 5:                    
          step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
          denom = exp_avg_sq.sqrt().add_(group['eps'])
          p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
        else:
          step_size = group['lr'] / (1 - beta1 ** state['step'])
          p_data_fp32.add_(-step_size, exp_avg)

        p.data.copy_(p_data_fp32)

    return loss

############################################################################################################################
### from https://github.com/Luolc/AdaBound/blob/master/adabound/adabound.py                                              ###
############################################################################################################################
class AdaBound(Optimizer):
  """Implements AdaBound algorithm.
  It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
  Arguments:
      params (iterable): iterable of parameters to optimize or dicts defining
          parameter groups
      lr (float, optional): Adam learning rate (default: 1e-3)
      betas (Tuple[float, float], optional): coefficients used for computing
          running averages of gradient and its square (default: (0.9, 0.999))
      final_lr (float, optional): final (SGD) learning rate (default: 0.1)
      gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
      eps (float, optional): term added to the denominator to improve
          numerical stability (default: 1e-8)
      weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
      amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
  .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
      https://openreview.net/forum?id=Bkg3g2R9FX
  """
  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
               eps=1e-8, weight_decay=0, amsbound=False):
    if not 0.0 <= lr:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if not 0.0 <= eps:
      raise ValueError("Invalid epsilon value: {}".format(eps))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
    if not 0.0 <= final_lr:
      raise ValueError("Invalid final learning rate: {}".format(final_lr))
    if not 0.0 <= gamma < 1.0:
      raise ValueError("Invalid gamma parameter: {}".format(gamma))
    defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                    weight_decay=weight_decay, amsbound=amsbound)
    super(AdaBound, self).__init__(params, defaults)

    self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

  def __setstate__(self, state):
    super(AdaBound, self).__setstate__(state)
    for group in self.param_groups:
      group.setdefault('amsbound', False)

  def step(self, closure=None):
    """Performs a single optimization step.
    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
      loss = closure()

    for group, base_lr in zip(self.param_groups, self.base_lrs):
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data
        if grad.is_sparse:
          raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
        amsbound = group['amsbound']

        state = self.state[p]

        # State initialization
        if len(state) == 0:
          state['step'] = 0
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(p.data)
          # Exponential moving average of squared gradient values
          state['exp_avg_sq'] = torch.zeros_like(p.data)
          if amsbound:
            # Maintains max of all exp. moving avg. of sq. grad. values
            state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsbound:
          max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        if group['weight_decay'] != 0:
          grad = grad.add(group['weight_decay'], p.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsbound:
          # Maintains the maximum of all 2nd moment running avg. till now
          torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
          # Use the max. for normalizing running avg. of gradient
          denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
          denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

        # Applies bounds on actual learning rate
        # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
        final_lr = group['final_lr'] * group['lr'] / base_lr
        lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
        upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
        step_size = torch.full_like(denom, step_size)
        step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

        p.data.add_(-step_size)

    return loss

############################################################################################################################
### from https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annearing_with_warmup.py  ###
############################################################################################################################
class CosineAnnealingWarmUpRestarts(_LRScheduler):
  def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
    if T_0 <= 0 or not isinstance(T_0, int):
      raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
    if T_mult < 1 or not isinstance(T_mult, int):
      raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
    if T_up < 0 or not isinstance(T_up, int):
      raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
    self.T_0 = T_0
    self.T_mult = T_mult
    self.base_eta_max = eta_max
    self.eta_max = eta_max
    self.T_up = T_up
    self.T_i = T_0
    self.gamma = gamma
    self.cycle = 0
    super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    self.T_cur = last_epoch
  
  def get_lr(self):
    if self.T_cur == -1:
      return self.base_lrs
    elif self.T_cur < self.T_up:
      return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
    else:
      return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
              for base_lr in self.base_lrs]

  def step(self, epoch=None):
    if epoch is None:
      epoch = self.last_epoch + 1
      self.T_cur = self.T_cur + 1
      if self.T_cur >= self.T_i:
        self.cycle += 1
        self.T_cur = self.T_cur - self.T_i
        self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
    else:
      if epoch >= self.T_0:
        if self.T_mult == 1:
          self.T_cur = epoch % self.T_0
          self.cycle = epoch // self.T_0
        else:
          n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
          self.cycle = n
          self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
          self.T_i = self.T_0 * self.T_mult ** (n)
      else:
        self.T_i = self.T_0
        self.T_cur = epoch
            
    self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
    self.last_epoch = math.floor(epoch)
    for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
      param_group['lr'] = lr