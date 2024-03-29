import os
import ast
import gym
import sys
import torch
import random
import logging
import argparse
import numpy as np
import torchvision.transforms as tvt

from tqdm import tqdm
from collections import deque
from torchvision.utils import make_grid, save_image

sys.path.append('../../../robot/')

sys.path.append(os.path.abspath(__file__).replace('USS-GL/rec_exps.py', ''))
import utils as u
import brain_exps as be


class AutoEncoder(torch.nn.Module):
  def __init__(self, gdn_act=False, add_body_infos=False, normalize_emb=False, dropout=0.2):
    super().__init__()
    self.normalize_emb = normalize_emb
    self.encoder = be.ImageEmbedder(gdn_act=gdn_act, dropout=dropout)
    self.decoder = be.ImageReconstructor(gdn_act=gdn_act, n_input_features=258 if add_body_infos else 256)
  
  def forward(self, x, body_infos=None):  # [B, C, H, W], [B, 2]
    code = self.encoder(x)  # -> [B, 256]

    if self.normalize_emb:
      code = torch.nn.functional.normalize(code, p=2, dim=1)

    if body_infos is not None:
      code = torch.cat([code, body_infos], dim=1)  # [B, 258]

    return self.decoder(code)
  
  @staticmethod
  def separate_saved_autoencoder(save_name='models/auto_encoder_state.pt'):
    autoencoder = AutoEncoder()
    u.load_model(autoencoder, save_name)
    u.save_checkpoint(autoencoder.encoder, None, 'models/autoencoder_state_encoder.pt')
    u.save_checkpoint(autoencoder.decoder, None, 'models/autoencoder_state_decoder.pt')


def get_game_env(game_view=False):
  env = gym.make('gym_robot_arm:robot-arm-v1')
  env.config['in_background'] = not game_view
  render_mode = 'human' if game_view else 'robot'
  env.reset()
  env.render(mode=render_mode)
  return env, render_mode


def get_state(my_env):
  screen = tvt.functional.to_tensor(my_env.get_screen())
  screen_cropped = tvt.functional.crop(screen, 140, 115, 180, 245)
  screen_resized = tvt.functional.resize(screen_cropped, [180, 180])
  return screen_resized


def random_apply_transform(batch, p=0.5):
  mask = torch.rand(batch.size(0))
  new_batch = batch.clone()
  for i, rv in enumerate(mask):
    if rv > p and rv <= p + (1-p) / 2:
      new_batch[i] = u.add_noise(new_batch[i])
    elif rv > p + (1-p) / 2:
      new_batch[i] = tvt.functional.gaussian_blur(new_batch[i], kernel_size=(5, 9), sigma=(0.1, 5))
  return new_batch, mask


def batch_random_black_rect(batch, p=0.5):
  mask = torch.rand(batch.size(0))
  new_batch = batch.clone()
  for i, rv in enumerate(mask):
    if rv > p:
      new_batch[i] = random_black_rect(new_batch[i])
  return new_batch, mask


def random_black_rect(img_tensor):
  x1, y1, x2, y2 = tvt.RandomCrop.get_params(img_tensor, (50, 50))
  img_tensor[:, x1:x1+x2, y1:y1+y2] = 0
  return img_tensor


def get_min_maxmin_joints():
  MIN_ANGLE0, MAX_ANGLE0 = 0, 90  # joint1
  MIN_ANGLE1, MAX_ANGLE1 = 0, 180  # joint2
  MAX_MIN_ANGLE0 = MAX_ANGLE0 - MIN_ANGLE0
  MAX_MIN_ANGLE1 = MAX_ANGLE1 - MIN_ANGLE1
  MINS = torch.FloatTensor([MIN_ANGLE0, MIN_ANGLE1])
  MAXS_MINS = torch.FloatTensor([MAX_MIN_ANGLE0, MAX_MIN_ANGLE1])
  return MINS, MAXS_MINS


def compute_separate_loss(states, rec_states, criterion, white_coef=0.9):
  white_mask = states.sum(1, keepdim=True).repeat(1, 3, 1, 1) == 3
  white_loss = criterion(rec_states[white_mask], states[white_mask])

  other_mask = ~white_mask
  other_loss = criterion(rec_states[other_mask], states[other_mask])

  loss = (1 - white_coef) * other_loss + white_coef * white_loss
  return loss


def collect_states(n_states, max_ep_len=50, env=None, render_mode=None, add_body_infos=False, MINS=None, MAXS_MINS=None):
  env, render_mode = get_game_env(game_view=False) if env is None else env

  if add_body_infos and (MINS == None or MAXS_MINS == None):
    MINS, MAXS_MINS = get_min_maxmin_joints()

  states = [get_state(env)]
  body_infos = []

  for i in range(n_states-1):
    action = random.randint(0, 4)
    joints_angle, reward, target_reached, _ = env.step(action)

    env.render(mode=render_mode)

    states.append(get_state(env))

    if add_body_infos:
      body_infos.append((torch.FloatTensor(joints_angle) - MINS) / MAXS_MINS)

    if target_reached or i % max_ep_len == 0:
      env.reset(to_reset='target')
  
  env.close()
  
  return (states, body_infos) if add_body_infos else states


def train_autoencoder(model, optimizer, criterion, n_epochs=10, batch_size=32, memory_size=1024, vp=None, start_epoch=0):
  states = collect_states(memory_size)
  random.shuffle(states)

  for epoch in tqdm(range(start_epoch, n_epochs + start_epoch)):
    losses = []
    for i in tqdm(range(0, memory_size, batch_size), leave=False):
      batch = torch.stack(states[i:i+batch_size])

      rec_batch = model(batch)

      optimizer.zero_grad()
      loss = criterion(rec_batch, batch)
      loss.backward()
      optimizer.step()

      losses.append(loss.item())
    
    epoch_loss = np.mean(losses)
    logging.info(f'Epoch {epoch} | loss={epoch_loss:.5f}')

    if vp is not None:
      vp.line_plot('Loss', 'Train', f'Training Loss={type(criterion).__name__}', epoch, epoch_loss, x_label='Epoch')

      tmp = torch.empty(6, 3, 180, 180)
      tmp[:3] = batch[:3]
      tmp[3:] = rec_batch[:3]
      vp.image_plot('img', make_grid(tmp, nrow=3), opts={'title': 'Image_n_reconstruction'})
    
    random.shuffle(states)


def reconstruction_experiment(use_visdom=True, batch_size=32, gdn_act=False, lr=1e-3, loss_type='ms_ssim',
                              n_epochs=10, memory_size=1024):
  if use_visdom:
    vp = u.VisdomPlotter()
  
  model = AutoEncoder(gdn_act=gdn_act)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  criterion = be.MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3) if loss_type == 'ms_ssim' else torch.nn.MSELoss()

  start_epoch = 0
  while True:
    train_autoencoder(model, optimizer, criterion, n_epochs=n_epochs, batch_size=batch_size, memory_size=memory_size, vp=vp,
                start_epoch=start_epoch)
    start_epoch += n_epochs


def visdom_plotting(vp, epoch, epoch_loss, loss_type, batch, rec_batch, gdn_act):
  vp.line_plot('Loss', 'Train', f'Training Loss={loss_type}', epoch, epoch_loss, x_label='Epoch')

  tmp = torch.empty(6, 3, 180, 180)
  tmp[:3] = batch[:3]
  tmp[3:] = rec_batch[:3]
  vp.image_plot('img', make_grid(tmp, nrow=3), opts={'title': 'Image_n_reconstruction'})
  save_image(make_grid(tmp, nrow=3), f'rec_img_exps_loss_{loss_type}_gdn_{gdn_act}.png')


def train_autoencoder_incr(model, optimizer, criterion, states, batch_size=32, n_epochs=5, start_epoch=0, vp=None,
                           gdn_act=False, device=None, add_augmentation=False, body_infos=None, use_separate_loss=True,
                           add_obfuscation=False, loss_type='mse'):  # loss_type = 'mse' or 'ce'
  if device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  if body_infos is None:
    random.shuffle(states)
    states = list(states)
  else:
    states, body_infos = u.shuffle_lists(list(states), list(body_infos))

  for epoch in tqdm(range(start_epoch, n_epochs + start_epoch)):
    losses = []
    for i in tqdm(range(0, len(states), batch_size), leave=False):
      batch_states = torch.stack(states[i:i+batch_size]).to(device)
      batch_states_to_rec = batch_states.clone()
      batch_body_infos = torch.stack(body_infos[i:i+batch_size]).to(device) if body_infos is not None else None

      if batch_states.size(0) <= 1:
        continue

      if add_augmentation:
        batch_states_to_rec, _ = random_apply_transform(batch_states_to_rec)
      
      if add_obfuscation:
        batch_states_to_rec, _ = batch_random_black_rect(batch_states_to_rec)

      rec_batch = model(batch_states_to_rec, body_infos=batch_body_infos)

      optimizer.zero_grad()
      if loss_type in ['mse', 'ms_ssim']:
        loss = compute_separate_loss(batch_states, rec_batch, criterion) if use_separate_loss else criterion(rec_batch, batch_states)
      else:
        rec_distri = torch.distributions.Independent(torch.distributions.Normal(rec_batch, 1), 3)
        loss = -torch.mean(rec_distri.log_prob(batch_states_to_rec))
      loss.backward()
      optimizer.step()

      losses.append(loss.item())

    epoch_loss = np.mean(losses)
    logging.info(f'Epoch {epoch} | loss={epoch_loss:.5f}')

    if vp is not None:
      with torch.no_grad():
        model.eval()
        rec_batch = model(batch_states, body_infos=batch_body_infos)
        model.train()
      visdom_plotting(vp, epoch, epoch_loss, type(criterion).__name__, batch_states.cpu(), rec_batch.cpu(), gdn_act)


def reconstruction_experiment_incr(use_visdom=True, batch_size=32, gdn_act=False, lr=1e-3, loss_type='ms_ssim', dropout=0.2,
                                   n_epochs=5, memory_size=1024, max_ep_len=200, add_augmentation=False, add_body_infos=False,
                                   use_separate_loss=True, normalize_emb=False, max_n_epochs=100, add_obfuscation=False,
                                   save_model=False, save_name='models/auto_encoder_state.pt'):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  vp = u.VisdomPlotter() if use_visdom else None
  
  model = AutoEncoder(gdn_act=gdn_act, add_body_infos=add_body_infos, normalize_emb=normalize_emb, dropout=dropout).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  criterion = be.MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3) if loss_type == 'ms_ssim' else torch.nn.MSELoss()
  
  env, render_mode = get_game_env(game_view=False)

  start_epoch = 0
  current_ep_len = 1
  target_reached = False
  MINS, MAXS_MINS = get_min_maxmin_joints()

  states = deque([get_state(env)], maxlen=memory_size)
  body_infos = deque([(torch.FloatTensor(env.joints_angle) - MINS) / MAXS_MINS], maxlen=memory_size)
  while True:
    action = random.randint(0, 4)
    joints_angle, reward, target_reached, _ = env.step(action)

    env.render(mode=render_mode)

    states.append(get_state(env))
    body_infos.append((torch.FloatTensor(joints_angle) - MINS) / MAXS_MINS)

    if target_reached or current_ep_len % max_ep_len == 0:
      train_autoencoder_incr(model, optimizer, criterion, states, batch_size=batch_size, n_epochs=n_epochs,
                             vp=vp, gdn_act=gdn_act, device=device, add_augmentation=add_augmentation,
                             add_obfuscation=add_obfuscation, start_epoch=start_epoch, loss_type=loss_type,
                             body_infos=body_infos if add_body_infos else None, use_separate_loss=use_separate_loss)

      start_epoch += n_epochs
      current_ep_len = 0

      if start_epoch >= max_n_epochs:
        break

      env.reset(to_reset='target')
    
    current_ep_len += 1
  
  env.close()

  if save_model:
    u.save_checkpoint(model, None, save_name)


def train_action_predictor(model, optimizer, criterion, training_data, batch_size=32, device=None, n_epochs=5,
                           start_epoch=0, vp=None):
  if device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  stack_fn = lambda x: torch.stack(x).to(device)
  
  random.shuffle(training_data)
  training_data = list(training_data)

  for epoch in tqdm(range(start_epoch, n_epochs + start_epoch)):
    n_good_preds = 0
    for i in tqdm(range(0, len(training_data), batch_size), leave=False):
      body_infos, next_body_infos, actions = map(list, zip(*training_data[i:i+batch_size]))
      body_infos, next_body_infos = map(stack_fn, [body_infos, next_body_infos])  # [B, 2]
      actions = torch.LongTensor(actions).to(device)  # [B]

      actions_logits = model(body_infos, next_body_infos)

      optimizer.zero_grad()
      loss = criterion(actions_logits, actions)
      loss.backward()
      optimizer.step()

      n_good_preds += (actions == actions_logits.argmax(-1)).sum().item()
    
    accuracy = n_good_preds/len(training_data)
    logging.info(f'Epoch {epoch} | accuracy={accuracy:.3f}')

    if accuracy > 0.98:
      for param_group in optimizer.param_groups:
        if param_group['lr'] == 1e-2:
          param_group['lr'] = 1e-3
          print('!LR UPDATED to 1e-3!!!!!!!!!!!!!!!!')

    if vp is not None:
      vp.line_plot('Accuracy', 'Train', 'Training accuracy', epoch, accuracy, x_label='Epoch')


def float_tensor_to_binary(my_tensor):
  int_to_bin = lambda x: torch.Tensor(list(map(int, f'{x:010b}')))
  return torch.cat([int_to_bin(int(el*1000)) for el in my_tensor])


def predict_action_experiment(use_visdom=True, lr=1e-2, memory_size=1024, max_ep_len=200, batch_size=32, n_epochs=5,
                              max_n_epochs=100, save_model=False, save_name='models/action_predictor_BI.pt'):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  vp = u.VisdomPlotter() if use_visdom else None

  model = be.ActionPredictorWbodyinfos().to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  criterion = torch.nn.CrossEntropyLoss()

  env, render_mode = get_game_env(game_view=False)
  MINS, MAXS_MINS = get_min_maxmin_joints()

  start_epoch = 0
  current_ep_len = 1
  target_reached = False
  training_data = deque([], maxlen=memory_size)
  body_infos = (torch.FloatTensor(env.joints_angle) - MINS) / MAXS_MINS
  # body_infos = float_tensor_to_binary((torch.FloatTensor(env.joints_angle) - MINS) / MAXS_MINS)

  while True:
    action = random.randint(0, 4)
    joints_angle, reward, target_reached, _ = env.step(action)

    env.render(mode=render_mode)

    next_body_infos = (torch.FloatTensor(joints_angle) - MINS) / MAXS_MINS
    # next_body_infos = float_tensor_to_binary((torch.FloatTensor(joints_angle) - MINS) / MAXS_MINS)

    if action != 0 and torch.equal(body_infos, next_body_infos):
      action = 0  # action that doesn't change the state so cannot be predicted from body_infos

    training_data.append([body_infos, next_body_infos, action])

    if target_reached or current_ep_len % max_ep_len == 0:
      train_action_predictor(model, optimizer, criterion, training_data, batch_size=batch_size, device=device, n_epochs=n_epochs,
                             start_epoch=start_epoch, vp=vp)

      current_ep_len = 0
      start_epoch += n_epochs

      if start_epoch >= max_n_epochs:
        break

      env.reset(to_reset='target')
      env.render(mode=render_mode)
      body_infos = (torch.FloatTensor(env.joints_angle) - MINS) / MAXS_MINS
      # body_infos = float_tensor_to_binary((torch.FloatTensor(env.joints_angle) - MINS) / MAXS_MINS)
    else:
      body_infos = next_body_infos.clone()
    
    current_ep_len += 1
  
  env.close()

  if save_model:
    u.save_checkpoint(model, None, save_name)


def train_next_body_infos_predictor(model, optimizer, criterion, training_data, batch_size=32, device=None, n_epochs=5,
                                    start_epoch=0, vp=None):
  if device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  stack_fn = lambda x: torch.stack(x).to(device)
  
  random.shuffle(training_data)
  training_data = list(training_data)

  for epoch in tqdm(range(start_epoch, n_epochs + start_epoch)):
    losses = []
    for i in tqdm(range(0, len(training_data), batch_size), leave=False):
      body_infos, next_body_infos, actions = map(list, zip(*training_data[i:i+batch_size]))
      body_infos, next_body_infos = map(stack_fn, [body_infos, next_body_infos])  # [B, 2]
      actions = torch.nn.functional.one_hot(torch.LongTensor(actions), num_classes=5).to(device)  # [B, 5]

      next_body_infos_predicted = model(body_infos, actions)

      optimizer.zero_grad()
      loss = criterion(next_body_infos_predicted, next_body_infos)
      loss.backward()
      optimizer.step()

      losses.append(loss.item())
    
    epoch_loss = np.mean(losses)
    logging.info(f'Epoch {epoch} | loss={epoch_loss:.6f}')

    if vp is not None:
      vp.line_plot('Loss', 'Train', 'Training MSE loss', epoch, epoch_loss, x_label='Epoch')


def predict_next_state_experiment(use_visdom=True, lr=1e-2, memory_size=1024, max_ep_len=200, batch_size=32, n_epochs=5,
                                  max_n_epochs=50, save_model=False, save_name='models/next_state_predictor_BI.pt'):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  vp = u.VisdomPlotter() if use_visdom else None

  model = be.NextBodyInfosPredictor().to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  criterion = torch.nn.MSELoss()

  env, render_mode = get_game_env(game_view=False)
  MINS, MAXS_MINS = get_min_maxmin_joints()

  start_epoch = 0
  current_ep_len = 1
  target_reached = False
  training_data = deque([], maxlen=memory_size)
  body_infos = (torch.FloatTensor(env.joints_angle) - MINS) / MAXS_MINS
  # body_infos = float_tensor_to_binary((torch.FloatTensor(env.joints_angle) - MINS) / MAXS_MINS)

  while True:
    action = random.randint(0, 4)
    joints_angle, reward, target_reached, _ = env.step(action)

    env.render(mode=render_mode)

    next_body_infos = (torch.FloatTensor(joints_angle) - MINS) / MAXS_MINS
    # next_body_infos = float_tensor_to_binary((torch.FloatTensor(joints_angle) - MINS) / MAXS_MINS)

    training_data.append([body_infos, next_body_infos, action])

    if target_reached or current_ep_len % max_ep_len == 0:
      train_next_body_infos_predictor(model, optimizer, criterion, training_data, batch_size=batch_size, device=device,
                                      n_epochs=n_epochs, start_epoch=start_epoch, vp=vp)

      current_ep_len = 0
      start_epoch += n_epochs

      if start_epoch >= max_n_epochs:
        break

      env.reset(to_reset='target')
      env.render(mode=render_mode)
      body_infos = (torch.FloatTensor(env.joints_angle) - MINS) / MAXS_MINS
      # body_infos = float_tensor_to_binary((torch.FloatTensor(env.joints_angle) - MINS) / MAXS_MINS)
    else:
      body_infos = next_body_infos.clone()
    
    current_ep_len += 1
  
  env.close()

  if save_model:
    u.save_checkpoint(model, None, save_name)


if __name__ == '__main__':
  # Observations :
  # -> if the learning batch is composed of followed step it will not be able to learn as
  #    the sgd assumption on iid is more than not respected
  # -> same in reconstruction_experiment_incr, if we not use random.shuffle, it also struggle to converge
  # -> perform training on small subset of same element before gradually increase dataset size
  #    seems to allow a fast convergence (reconstruction_experiment_incr >> reconstruction_experiment)
  #    small overfitting to start learning is beneficial?
  # -> reducing n_epochs on reconstruction_experiment_incr seems to allow a faster convergence,
  #    it may be because it avoid temporary small overfitting?
  #    But leaving n_epochs higher allow the network to more easily reconstruct the red target
  # -> reconstruction_experiment_incr converge in ~20 epochs achieved in ~6min
  # -> reconstruction_experiment struggle to reconstruct the red target and is not able to do it
  #    even after 20 epochs achieved in ~10min
  # -> ms_ssim loss is slower and struggle to reconstruct the red target
  # -> gde activation with same hyperparameter seems to make convergence slower and is computationnally heavier
  # -> between 2 runs, the reconstruction of the red point vary a lot and often it struggles to reconstruct it
  # -> By using a separate loss for white pixels and other pixels and keeping a ratio giving more importance
  #    to the white loss (as there is way more white pixels) allow a fast convergence to a good reconstruction
  # -> Adding batch_norm in ImageEmbedder & ImageReconstructor slow down the color learning but stabilize the
  #    shape learning
  argparser = argparse.ArgumentParser(prog='rec_exps.py', description='')
  argparser.add_argument('--log_file', default='_tmp_rec_exps_logs.txt', type=str)
  argparser.add_argument('--gdn_act', default=False, type=ast.literal_eval)
  argparser.add_argument('--use_visdom', default=True, type=ast.literal_eval)
  argparser.add_argument('--save_model', default=True, type=ast.literal_eval)
  argparser.add_argument('--load_model', default=True, type=ast.literal_eval)
  argparser.add_argument('--normalize_emb', default=False, type=ast.literal_eval)
  argparser.add_argument('--train_embedder', default=False, type=ast.literal_eval)
  argparser.add_argument('--add_body_infos', default=False, type=ast.literal_eval)
  argparser.add_argument('--add_obfuscation', default=False, type=ast.literal_eval)
  argparser.add_argument('--add_augmentation', default=False, type=ast.literal_eval)
  argparser.add_argument('--use_separate_loss', default=True, type=ast.literal_eval)
  argparser.add_argument('--seed', default=42, type=int)
  argparser.add_argument('--n_epochs', default=5, type=int)
  argparser.add_argument('--batch_size', default=32, type=int)
  argparser.add_argument('--max_ep_len', default=200, type=int)
  argparser.add_argument('--memory_size', default=1024, type=int)
  argparser.add_argument('--max_n_epochs', default=100, type=int)
  argparser.add_argument('--lr', default=1e-3, type=float)
  argparser.add_argument('--dropout', default=0.2, type=float)
  argparser.add_argument('--loss_type', default='mse', type=str, choices=['mse', 'ms_ssim', 'ce'])
  argparser.add_argument('--force_training', default=False, type=ast.literal_eval)
  args = argparser.parse_args()

  logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')
  
  rep = input('Start AutoEncoder training? (y or n): ') if not args.force_training else 'y'
  if rep == 'y':
    logging.info('>AutoEncoder Training<')
    # seeding for reproducibility
    random.seed(args.seed * args.seed)
    torch.manual_seed(args.seed)

    u.dump_argparser_parameters(args)

    # reconstruction_experiment(use_visdom=args.use_visdom, batch_size=args.batch_size, gdn_act=args.gdn_act,
    #                           lr=args.lr, loss_type=args.loss_type, n_epochs=args.n_epochs, memory_size=args.memory_size)
    reconstruction_experiment_incr(use_visdom=args.use_visdom, batch_size=args.batch_size, gdn_act=args.gdn_act, lr=args.lr,
                                   loss_type=args.loss_type, n_epochs=args.n_epochs, memory_size=args.memory_size,
                                   max_ep_len=args.max_ep_len, add_augmentation=args.add_augmentation,
                                   add_body_infos=args.add_body_infos, use_separate_loss=args.use_separate_loss,
                                   normalize_emb=args.normalize_emb, max_n_epochs=args.max_n_epochs, dropout=args.dropout,
                                   add_obfuscation=args.add_obfuscation, save_model=args.save_model)
  
  rep = input('Start ActionPredictor training? (y or n): ')
  if rep == 'y':
    logging.info('>ActionPredictor Training<')  # LR = 1e-2
    # seeding for reproducibility
    random.seed(args.seed * args.seed)
    torch.manual_seed(args.seed)

    u.dump_argparser_parameters(args)

    predict_action_experiment(use_visdom=args.use_visdom, batch_size=args.batch_size, lr=args.lr, n_epochs=args.n_epochs,
                              memory_size=args.memory_size, max_ep_len=args.max_ep_len, max_n_epochs=args.max_n_epochs,
                              save_model=args.save_model)
  
  rep = input('Start NextBodyInfosPredictor training? (y or n): ')
  if rep == 'y':
    logging.info('>NextBodyInfosPredictor Training<')
    # seeding for reproducibility
    random.seed(args.seed * args.seed)
    torch.manual_seed(args.seed)

    u.dump_argparser_parameters(args)

    predict_next_state_experiment(use_visdom=args.use_visdom, batch_size=args.batch_size, lr=args.lr, n_epochs=args.n_epochs,
                                  memory_size=args.memory_size, max_ep_len=args.max_ep_len, max_n_epochs=args.max_n_epochs,
                                  save_model=args.save_model)
  

  # Example to update lr if threshold
  # if loss < 0.005:
  #   for param_group in optimizer.param_groups:
  #     if param_group['lr'] == 1e-3:
  #       param_group['lr'] = 1e-4
  #       print('!LR UPDATED!!!!!!!!!!!!!!!!')