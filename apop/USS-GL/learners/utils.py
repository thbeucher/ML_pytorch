import os
import torch
import numpy as np
import torchvision.transforms as tvt


def get_state(my_env):
  screen = tvt.functional.to_tensor(my_env.get_screen())
  screen_cropped = tvt.functional.crop(screen, 140, 115, 180, 245)
  screen_resized = tvt.functional.resize(screen_cropped, [180, 180])
  return screen_resized


def save_model(model, models_folder='models/', save_name='global_trainer_model.pt', put_in_models_folder=True):
  save_name = os.path.join(models_folder, save_name) if put_in_models_folder else save_name
  if not os.path.isdir(os.path.dirname(save_name)):
    os.makedirs(os.path.dirname(save_name))
  torch.save({'model': model.state_dict()}, save_name)
  print(f'Model saved in {save_name}')
  

def load_model(model, models_folder='models/', save_name='global_trainer_model.pt', put_in_models_folder=True, device=None):
  save_name = os.path.join(models_folder, save_name) if put_in_models_folder else save_name
  if os.path.isfile(save_name):
    data = torch.load(save_name, map_location=device)
    model.load_state_dict(data['model'])
  else:
    print(f"File {save_name} doesn't exist")


def plot_metric(vis, metric, iteration, win='loss', title='loss evolution', ylabel='loss', xlabel='iteration'):
  if iteration == 0:
    vis.line(X=np.array([iteration, iteration]), Y=np.array([metric, metric]), win=win,
                              opts={'ylabel': ylabel, 'xlabel': xlabel, 'title': title})
  else:
    vis.line(X=np.array([iteration]), Y=np.array([metric]), win=win, update='append')


def int_to_bin(my_int, n_digit=8):
  b = format(int(my_int), 'b')
  return [0] * (n_digit - len(b)) + list(map(int, b))