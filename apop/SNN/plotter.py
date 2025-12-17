import re
import h5py
import torch
import imageio
import argparse
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict


def plot_lr(log_file):
  with open(log_file, 'r') as f:
    log_data = [l.strip() for l in f.read().splitlines()]

  # 2021-06-03 10:44:47,591 - INFO - New learning rates = (0.00800000037997961,-0.003000000026077032)
  lrs = [el for el in log_data if 'New learning rates' in el]
  lrs = [re.findall(r"0\.\d+", el) for el in lrs]
  
  data = defaultdict(list)
  for i, (ap, an) in enumerate(lrs):
    data['lr'].append(float(ap))
    data['label'].append('ap')
    data['lr'].append(float(an))
    data['label'].append('an')
    data['x'] += [i, i]

  df = pd.DataFrame.from_dict(data)
  
  sns.lineplot(x='x', y='lr', data=df.head(20), hue='label')
  plt.show()


def plot_sparsity_ratio(log_data, mean_grouping_size=5000, layers_idx=[0, 1]):
  # 2021-06-03 10:44:47,587 - INFO - layer_idx=0 | sparsity_ratio=0.9836973547935486
  for layer_idx in layers_idx:
    layer = [el for el in log_data if 'sparsity_ratio' in el and f'layer_idx={layer_idx}' in el]
    sr = [float(re.findall(r"0\.\d+", el)[0]) for el in layer]

    mean_sr = [np.mean(sr[i:i+mean_grouping_size]) for i in range(0, len(sr), mean_grouping_size)]
    mean_x = [i for i in range(0, len(sr), mean_grouping_size)]

    max_sr, min_sr = max(mean_sr), min(mean_sr)
    norm_mean_sr = [(el - min_sr) / (max_sr - min_sr) for el in mean_sr]
    max_x, min_x = max(mean_x), min(mean_x)
    norm_mean_x = [(el - min_x) / (max_x - min_x) for el in mean_x]
    slopes = [(norm_mean_sr[i] - norm_mean_sr[i-1]) / (norm_mean_x[i] - norm_mean_x[i-1]) for i in range(1, len(norm_mean_sr))]
    # print(slopes)
    # print([s < 0.15 for s in slopes])
    stability, patience = False, 0
    for s in slopes:
      if s < 0.15:
        patience += 1
      else:
        patience = 0
      if patience > 5:
        stability = True
    print(f'STABILITY REACHED = {stability}')
    
    data = {'x': mean_x, 'y': mean_sr}
    df = pd.DataFrame.from_dict(data)

    sns.lineplot(x='x', y='y', data=df)
    plt.grid()
    plt.show()


def plot_convergence_value(log_data):
  # 2021-06-08 11:58:07,755 - INFO - EXP2 - epoch=0 - step=25000 - C=0.014322965405881405
  data = [el for el in log_data if 'C=' in el]
  convergence_value = [float(re.findall(r"0\.\d+", el)[0]) for el in data]
  x = list(range(len(convergence_value)))

  data = {'x': x, 'convergence_value': convergence_value}
  df = pd.DataFrame.from_dict(data)

  sns.lineplot(x='x', y='convergence_value', data=df)
  plt.grid()
  plt.show()


def test_animation():
  arrays = [torch.Tensor(32, 2, 5, 5).normal_(i, 0.02) for i in np.linspace(0.1, 0.9, 30)]
  x = np.linspace(0, 1, 20)
  data = []
  for arr in arrays:
    xx, yy = [], []
    for i, el in enumerate(x[:-1]):
      xx.append(round((el + x[i+1])/2, 2))
      yy.append(arr[(arr >= el) & (arr < x[i+1])].count_nonzero().item())
    data.append([xx, yy])

  filenames = []
  for i in range(30):
    fig = matplotlib.figure.Figure(figsize=(10,6))
    plt.xlim(0, 1)
    plt.ylim(0, max([max(el[1]) for el in data]))
    plt.xlabel('W', fontsize=20)
    plt.ylabel('n', fontsize=20)
    plt.title('Weights distribution', fontsize=20)

    sns.barplot(x='x', y='y', data=pd.DataFrame.from_dict({'x': data[i][0], 'y': data[i][1]}))
    plt.savefig(f'plots/plot_{i}.png')
    plt.clf()
    filenames.append(f'plots/plot_{i}.png')
  
  with imageio.get_writer('weights_distribution.gif', mode='I') as writer:
    for filename in filenames:
      image = imageio.imread(filename)
      writer.append_data(image)


def get_distri_from_kernel_arr(kernel, min_distri=0, max_distri=1, n_bins=20):
  x = np.linspace(min_distri, max_distri, n_bins)
  xx, yy = [], []
  for i, el in enumerate(x[:-1]):
    xx.append(round((el + x[i+1])/2, 2))
    # yy.append(kernel[(kernel >= el) & (kernel < x[i+1])].count_nonzero().item())
    yy.append(np.count_nonzero(kernel[(kernel >= el) & (kernel < x[i+1])]))
  return xx, yy


def create_figures(xx_yy, fname_root='plots/plot_{}.png', x_lim_min=0, x_lim_max=1, fig_size=(10, 6)):
  filenames = []
  for i in tqdm(range(len(xx_yy))):
    fig = matplotlib.figure.Figure(figsize=fig_size)
    plt.xlim(x_lim_min, x_lim_max)
    plt.ylim(0, max([max(el[1]) for el in xx_yy]))
    plt.xlabel('W', fontsize=20)
    plt.ylabel('n', fontsize=20)
    plt.title(f'Weights distribution {i}', fontsize=20)

    sns.barplot(x='x', y='y', data=pd.DataFrame.from_dict({'x': xx_yy[i][0], 'y': xx_yy[i][1]}))
    plt.savefig(fname_root.format(i))
    plt.clf()
    filenames.append(fname_root.format(i))
  return filenames


def create_gif(filenames, gif_name='weights_distribution.gif'):
  with imageio.get_writer(gif_name, mode='I') as writer:
    for filename in filenames:
      image = imageio.imread(filename)
      writer.append_data(image)


def create_gif_kernel_distri_training(log_file, save_file):
  data = h5py.File(log_file, 'r')
  xx_yy = [get_distri_from_kernel_arr(data[k][()]) for k in data.keys()]
  print(f'Number of data points = {len(xx_yy)}')
  fnames = create_figures(xx_yy, fname_root='plots/weights_distri_{}.png')
  create_gif(fnames, gif_name=save_file)


def create_gif_distri_nonzero(data_fname='_tmp_nonzero_elmt_timestep_along_training_layer_0.h5',
                              save_gif_file='nonzero_progtimestep_layer_0.gif', choose_n=100, prop=True):
  data = h5py.File(data_fname, 'r')
  epoch_iter = [list(map(lambda x: int(x), k.split('_'))) for k in data.keys()]
  epoch_iter = sorted(epoch_iter, key=lambda x: (x[0], x[1]))
  epoch_iter = epoch_iter[::len(epoch_iter)//choose_n]
  xx_yy = [[list(range(len(data[f'{ei[0]}_{ei[1]}']))), data[f'{ei[0]}_{ei[1]}'][()]] for ei in epoch_iter]
  # xx_yy = xx_yy[::len(xx_yy)//choose_n]
  if prop:
    new_xx_yy = []
    for x_y in xx_yy:
      max_y = sum(x_y[1])
      new_xx_yy.append([x_y[0], [y/max_y for y in x_y[1]]])
    xx_yy = new_xx_yy
  fnames = create_figures(xx_yy, fname_root='plots/nonzero_distri_{}.png')
  create_gif(fnames, gif_name=save_gif_file)


def plot_lr_anti_lr(logfile):
  with open(logfile, 'r') as f:
    data = f.read().splitlines()
  
  lrs = [el for el in data if 'new_ap =' in el]
  anti_lrs = [el for el in data if 'new_anti_ap' in el]

  ap, an = zip(*[(float(el.split(' | ')[0].split(' = ')[-1]), float(el.split(' | ')[-1].split(' = ')[-1])) for el in lrs])
  anti_ap, anti_an = zip(*[(float(el.split(' | ')[0].split(' = ')[-1]), float(el.split(' | ')[-1].split(' = ')[-1])) for el in anti_lrs])

  df = pd.DataFrame.from_dict({'x': list(range(len(ap))) * 4,
                               'y': ap + an + anti_ap + anti_an,
                               'label': ['ap'] * len(ap) + ['an'] * len(ap) + ['anti_ap'] * len(ap) + ['anti_an'] * len(ap)})
  
  sns.lineplot(x='x', y='y', data=df, hue='label')
  plt.show()


def plot_network_performance(logfile, take_last=True):
  with open(logfile, 'r') as f:
    data = f.read().splitlines()

  if take_last:
    data = data[max([i for i, d in enumerate(data) if 'Layer 0 - Epoch 0 - n_data 0' in d]):]
  
  network_perf = [float(el.split(' = ')[-1]) for el in data if 'Network performance' in el]
  layer2_epochs = [int(el.split(' - ')[3].replace('Epoch ', '')) for el in data if 'Layer 2' in el]
  layer2_epochs = [e for i, e in enumerate(layer2_epochs) if i == 0 or layer2_epochs[i-1] != e]

  perf_per_epoch = [p for i, p in enumerate(network_perf) if i % 3 == 0]  # 3 network perf for each epoch - 0, 20K, 40k

  df = pd.DataFrame.from_dict({'epoch': list(range(len(perf_per_epoch))), 'f1': perf_per_epoch})
  sns.lineplot(x='epoch', y='f1', data=df)
  plt.show()


def plot_lr_stdp():
  # new_ap = torch.min(self.learning_rates[i][0][0] * 2, self.config['max_ap'])
  # new_an = self.learning_rates[i][0][0] * self.config['an_update']
  ap, an = 0.004, -0.003

  lrs = [[ap, an]]
  for i in range(100):
    an = ap * -0.75
    ap = min(ap * 2, 0.15)
    lrs.append([ap, an])

  data = defaultdict(list)
  for i, (ap, an) in enumerate(lrs):
    data['lr'].append(float(ap))
    data['label'].append('ap')
    data['lr'].append(float(an))
    data['label'].append('an')
    data['x'] += [i, i]

  df = pd.DataFrame.from_dict(data)
  
  sns.lineplot(x='x', y='lr', data=df.head(20), hue='label')
  plt.show()


if __name__ == "__main__":
  argparser = argparse.ArgumentParser(prog='plotter.py', description='')
  argparser.add_argument('--log_file', default='_tmp_sdcnnOR_logs.txt', type=str)
  argparser.add_argument('--save_file', default='weights_distribution.gif', type=str)
  args = argparser.parse_args()

  rep = input('Plot learning rates? (y or n): ')
  if rep == 'y':
    # plot_lr(args.log_file)
    plot_lr_stdp()
  
  # rep = input('Plot sparsity ratio? (y or n): ')
  # if rep == 'y':
  #   plot_sparsity_ratio(log_data)
  
  # rep = input('Plot convergence value? (y or n): ')
  # if rep == 'y':
  #   plot_convergence_value(log_data)
  
  # test_animation()
  # create_gif_kernel_distri_training(args.log_file, args.save_file)

  rep = input('Plot lr and anti-lr? (y or n): ')
  if rep == 'y':
    plot_lr_anti_lr(args.log_file)
  # plot_network_performance(args.log_file)

  rep = input('Create gif distribution of nonzero elmt? (y or n): ')
  if rep == 'y':
    # create_gif_distri_nonzero()
    # create_gif_distri_nonzero(data_fname='_tmp_nonzero_elmt_timestep_along_training_layer_1.h5',
    #                           save_gif_file='nonzero_progtimestep_layer_1.gif')
    create_gif_distri_nonzero(data_fname='_tmp_nonzero_elmt_timestep_along_training_layer_2.h5',
                              save_gif_file='nonzero_progtimestep_layer_2.gif')