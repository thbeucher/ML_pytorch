import os
import imageio
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from natsort import natsorted


def plot_generated(generated_imgs, dim=(1, 10), figsize=(12, 2), save_name=None):
  plt.figure(figsize=figsize)
  for i in range(generated_imgs.shape[0]):
    plt.subplot(dim[0], dim[1], i+1)
    plt.imshow(generated_imgs[i], interpolation='nearest', cmap='gray_r')
    plt.axis('off')
  plt.tight_layout()
  # plt.draw()
  # plt.pause(0.01)
  if save_name is None:
    plt.show()
  else:
    plt.savefig(save_name)


def create_gif(img_folder, save_gif_name=None, fps=5):
  images = [f'{img_folder}{img}' for img in natsorted(os.listdir(img_folder))]

  gif = []
  for image in images:
      gif.append(imageio.imread(image))
  
  if save_gif_name is None:
    save_gif_namef = f'{img_folder[:-1]}_animated.gif'

  imageio.mimsave(save_gif_namef, gif, fps=fps)


def read_subset_classif_results(log_file):
  with open(log_file, 'r') as f:
    data = f.read().splitlines()
  
  starts_ends = [i for i, el in enumerate(data) if 'START' in el or 'END' in el]
  starts_ends = starts_ends[:len(starts_ends) - len(starts_ends) % 2]

  results = {'epoch': [], 'f1': [], 'label': []}
  percents = []
  for i, idx in enumerate(range(0, len(starts_ends), 2)):
    if i == 0:
      continue
    tmp = data[starts_ends[idx]:starts_ends[idx+1]]
    epochs, f1s = zip(*[(int(l.split(' | ')[0].split('Epoch ')[-1]), float(l.split(' = ')[-1])) for l in tmp if ' | f1 = ' in l])
    results['epoch'] += epochs
    results['f1'] += f1s
    results['label'] += [i] * len(epochs)
    percents.append(tmp[0].split(' = ')[-1])
    print(f'Percent = {percents[-1]} | max f1 = {max(f1s)}')
    
  lp = sns.lineplot(x='epoch', y='f1', hue='label', data=pd.DataFrame.from_dict(results),
                    legend=False, palette=sns.color_palette()[:len(percents)])
  # lp.set(yscale="log")
  plt.legend(title='percent', loc='lower right', labels=percents)
  plt.show()


if __name__ == '__main__':
  argparser = argparse.ArgumentParser(prog='plotter.py', description='')
  argparser.add_argument('--log_file', default='_tmp_classif_exps_mnist_logs.txt', type=str)
  args = argparser.parse_args()

  rep = input('Read subset classification results? (y or n): ')
  if rep == 'y':
    read_subset_classif_results(args.log_file)