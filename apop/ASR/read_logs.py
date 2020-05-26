import os
import re
import ast
import sys
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(__file__).replace('ASR/read_logs.py', ''))
import utils as u

from collections import defaultdict


def fix_logfile(filename):
  with open(filename, 'r') as f:
    data = f.read().splitlines()
  
  data = [el.replace('Epoch = ', 'Epoch ') if 'test_' in el else el for el in data]

  with open(filename, 'w') as f:
    f.write('\n'.join(data))


def get_train_test_epoch_acc(filename, pattern=r'word_accuracy = 0.\d+'):
  # pattern = r'greedy_word_accuracy = 0.\d+' | r'Word accuracy = 0.\d+'
  print(f'Handle {filename}')
  with open(filename, 'r') as f:
    data = f.read().splitlines()
  
  train_lines = [el for el in data if 'train_' in el and '_acc' in el]
  train_epoch_acc = [(int(el.split(' | ')[0].split('Epoch ')[-1]),
                      round(float(re.search(pattern, el).group(0).split(' = ')[-1]), 3))
                        for el in train_lines]
  train_epoch_acc = train_epoch_acc[[i for i, (e, acc) in enumerate(train_epoch_acc) if e == 0][-1]:]

  test_lines = [el for el in data if 'test_' in el and not 'None' in el and 'accuracy' in el]
  test_epoch_acc = [(int(el.split(' | ')[0].split('Epoch ')[-1]),
                      round(float(re.search(pattern, el).group(0).split(' = ')[-1]), 3))
                        for el in test_lines]
  test_epoch_acc = test_epoch_acc[[i for i, (e, acc) in enumerate(test_epoch_acc) if e == 0][-1]:]
  

  return train_epoch_acc, test_epoch_acc


def align_train_test_epochs(train_epoch_acc, test_epoch_acc):
  epochs = list(map(lambda x: x[0], test_epoch_acc))
  train_acc = [el[1] for el in train_epoch_acc if el[0] in epochs]
  test_acc = list(map(lambda x: x[1], test_epoch_acc))
  return epochs, train_acc, test_acc


def analyze(filename):
  print(f'Handle {filename}')

  train_epoch_acc, test_epoch_acc = get_train_test_epoch_acc(filename, pattern=r'Word accuracy = 0.\d+')
  epochs, train_acc, test_acc = align_train_test_epochs(train_epoch_acc, test_epoch_acc)

  ## DUMP train & test accuracy
  top = {'Train acc': train_acc, 'Test acc': test_acc}

  u.board_printing(top)

  ## Plot train & test accuracy
  train_epochs, train_acc = zip(*train_epoch_acc)
  test_epochs, test_acc = zip(*test_epoch_acc)
  df_data = {'epoch': train_epochs + test_epochs, 'accuracy': train_acc + test_acc,
             'label': ['train'] * len(train_epochs) + ['test'] * len(test_epochs)}
  df = pd.DataFrame.from_dict(df_data)
  sns.lineplot(x='epoch', y='accuracy', hue='label', data=df)
  plt.title(f'{filename}')
  plt.show()


def compare_all(folder, read_from_new=True):
  df_data = defaultdict(list)
  top = defaultdict(list)
  names = []
  for fname in os.listdir(folder):
    if not '.txt' in fname:
      continue

    if read_from_new:
      name = fname.replace('_logs_', '').replace('.txt', '')
    else:
      name = fname.replace('_convnet_experiments_', '').replace('.txt', '')  # READ _OLD_LOGS/
    print(f'Reading results from {name}')

    try:
      train_epoch_acc, test_epoch_acc = get_train_test_epoch_acc(f'{folder}{fname}', pattern=r'word_accuracy = 0.\d+')
      epochs, train_acc, test_acc = align_train_test_epochs(train_epoch_acc, test_epoch_acc)
    except:
      continue
    
    df_data['epoch'] += epochs + epochs
    df_data['accuracy'] += train_acc
    df_data['accuracy'] += test_acc
    df_data['label'] += [name] * len(epochs) + [name] * len(epochs)
    df_data['tr_te'] += ['train'] * len(epochs) + ['test'] * len(epochs)

    top[f'{name}_train'] += train_acc
    top[f'{name}_test'] += test_acc
    names += [f'{name}_train', f'{name}_test']
  
  max_len = max(map(len, top.values()))
  top = {names.index(k): v + [0] * (max_len - len(v)) for k, v in top.items()}
  u.board_printing(top)
  print({k: i for i, k in enumerate(names)})
  print({names[k]: max(v) for k, v in top.items() if '_test' in names[k]})
  
  df = pd.DataFrame.from_dict(df_data)

  fig, ax =plt.subplots(1,2)
  ax[0].title.set_text('Train')
  sns.lineplot(x='epoch', y='accuracy', hue='label', data=df[df['tr_te']=='train'], ax=ax[0])
  ax[1].title.set_text('Test')
  sns.lineplot(x='epoch', y='accuracy', hue='label', data=df[df['tr_te']=='test'], ax=ax[1])
  plt.show()


if __name__ == "__main__":
  argparser = argparse.ArgumentParser(prog='read_logs.py', description='Analyze logs')
  argparser.add_argument('--folder', default='_logs/', type=str)
  argparser.add_argument('--logfile', default='_logs/_convnet_experiments_feedback_logs.txt', type=str)
  argparser.add_argument('--read_all', default=False, type=ast.literal_eval)
  argparser.add_argument('--read_from_new', default=True, type=ast.literal_eval)
  args = argparser.parse_args()

  if args.read_all:
    compare_all(args.folder, args.read_from_new)
  else:
    analyze(args.logfile)