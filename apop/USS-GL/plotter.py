import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from collections import defaultdict


def check_interaction_time(logfile):
  with open(logfile, 'r') as f:
    data = f.read().splitlines()
  
  interactions = [l for l in data if 'interaction with the environment' in l]
  interactions = [l.split('achieved with ')[-1].split(' interaction')[0] for l in interactions]
  print(f'interactions:\n{interactions}')

  run_time = [l.split('Run done in ')[-1] for l in data if 'Run done in' in l]
  print(f'run_time:\n{run_time}')


def runs_to_df_pg_exps(logfile, df_dict=None, name=None):
  runs = parse_pg_exps_log_file(filename=logfile)

  if df_dict is None:
    df_dict = defaultdict(list)

  for run_n, run_data in runs.items():
    df_dict['episode'] += run_data['episodes']
    df_dict['time_to_target'] += run_data['time_to_target']
    df_dict['run_number'] += [run_n] * len(run_data['episodes'])

    if name is not None:
      df_dict['name'] += [name] * len(run_data['episodes'])

  df = pd.DataFrame.from_dict(df_dict)
  return df, df_dict


def runs_to_df_rec_exps(logfile, df_dict=None, name=None):
  runs = parse_rec_exps_log_file(filename=logfile)

  if df_dict is None:
    df_dict = defaultdict(list)

  for run_n, run_data in runs.items():
    df_dict['epoch'] += run_data['epoch']
    df_dict['loss'] += run_data['loss']
    df_dict['run_number'] += [run_n] * len(run_data['epoch'])

    if name is not None:
      df_dict['name'] += [name] * len(run_data['epoch'])

  df = pd.DataFrame.from_dict(df_dict)
  return df, df_dict


def plot_pg_seed_exps(logfiles=[], names=[]):
  df_dict = defaultdict(list)
  for logfile, name in zip(logfiles, names):
    _, df_dict = runs_to_df_pg_exps(logfile, df_dict=df_dict, name=name)
  
  df = pd.DataFrame.from_dict(df_dict)
  sns.lineplot(data=df, x='episode', y='time_to_target', hue='name')
  plt.show()


def plot_rec_seed_exps(logfiles=[], names=[]):
  df_dict = defaultdict(list)
  for logfile, name in zip(logfiles, names):
    _, df_dict = runs_to_df_rec_exps(logfile, df_dict=df_dict, name=name)
  
  df = pd.DataFrame.from_dict(df_dict)
  sns.lineplot(data=df, x='epoch', y='loss', hue='name')
  plt.show()


def parse_rec_exps_log_file(filename='_tmp_rec_exps_logs.txt'):
  # 2023-01-25 09:08:52,805 - INFO - Epoch 0 | loss=0.09731
  with open(filename, 'r') as f:
    data = f.read().splitlines()
  
  data = [l for l in data if 'Epoch' in l]

  runs, run_number = {}, 0
  for l in data:
    l = l.split(' | ')
    epoch = int(l[-2].split('Epoch ')[-1])
    loss = float(l[-1].split('=')[-1])

    if epoch == 0:
      run_number += 1
      runs[run_number] = defaultdict(list)
    
    runs[run_number]['epoch'].append(epoch)
    runs[run_number]['loss'].append(loss)
  
  return runs


def parse_pg_exps_log_file(filename='_tmp_pg_exps_logs.txt'):
  # 2022-12-28 01:02:40,728 - INFO - Episode 1(x100) | Average time-to-target=1.078
  with open(filename, 'r') as f:
    data = f.read().splitlines()
  
  data = [l for l in data if 'Average time-to-target' in l]
  
  runs, run_number = {}, 0
  for l in data:
    ttt = float(l.split('=')[-1])
    ep = int(l.split('Episode ')[-1].split('(')[0])
    t = datetime.strptime(l.split(' - ')[0], "%Y-%m-%d %H:%M:%S,%f")

    if ep == 1:
      run_number += 1
      runs[run_number] = defaultdict(list)
    
    runs[run_number]['time_to_target'].append(ttt)
    runs[run_number]['episodes'].append(ep)
    runs[run_number]['times'].append(t)
  
  return runs


def plot_pg_exps_log_file(filename='_tmp_pg_exps_logs.txt'):
  # times, episodes, time_to_target = parse_pg_exps_log_file(filename=filename)
  runs = parse_pg_exps_log_file(filename=filename)

  starts_time = []
  steps03 = []

  fig, ax = plt.subplots()

  for run_nb, run_data in runs.items():
    min_ttt = min(run_data['time_to_target'])
    ep_min_ttt = run_data['episodes'][run_data['time_to_target'].index(min_ttt)]
    print(f'Run {run_nb} -> Min time-to-target={min_ttt} reached in episode {ep_min_ttt}')

    step03 = [t for i, t in enumerate(run_data['times']) if run_data['time_to_target'][i] <= 0.3]
    if len(step03) > 0:
      starts_time.append(run_data['times'][0])
      steps03.append(step03[0])

    ax.plot(run_data['episodes'], run_data['time_to_target'])
  
  steps03_str = '\n'.join([f"Time taken to reach 0.3 = {step03 - start_time}" for start_time, step03 in zip(starts_time, steps03)])
  ax.text(0.85, 0.85, steps03_str, transform=ax.transAxes, ha="right", va="top")

  plt.xlabel('Episode')
  plt.ylabel('Time-to-Target')
  plt.title('Policy Performance')
  plt.show()


if __name__ == '__main__':
  argparser = argparse.ArgumentParser(prog='plotter.py', description='')
  argparser.add_argument('--log_file', default='_tmp_pg_exps_logs.txt', type=str)
  args = argparser.parse_args()

  rep = input(f'Plot data from {args.log_file} (supposed to come from pg_exps.py)? (y or n): ')
  if rep == 'y':
    plot_pg_exps_log_file(filename=args.log_file)