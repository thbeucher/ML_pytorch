import os
import ast
import gym
import sys
import torch
import pygame
import random
import argparse
import torchvision.transforms as tvt

sys.path.append('../../../robot/')

MIN_ANGLE0, MAX_ANGLE0 = 0, 90  # joint1
MIN_ANGLE1, MAX_ANGLE1 = 0, 180  # joint2
MIN_X_TARGET, MAX_X_TARGET = 125, 350
MIN_Y_TARGET, MAX_Y_TARGET = 150, 300
MAX_MIN_ANGLE0 = MAX_ANGLE0 - MIN_ANGLE0
MAX_MIN_ANGLE1 = MAX_ANGLE1 - MIN_ANGLE1
MAX_MIN_X_TARGET = MAX_X_TARGET - MIN_X_TARGET
MAX_MIN_Y_TARGET = MAX_Y_TARGET - MIN_Y_TARGET
MAXS_MINS = torch.FloatTensor([MAX_MIN_ANGLE0, MAX_MIN_ANGLE1, MAX_MIN_X_TARGET, MAX_MIN_Y_TARGET])
MINS = torch.FloatTensor([MIN_ANGLE0, MIN_ANGLE1, MIN_X_TARGET, MIN_Y_TARGET])
MAXS_MINS_ANGLES = torch.FloatTensor([MAX_MIN_ANGLE0, MAX_MIN_ANGLE1])
MINS_ANGLES = torch.FloatTensor([MIN_ANGLE0, MIN_ANGLE1])


def get_state(my_env):
  screen = tvt.functional.to_tensor(my_env.get_screen())
  screen_cropped = tvt.functional.crop(screen, 140, 115, 180, 245)
  screen_resized = tvt.functional.resize(screen_cropped, [180, 180])
  return screen_resized


def save_img_state(folder='simulation_img_examples/'):
  if not os.path.isdir(folder):
    os.makedirs(folder)
  
  env = gym.make('gym_robot_arm:robot-arm-v1')
  env.reset()
  env.render()

  target_pos = env.target_pos  # [int, int]
  body_info = env.joints_angle  # [float, float]
  state = get_state(env)  # torch.Tensor [3, 180, 180]

  tpx, tpy = target_pos
  ja1, ja2 = round(body_info[0], 2), round(body_info[1], 2)
  torch.save(state, os.path.join(folder, f'target_pos_{tpx}_{tpy}_joints_angle_{ja1}_{ja2}.pt'))

  done = False
  act = False
  action = 0
  while not done:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        done = True
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_a:
          action = 1
          body_info, *_ = env.step(action)  # INC_J1
          act = True
        elif event.key == pygame.K_z:
          action = 2
          body_info, *_ = env.step(action)  # DEC_J1
          act = True
        elif event.key == pygame.K_o:
          action = 3
          body_info, *_ = env.step(action)  # INC_J2
          act = True
        elif event.key == pygame.K_p:
          action = 4
          body_info, *_ = env.step(action)  # DEC_J2
          act = True
    
    env.render()

    if act:
      state = get_state(env)
      ja1, ja2 = round(body_info[0], 2), round(body_info[1], 2)
      torch.save(state, os.path.join(folder, f'target_pos_{tpx}_{tpy}_joints_angle_{ja1}_{ja2}.pt'))


def goal_reaching(save_folder='goal_reaching_dataset/', to_reset='arm', screen_state=False):
  if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

  env = gym.make('gym_robot_arm:robot-arm-v1')
  env.reset()
  env.render()

  act = False
  done = False
  action = 0

  episode_len = 0
  cum_reward = 0
  end_ep = False

  target_pos = env.target_pos

  states = [get_state(env)] if screen_state else [(torch.FloatTensor(env.joints_angle + list(target_pos)) - MINS) / MAXS_MINS]
  body_infos = [(torch.FloatTensor(env.joints_angle) - MINS_ANGLES) / MAXS_MINS_ANGLES]
  actions = []
  rewards = []

  while not done:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        done = True
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_a:
          action = 1
          body_info, reward, end_ep, _ = env.step(action)  # INC_J1
          act = True
        elif event.key == pygame.K_z:
          action = 2
          body_info, reward, end_ep, _ = env.step(action)  # DEC_J1
          act = True
        elif event.key == pygame.K_o:
          action = 3
          body_info, reward, end_ep, _ = env.step(action)  # INC_J2
          act = True
        elif event.key == pygame.K_p:
          action = 4
          body_info, reward, end_ep, _ = env.step(action)  # DEC_J2
          act = True
    
    env.render()

    if act:
      act = False
      cum_reward += reward
      episode_len += 1

      actions.append(action)
      rewards.append(reward)

      if end_ep:
        print(f'End episode with reward/timestep = {cum_reward/episode_len:.3f}')
        torch.save([states, body_infos, actions, rewards],
                   os.path.join(save_folder, f'ep_{len(os.listdir(save_folder))}_len_{episode_len}.pt'))

        cum_reward = 0
        episode_len = 0

        env.reset(to_reset=to_reset)

        states.clear()
        body_infos.clear()
        actions.clear()
        rewards.clear()
        target_pos = env.target_pos
      
      states.append(get_state(env) if screen_state else (torch.FloatTensor(body_info + list(target_pos)) - MINS) / MAXS_MINS)
      body_infos.append((torch.FloatTensor(body_info) - MINS_ANGLES) / MAXS_MINS_ANGLES)

  env.close()


if __name__ == '__main__':
  argparser = argparse.ArgumentParser(prog='behaviour_dataset_maker.py', description='')
  argparser.add_argument('--folder', default='goal_reaching_dataset/', type=str)
  argparser.add_argument('--to_reset', default='target', type=str)
  argparser.add_argument('--screen_state', default=False, type=ast.literal_eval)
  args = argparser.parse_args()

  rep = input(f'Create data for goal reaching experiment? (y or n): ')
  if rep == 'y':
    goal_reaching(save_folder=args.folder, to_reset=args.to_reset, screen_state=args.screen_state)
  
  rep = input(f'Save image state at each of your move? (saved in {args.folder}) (y or n): ')
  if rep == 'y':
    save_img_state(args.folder)