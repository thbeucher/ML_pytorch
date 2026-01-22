import os
import sys
import torch
import random
import warnings
import gymnasium as gym

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from trainers_zoo import GANGoalImagePredictor, FlowGoalImagePredictor

sys.path.append('../../../robot/')
warnings.filterwarnings("ignore")


class MultiStepsNSPOrchestrator:
  CONFIG = {
    'experiment_name':          'multisteps_nextstatepredictor_orchestrator',
    'experiment_save_dir':      'experiments/',
    'internal_state_dim':       2,
    'action_dim':               1,
    'image_size':               256,
    'resize_to':                32,
    'normalize_img':            True,
    'n_train_episodes':         100,
    'train_replay_buffer_size': 100*60,
    'n_test_episodes':          10,
    'test_replay_buffer_size':  10*60,  # 10 episode of max size 100, test buffer will certainly not be full
    'render_mode':              'rgb_array',
    'use_tf_logger':            True,
  }
  def __init__(self, config={}):
    self.config = {**MultiStepsNSPOrchestrator.CONFIG, **config}
    self.device = torch.device('cuda' if torch.cuda.is_available() else
                               'mps' if torch.backends.mps.is_available() else
                               'cpu')
    self.env = gym.make("gymnasium_env:RobotArmEnv", render_mode=self.config['render_mode'], cropping=True)
    self.instanciate_trainers()
    self.instanciate_utils()

    save_dir_run = os.path.join(self.config['experiment_save_dir'], self.config['experiment_name'], 'runs/')
    self.tf_logger = SummaryWriter(save_dir_run) if self.config['use_tf_logger'] else None

  def instanciate_trainers(self):
    self.gan_goal_image_trainer = GANGoalImagePredictor()
    self.flow_goal_image_trainer = FlowGoalImagePredictor()
  
  def instanciate_utils(self):
    self.train_replay_buffer = ReplayBuffer(self.config['internal_state_dim'],
                                            self.config['action_dim'],
                                            self.config['image_size'],
                                            resize_to=self.config['resize_to'],
                                            normalize_img=self.config['normalize_img'],
                                            capacity=self.config['train_replay_buffer_size'],
                                            device='cpu',
                                            target_device=self.device)
    self.test_replay_buffer = ReplayBuffer(self.config['internal_state_dim'],
                                           self.config['action_dim'],
                                           self.config['image_size'],
                                           resize_to=self.config['resize_to'],
                                           normalize_img=self.config['normalize_img'],
                                           capacity=self.config['train_replay_buffer_size'],
                                           device='cpu',
                                           target_device=self.device)
  
  def fill_memory(self, replay_buffer, act='random', n_episodes=100, max_episode_steps=60):
    '''act=(random or best or policy)'''
    print(f'Filling memory buffer... ({act=})')
    obs, _ = self.env.reset()
    img = self.env.render()

    for _ in tqdm(range(n_episodes)):
      episode_step = 0
      for _ in range(max_episode_steps):
        if act == 'policy':
          action = random.randint(0, 4)
        elif act == 'best':
          action = self.env.unwrapped.get_best_action()
        else:
          action = random.randint(0, 4)

        next_obs, reward, terminated, truncated, info = self.env.step(action)
        next_img = self.env.render()

        episode_step += 1

        replay_buffer.add(obs//5, action, img, reward, terminated or episode_step > max_episode_steps,
                          next_obs//5, next_img)
        obs, img = next_obs, next_img

        if terminated or episode_step > max_episode_steps:
          obs, _ = self.env.reset()
          img = self.env.render()
          episode_step = 0
          break
  
  def run(self):
    self.fill_memory(self.train_replay_buffer, n_episodes=self.config['n_train_episodes'])
    self.fill_memory(self.test_replay_buffer, n_episodes=self.config['n_test_episodes'], act='best')

    # --- Goal Image prediction experiment using GAN --- #
    # self.gan_goal_image_trainer.load()
    # losses = self.gan_goal_image_trainer.train(lambda x: self.train_replay_buffer.sample_image_is_goal_batch(x),
    #                                            tf_logger=self.tf_logger)
    # rec_loss = self.gan_goal_image_trainer.evaluate(lambda x: self.test_replay_buffer.sample_image_is_goal_batch(x),
    #                                                 tf_logger=self.tf_logger)
    # self.gan_goal_image_trainer.save()
    # -------------------------------------------------- #

    # --- Goal Image prediction experiment using FLOW --- #
    self.flow_goal_image_trainer.load()
    loss = self.flow_goal_image_trainer.train(lambda x: self.train_replay_buffer.sample_image_is_goal_batch(x),
                                              tf_logger=self.tf_logger)
    rec_loss = self.flow_goal_image_trainer.evaluate(lambda x: self.test_replay_buffer.sample_image_is_goal_batch(x),
                                                     tf_logger=self.tf_logger)
    self.flow_goal_image_trainer.save()
    # --------------------------------------------------- #


if __name__ == '__main__':
  ms_nsp_orchestrator = MultiStepsNSPOrchestrator()
  ms_nsp_orchestrator.run()

  # import pygame

  # pygame.init()
  # env = gym.make("gymnasium_env:RobotArmEnv", render_mode='human', cropping=True)

  # obs, _ = env.reset()
  # img = env.render()

  # n_step_ep = 0
  # running = True

  # while running:
  #     # Capture keyboard events
  #     for event in pygame.event.get():
  #       if event.type == pygame.QUIT:  # Close window
  #         running = False
  #       elif event.type == pygame.KEYDOWN:
  #         if event.key == pygame.K_q:  # Quit when 'Q' is pressed
  #           running = False
  #         elif event.key == pygame.K_r:
  #           force_reset = True

  #     # --- Agent action ---
  #     action = env.unwrapped.get_best_action()

  #     obs, reward, terminated, truncated, info = env.step(action)
  #     img = env.render()

  #     n_step_ep += 1

  #     if terminated or truncated:
  #       obs, _ = env.reset(options={'only_target': True})
  #       img = env.render()

  #       if n_step_ep > 50:
  #         print(f'Episode ended in {n_step_ep} steps')
  #       n_step_ep = 0

  # env.close()
  # pygame.quit()