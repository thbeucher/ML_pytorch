import os
import sys
import torch
import random
import warnings
import gymnasium as gym

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import trainers_zoo as tz
from replay_buffer import ReplayBuffer

sys.path.append('../../../robot/')
warnings.filterwarnings("ignore")


class MultiStepsNSPOrchestrator:
  CONFIG = {
    'experiment_name':          'multisteps_nextstatepredictor_orchestrator_nspe',
    'experiment_save_dir':      'experiments/',
    'internal_state_dim':       2,
    'action_dim':               1,
    'image_size':               256,
    'resize_to':                32,
    'normalize_img':            True,
    'n_train_episodes':         100,
    'train_buffer_size':        100*60,
    'n_test_episodes':          10,
    'test_buffer_size':         10*60,  # 10 episode of max size 100, test buffer will certainly not be full
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
    self.gan_goal_image_trainer = tz.GANGoalImagePredictorTrainer()
    self.flow_goal_image_trainer = tz.FlowGoalImagePredictorTrainer()
    self.is_predictor_trainer = tz.ISPredictorTrainer()
    self.ae_goal_image_trainer = tz.CNNAETrainer({'experiment_name': 'ae_goal_image_predictor',
                                                  'model_config': {'encoder_archi': 'BigCNNEncoder',
                                                                   'skip_connection': False,
                                                                   'linear_bottleneck': True}})
    self.ae_ensp_trainer = tz.CNNAEEmbNSPTrainer()
    self.mb_gep_trainer = tz.MemoryBankGoalEmbeddingPredictorTrainer()
  
  def instanciate_utils(self):
    self.train_buffer = ReplayBuffer(self.config['internal_state_dim'],
                                     self.config['action_dim'],
                                     self.config['image_size'],
                                     resize_to=self.config['resize_to'],
                                     normalize_img=self.config['normalize_img'],
                                     capacity=self.config['train_buffer_size'],
                                     device='cpu',
                                     target_device=self.device)
    self.test_buffer = ReplayBuffer(self.config['internal_state_dim'],
                                    self.config['action_dim'],
                                    self.config['image_size'],
                                    resize_to=self.config['resize_to'],
                                    normalize_img=self.config['normalize_img'],
                                    capacity=self.config['train_buffer_size'],
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

        replay_buffer.add(obs//5, action, img, reward, terminated or episode_step >= max_episode_steps,
                          next_obs//5, next_img)
        obs, img = next_obs, next_img

        if terminated or episode_step >= max_episode_steps:
          obs, _ = self.env.reset()
          img = self.env.render()
          episode_step = 0
          break
  
  def add_var_to_memory(self, buffer, var_generator, var_name, source_tensor, tensor_like=None, batch_size=128):
    '''Add a variable to the replay buffer by generating it with the provided generator and
    adding it in batches.'''
    print(f'Add {var_name} to buffer...')
    new_var = torch.zeros_like(source_tensor if tensor_like is None else tensor_like, device=self.device)
    for i in tqdm(range(0, buffer.size, batch_size)):
      new_var[i:i+batch_size] = var_generator.infer(source_tensor[i:i+batch_size].to(self.device))
    buffer.add_variable(new_var, var_name)

  def run(self):
    self.fill_memory(self.train_buffer, n_episodes=self.config['n_train_episodes'])
    self.fill_memory(self.test_buffer, n_episodes=self.config['n_test_episodes'], act='best')

    ######################################################
    # --- Goal Image prediction experiment using GAN --- #
    ######################################################
    # self.gan_goal_image_trainer.load()
    # losses = self.gan_goal_image_trainer.train(lambda x: self.train_buffer.sample_image_is_goal_batch(x),
    #                                            tf_logger=self.tf_logger)
    # rec_loss = self.gan_goal_image_trainer.evaluate(lambda x: self.test_buffer.sample_image_is_goal_batch(x),
    #                                                 tf_logger=self.tf_logger)
    # self.gan_goal_image_trainer.save()
    # -------------------------------------------------- #

    #######################################################
    # --- Goal Image prediction experiment using FLOW --- #
    #######################################################
    # self.flow_goal_image_trainer.load()
    # loss = self.flow_goal_image_trainer.train(lambda x: self.train_buffer.sample_image_is_goal_batch(x),
    #                                           tf_logger=self.tf_logger)
    # rec_loss = self.flow_goal_image_trainer.evaluate(lambda x: self.test_buffer.sample_image_is_goal_batch(x),
    #                                                  tf_logger=self.tf_logger)
    # self.flow_goal_image_trainer.save()
    # print('Filling Replay Buffer with Generated Goal Image...')
    # self.add_var_to_memory(self.train_buffer,
    #                        self.flow_goal_image_trainer,
    #                        'goal_image_generated',
    #                        self.train_buffer.image)
    # self.tf_logger.add_images('goal_image_generated_train_examples',
    #                           self.train_buffer.other_stored_obj['goal_image_generated'][:24], 1)
    
    # self.add_var_to_memory(self.test_buffer,
    #                        self.flow_goal_image_trainer,
    #                        'goal_image_generated',
    #                        self.test_buffer.image)
    # self.tf_logger.add_images('goal_image_generated_test_examples',
    #                           self.test_buffer.other_stored_obj['goal_image_generated'][:24], 1)
    # --------------------------------------------------- #

    ######################################################################
    # --- Internal State prediction from Generated Image using a CNN --- #
    ######################################################################
    # self.is_predictor_trainer.load()
    # loss = self.is_predictor_trainer.train(lambda x: self.train_buffer.sample_image_is_goal_batch(x),
    #                                        tf_logger=self.tf_logger)
    # acc1, acc2 = self.is_predictor_trainer.evaluate(lambda x: self.test_buffer.sample_image_is_goal_batch(x),
    #                                                 tf_logger=self.tf_logger)
    # print(f'Test mean accuracy: {(acc1+acc2)/2:.3f}')
    # self.is_predictor_trainer.save()
    # -------------------------------------------------------- #

    ########################################################################
    # --- Goal Internal State Prediction from image and internal_state --- #
    ########################################################################
    # self.fill_memory(self.train_buffer, n_episodes=20, act='best')
    # for epoch in range(100):
    #   loss, acc = self.is_predictor_trainer.train(lambda x: self.train_buffer.sample_image_is_goal_batch(x),
    #                                               tf_logger=self.tf_logger, n_max_steps=50, image_key='image',
    #                                               hinge_weight=1.0, margin=2.0)
    #   acc1, acc2 = self.is_predictor_trainer.evaluate(lambda x: self.test_buffer.sample_image_is_goal_batch(x),
    #                                                   tf_logger=self.tf_logger, image_key='image')
    #   print(f'Epoch: {epoch} - Train loss: {loss:.4f}'
    #         f' - Train accuracy: {acc:.3f} - Test mean accuracy: {(acc1+acc2)/2:.3f}')
    # --------------------------------------------------------------------- #

    #####################################################
    # --- Goal Image prediction experiment using AE --- #
    #####################################################
    # self.ae_goal_image_trainer.load()
    # losses = self.ae_goal_image_trainer.train(lambda x: self.train_buffer.sample_image_is_goal_batch(x),
    #                                           tf_logger=self.tf_logger, target_image_key='goal_image',
    #                                           n_max_steps=150, augment=True)
    # print(f'TRAIN:\nMSE: {losses[0]:.4f} | SSIM: {losses[1]:.4f} | REC: {losses[2]:.4f}')
    # losses = self.ae_goal_image_trainer.evaluate(lambda x: self.test_buffer.sample_image_is_goal_batch(x),
    #                                              tf_logger=self.tf_logger, target_image_key='goal_image')
    # print(f'TEST:\nMSE: {losses[0]:.4f} | SSIM: {losses[1]:.4f} | REC: {losses[2]:.4f}')
    # self.ae_goal_image_trainer.save()
    # # ------------------------------- #
    # self.goal_image_embedding_trainer = tz.EmbeddingOptimizerTrainer(self.ae_goal_image_trainer.model)
    # self.goal_image_embedding_trainer.load()
    # losses = self.goal_image_embedding_trainer.train(lambda x: self.train_buffer.sample_image_is_goal_batch(x),
    #                                                  tf_logger=self.tf_logger, target_image_key='goal_image',
    #                                                  n_max_steps=200)
    # print(f'TRAIN:\nREC: {losses[0]:.4f} | DIVERSITY: {losses[1]:.4f} | TOTAL: {losses[2]:.4f}')
    # losses = self.goal_image_embedding_trainer.evaluate(lambda x: self.test_buffer.sample_image_is_goal_batch(x),
    #                                                     tf_logger=self.tf_logger, target_image_key='goal_image')
    # print(f'TEST:\nREC: {losses[0]:.4f} | DIVERSITY: {losses[1]:.4f} | TOTAL: {losses[2]:.4f}')
    # self.goal_image_embedding_trainer.save()

    ################################################
    # --- Next state prediction from embedding --- #
    ################################################
    self.ae_ensp_trainer.load()
    # best_rec_loss, best_nsp_loss = float('inf'), float('inf')
    # for epoch in range(10):
    #   losses = self.ae_ensp_trainer.train(lambda x: self.train_buffer.sample(x),
    #                                     tf_logger=self.tf_logger, target_image_key='next_image',
    #                                     n_max_steps=100, start_step=epoch*100)
    #   print(f'TRAIN:\nMSE: {losses[0]:.4f} | SSIM: {losses[1]:.4f}'
    #         f' | REC: {losses[2]:.4f} | NSP: {losses[3]:.4f} | TOTAL: {losses[4]:.4f}')
    #   losses = self.ae_ensp_trainer.evaluate(lambda x: self.test_buffer.sample(x),
    #                                         tf_logger=self.tf_logger, target_image_key='next_image',
    #                                         step=epoch)
    #   print(f'TEST:\nMSE: {losses[0]:.4f} | SSIM: {losses[1]:.4f} | REC: {losses[2]:.4f} | NSP: {losses[3]:.4f}')
      
    #   if losses[2] < best_rec_loss and losses[3] < best_nsp_loss:
    #     best_rec_loss = losses[2]
    #     best_nsp_loss = losses[3]
    #     self.ae_ensp_trainer.save()
    # -------------------------------------------- #
    # --- Goal Embedding prediction            --- #
    # -------------------------------------------- #
    self.memory_bank = ReplayBuffer(2, 1, 256, resize_to=32, normalize_img=True, capacity=5*60, device='cpu',
                                    target_device=self.device)
    self.fill_memory(self.memory_bank, n_episodes=5, act='best')
    self.add_var_to_memory(self.memory_bank, self.ae_ensp_trainer.model, 'image_embedding',
                           self.memory_bank.image,
                           tensor_like=torch.zeros(
                             (self.memory_bank.capacity,
                              self.ae_ensp_trainer.config['model_config']['ae_config']['latent_dim']),
                              device=self.device))
    self.add_var_to_memory(self.train_buffer, self.ae_ensp_trainer.model, 'image_embedding',
                           self.train_buffer.image,
                           tensor_like=torch.zeros(
                             (self.train_buffer.capacity,
                              self.ae_ensp_trainer.config['model_config']['ae_config']['latent_dim']),
                              device=self.device))
    self.add_var_to_memory(self.test_buffer, self.ae_ensp_trainer.model, 'image_embedding',
                           self.test_buffer.image,
                           tensor_like=torch.zeros(
                             (self.test_buffer.capacity,
                              self.ae_ensp_trainer.config['model_config']['ae_config']['latent_dim']),
                               device=self.device))
    n_train_finished_episodes = len(self.train_buffer.successful_episodes)
    print(f'{n_train_finished_episodes} finished episodes in train buffer')
    self.mb_gep_trainer.load()
    best_loss = float('inf')
    pbar = tqdm(range(1000))
    for epoch in pbar:
      loss = self.mb_gep_trainer.train(self.train_buffer, self.memory_bank, start_step=100*epoch,
                                       n_training_episodes=n_train_finished_episodes,
                                       tf_logger=self.tf_logger, memory_size=5)
      losses = self.mb_gep_trainer.evaluate(self.test_buffer, self.memory_bank, step=epoch,
                                            tf_logger=self.tf_logger, memory_size=5,
                                            n_trajectories=self.config['n_test_episodes'])
      if losses[0] < best_loss:
        best_loss = losses[0]
        self.mb_gep_trainer.save()
      
      train_descr = f'Epoch {epoch} - Train loss: {loss:.4f}'
      eval_descr = f'Gen: Err={losses[0]:.3f}, Adv={losses[1]:.3f} | Ret: Err={losses[2]:.3f}, Adv={losses[3]:.3f}'
      pbar.set_description(f'{train_descr} | {eval_descr}')

    # for each element of the batch, compute the distance between step t and last step
    # add plot of the distance at each step, one line per episode
    # 1. Get last valid embedding per batch
    batch = self.memory_bank.sample_episode_batch(5, 60)
    emb = batch['image_embedding'].cpu()  # [B, T, D]
    sizes = batch['episode_size'].cpu()  # [B]
    B, T, D = emb.shape
    last = emb[torch.arange(B, device=emb.device), sizes - 1]  # [B, D]
    # 2. Compute distances (broadcasting)
    diff = emb - last.unsqueeze(1)                      # [B, T, D]
    distances = torch.linalg.norm(diff, dim=-1)         # [B, T]
    cos_dists = torch.nn.functional.cosine_similarity(emb, last.unsqueeze(1), dim=-1)  # [B, T]
    print(cos_dists)
    print(f'{emb[0, 0]=} \n{last[0]=}')

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))

    for i in range(B):
      ep_len = sizes[i].item()
      
      if ep_len <= 1:
        continue  # skip degenerate episodes
      
      plt.plot(
        distances[i, :ep_len-1],
        alpha=0.7
      )
      plt.plot(
        cos_dists[i, :ep_len-1],
        alpha=0.7,
        linestyle='dashed',
      )

    plt.xlabel('Step')
    plt.ylabel('Distance to goal embedding')
    plt.title('Distance to goal embedding per episode')
    plt.grid(True)
    plt.show()


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