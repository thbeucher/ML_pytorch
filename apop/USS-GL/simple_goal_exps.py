import os
import ast
import argparse

from learners.action_predictor import APTrainer
from learners.next_state_predictor import NSPTrainer
from learners.visual_feature_learner import VFLTrainer


def training_routine(learners, learner, force_retrain=False, **kwargs):
  trainer = learners[learner](config=kwargs)
  save_name = os.path.join(trainer.config['models_folder'], trainer.config['save_name'])

  if not os.path.isfile(save_name):
    trainer.train(**kwargs)
  else:
    if force_retrain:
      print(f'Loading {learner} model...')
      trainer.load_model()
      trainer.train(**kwargs)


if __name__ == '__main__':
  argparser = argparse.ArgumentParser(prog='simple_goal_exps.py', description='')
  argparser.add_argument('--learner', default='vfl', type=str, choices=['vfl', 'nsp', 'ap'])
  argparser.add_argument('--force_retrain', default=False, type=ast.literal_eval)
  args = argparser.parse_args()

  learners = {'vfl': VFLTrainer, 'nsp': NSPTrainer, 'ap': APTrainer}

  # training_routine(learners, args.learner, force_retrain=args.force_retrain)
  training_routine(learners, args.learner, force_retrain=args.force_retrain,
                   save_name='test_ap.pt', memory_size=120, n_evaluated_samples=30, n_iterations=1)