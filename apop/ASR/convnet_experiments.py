import os
import re
import sys
import torch
import random
import logging
import numpy as np

from tqdm import tqdm
from fairseq.models.wav2vec import Wav2VecModel

sys.path.append(os.path.abspath(__file__).replace('ASR/convnet_experiments.py', ''))
import utils as u

from data import Data
from convnet_trainer import ConvnetTrainer


## STATUS = FAILURE
class Experiment1(ConvnetTrainer):
  '''Encoder-Decoder Convnet for syllables prediction, adam optimizer, CrossEntropy loss, window-sliced'''
  def __init__(self, logfile='_logs/_logs_experiment1.txt', save_name_model='convnet/convnet_experiment1.pt',
               encoding_fn=Data.syllables_encoding, decay_factor=0, metadata_file='_Data_metadata_syllables_raw0025.pk'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, encoding_fn=encoding_fn, decay_factor=decay_factor,
                     metadata_file=metadata_file)


## STATUS = FAILURE
class Experiment2(ConvnetTrainer):
  '''Encoder-Decoder Convnet for syllables prediction, adam optimizer, Attention-CrossEntropy loss, window-sliced'''
  def __init__(self, logfile='_logs/_logs_experiment2.txt', save_name_model='convnet/convnet_experiment2.pt',
               encoding_fn=Data.syllables_encoding, metadata_file='_Data_metadata_syllables_raw0025.pk'):
    super().__init__(save_name_model=save_name_model, logfile=logfile, encoding_fn=encoding_fn, metadata_file=metadata_file)


## STATUS = FAILURE
class Experiment3(ConvnetTrainer):
  '''Encoder-Decoder Convnet for syllables prediction, adam optimizer, CrossEntropy loss, std-threshold-selected'''
  def __init__(self, logfile='_logs/_logs_experiment3.txt', decay_factor=0, save_name_model='convnet/convnet_experiment3.pt',
               signal_type='std-threshold-selected', encoding_fn=Data.syllables_encoding,
               metadata_file='_Data_metadata_syllables_raw0025.pk'):
    super().__init__(save_name_model=save_name_model, logfile=logfile, signal_type=signal_type, decay_factor=decay_factor,
                     encoding_fn=encoding_fn, metadata_file=metadata_file)


## STATUS = FAILURE
class Experiment5(ConvnetTrainer):
  '''Encoder-Decoder Convnet for phonemes prediction, adam optimizer, CrossEntropy loss, window-sliced'''
  def __init__(self, logfile='_logs/_logs_experiment5.txt', decay_factor=0, encoding_fn=Data.phonemes_encoding,
               save_name_model='convnet/convnet_experiment5.pt', metadata_file='_Data_metadata_phonemes_raw0025.pk'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, encoding_fn=encoding_fn, metadata_file=metadata_file,
                     decay_factor=decay_factor)


## STATUS = FAILURE
class Experiment6(ConvnetTrainer):
  '''Encoder-Decoder Convnet for syllables prediction, adam optimizer, CrossEntropy loss, window-sliced, sigmoid score_fn'''
  def __init__(self, logfile='_logs/_logs_experiment6.txt', decay_factor=0, save_name_model='convnet/convnet_experiment6.pt',
               encoding_fn=Data.syllables_encoding, metadata_file='_Data_metadata_syllables_raw0025.pk', score_fn=u.sigmoid_energy):
    super().__init__(logfile=logfile, save_name_model=save_name_model, score_fn=score_fn, decay_factor=decay_factor,
                     encoding_fn=encoding_fn, metadata_file=metadata_file)


## STATUS = FAILURE
class Experiment7(ConvnetTrainer):
  '''Encoder-Decoder Convnet for syllables prediction, adam optimizer, CrossEntropy loss, window-sliced, MultiHeadAttention'''
  def __init__(self, logfile='_logs/_logs_experiment7.txt', decay_factor=0, save_name_model='convnet/convnet_experiment7.pt',
               encoding_fn=Data.syllables_encoding, metadata_file='_Data_metadata_syllables_raw0025.pk', multi_head=True, d_keys_values=64):
    super().__init__(logfile=logfile, save_name_model=save_name_model, multi_head=multi_head, d_keys_values=d_keys_values,
                     decay_factor=decay_factor, encoding_fn=encoding_fn, metadata_file=metadata_file)


# STATUS = FAILURE
class Experiment8(ConvnetTrainer):
  '''Encoder-Decoder Convnet for syllables prediction, adam optimizer, Attention-CrossEntropy loss, window-sliced, MultiHeadAttention'''
  def __init__(self, logfile='_logs/_logs_experiment8.txt', save_name_model='convnet/convnet_experiment8.pt',
               encoding_fn=Data.syllables_encoding, metadata_file='_Data_metadata_syllables_raw0025.pk', multi_head=True, d_keys_values=64):
    super().__init__(logfile=logfile, save_name_model=save_name_model, multi_head=multi_head, d_keys_values=d_keys_values,
                     encoding_fn=encoding_fn, metadata_file=metadata_file)


## STATUS = FAILURE
class Experiment9(ConvnetTrainer):
  '''Encoder-Decoder Convnet for syllables prediction, adam optimizer, Attention-CrossEntropy loss, window-sliced=0.05'''
  def __init__(self, logfile='_logs/_logs_experiment9.txt', save_name_model='convnet/convnet_experiment9.pt',
               encoding_fn=Data.syllables_encoding, metadata_file='_Data_metadata_syllables_raw0025.pk', window_size=0.05):
    super().__init__(logfile=logfile, save_name_model=save_name_model, encoding_fn=encoding_fn, metadata_file=metadata_file,
                     window_size=window_size)


## STATUS = GOOD, but slower convergence than experiment12&14
class Experiment10(ConvnetTrainer):
  '''Encoder-Decoder Convnet for letters prediction, adam optimizer, Attention-CrossEntropy loss, mfcc, n_fft=2048, hop_length=512'''
  def __init__(self, logfile='_logs/_logs_experiment10.txt', save_name_model='convnet/convnet_experiment10.pt', batch_size=8,
               slice_fn=Data.mfcc_extraction, n_fft=2048, hop_length=512, scorer=Data.compute_scores,
               metadata_file='_Data_metadata_letters_mfcc0128.pk'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, slice_fn=slice_fn, batch_size=batch_size,
                     n_fft=n_fft, hop_length=hop_length, scorer=scorer, metadata_file=metadata_file)


class Experiment12(ConvnetTrainer):
  '''Convnet letters prediction, adam, Attention-CrossEntropy loss, mfcc, n_fft=2048, hop_length=512, MultiHead'''
  def __init__(self, logfile='_logs/_logs_experiment12.txt', save_name_model='convnet/convnet_experiment12.pt', batch_size=8,
               slice_fn=Data.mfcc_extraction, n_fft=2048, hop_length=512, scorer=Data.compute_scores, multi_head=True,
               metadata_file='_Data_metadata_letters_mfcc0128.pk'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, slice_fn=slice_fn, batch_size=batch_size, n_fft=n_fft,
                     hop_length=hop_length, scorer=scorer, multi_head=multi_head, metadata_file=metadata_file)
    # u.load_model(self.model, save_name_model, restore_only_similars=True)


## STATUS = extremely slow convergence, wait only till epoch 100
class Experiment13(ConvnetTrainer):
  '''Convnet letters prediction, adam, Attention-CrossEntropy loss, window-raw-sliced'''
  def __init__(self, logfile='_logs/_logs_experiment13.txt', save_name_model='convnet/convnet_experiment13.pt', batch_size=8,
               scorer=Data.compute_scores, metadata_file='_Data_metadata_letters_raw0025.pk'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=batch_size, scorer=scorer, metadata_file=metadata_file)


## STATUS = stopped, epoch 300, good but slower convergence than experiment12
class Experiment14(ConvnetTrainer):
  '''Convnet letters prediction, adam, Attention-CrossEntropy loss, mfcc, n_fft=2048, hop_length=512, MultiHead, ReLU'''
  def __init__(self, logfile='_logs/_logs_experiment14.txt', save_name_model='convnet/convnet_experiment14.pt', batch_size=8,
               slice_fn=Data.mfcc_extraction, n_fft=2048, hop_length=512, scorer=Data.compute_scores, multi_head=True,
               metadata_file='_Data_metadata_letters_mfcc0128.pk', relu=True):
    super().__init__(logfile=logfile, save_name_model=save_name_model, slice_fn=slice_fn, batch_size=batch_size, relu=relu,
                     n_fft=n_fft, hop_length=hop_length, scorer=scorer, multi_head=multi_head, metadata_file=metadata_file)


class Experiment17(ConvnetTrainer):
  '''Convnet, letters prediction, adam, Attention-CrossEntropy, MultiHead, window-raw-slice with win=0.128'''
  def __init__(self, logfile='_logs/_logs_experiment17.txt', save_name_model='convnet/convnet_experiment17.pt', batch_size=8,
               metadata_file='_Data_metadata_letters_raw0128.pk', slice_fn=Data.overlapping_window_slicing_signal, multi_head=True):
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, slice_fn=slice_fn,
                     batch_size=batch_size, multi_head=multi_head)


## STATUS = stopped, best train_acc=0.74,test_acc=0.472 epoch 460, then it diverges
class Experiment22(ConvnetTrainer):
  '''Convnet phonemes prediction, adam, Attention-CrossEntropy loss, mfcc, n_fft=2048, hop_length=512, MultiHead'''
  def __init__(self, logfile='_logs/_logs_experiment22.txt', save_name_model='convnet/convnet_experiment22.pt', batch_size=8,
               slice_fn=Data.mfcc_extraction, n_fft=2048, hop_length=512, scorer=Data.compute_scores, multi_head=True,
               metadata_file='_Data_metadata_phonemes_mfcc0128.pk', encoding_fn=Data.phonemes_encoding):
    super().__init__(logfile=logfile, save_name_model=save_name_model, slice_fn=slice_fn, batch_size=batch_size, encoding_fn=encoding_fn,
                     n_fft=n_fft, hop_length=hop_length, scorer=scorer, multi_head=multi_head, metadata_file=metadata_file)


# STATUS = stopped, epoch 420
class Experiment23(ConvnetTrainer):
  '''Convnet words prediction, adam, Attention-CrossEntropy loss, mfcc, n_fft=2048, hop_length=512, MultiHead'''
  def __init__(self, logfile='_logs/_logs_experiment23.txt', save_name_model='convnet/convnet_experiment23.pt',
               slice_fn=Data.mfcc_extraction, n_fft=2048, hop_length=512, multi_head=True, batch_size=32,
               metadata_file='_Data_metadata_words_mfcc0128.pk', encoding_fn=Data.words_encoding):
    super().__init__(logfile=logfile, save_name_model=save_name_model, slice_fn=slice_fn, batch_size=batch_size, n_fft=n_fft,
                     encoding_fn=encoding_fn, hop_length=hop_length, multi_head=multi_head, metadata_file=metadata_file)


class Experiment28(ConvnetTrainer):
  '''Convnet bigrams prediction, adam, Attention-CrossEntropy loss, mfcc, n_fft=2048, hop_length=512, MultiHead'''
  def __init__(self, logfile='_logs/_logs_experiment28.txt', save_name_model='convnet/convnet_experiment28.pt', batch_size=8,
               slice_fn=Data.mfcc_extraction, n_fft=2048, hop_length=512, scorer=Data.compute_scores, multi_head=True,
               metadata_file='_Data_metadata_bigrams_mfcc0128.pk', encoding_fn=Data.ngrams_encoding):
    super().__init__(logfile=logfile, save_name_model=save_name_model, slice_fn=slice_fn, batch_size=batch_size, n_fft=n_fft,
                     hop_length=hop_length, scorer=scorer, multi_head=multi_head, metadata_file=metadata_file, encoding_fn=encoding_fn)


class Experiment29(ConvnetTrainer):
  '''Convnet letters prediction, adam, CrossEntropy loss, wav2letter, MultiHead'''
  def __init__(self, logfile='_logs/_logs_experiment29.txt', save_name_model='convnet/convnet_experiment29.pt', batch_size=8,
               slice_fn=Data.wav2vec_extraction, scorer=Data.compute_scores, multi_head=True, decay_factor=0,
               metadata_file='_Data_metadata_letters_wav2vec.pk'):
    convnet_config = {'emb_dim': 384, 'hid_dim': 512}
    cp = torch.load('wav2vec_large.pt')
    wav2vec_model = Wav2VecModel.build_model(cp['args'], task=None)
    wav2vec_model.load_state_dict(cp['model'])
    wav2vec_model.eval()
    super().__init__(logfile=logfile, save_name_model=save_name_model, slice_fn=slice_fn, batch_size=batch_size,
                     scorer=scorer, multi_head=multi_head, metadata_file=metadata_file, convnet_config=convnet_config,
                     wav2vec_model=wav2vec_model, save_features=True, decay_factor=decay_factor)
    u.load_model(self.model, self.save_name_model, restore_only_similars=True)


class Experiment30(ConvnetTrainer):
  '''Convnet letters prediction, adam, CrossEntropy loss, wav2letter, MultiHead'''
  def __init__(self, logfile='_logs/_logs_experiment29bigLR.txt', save_name_model='convnet/convnet_experiment29bigLR.pt', batch_size=8,
               slice_fn=Data.wav2vec_extraction, scorer=Data.compute_scores, multi_head=True, decay_factor=0,
               metadata_file='_Data_metadata_letters_wav2vec.pk'):
    convnet_config = {'emb_dim': 384, 'hid_dim': 512}
    cp = torch.load('wav2vec_large.pt')
    wav2vec_model = Wav2VecModel.build_model(cp['args'], task=None)
    wav2vec_model.load_state_dict(cp['model'])
    wav2vec_model.eval()
    super().__init__(logfile=logfile, save_name_model=save_name_model, slice_fn=slice_fn, batch_size=batch_size,
                     scorer=scorer, multi_head=multi_head, metadata_file=metadata_file, convnet_config=convnet_config,
                     wav2vec_model=wav2vec_model, save_features=True, decay_factor=decay_factor, lr=1e-4)
    self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 200, eta_min=1e-6, last_epoch=-1)
    u.load_model(self.model, self.save_name_model, restore_only_similars=True)
  
  def train(self):
    print('Start Training...')
    eval_accuracy_memory = 0
    for epoch in tqdm(range(self.n_epochs)):
      epoch_loss, accs = self.train_pass(only_loss=epoch % self.scores_step != 0)
      logging.info(f"Epoch {epoch} | train_loss = {epoch_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")
      eval_loss, accs = self.evaluation(scores=epoch % self.scores_step == 0, greedy_scores=epoch % self.eval_step ==0)
      logging.info(f"Epoch {epoch} | test_loss = {eval_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")

      oea = accs['preds_acc'] if 'preds_acc' in accs else accs.get('word_accuracy', None)

      self.criterion.step(999 if self.decay_factor == 0 else epoch)
      self.lr_scheduler.step()

      if oea is not None and oea > eval_accuracy_memory:
        logging.info(f'Save model with eval_accuracy = {oea:.3f}')
        u.save_checkpoint(self.model, None, self.save_name_model)
        eval_accuracy_memory = oea


if __name__ == "__main__":
  ## SEEDING FOR REPRODUCIBILITY
  SEED = 42
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)

  experiments = {k.replace('Experiment', ''): v for k, v in locals().items() if re.search(r'Experiment\d+', k) is not None}
  
  rep = input(f'Which Experiment do you want to start? ({",".join(experiments.keys())}): ')
  exp = experiments[rep]()
  exp.train()