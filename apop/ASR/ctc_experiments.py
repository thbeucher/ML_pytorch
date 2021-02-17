import os
import re
import sys
import torch
import random
import logging
import numpy as np
import pickle as pk
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(__file__).replace('ASR/ctc_experiments.py', ''))

import utils as u

from data import Data
from models.stt_net import Encoder
from optimizer import CosineAnnealingWarmUpRestarts
from models.stt_net_configs import get_encoder_config


class CustomDataset(Dataset):
  def __init__(self, ids_to_audiofile, ids_to_encodedsources, sort_by_target_len=True, **kwargs):
    self.ids_to_audiofilefeatures = {i: f for i, f in ids_to_audiofile.items()}
    self.ids_to_encodedsources = ids_to_encodedsources
    self.identities = list(sorted(ids_to_encodedsources.keys()))

    self.process_file_fn = kwargs.get('process_file_fn', Data.read_and_slice_signal)
    kwargs['slice_fn'] = kwargs.get('slice_fn', Data.wav2vec_extraction)
    kwargs['save_features'] = kwargs.get('save_features', True)
    self.process_file_fn_args = kwargs
    # self.client = hc.HTTPConnection('localhost', 8080)

    self.ids_input_lens = {}
    if kwargs['slice_fn'] == Data.wav2vec_extraction or kwargs.get('use_wav2vec', True):
      for i, f in self.ids_to_audiofilefeatures.items():
        fname = f.replace('.flac', '.wav2vec_shape.pk')
        if os.path.isfile(fname):
          self.ids_input_lens[i] = pk.load(open(fname, 'rb'))[0]

    if sort_by_target_len:
      self.identities = CustomDataset._sort_by_targets_len(self.identities, ids_to_encodedsources)
  
  @staticmethod
  def _sort_by_targets_len(ids, ids2es):
    return list(map(lambda x: x[0], sorted([(i, len(ids2es[i])) for i in ids], key=lambda x: x[1])))
  
  def __len__(self):
    return len(self.identities)
  
  def __getitem__(self, idx):
    signal = self.process_file_fn(self.ids_to_audiofilefeatures[self.identities[idx]], **self.process_file_fn_args)
    input_ = torch.Tensor(signal) if isinstance(signal, np.ndarray) else signal
    # self.client.request('POST', '/process', '{"filename": "' + self.ids_to_audiofilefeatures[self.identities[idx]] + '"}')
    # input_ = torch.Tensor(np.array(json.loads(self.client.getresponse().read())))

    target = torch.LongTensor(self.ids_to_encodedsources[self.identities[idx]])
    input_len = self.ids_input_lens[self.identities[idx]] if self.identities[idx] in self.ids_input_lens else len(input_)
    target_len = len(target)
    return input_, target, input_len, target_len


class CustomCollator(object):
  def __init__(self, audio_pad, text_pad):
    self.audio_pad = audio_pad
    self.text_pad = text_pad

  def __call__(self, batch):
    inputs, targets, input_lens, target_lens = zip(*batch)
    inputs_batch = pad_sequence(inputs, batch_first=True, padding_value=self.audio_pad).float()
    targets_batch = pad_sequence(targets, batch_first=True, padding_value=self.text_pad)
    input_lens = torch.LongTensor(input_lens)
    target_lens = torch.LongTensor(target_lens)
    return inputs_batch, targets_batch, input_lens, target_lens


class CTCTrainer(object):
  # ratios audio_len/text_len -> min = 4.75 | max = 9.29 | mean = 5.96
  def __init__(self, device=None, logfile='_logs/_logs_CTC.txt', save_metadata='_Data_metadata.pk', batch_size=64,
               lr=1e-4, load_model=True, n_epochs=500, eval_step=1, config={}, save_name_model='convnet/ctc_model.pt',
               lr_scheduling=True, lr_scheduler_type='plateau', **kwargs):
    if not os.path.isdir(os.path.dirname(logfile)):
      os.makedirs(os.path.dirname(logfile))
    logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.batch_size = batch_size
    self.n_epochs = n_epochs
    self.eval_step = eval_step
    self.save_name_model = save_name_model
    self.lr_scheduling = lr_scheduling
    self.kwargs = kwargs

    self.set_metadata(save_metadata)
    self.set_data_loader()

    self.config = {'output_dim': len(self.idx_to_tokens), 'emb_dim': 512, 'd_model': 512, 'n_heads': 8, 'd_ff': 1024,
                   'kernel_size': 3, 'n_blocks': 6, 'n_blocks_strided': 2, 'dropout': 0., 'block_type': 'dilated'}
    self.config = {**self.config, **config}
    self.model = self.instanciate_model(**self.config)
    self.model = nn.DataParallel(self.model)

    u.dump_dict({'batch_size': batch_size, 'lr': lr, 'lr_scheduling': lr_scheduling,
                 'lr_scheduler_type': lr_scheduler_type, 'load_model': load_model}, 'CTCModel Hyperparameters')
    u.dump_dict(self.config, 'CTCModel PARAMETERS')
    logging.info(self.model)
    logging.info(f'The model has {u.count_trainable_parameters(self.model):,} trainable parameters')

    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.criterion = nn.CTCLoss(zero_infinity=True)

    if self.lr_scheduling:
      if lr_scheduler_type == 'cosine':
        self.lr_scheduler = CosineAnnealingWarmUpRestarts(self.optimizer, T_0=150, T_mult=2, eta_max=1e-2, T_up=50, gamma=0.5)
      else:
        patience, min_lr, threshold = kwargs.get('patience', 50), kwargs.get('min_lr', 1e-5), kwargs.get('lr_threshold', 0.003)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=patience, verbose=True,
                                                                 min_lr=min_lr, threshold_mode='abs', threshold=threshold)

    if load_model:
      u.load_model(self.model, self.save_name_model, restore_only_similars=True)
  
  def set_metadata(self, save_metadata):
    self.data = Data()

    if not os.path.isfile(save_metadata):
      train_folder = self.kwargs.get('train_folder', '../../../datasets/openslr/LibriSpeech/train-clean-100/')
      test_folder = self.kwargs.get('test_folder', '../../../datasets/openslr/LibriSpeech/test-clean/')
      self.data.process_all_transcripts(train_folder, test_folder, encoding_fn=Data.letters_encoding)
      self.data.idx_to_tokens = ['<blank>'] + self.data.idx_to_tokens[3:]  # data add by default SOS/EOS/PAD tokens
      self.data.tokens_to_idx = {t: i for i, t in enumerate(self.data.idx_to_tokens)}
      self.data.ids_to_encodedsources_train = {k: (np.array(v[1:-1])-2).tolist() for k, v in self.data.ids_to_encodedsources_train.items()}
      self.data.ids_to_encodedsources_test = {k: (np.array(v[1:-1])-2).tolist() for k, v in self.data.ids_to_encodedsources_test.items()}
      self.data.set_audio_metadata(train_folder, test_folder, slice_fn=Data.wav2vec_extraction, save_features=True, **self.kwargs)
      self.data.save_metadata(save_name=save_metadata)
    else:
      self.data.load_metadata(save_name=save_metadata)
    
    self.idx_to_tokens = self.data.idx_to_tokens
    self.tokens_to_idx = self.data.tokens_to_idx
    self.ids_to_encodedsources_train = self.data.ids_to_encodedsources_train
    self.ids_to_encodedsources_test = self.data.ids_to_encodedsources_test
    self.ids_to_audiofile_train = self.data.ids_to_audiofile_train
    self.ids_to_audiofile_test = self.data.ids_to_audiofile_test
  
  def set_data_loader(self):
    train_dataset = CustomDataset(self.ids_to_audiofile_train, self.ids_to_encodedsources_train)
    test_dataset = CustomDataset(self.ids_to_audiofile_test, self.ids_to_encodedsources_test)

    collator = CustomCollator(0, 0)

    self.train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=8, collate_fn=collator,
                                        shuffle=True, pin_memory=True)
    self.test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=8, collate_fn=collator,
                                       shuffle=True, pin_memory=True)
  
  def instanciate_model(self, **kwargs):
    return Encoder(config=get_encoder_config(config='conv_attention_deep'), output_size=kwargs['output_dim'],
                   input_proj='base').to(self.device)

  def get_input_lens(self, input_lens):
    return u.compute_out_conv(u.compute_out_conv(input_lens, kernel=3, stride=2, padding=1, dilation=1),
                              kernel=3, stride=2, padding=1, dilation=1)

  def train(self):
    print('Start Training...')
    eval_accuracy_memory = 0
    for epoch in tqdm(range(self.n_epochs)):
      epoch_loss, accs = self.train_pass(only_loss=epoch % self.eval_step != 0)
      logging.info(f"Epoch {epoch} | train_loss = {epoch_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")
      eval_loss, accs = self.evaluation(only_loss=epoch % self.eval_step != 0)
      logging.info(f"Epoch {epoch} | test_loss = {eval_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")

      oea = accs.get('word_accuracy', None)

      if self.lr_scheduling and oea is not None:
        self.lr_scheduler.step(oea)

      if oea is not None and oea > eval_accuracy_memory:
        logging.info(f'Save model with eval_accuracy = {oea:.3f}')
        u.save_checkpoint(self.model, None, self.save_name_model)
        eval_accuracy_memory = oea
  
  @torch.no_grad()
  def evaluation(self, only_loss=True):
    losses, accs = 0, {}
    all_targets, all_preds = [], []

    self.model.eval()

    for inputs, targets, input_lens, target_lens in tqdm(self.test_data_loader):
      input_lens = self.get_input_lens(input_lens)

      inputs, targets = inputs.to(self.device), targets.to(self.device)
      input_lens, target_lens = input_lens.to(self.device), target_lens.to(self.device)

      preds = self.model(inputs)  # [batch_size, seq_len, output_dim]

      losses += self.criterion(preds.permute(1, 0, 2).log_softmax(-1), targets, input_lens, target_lens).item()

      all_targets += targets.tolist()
      all_preds += preds.argmax(dim=-1).tolist()
    
    self.model.train()

    if not only_loss:
      accs = Data.ctc_scorer(all_targets, all_preds, self.idx_to_tokens, self.tokens_to_idx)

    return losses / len(self.test_data_loader), accs
  
  def train_pass(self, only_loss=True):
    losses, accs = 0, {}
    all_targets, all_preds = [], []

    for inputs, targets, input_lens, target_lens in tqdm(self.train_data_loader):
      input_lens = self.get_input_lens(input_lens)

      inputs, targets = inputs.to(self.device), targets.to(self.device)
      input_lens, target_lens = input_lens.to(self.device), target_lens.to(self.device)
      
      preds = self.model(inputs)  # [batch_size, seq_len, output_dim]

      self.optimizer.zero_grad()
      current_loss = self.criterion(preds.permute(1, 0, 2).log_softmax(-1), targets, input_lens, target_lens)
      current_loss.backward()
      self.optimizer.step()

      all_targets += targets.tolist()
      all_preds += preds.argmax(dim=-1).tolist()

      losses += current_loss.item()
    
    if not only_loss:
      accs = Data.ctc_scorer(all_targets, all_preds, self.idx_to_tokens, self.tokens_to_idx)
    
    return losses / len(self.train_data_loader), accs
  
  @torch.no_grad()
  def dump_predictions(self, save_file='_ctc_experiment_results.pk'):
    all_targets, predictions = [], []
    self.model.eval()

    for inputs, targets, input_lens, target_lens in tqdm(self.test_data_loader):
      input_lens = self.get_input_lens(input_lens)

      inputs, targets = inputs.to(self.device), targets.to(self.device)
      input_lens, target_lens = input_lens.to(self.device), target_lens.to(self.device)

      preds = self.model(inputs)

      all_targets += targets.tolist()
      predictions += preds.tolist()
    
    with open(save_file, 'wb') as f:
      pk.dump({'targets': all_targets, 'predictions': predictions}, f)
    
    self.model.train()


class Experiment1(CTCTrainer):
  '''Stop epoch 282, best epoch 265 : word_acc = 0.861, WER = 0.058'''
  def __init__(self, logfile='_logs/_logs_CTC1.txt', save_name_model='convnet/ctc_conv_attention1.pt',
               save_metadata='_Data_metadata_ctc_500-360-100_letters_wav2vec.pk'):
    wav2vec_model = Data.get_wav2vec_model().cuda()
    train_folder = ['../../../datasets/openslr/LibriSpeech/train-clean-100/', '../../../datasets/openslr/LibriSpeech/train-clean-360/',
                    '../../../datasets/openslr/LibriSpeech/train-other-500/']
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=48, lr=1e-4, save_metadata=save_metadata,
                     wav2vec_model=wav2vec_model, train_folder=train_folder)


class Experiment2(CTCTrainer):
  def __init__(self, logfile='_logs/_logs_CTC2.txt', save_name_model='convnet/ctc_conv_attention2.pt',
               save_metadata='_Data_metadata_ctc_100_letters_wav2vec.pk'):
    wav2vec_model = Data.get_wav2vec_model().cuda()
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=48, lr=1e-4, save_metadata=save_metadata,
                     wav2vec_model=wav2vec_model)


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