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
import torch.nn.functional as F

from tqdm import tqdm
from itertools import groupby
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Subset

sys.path.append(os.path.abspath(__file__).replace('ASR/ctc_experiments.py', ''))
import utils as u

from data import Data
from models.divers_models import ConvLayer
from optimizer import CosineAnnealingWarmUpRestarts


class CTCModel(nn.Module):
  def __init__(self, output_dim, emb_dim, d_model, n_heads, d_ff, kernel_size, n_blocks, n_blocks_strided=None, dropout=0.):
    super().__init__()
    embedder = lambda x: x
    self.net = ConvLayer(emb_dim, d_model, n_heads, d_ff, kernel_size, n_blocks, embedder, dropout=dropout,
                         only_see_past=False, block_type='dilated', n_blocks_strided=n_blocks_strided)
    self.output_proj = nn.Linear(d_model, output_dim)
  
  def forward(self, x):
    x = self.net(x)
    return self.output_proj(x)


class CustomDataset(Dataset):
  def __init__(self, ids_to_audiofile, ids_to_encodedsources, sort_by_target_len=True):
    self.ids_to_audiofilefeatures = {i: f.replace('.flac', '.features.npy') for i, f in ids_to_audiofile.items()}
    self.ids_to_encodedsources = ids_to_encodedsources
    self.identities = list(sorted(ids_to_encodedsources.keys()))

    if sort_by_target_len:
      self.identities = CustomDataset._sort_by_targets_len(self.identities, ids_to_encodedsources)
  
  @staticmethod
  def _sort_by_targets_len(ids, ids2es):
    return list(map(lambda x: x[0], sorted([(i, len(ids2es[i])) for i in ids], key=lambda x: x[1])))
  
  def __len__(self):
    return len(self.identities)
  
  def __getitem__(self, idx):
    input_ = torch.tensor(np.load(self.ids_to_audiofilefeatures[self.identities[idx]]))
    target = torch.LongTensor(self.ids_to_encodedsources[self.identities[idx]])
    input_len = len(input_)
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
  def __init__(self, device=None, logfile='_logs/_logs_CTC.txt', metadata_file='_Data_metadata_letters_wav2vec.pk', batch_size=64,
               lr=1e-4, load_model=True, n_epochs=500, eval_step=1, config={}, save_name_model='convnet/ctc_convDilated.pt',
               lr_scheduling=False):
    logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.batch_size = batch_size
    self.n_epochs = n_epochs
    self.eval_step = eval_step
    self.save_name_model = save_name_model
    self.lr_scheduling = lr_scheduling

    self.set_metadata(metadata_file)
    self.set_data_loader()

    self.config = {'output_dim': len(self.idx_to_tokens), 'emb_dim': 512, 'd_model': 512, 'n_heads': 8, 'd_ff': 1024,
                   'kernel_size': 3, 'n_blocks': 6, 'n_blocks_strided': 2, 'dropout': 0.}
    self.config = {**self.config, **config}
    self.model = self.instanciate_model(**self.config)
    self.model = nn.DataParallel(self.model)

    u.dump_dict(self.config, 'CTCModel PARAMETERS')
    logging.info(f'The model has {u.count_trainable_parameters(self.model):,} trainable parameters')

    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.criterion = nn.CTCLoss()

    if self.lr_scheduling:
      self.lr_scheduler = CosineAnnealingWarmUpRestarts(self.optimizer, T_0=150, T_mult=1, eta_max=1e-2, T_up=10, gamma=0.5)

    if load_model:
      u.load_model(self.model, self.save_name_model, restore_only_similars=True)
  
  def set_metadata(self, metadata_file):
    with open(metadata_file, 'rb') as f:
      data = pk.load(f)

    self.idx_to_tokens = ['<blank>'] + data['idx_to_tokens'][3:]
    self.tokens_to_idx = {t: i for i, t in enumerate(self.idx_to_tokens)}

    self.ids_to_encodedsources_train = {k: (np.array(v[1:-1])-2).tolist() for k, v in data['ids_to_encodedsources_train'].items()}
    self.ids_to_encodedsources_test = {k: (np.array(v[1:-1])-2).tolist() for k, v in data['ids_to_encodedsources_test'].items()}

    self.ids_to_audiofile_train = data['ids_to_audiofile_train']
    self.ids_to_audiofile_test = data['ids_to_audiofile_test']
  
  def set_data_loader(self):
    train_dataset = CustomDataset(self.ids_to_audiofile_train, self.ids_to_encodedsources_train)
    test_dataset = CustomDataset(self.ids_to_audiofile_test, self.ids_to_encodedsources_test)

    collator = CustomCollator(0, 0)

    self.train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=8, collate_fn=collator,
                                        shuffle=True, pin_memory=True)
    self.test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=8, collate_fn=collator,
                                       shuffle=True, pin_memory=True)
  
  def instanciate_model(self, output_dim=32, emb_dim=512, d_model=512, n_heads=8, d_ff=2048, kernel_size=3, n_blocks=10,
                        n_blocks_strided=None, dropout=0.):
    return CTCModel(output_dim, emb_dim, d_model, n_heads, d_ff, kernel_size, n_blocks, n_blocks_strided=n_blocks_strided,
                    dropout=dropout).to(self.device)

  def train(self):
    print('Start Training...')
    eval_accuracy_memory = 0
    for epoch in tqdm(range(self.n_epochs)):
      epoch_loss, accs = self.train_pass(only_loss=epoch % self.eval_step != 0)
      logging.info(f"Epoch {epoch} | train_loss = {epoch_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")
      eval_loss, accs = self.evaluation(only_loss=epoch % self.eval_step != 0)
      logging.info(f"Epoch {epoch} | test_loss = {eval_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")

      oea = accs.get('word_accuracy', None)

      if self.lr_scheduling:
        self.lr_scheduler.step()

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
      input_lens = u.compute_out_conv(u.compute_out_conv(input_lens, kernel=3, stride=2, padding=1, dilation=1),
                                      kernel=3, stride=2, padding=1, dilation=1)

      inputs, targets = inputs.to(self.device), targets.to(self.device)
      input_lens, target_lens = input_lens.to(self.device), target_lens.to(self.device)

      preds = self.model(inputs)  # [batch_size, seq_len, output_dim]

      losses += self.criterion(preds.permute(1, 0, 2).log_softmax(-1), targets, input_lens, target_lens).item()

      all_targets += targets.tolist()
      all_preds += preds.argmax(dim=-1).tolist()
    
    self.model.train()

    if not only_loss:
      accs = CTCTrainer.scorer(all_targets, all_preds, self.idx_to_tokens, self.tokens_to_idx)

    return losses / len(self.test_data_loader), accs
  
  def train_pass(self, only_loss=True):
    losses, accs = 0, {}
    all_targets, all_preds = [], []

    for inputs, targets, input_lens, target_lens in tqdm(self.train_data_loader):
      input_lens = u.compute_out_conv(u.compute_out_conv(input_lens, kernel=3, stride=2, padding=1, dilation=1),
                                      kernel=3, stride=2, padding=1, dilation=1)

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
      accs = CTCTrainer.scorer(all_targets, all_preds, self.idx_to_tokens, self.tokens_to_idx)
    
    return losses / len(self.train_data_loader), accs
  
  @staticmethod
  def scorer(targets, predictions, idx_to_tokens, tokens_to_idx):
    target_sentences = [''.join([idx_to_tokens[i] for i in t[:t.index(0) if 0 in t else None]]) for t in targets]
    predicted_sentences = [[i for i, _ in groupby(p)] for p in predictions]
    predicted_sentences = [''.join([idx_to_tokens[i] for i in p if i != 0]) for p in predicted_sentences]
    return Data.compute_scores(targets=target_sentences, predictions=predicted_sentences, rec=False)


class Experiment1(CTCTrainer):
  def __init__(self):
    super().__init__()


class Experiment2(CTCTrainer):
  def __init__(self, logfile='_logs/_logs_CTC2.txt', save_name_model='convnet/ctc_convDilated2.pt', dropout=0.25):
    super().__init__(logfile=logfile, save_name_model=save_name_model, config={'dropout': dropout})
    logging.info(self.model)


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