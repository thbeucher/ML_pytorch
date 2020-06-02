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
from models.final_net import Encoder
from models.divers_models import ConvLayer
from optimizer import CosineAnnealingWarmUpRestarts
from models.final_net_configs import get_encoder_config


class CTCModel(nn.Module):
  def __init__(self, output_dim, emb_dim, d_model, n_heads, d_ff, kernel_size, n_blocks, n_blocks_strided=None, dropout=0.,
               block_type='dilated', network_type='conv_layer', network_config='attention_glu'):
    super().__init__()
    embedder = lambda x: x
    if network_type == 'encoder_attention':
      self.net = Encoder(config=get_encoder_config(config=network_config))
    else:
      self.net = ConvLayer(emb_dim, d_model, n_heads, d_ff, kernel_size, n_blocks, embedder, dropout=dropout,
                          only_see_past=False, block_type=block_type, n_blocks_strided=n_blocks_strided)
    self.output_proj = nn.Linear(d_model, output_dim)
  
  def forward(self, x):
    x = self.net(x)
    return self.output_proj(x)


class CustomDataset(Dataset):
  def __init__(self, ids_to_audiofile, ids_to_encodedsources, sort_by_target_len=True, **kwargs):
    self.ids_to_audiofilefeatures = {i: f for i, f in ids_to_audiofile.items()}
    self.ids_to_encodedsources = ids_to_encodedsources
    self.identities = list(sorted(ids_to_encodedsources.keys()))

    self.process_file_fn = kwargs.get('process_file_fn', Data.read_and_slice_signal)
    kwargs['slice_fn'] = kwargs.get('slice_fn', Data.wav2vec_extraction)
    kwargs['save_features'] = kwargs.get('save_features', True)
    self.process_file_fn_args = kwargs

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
               lr_scheduling=True, lr_scheduler_type='plateau', **kwargs):
    logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.batch_size = batch_size
    self.n_epochs = n_epochs
    self.eval_step = eval_step
    self.save_name_model = save_name_model
    self.lr_scheduling = lr_scheduling

    if kwargs.get('augmented', False):
      self.new_metadata_file = kwargs.get('new_metadata_file', None)
      assert self.new_metadata_file is not None, 'new_metadata_file must be provide'
      self.set_metadata_augmented(metadata_file)
    else:
      self.set_metadata(metadata_file)
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
  
  def set_metadata(self, metadata_file):
    with open(metadata_file, 'rb') as f:
      data = pk.load(f)

    self.idx_to_tokens = ['<blank>'] + data['idx_to_tokens'][3:]
    self.tokens_to_idx = {t: i for i, t in enumerate(self.idx_to_tokens)}

    self.ids_to_encodedsources_train = {k: (np.array(v[1:-1])-2).tolist() for k, v in data['ids_to_encodedsources_train'].items()}
    self.ids_to_encodedsources_test = {k: (np.array(v[1:-1])-2).tolist() for k, v in data['ids_to_encodedsources_test'].items()}

    self.ids_to_audiofile_train = data['ids_to_audiofile_train']
    self.ids_to_audiofile_test = data['ids_to_audiofile_test']
  
  def set_metadata_augmented(self, metadata_file):
    if os.path.isfile(self.new_metadata_file):
      with open(self.new_metadata_file, 'rb') as f:
        data = pk.load(f)
      self.idx_to_tokens = data['idx_to_tokens']
      self.tokens_to_idx = data['tokens_to_idx']
      self.ids_to_encodedsources_train = data['ids_to_encodedsources_train']
      self.ids_to_encodedsources_test = data['ids_to_encodedsources_test']
      self.ids_to_audiofile_train = data['ids_to_audiofile_train']
      self.ids_to_audiofile_test = data['ids_to_audiofile_test']
    else:
      data = Data()
      data.load_metadata(save_name=metadata_file)

      wav2vec = Data.get_wav2vec_model()
      data.data_augmentation_create_n_add(wav2vec_model=wav2vec, save_features=True)

      self.idx_to_tokens = ['<blank>'] + data.idx_to_tokens[3:]
      self.tokens_to_idx = {t: i for i, t in enumerate(self.idx_to_tokens)}

      self.ids_to_encodedsources_train = {k: (np.array(v[1:-1])-2).tolist() for k, v in data.ids_to_encodedsources_train.items()}
      self.ids_to_encodedsources_test = {k: (np.array(v[1:-1])-2).tolist() for k, v in data.ids_to_encodedsources_test.items()}

      self.ids_to_audiofile_train = data.ids_to_audiofile_train
      self.ids_to_audiofile_test = data.ids_to_audiofile_test

      with open(self.new_metadata_file, 'wb') as f:
        pk.dump({'idx_to_tokens': self.idx_to_tokens, 'tokens_to_idx': self.tokens_to_idx,
                 'ids_to_encodedsources_train': self.ids_to_encodedsources_train,
                 'ids_to_encodedsources_test': self.ids_to_encodedsources_test,
                 'ids_to_audiofile_train': self.ids_to_audiofile_train,
                 'ids_to_audiofile_test': self.ids_to_audiofile_test}, f)
  
  def set_data_loader(self):
    train_dataset = CustomDataset(self.ids_to_audiofile_train, self.ids_to_encodedsources_train)
    test_dataset = CustomDataset(self.ids_to_audiofile_test, self.ids_to_encodedsources_test)

    collator = CustomCollator(0, 0)

    self.train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collator,
                                        shuffle=True, pin_memory=True)
    self.test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collator,
                                       shuffle=True, pin_memory=True)
  
  def instanciate_model(self, output_dim=32, emb_dim=512, d_model=512, n_heads=8, d_ff=2048, kernel_size=3, n_blocks=10,
                        n_blocks_strided=None, dropout=0., block_type='dilated', **kwargs):
    return CTCModel(output_dim, emb_dim, d_model, n_heads, d_ff, kernel_size, n_blocks, n_blocks_strided=n_blocks_strided,
                    dropout=dropout, block_type=block_type, **kwargs).to(self.device)

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
  def reconstruct_sentences(targets, predictions, idx_to_tokens, tokens_to_idx):
    target_sentences = [''.join([idx_to_tokens[i] for i in t[:t.index(0) if 0 in t else None]]) for t in targets]
    predicted_sentences = [[i for i, _ in groupby(p)] for p in predictions]
    predicted_sentences = [''.join([idx_to_tokens[i] for i in p if i != 0]) for p in predicted_sentences]
    return target_sentences, predicted_sentences

  @staticmethod
  def scorer(targets, predictions, idx_to_tokens=None, tokens_to_idx=None, rec=True):
    if rec:
      targets, predictions = CTCTrainer.reconstruct_sentences(targets, predictions, idx_to_tokens, tokens_to_idx)
    return Data.compute_scores(targets=targets, predictions=predictions, rec=False)
  
  @torch.no_grad()
  def dump_predictions(self, save_file='_ctc_experiment_results.pk'):
    all_targets, predictions = [], []
    self.model.eval()

    for inputs, targets, input_lens, target_lens in tqdm(self.test_data_loader):
      input_lens = u.compute_out_conv(u.compute_out_conv(input_lens, kernel=3, stride=2, padding=1, dilation=1),
                                      kernel=3, stride=2, padding=1, dilation=1)

      inputs, targets = inputs.to(self.device), targets.to(self.device)
      input_lens, target_lens = input_lens.to(self.device), target_lens.to(self.device)

      preds = self.model(inputs)

      all_targets += targets.tolist()
      predictions += preds.softmax(-1).tolist()
    
    with open(save_file, 'wb') as f:
      pk.dump({'targets': all_targets, 'predictions': predictions}, f)
    
    self.model.train()


class Experiment1(CTCTrainer):
  '''FINISHED, Best word_accuracy=0.543, WER=0.252'''
  def __init__(self):
    super().__init__()


class Experiment2(CTCTrainer):
  '''FINISHED, Best word_accuracy=0.657, WER=0.169'''
  def __init__(self, logfile='_logs/_logs_CTC2.txt', save_name_model='convnet/ctc_convDilated2.pt', dropout=0.25):
    super().__init__(logfile=logfile, save_name_model=save_name_model, config={'dropout': dropout})


class Experiment3(CTCTrainer):
  '''Epoch 492: train_word_acc=0.853, test_word_acc=0.69, test_WER=0.15'''
  def __init__(self, logfile='_logs/_logs_CTC3.txt', save_name_model='convnet/ctc_convDilated3.pt'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=64,
                     config={'dropout': 0.25, 'block_type': 'dilated_bnd'})


class Experiment4(CTCTrainer):
  def __init__(self, logfile='_logs/_logs_CTC4.txt', save_name_model='convnet/ctc_attention4.pt'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=128, config={'network_type': 'encoder_attention'})


class Experiment5(CTCTrainer):
  def __init__(self, logfile='_logs/_logs_CTC5.txt', save_name_model='convnet/ctc_attention5.pt'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=128, lr=1e-2)
  
  def instanciate_model(self, **kwargs):
    return Encoder(config=get_encoder_config(config='attention_glu'), output_size=kwargs['output_dim']).to(self.device)


class Experiment6(CTCTrainer):
  def __init__(self, logfile='_logs/_logs_CTC6.txt', save_name_model='convnet/ctc_DepthWise6.pt'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=32, lr=1e-2)
  
  def instanciate_model(self, **kwargs):
    return Encoder(config=get_encoder_config(config='separable'), output_size=kwargs['output_dim']).to(self.device)


class Experiment7(CTCTrainer):
  def __init__(self, logfile='_logs/_logs_CTC7.txt', save_name_model='convnet/ctc_conv_attention7.pt'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=32, lr=1e-4)
  
  def instanciate_model(self, **kwargs):
    return Encoder(config=get_encoder_config(config='conv_attention'), output_size=kwargs['output_dim']).to(self.device)


class Experiment8(CTCTrainer):
  def __init__(self, logfile='_logs/_logs_CTC8.txt', save_name_model='convnet/ctc_conv_gru8.pt'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=32, lr=1e-4, lr_scheduling=False)
  
  def instanciate_model(self, **kwargs):
    return Encoder(config=get_encoder_config(config='rnn_base'), output_size=kwargs['output_dim']).to(self.device)


class Experiment9(CTCTrainer):
  def __init__(self, logfile='_logs/_logs_CTC9.txt', save_name_model='convnet/ctc_conv9.pt', new_metadata_file='_CTC_EXP9_metadata.pk'):
    self.new_metadata_file = new_metadata_file
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=32, lr=1e-4, lr_scheduling=False)
  
  def instanciate_model(self, **kwargs):
    return Encoder(config=get_encoder_config(config='conv_attention'), output_size=kwargs['output_dim']).to(self.device)
  
  def set_metadata(self, metadata_file):
    if os.path.isfile(self.new_metadata_file):
      with open(self.new_metadata_file, 'rb') as f:
        data = pk.load(f)
      self.idx_to_tokens = data['idx_to_tokens']
      self.tokens_to_idx = data['tokens_to_idx']
      self.ids_to_encodedsources_train = data['ids_to_encodedsources_train']
      self.ids_to_encodedsources_test = data['ids_to_encodedsources_test']
      self.ids_to_audiofile_train = data['ids_to_audiofile_train']
      self.ids_to_audiofile_test = data['ids_to_audiofile_test']
    else:
      data = Data()
      data.load_metadata(save_name=metadata_file)

      wav2vec = Data.get_wav2vec_model()
      data.data_augmentation_create_n_add(wav2vec_model=wav2vec, save_features=True)

      self.idx_to_tokens = ['<blank>'] + data.idx_to_tokens[3:]
      self.tokens_to_idx = {t: i for i, t in enumerate(self.idx_to_tokens)}

      self.ids_to_encodedsources_train = {k: (np.array(v[1:-1])-2).tolist() for k, v in data.ids_to_encodedsources_train.items()}
      self.ids_to_encodedsources_test = {k: (np.array(v[1:-1])-2).tolist() for k, v in data.ids_to_encodedsources_test.items()}

      self.ids_to_audiofile_train = data.ids_to_audiofile_train
      self.ids_to_audiofile_test = data.ids_to_audiofile_test

      with open(self.new_metadata_file, 'wb') as f:
        pk.dump({'idx_to_tokens': self.idx_to_tokens, 'tokens_to_idx': self.tokens_to_idx,
                 'ids_to_encodedsources_train': self.ids_to_encodedsources_train,
                 'ids_to_encodedsources_test': self.ids_to_encodedsources_test,
                 'ids_to_audiofile_train': self.ids_to_audiofile_train,
                 'ids_to_audiofile_test': self.ids_to_audiofile_test}, f)


class Experiment10(CTCTrainer):  # better than Exp7
  '''EPOCH 367: Train_word_acc = 0.934, Test_word_acc = 0.683, WER = 0.148'''
  def __init__(self, logfile='_logs/_logs_CTC10.txt', save_name_model='convnet/ctc_conv_attention10.pt'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=32, lr=1e-4)
  
  def instanciate_model(self, **kwargs):
    return Encoder(config=get_encoder_config(config='conv_attention'), output_size=kwargs['output_dim'],
                   input_proj='base').to(self.device)


class Experiment11(CTCTrainer):
  def __init__(self, logfile='_logs/_logs_CTC11.txt', save_name_model='convnet/ctc_conv_dilated11.pt'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=64, lr=1e-5, lr_scheduler_type='cosine')
  
  def instanciate_model(self, **kwargs):
    return Encoder(config=get_encoder_config(config='base'), output_size=kwargs['output_dim'], input_proj='base').to(self.device)


class Experiment12(CTCTrainer):
  def __init__(self, logfile='_logs/_logs_CTC12.txt', save_name_model='convnet/ctc_conv_dilated12.pt'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=48, lr=1e-4, patience=75, augmented=True,
                     new_metadata_file='_CTC_EXP9_metadata.pk')
  
  def instanciate_model(self, **kwargs):
    return Encoder(config=get_encoder_config(config='base'), output_size=kwargs['output_dim'], input_proj='base').to(self.device)


class Experiment13(CTCTrainer):
  def __init__(self, logfile='_logs/_logs_CTC13.txt', save_name_model='convnet/ctc_conv_attention13.pt',
               metadata_file='_Data_metadata_multigrams_wav2vec.pk'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, batch_size=48, lr=1e-4)
  
  def instanciate_model(self, **kwargs):
    return Encoder(config=get_encoder_config(config='conv_attention'), output_size=kwargs['output_dim'],
                   input_proj='base').to(self.device)


class Experiment14(CTCTrainer):
  def __init__(self, logfile='_logs/_logs_CTC14.txt', save_name_model='convnet/ctc_conv_attention14.pt'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=32, lr=1e-4)
  
  def instanciate_model(self, **kwargs):
    return Encoder(config=get_encoder_config(config='conv_attention_large'), output_size=kwargs['output_dim'],
                   input_proj='base').to(self.device)


class Experiment15(CTCTrainer):
  def __init__(self, logfile='_logs/_logs_CTC15.txt', save_name_model='convnet/ctc_conv_dilated15.pt'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=48, lr=1e-4, patience=75, augmented=True,
                     new_metadata_file='_CTC_EXP9_metadata.pk')
  
  def instanciate_model(self, **kwargs):
    return Encoder(config=get_encoder_config(config='conv_attention'), output_size=kwargs['output_dim'], input_proj='base').to(self.device)


class Experiment16(CTCTrainer):
  def __init__(self, logfile='_logs/_logs_CTC16.txt', save_name_model='convnet/ctc_conv_attention16.pt'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=32, lr=1e-4)
  
  def instanciate_model(self, **kwargs):
    return Encoder(config=get_encoder_config(config='conv_transformer'), output_size=kwargs['output_dim'],
                   input_proj='base').to(self.device)


class Experiment17(CTCTrainer):
  def __init__(self, logfile='_logs/_logs_CTC17.txt', save_name_model='convnet/ctc_conv_attention17.pt'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=32, lr=1e-4)
    self.query_embedder = nn.Embedding(100, 512)
  
  def instanciate_model(self, **kwargs):
    return Encoder(config=get_encoder_config(config='conv_attention2'), output_size=kwargs['output_dim'],
                   input_proj='base2').to(self.device)

  def train_pass(self, only_loss=True):
    losses, accs = 0, {}
    all_targets, all_preds = [], []

    for inputs, targets, input_lens, target_lens in tqdm(self.train_data_loader):
      input_lens = u.compute_out_conv(u.compute_out_conv(input_lens, kernel=3, stride=2, padding=1, dilation=1),
                                      kernel=3, stride=2, padding=1, dilation=1)

      inputs, targets = inputs.to(self.device), targets.to(self.device)
      input_lens, target_lens = input_lens.to(self.device), target_lens.to(self.device)

      ## START AUGMENTATION
      idxs = [(el == self.tokens_to_idx[' ']).nonzero().reshape(-1) for el in targets]
      words_groups = [[el[idxs[i][j]:idxs[i][j+1]] for j in range(len(idxs[i]) - 1)] for i, el in enumerate(targets)]
      for i in range(min([len(wg) for wg in words_groups])):
        new_inputs = torch.cat((inputs, torch.ones(inputs.size(0), inputs.size(1), 8).to(inputs.device) * (i + 1)), dim=-1)
        new_targets = pad_sequence([el[i] for el in words_groups], batch_first=True, padding_value=0)
        new_target_lens = torch.LongTensor([len(el[i]) for el in words_groups]).to(inputs.device)

        preds = self.model(new_inputs)

        self.optimizer.zero_grad()
        current_loss = self.criterion(preds.permute(1, 0, 2).log_softmax(-1), new_targets, input_lens, new_target_lens)
        current_loss.backward()
        self.optimizer.step()
      inputs = torch.cat((inputs, torch.ones(inputs.size(0), inputs.size(1), 8).to(inputs.device) * -1), dim=-1)
      ## END AUGMENTATION
      
      preds = self.model(inputs, y=self.query_embedder(torch.LongTensor([99])).repeat(inputs.size(0), max(input_lens), 1))

      self.optimizer.zero_grad()
      current_loss = self.criterion(preds.permute(1, 0, 2).log_softmax(-1), targets, input_lens, target_lens)
      current_loss.backward()
      self.optimizer.step()

      all_targets += targets.tolist()
      all_preds += preds.argmax(dim=-1).tolist()

      losses += current_loss.item()
      print(f'loss = {current_loss.item()}')
    
    if not only_loss:
      accs = CTCTrainer.scorer(all_targets, all_preds, self.idx_to_tokens, self.tokens_to_idx)
    
    return losses / len(self.train_data_loader), accs
  
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

      inputs = torch.cat((inputs, torch.ones(inputs.size(0), inputs.size(1), 8).to(inputs.device) * -1), dim=-1)

      preds = self.model(inputs)  # [batch_size, seq_len, output_dim]

      losses += self.criterion(preds.permute(1, 0, 2).log_softmax(-1), targets, input_lens, target_lens).item()

      all_targets += targets.tolist()
      all_preds += preds.argmax(dim=-1).tolist()
    
    self.model.train()

    if not only_loss:
      accs = CTCTrainer.scorer(all_targets, all_preds, self.idx_to_tokens, self.tokens_to_idx)

    return losses / len(self.test_data_loader), accs


class Experiment18(CTCTrainer):
  def __init__(self, logfile='_logs/_logs_CTC18.txt', save_name_model='convnet/ctc_conv_attention18.pt'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=64, lr=1e-4)
  
  def instanciate_model(self, **kwargs):
    return Encoder(config=get_encoder_config(config='conv_attention_deep'), output_size=kwargs['output_dim'],
                   input_proj='base').to(self.device)


class Experiment19(CTCTrainer):
  def __init__(self, logfile='_logs/_logs_CTC19.txt', save_name_model='convnet/ctc_conv_attention19.pt'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=128, lr=1e-4)
  
  def instanciate_model(self, **kwargs):
    return Encoder(config=get_encoder_config(config='conv_attention_deep'), output_size=kwargs['output_dim'],
                   input_proj='base').to(self.device)
  
  def set_metadata(self, metadata_file, save_metadata='_Data_metadata_ctc_360-100_letters_wav2vec.pk'):
    self.data = Data()

    if not os.path.isfile(save_metadata):
      train_folder = ['../../../datasets/openslr/LibriSpeech/train-clean-100/', '../../../datasets/openslr/LibriSpeech/train-clean-360/']
      test_folder = '../../../datasets/openslr/LibriSpeech/test-clean/'
      self.data.process_all_transcripts(train_folder, test_folder, encoding_fn=Data.letters_encoding)
      self.data.idx_to_tokens = ['<blank>'] + self.data.idx_to_tokens[3:]
      self.data.tokens_to_idx = {t: i for i, t in enumerate(self.data.idx_to_tokens)}
      self.data.ids_to_encodedsources_train = {k: (np.array(v[1:-1])-2).tolist() for k, v in self.data.ids_to_encodedsources_train.items()}
      self.data.ids_to_encodedsources_test = {k: (np.array(v[1:-1])-2).tolist() for k, v in self.data.ids_to_encodedsources_test.items()}
      self.data.set_audio_metadata(train_folder, test_folder, slice_fn=Data.wav2vec_extraction, save_features=True)
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


def read_preds_greedy_n_beam_search(res_file='_ctc_exp3_predictions.pk', data_file='_Data_metadata_letters_wav2vec.pk', beam_size=10):
  from fast_ctc_decode import beam_search

  with open(data_file, 'rb') as f:
    data = pk.load(f)
  
  with open(res_file, 'rb') as f:
    res = pk.load(f)
  
  idx_to_tokens = ['<blank>'] + data['idx_to_tokens'][3:]
  tokens_to_idx = {t: i for i, t in enumerate(idx_to_tokens)}

  greedy_preds = [np.array(p).argmax(-1).tolist() for p in res['predictions']]

  print('Beam search...')  # beam_cut_threshold
  bs_res = [beam_search(np.array(p, dtype=np.float32), idx_to_tokens, beam_size=beam_size) for p in tqdm(res['predictions'])]
  bs_s = [el[0] for el in bs_res]

  targets, _ = CTCTrainer.reconstruct_sentences(res['targets'], res['targets'], idx_to_tokens, tokens_to_idx)

  print(f'GREEDY PREDICTION SCORES:\n{CTCTrainer.scorer(res["targets"], greedy_preds, idx_to_tokens, tokens_to_idx)}\n')
  print(f'BEAM SEARCH PREDICTION SCORES:\n{CTCTrainer.scorer(targets, bs_s, rec=False)}')


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

  ## START AUGMENTATION
  # words_groups = [[el[:i] for i in (el == self.tokens_to_idx[' ']).nonzero().reshape(-1)] for el in targets]
  # for i in range(min([len(wg) for wg in words_groups])):
  #   new_inputs = torch.cat((inputs, torch.ones(inputs.size(0), inputs.size(1), 8).to(inputs.device) * (i + 1)), dim=-1)
  #   # query = [int(el) for el in bin(i).split('b')[-1]]
  #   # query = [0] * (8 - len(query)) + query
  #   # new_inputs = torch.cat((inputs, torch.Tensor(query).to(inputs.device).repeat(inputs.size(0), inputs.size(1), 1)+1), dim=-1)
  #   new_targets = pad_sequence([el[i] for el in words_groups], batch_first=True, padding_value=0)
  #   new_target_lens = torch.LongTensor([len(el[i]) for el in words_groups]).to(inputs.device)

  #   preds = self.model(new_inputs)

  #   self.optimizer.zero_grad()
  #   current_loss = self.criterion(preds.permute(1, 0, 2).log_softmax(-1), new_targets, input_lens, new_target_lens)
  #   current_loss.backward()
  #   self.optimizer.step()
  # inputs = torch.cat((inputs, torch.ones(inputs.size(0), inputs.size(1), 8).to(inputs.device) * -1), dim=-1)
  ## END AUGMENTATION

  ## START AUGMENTATION
  # idxs = [(el == self.tokens_to_idx[' ']).nonzero().reshape(-1) for el in targets]
  # words_groups = [[el[idxs[i][j]:idxs[i][j+1]] for j in range(len(idxs[i]) - 1)] for i, el in enumerate(targets)]
  # for i in range(min([len(wg) for wg in words_groups])):
  #   new_targets = pad_sequence([el[i] for el in words_groups], batch_first=True, padding_value=0)
  #   new_target_lens = torch.LongTensor([len(el[i]) for el in words_groups]).to(inputs.device)

  #   # preds = self.model(inputs, y=torch.ones(inputs.size(0), max(input_lens), inputs.size(2)).to(inputs.device) * (i + 1))
  #   preds = self.model(inputs, y=self.query_embedder(torch.LongTensor([i])).repeat(inputs.size(0), max(input_lens), 1))

  #   self.optimizer.zero_grad()
  #   current_loss = self.criterion(preds.permute(1, 0, 2).log_softmax(-1), new_targets, input_lens, new_target_lens)
  #   current_loss.backward()
  #   self.optimizer.step()
  ## END AUGMENTATION