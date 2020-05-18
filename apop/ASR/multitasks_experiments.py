import os
import re
import sys
import json
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from fairseq.models.wav2vec import Wav2VecModel
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(__file__).replace('ASR/multitasks_experiments.py', ''))
import utils as u

from data import Data
from ngrams_experiments import multigrams_encoding
from optimizer import CosineAnnealingWarmUpRestarts
from models.divers_models import TextEmbedder, Seq2Seq, Seq2SeqReview, ConvLayer


class TextModel(nn.Module):
  def __init__(self, output_dim, emb_dim, d_model, n_heads, d_ff, kernel_size, n_blocks, max_seq_len,
               dropout=0., only_see_past=True, n_step_aheads=1):
    super().__init__()
    self.n_step_aheads = n_step_aheads
    embedder = TextEmbedder(output_dim, emb_dim=emb_dim, max_seq_len=max_seq_len)
    self.net = ConvLayer(emb_dim, d_model, n_heads, d_ff, kernel_size, n_blocks, embedder, dropout=dropout, only_see_past=only_see_past)
    self.output_proj = nn.Linear(d_model, n_step_aheads*output_dim)
  
  def forward(self, x):
    x = self.net(x)
    return self.output_proj(x)


class TextDataset(Dataset):
  def __init__(self, ids_to_encodedsources, sort_by_target_len=True):
    self.ids_to_encodedsources = ids_to_encodedsources
    self.identities = list(sorted(ids_to_encodedsources.keys()))

    if sort_by_target_len:
      self.identities = TextDataset._sort_by_targets_len(self.identities, ids_to_encodedsources)
  
  @staticmethod
  def _sort_by_targets_len(ids, ids2es):
    return list(map(lambda x: x[0], sorted([(i, len(ids2es[i])) for i in ids], key=lambda x: x[1])))
  
  def __len__(self):
    return len(self.identities)
  
  def __getitem__(self, idx):
    return torch.LongTensor(self.ids_to_encodedsources[self.identities[idx]])


class TextCollator(object):
  def __init__(self, pad_idx):
    self.pad_idx = pad_idx
  
  def __call__(self, batch):
    return pad_sequence(batch, batch_first=True, padding_value=self.pad_idx)


class TextTrainer(object):
  def __init__(self, device=None, encoding_fn=Data.letters_encoding, metadata_file='_Data_metadata_letters.pk', smoothing_eps=0.1,
               train_folder='../../../datasets/openslr/LibriSpeech/train-clean-100/', batch_size=32, lr=1e-3, n_epochs=500,
               test_folder='../../../datasets/openslr/LibriSpeech/test-clean/', logfile='_logs/_textTrainer_logs.txt',
               scorer=Data.compute_accuracy, save_name_model='convnet/text_trainer.pt'):
    logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.encoding_fn = encoding_fn
    self.metadata_file = metadata_file
    self.train_folder = train_folder
    self.test_folder = test_folder
    self.batch_size = batch_size
    self.save_name_model = save_name_model
    self.smoothing_eps = smoothing_eps
    self.n_epochs = n_epochs
    self.scorer = scorer

    self.set_data()
    self.sos_idx = self.data.tokens_to_idx['<sos>']
    self.eos_idx = self.data.tokens_to_idx['<eos>']
    self.pad_idx = self.data.tokens_to_idx['<pad>']
    self.mask_idx = self.data.tokens_to_idx['<mask>']

    self.set_data_loader()

    self.model = self.instanciate_model()
    logging.info(f'The model has {u.count_trainable_parameters(self.model):,} trainable parameters')

    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.criterion = u.CrossEntropyLoss(self.pad_idx)
  
  def set_data(self):
    self.data = Data()

    if not os.path.isfile(self.metadata_file):
      self.data.process_all_transcripts(self.train_folder, self.test_folder, encoding_fn=self.encoding_fn)
      self.data.idx_to_tokens.append('<mask>')
      self.data.tokens_to_idx['<mask>'] = self.data.idx_to_tokens.index('<mask>')
      self.data.save_metadata(save_name=self.metadata_file)
    else:
      self.data.load_metadata(save_name=self.metadata_file)
  
  def set_data_loader(self):
    train_dataset = TextDataset(self.data.ids_to_encodedsources_train)
    test_dataset = TextDataset(self.data.ids_to_encodedsources_test)
    collator = TextCollator(self.pad_idx)
    self.train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collator,
                                        pin_memory=True, drop_last=True)
    self.test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collator,
                                       pin_memory=True, drop_last=True)

  def instanciate_model(self, emb_dim=100, d_model=512, n_heads=8, d_ff=1024, kernel_size=3, n_blocks=6):
    self.output_dim = len(self.data.idx_to_tokens)
    self.n_step_aheads = 1
    return TextModel(self.output_dim, emb_dim, d_model, n_heads, d_ff, kernel_size, n_blocks, self.data.max_source_len,
                     dropout=0., only_see_past=False, n_step_aheads=self.n_step_aheads).to(self.device)
  
  def train(self):
    print('Start Training...')
    eval_accuracy_memory = 0
    for epoch in tqdm(range(self.n_epochs)):
      epoch_loss, accs = self.train_pass()
      logging.info(f"Epoch {epoch} | train_loss = {epoch_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")
      eval_loss, accs = self.evaluation()
      logging.info(f"Epoch {epoch} | test_loss = {eval_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")

      oea = accs.get('preds_acc', None)

      if oea is not None and oea > eval_accuracy_memory:
        logging.info(f'Save model with eval_accuracy = {oea:.3f}')
        u.save_checkpoint(self.model, None, self.save_name_model)
        eval_accuracy_memory = oea
  
  @torch.no_grad()
  def evaluation(self):
    losses = 0
    targets, predictions = [], []

    self.model.eval()

    for enc_text in tqdm(self.test_data_loader):
      batch_size, seq_len = enc_text.shape
      mask = np.random.choice([True, False], p=[0.15, 0.85], size=(seq_len-2)*batch_size).reshape((batch_size, -1))
      mask[(tuple(range(batch_size)), np.random.randint(0, seq_len-2, batch_size))] = True
      enc_text[:, 1:-1][mask] = self.mask_idx
      enc_text = enc_text.to(self.device)

      target = enc_text[:, 2:].clone()
      if self.n_step_aheads > 1:
        target = torch.cat([target[:, i:i+self.n_step_aheads] for i in range(target.shape[-1])], dim=-1)
        target = F.pad(target, (0, self.n_step_aheads * (enc_text.shape[-1] - 2) - target.shape[-1]))
      target[mask==False] = self.pad_idx
      

      preds = self.model(enc_text[:, 1:-1])
      preds_sep = preds.reshape(preds.shape[0], preds.shape[1] * self.n_step_aheads, self.output_dim)

      losses += self.criterion(preds.reshape(-1, preds_sep.shape[-1]), target.reshape(-1), epsilon=self.smoothing_eps).item()

      targets += [et[m].tolist() for et, m in zip(enc_text[:, 2:], mask)]
      predictions += [p[m].tolist() for p, m in zip(preds[:, :, :self.output_dim].argmax(dim=-1), mask)]
    
    self.model.train()

    accs = self.scorer(**{'targets': targets, 'predictions': predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                          'idx_to_tokens': self.data.idx_to_tokens})

    return losses / len(self.test_data_loader), accs
  
  def train_pass(self, repeat_batch=1):
    # Task1 = predict masked words
    losses = 0
    targets, predictions = [], []

    for enc_text in tqdm(self.train_data_loader):
      batch_size, seq_len = enc_text.shape
      mask = np.random.choice([True, False], p=[0.15, 0.85], size=(seq_len-2)*batch_size).reshape((batch_size, -1))
      mask[(tuple(range(batch_size)), np.random.randint(0, seq_len-2, batch_size))] = True
      enc_text[:, 1:-1][mask] = self.mask_idx
      enc_text = enc_text.to(self.device)

      target = enc_text[:, 2:].clone()
      if self.n_step_aheads > 1:
        target = torch.cat([target[:, i:i+self.n_step_aheads] for i in range(target.shape[-1])], dim=-1)
        target = F.pad(target, (0, self.n_step_aheads * (enc_text.shape[-1] - 2) - target.shape[-1]))
      # target[mask==False] = self.pad_idx

      for _ in range(repeat_batch):
        preds = self.model(enc_text[:, 1:-1])
        preds_sep = preds.reshape(preds.shape[0], preds.shape[1] * self.n_step_aheads, self.output_dim)

        self.optimizer.zero_grad()
        current_loss = self.criterion(preds_sep.reshape(-1, preds_sep.shape[-1]), target.reshape(-1), epsilon=self.smoothing_eps)
        current_loss.backward()
        self.optimizer.step()

      # targets += [et[m].tolist() for et, m in zip(enc_text[:, 2:], mask)]
      # predictions += [p[m].tolist() for p, m in zip(preds[:, :, :self.output_dim].argmax(dim=-1), mask)]
      targets += enc_text[:, 2:].tolist()
      predictions += preds[:, :, :self.output_dim].argmax(dim=-1).tolist()

      losses += current_loss.item()

    accs = self.scorer(**{'targets': targets, 'predictions': predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                          'idx_to_tokens': self.data.idx_to_tokens})
    
    return losses / len(self.train_data_loader), accs


class STTTrainer(object):
  def __init__(self, device=None, metadata_file='_Data_metadata_multigrams_wav2vec.pk', logfile='_logs/_logs_sttTrainer.txt',
               train_folder='../../../datasets/openslr/LibriSpeech/train-clean-100/', lr=1e-4, batch_size=8, scores_step=5,
               test_folder='../../../datasets/openslr/LibriSpeech/test-clean/', scorer=Data.compute_scores, eval_step=10,
               save_name_model='convnet/stt_trainer.pt', smoothing_eps=0.1, n_epochs=500, load_model=True,
               encoding_fn=multigrams_encoding, block_type='self_attn', lr_scheduling=False):
    logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.train_folder = train_folder
    self.test_folder = test_folder
    self.metadata_file = metadata_file
    self.batch_size = batch_size
    self.scorer = scorer
    self.scores_step = scores_step
    self.eval_step = eval_step
    self.save_name_model = save_name_model
    self.smoothing_eps = smoothing_eps
    self.n_epochs = n_epochs
    self.encoding_fn = encoding_fn
    self.block_type = block_type
    self.lr_scheduling = lr_scheduling

    self.set_wav2vec()
    self.set_data()
    self.sos_idx = self.data.tokens_to_idx['<sos>']
    self.eos_idx = self.data.tokens_to_idx['<eos>']
    self.pad_idx = self.data.tokens_to_idx['<pad>']

    self.set_data_loader()

    self.model = self.instanciate_model()
    logging.info(f'The model has {u.count_trainable_parameters(self.model):,} trainable parameters')

    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.criterion = u.CrossEntropyLoss(self.pad_idx)

    if lr_scheduling:
      self.lr_scheduler = CosineAnnealingWarmUpRestarts(self.optimizer, T_0=150, T_mult=1, eta_max=1e-3, T_up=10, gamma=0.5)

    if load_model:
      u.load_model(self.model, self.save_name_model, restore_only_similars=True)
  
  def set_wav2vec(self):
    cp = torch.load('wav2vec_large.pt')
    self.wav2vec_model = Wav2VecModel.build_model(cp['args'], task=None)
    self.wav2vec_model.load_state_dict(cp['model'])
    self.wav2vec_model.eval()
  
  def set_data(self):
    self.data = Data()

    if not os.path.isfile(self.metadata_file):
      self.data.set_audio_metadata(self.train_folder, self.test_folder, list_files_fn=Data.get_openslr_files,
                                   process_file_fn=Data.wav2vec_extraction, save_features=True, wav2vec_model=self.wav2vec_model)
      self.data.process_all_transcripts(self.train_folder, self.test_folder, encoding_fn=self.encoding_fn)
      self.data.save_metadata(save_name=self.metadata_file)
    else:
      self.data.load_metadata(save_name=self.metadata_file)
  
  def set_data_loader(self):
    self.train_data_loader = self.data.get_dataset_generator(batch_size=self.batch_size, pad_idx=self.pad_idx,
                                                             pin_memory=True, save_features=True, wav2vec_model=self.wav2vec_model,
                                                             slice_fn=Data.wav2vec_extraction)
    self.test_data_loader = self.data.get_dataset_generator(train=False, batch_size=self.batch_size, pad_idx=self.pad_idx,
                                                            pin_memory=True, shuffle=False, save_features=True,
                                                            wav2vec_model=self.wav2vec_model, slice_fn=Data.wav2vec_extraction)

  def instanciate_model(self, emb_dim=50, d_model=256, n_heads=4, d_ff=512, kernel_size=3, n_blocks=6):
    self.output_dim = len(self.data.idx_to_tokens)
    self.n_step_aheads = 1
    return Seq2Seq(self.output_dim, self.data.n_signal_feats, emb_dim, d_model, n_heads, d_ff, d_ff, kernel_size, n_blocks,
                   self.data.max_signal_len, self.data.max_source_len, dropout=0., n_step_aheads=self.n_step_aheads,
                   enc_block_type=self.block_type, dec_block_type=self.block_type).to(self.device)

  def train(self):
    print('Start Training...')
    eval_accuracy_memory = 0
    for epoch in tqdm(range(self.n_epochs)):
      epoch_loss, accs = self.train_pass(only_loss=epoch % self.scores_step != 0)
      logging.info(f"Epoch {epoch} | train_loss = {epoch_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")
      eval_loss, accs = self.evaluation(scores=epoch % self.scores_step == 0, greedy_scores=epoch % self.eval_step ==0)
      logging.info(f"Epoch {epoch} | test_loss = {eval_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")

      oea = accs['preds_acc'] if 'preds_acc' in accs else accs.get('word_accuracy', None)

      if self.lr_scheduling:
        self.lr_scheduler.step()

      if oea is not None and oea > eval_accuracy_memory:
        logging.info(f'Save model with eval_accuracy = {oea:.3f}')
        u.save_checkpoint(self.model, None, self.save_name_model)
        eval_accuracy_memory = oea

  @torch.no_grad()
  def evaluation(self, scores=False, greedy_scores=False):
    losses, accs = 0, {}
    targets, predictions, greedy_predictions = [], [], []

    self.model.eval()

    for enc_in, dec_in in tqdm(self.test_data_loader):
      enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
      preds = self.model(enc_in, dec_in[:, :-1])

      losses += self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1), epsilon=self.smoothing_eps).item()

      targets += dec_in[:, 1:].tolist()
      predictions += preds.argmax(dim=-1).tolist()
      
      if greedy_scores:
        preds = self.model.greedy_decoding(enc_in, self.sos_idx, self.eos_idx, max_seq_len=dec_in.shape[1])
        greedy_predictions += preds.tolist()
    
    self.model.train()

    if scores:
      accs = self.scorer(**{'targets': targets, 'predictions': predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.data.idx_to_tokens})

    if greedy_scores:
      greedy_accs = self.scorer(**{'targets': targets, 'predictions': greedy_predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.data.idx_to_tokens})
      accs = {**accs, **{'greedy_'+k: v for k, v in greedy_accs.items()}}

    return losses / len(self.test_data_loader), accs
  
  def train_pass(self, only_loss=True):
    losses, accs = 0, {}
    targets, predictions = [], []

    for enc_in, dec_in in tqdm(self.train_data_loader):
      enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
      preds = self.model(enc_in, dec_in[:, :-1])

      self.optimizer.zero_grad()

      current_loss = self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1), epsilon=self.smoothing_eps)

      current_loss.backward()

      self.optimizer.step()

      targets += dec_in[:, 1:].tolist()
      predictions += preds.argmax(dim=-1).tolist()

      losses += current_loss.item()
    
    if not only_loss:
      accs = self.scorer(**{'targets': targets, 'predictions': predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.data.idx_to_tokens})
    
    return losses / len(self.train_data_loader), accs


class STTReviewTrainer(STTTrainer):
  def __init__(self):
    super().__init__(logfile='_logs/_logs_sttReviewTrainer.txt', save_name_model='convnet/sttReview_trainer.pt')
  
  def instanciate_model(self, emb_dim=50, d_model=256, n_heads=4, d_ff=384, kernel_size=3, n_blocks=4):
    self.output_dim = len(self.data.idx_to_tokens)
    self.n_step_aheads = 1
    return Seq2SeqReview(self.output_dim, self.data.n_signal_feats, emb_dim, d_model, n_heads, d_ff, kernel_size, n_blocks,
                         self.data.max_signal_len, self.data.max_source_len, dropout=0.).to(self.device)
  
  def train_pass(self, only_loss=True):
    losses, accs = 0, {}
    targets, predictions = [], []

    for enc_in, dec_in in tqdm(self.train_data_loader):
      enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
      preds, _ = self.model(enc_in, dec_in[:, :-1])

      self.optimizer.zero_grad()
      current_loss = self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1), epsilon=self.smoothing_eps)
      current_loss.backward()
      self.optimizer.step()

      targets += dec_in[:, 1:].tolist()
      predictions += preds.argmax(dim=-1).tolist()

      losses += current_loss.item()
    
    if not only_loss:
      accs = self.scorer(**{'targets': targets, 'predictions': predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.data.idx_to_tokens})
    
    return losses / len(self.train_data_loader), accs
  
  @torch.no_grad()
  def evaluation(self, scores=False, greedy_scores=False):
    losses, accs = 0, {}
    targets, predictions, greedy_predictions = [], [], []

    self.model.eval()

    for enc_in, dec_in in tqdm(self.test_data_loader):
      enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
      preds, _ = self.model(enc_in, dec_in[:, :-1])

      losses += self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1), epsilon=self.smoothing_eps).item()

      targets += dec_in[:, 1:].tolist()
      predictions += preds.argmax(dim=-1).tolist()
      
      if greedy_scores:
        preds, _ = self.model.greedy_decoding(enc_in, self.sos_idx, self.eos_idx, max_seq_len=dec_in.shape[1])
        greedy_predictions += preds.tolist()
    
    self.model.train()

    if scores:
      accs = self.scorer(**{'targets': targets, 'predictions': predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.data.idx_to_tokens})

    if greedy_scores:
      greedy_accs = self.scorer(**{'targets': targets, 'predictions': greedy_predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.data.idx_to_tokens})
      accs = {**accs, **{'greedy_'+k: v for k, v in greedy_accs.items()}}

    return losses / len(self.test_data_loader), accs


class STTReviewTrainer2(STTTrainer):
  def __init__(self, logfile='_logs/_logs_sttReviewTrainer2.txt'):
    super().__init__(logfile=logfile, save_name_model='convnet/sttReview_trainer2.pt')
    # u.load_model(self.model, 'convnet/stt_trainer.pt', restore_only_similars=True)
  
  def instanciate_model(self, emb_dim=50, d_model=256, n_heads=4, d_ff=320, kernel_size=3, n_blocks=5):
    self.output_dim = len(self.data.idx_to_tokens)
    self.n_step_aheads = 1
    return Seq2SeqReview(self.output_dim, self.data.n_signal_feats, emb_dim, d_model, n_heads, d_ff, kernel_size, n_blocks,
                         self.data.max_signal_len, self.data.max_source_len, dropout=0.).to(self.device)
  
  def train_pass(self, only_loss=True):
    losses, accs = 0, {}
    targets, predictions = [], []

    for enc_in, dec_in in tqdm(self.train_data_loader):
      enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
      preds, first_preds = self.model(enc_in, dec_in[:, :-1])

      self.optimizer.zero_grad()

      target = dec_in[:, 1:].reshape(-1)

      first_preds_loss = self.criterion(first_preds.reshape(-1, first_preds.shape[-1]), target, epsilon=self.smoothing_eps)
      first_preds_loss.backward(retain_graph=True)

      current_loss = self.criterion(preds.reshape(-1, preds.shape[-1]), target, epsilon=self.smoothing_eps)
      current_loss.backward()

      self.optimizer.step()

      targets += dec_in[:, 1:].tolist()
      predictions += preds.argmax(dim=-1).tolist()

      losses += current_loss.item()
    
    if not only_loss:
      accs = self.scorer(**{'targets': targets, 'predictions': predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.data.idx_to_tokens})
    
    return losses / len(self.train_data_loader), accs
  
  @torch.no_grad()
  def evaluation(self, scores=False, greedy_scores=False):
    losses, accs = 0, {}
    targets, predictions, greedy_predictions = [], [], []

    self.model.eval()

    for enc_in, dec_in in tqdm(self.test_data_loader):
      enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
      preds, _ = self.model(enc_in, dec_in[:, :-1])

      losses += self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1), epsilon=self.smoothing_eps).item()

      targets += dec_in[:, 1:].tolist()
      predictions += preds.argmax(dim=-1).tolist()
      
      if greedy_scores:
        preds, _ = self.model.greedy_decoding(enc_in, self.sos_idx, self.eos_idx, max_seq_len=dec_in.shape[1])
        greedy_predictions += preds.tolist()
    
    self.model.train()

    if scores:
      accs = self.scorer(**{'targets': targets, 'predictions': predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.data.idx_to_tokens})

    if greedy_scores:
      greedy_accs = self.scorer(**{'targets': targets, 'predictions': greedy_predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.data.idx_to_tokens})
      accs = {**accs, **{'greedy_'+k: v for k, v in greedy_accs.items()}}

    return losses / len(self.test_data_loader), accs
  
  @torch.no_grad()
  def dump_predictions(self, save_name='_multitaks_preds.json', data_loader=None):
    data_loader = self.test_data_loader if data_loader is None else data_loader
    u.load_model(self.model, self.save_name_model, restore_only_similars=True)
    self.model.eval()

    targets, predictions, greedy_predictions = [], [], []
    first_predictions, first_greedy_predictions = [], []
    for enc_in, dec_in in tqdm(data_loader):
      enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
      preds, first_preds = self.model(enc_in, dec_in[:, :-1])
      greedy_preds, first_greedy_preds = self.model.greedy_decoding(enc_in, self.sos_idx, self.eos_idx, max_seq_len=dec_in.shape[1])
      targets += dec_in[:, 1:].tolist()
      predictions += preds.argmax(dim=-1).tolist()
      first_predictions += first_preds.argmax(dim=-1).tolist()
      greedy_predictions += greedy_preds.tolist()
      first_greedy_predictions += first_greedy_preds.tolist()
    
    targets_sentences = Data.reconstruct_sources(targets, self.data.idx_to_tokens, self.eos_idx, joiner='')
    predictions_sentences = Data.reconstruct_sources(predictions, self.data.idx_to_tokens, self.eos_idx, joiner='')
    first_predictions_sentences = Data.reconstruct_sources(first_predictions, self.data.idx_to_tokens, self.eos_idx, joiner='')
    greedy_predictions_sentences = Data.reconstruct_sources(greedy_predictions, self.data.idx_to_tokens, self.eos_idx, joiner='')
    first_greedy_predictions_sentences = Data.reconstruct_sources(first_greedy_predictions, self.data.idx_to_tokens, self.eos_idx, joiner='')

    with open(save_name, 'w') as f:
      json.dump([{'target': t, 'prediction': p, 'greedy_prediction': gp, 'first_prediction': fp, 'first_greedy_prediction': fgp}
                    for t, p, gp, fp, fgp in zip(targets_sentences, predictions_sentences, greedy_predictions_sentences,
                                                 first_predictions_sentences, first_greedy_predictions_sentences)], f)


class STTReviewTrainer3(STTTrainer):
  def __init__(self):
    super().__init__(logfile='_logs/_logs_sttReviewTrainer3.txt', save_name_model='convnet/sttReview_trainer3.pt')
  
  def instanciate_model(self, emb_dim=50, d_model=256, n_heads=4, d_ff=320, kernel_size=3, n_blocks=4):
    self.output_dim = len(self.data.idx_to_tokens)
    self.n_step_aheads = 1
    return Seq2SeqReview(self.output_dim, self.data.n_signal_feats, emb_dim, d_model, n_heads, d_ff, kernel_size, n_blocks,
                         self.data.max_signal_len, self.data.max_source_len, dropout=0.).to(self.device)
  
  def train_pass(self, only_loss=True):
    losses, accs = 0, {}
    targets, predictions = [], []

    for enc_in, dec_in in tqdm(self.train_data_loader):
      enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
      preds, first_preds = self.model(enc_in, dec_in[:, :-1])

      self.optimizer.zero_grad()

      target = dec_in[:, 1:].reshape(-1)

      first_preds_loss = self.criterion(first_preds.reshape(-1, first_preds.shape[-1]), target, epsilon=self.smoothing_eps)
      first_preds_loss.backward(retain_graph=True)

      current_loss = self.criterion(preds.reshape(-1, preds.shape[-1]), target, epsilon=self.smoothing_eps)
      current_loss.backward()

      self.optimizer.step()

      targets += dec_in[:, 1:].tolist()
      predictions += preds.argmax(dim=-1).tolist()

      losses += current_loss.item()
    
    if not only_loss:
      accs = self.scorer(**{'targets': targets, 'predictions': predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.data.idx_to_tokens})
    
    return losses / len(self.train_data_loader), accs
  
  @torch.no_grad()
  def evaluation(self, scores=False, greedy_scores=False):
    losses, accs = 0, {}
    targets, predictions, greedy_predictions = [], [], []

    self.model.eval()

    for enc_in, dec_in in tqdm(self.test_data_loader):
      enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
      preds, _ = self.model(enc_in, dec_in[:, :-1])

      losses += self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1), epsilon=self.smoothing_eps).item()

      targets += dec_in[:, 1:].tolist()
      predictions += preds.argmax(dim=-1).tolist()
      
      if greedy_scores:
        preds, _ = self.model.greedy_decoding(enc_in, self.sos_idx, self.eos_idx, max_seq_len=dec_in.shape[1])
        greedy_predictions += preds.tolist()
    
    self.model.train()

    if scores:
      accs = self.scorer(**{'targets': targets, 'predictions': predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.data.idx_to_tokens})

    if greedy_scores:
      greedy_accs = self.scorer(**{'targets': targets, 'predictions': greedy_predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.data.idx_to_tokens})
      accs = {**accs, **{'greedy_'+k: v for k, v in greedy_accs.items()}}

    return losses / len(self.test_data_loader), accs


class STTReviewTrainer4(STTTrainer):
  def __init__(self, logfile='_logs/_logs_sttReviewTrainer4.txt'):
    super().__init__(logfile=logfile, save_name_model='convnet/sttReview_trainer4.pt')
  
  def instanciate_model(self, emb_dim=50, d_model=256, n_heads=4, d_ff=320, kernel_size=3, n_blocks=4):
    self.output_dim = len(self.data.idx_to_tokens)
    self.n_step_aheads = 1
    return Seq2SeqReview(self.output_dim, self.data.n_signal_feats, emb_dim, d_model, n_heads, d_ff, kernel_size, n_blocks,
                         self.data.max_signal_len, self.data.max_source_len, dropout=0.25).to(self.device)
  
  def train_pass(self, only_loss=True):
    losses, accs = 0, {}
    targets, predictions = [], []

    for enc_in, dec_in in tqdm(self.train_data_loader):
      enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
      preds, first_preds = self.model(enc_in, dec_in[:, :-1])

      self.optimizer.zero_grad()

      target = dec_in[:, 1:].reshape(-1)

      first_preds_loss = self.criterion(first_preds.reshape(-1, first_preds.shape[-1]), target, epsilon=self.smoothing_eps)
      first_preds_loss.backward()

      current_loss = self.criterion(preds.reshape(-1, preds.shape[-1]), target, epsilon=self.smoothing_eps)
      current_loss.backward()

      self.optimizer.step()

      targets += dec_in[:, 1:].tolist()
      predictions += preds.argmax(dim=-1).tolist()

      losses += current_loss.item()
    
    if not only_loss:
      accs = self.scorer(**{'targets': targets, 'predictions': predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.data.idx_to_tokens})
    
    return losses / len(self.train_data_loader), accs
  
  @torch.no_grad()
  def evaluation(self, scores=False, greedy_scores=False):
    losses, accs = 0, {}
    targets, predictions, greedy_predictions = [], [], []

    self.model.eval()

    for enc_in, dec_in in tqdm(self.test_data_loader):
      enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
      preds, _ = self.model(enc_in, dec_in[:, :-1])

      losses += self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1), epsilon=self.smoothing_eps).item()

      targets += dec_in[:, 1:].tolist()
      predictions += preds.argmax(dim=-1).tolist()
      
      if greedy_scores:
        preds, _ = self.model.greedy_decoding(enc_in, self.sos_idx, self.eos_idx, max_seq_len=dec_in.shape[1])
        greedy_predictions += preds.tolist()
    
    self.model.train()

    if scores:
      accs = self.scorer(**{'targets': targets, 'predictions': predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.data.idx_to_tokens})

    if greedy_scores:
      greedy_accs = self.scorer(**{'targets': targets, 'predictions': greedy_predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.data.idx_to_tokens})
      accs = {**accs, **{'greedy_'+k: v for k, v in greedy_accs.items()}}

    return losses / len(self.test_data_loader), accs

  @torch.no_grad()
  def dump_predictions(self, save_name='_multitaks_preds.json', data_loader=None):
    data_loader = self.test_data_loader if data_loader is None else data_loader
    u.load_model(self.model, self.save_name_model, restore_only_similars=True)
    self.model.eval()

    targets, predictions, greedy_predictions = [], [], []
    first_predictions, first_greedy_predictions = [], []
    for enc_in, dec_in in tqdm(data_loader):
      enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
      preds, first_preds = self.model(enc_in, dec_in[:, :-1])
      greedy_preds, first_greedy_preds = self.model.greedy_decoding(enc_in, self.sos_idx, self.eos_idx, max_seq_len=dec_in.shape[1])
      targets += dec_in[:, 1:].tolist()
      predictions += preds.argmax(dim=-1).tolist()
      first_predictions += first_preds.argmax(dim=-1).tolist()
      greedy_predictions += greedy_preds.tolist()
      first_greedy_predictions += first_greedy_preds.tolist()
    
    targets_sentences = Data.reconstruct_sources(targets, self.data.idx_to_tokens, self.eos_idx, joiner='')
    predictions_sentences = Data.reconstruct_sources(predictions, self.data.idx_to_tokens, self.eos_idx, joiner='')
    first_predictions_sentences = Data.reconstruct_sources(first_predictions, self.data.idx_to_tokens, self.eos_idx, joiner='')
    greedy_predictions_sentences = Data.reconstruct_sources(greedy_predictions, self.data.idx_to_tokens, self.eos_idx, joiner='')
    first_greedy_predictions_sentences = Data.reconstruct_sources(first_greedy_predictions, self.data.idx_to_tokens, self.eos_idx, joiner='')

    with open(save_name, 'w') as f:
      json.dump([{'target': t, 'prediction': p, 'greedy_prediction': gp, 'first_prediction': fp, 'first_greedy_prediction': fgp}
                    for t, p, gp, fp, fgp in zip(targets_sentences, predictions_sentences, greedy_predictions_sentences,
                                                 first_predictions_sentences, first_greedy_predictions_sentences)], f)


class STTTrainer2(STTTrainer):
  def __init__(self, logfile='_logs/_logs_sttTrainer2.txt'):
    super().__init__(logfile=logfile, save_name_model='convnet/stt_trainer2.pt', encoding_fn=Data.letters_encoding,
                     metadata_file='_Data_metadata_letters_wav2vec.pk')
  
  def train_pass(self, only_loss=True):
    losses, accs = 0, {}
    targets, predictions = [], []

    for enc_in, dec_in in tqdm(self.train_data_loader):
      enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
      preds = self.model(enc_in, dec_in[:, :-1])

      self.optimizer.zero_grad()

      current_loss = self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1), epsilon=self.smoothing_eps)
      current_loss += u.l1_regularization(self.model, _lambda=0.01, device=self.device)

      current_loss.backward()

      self.optimizer.step()

      targets += dec_in[:, 1:].tolist()
      predictions += preds.argmax(dim=-1).tolist()

      losses += current_loss.item()
    
    if not only_loss:
      accs = self.scorer(**{'targets': targets, 'predictions': predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.data.idx_to_tokens})
    
    return losses / len(self.train_data_loader), accs


class STTTrainer3(STTTrainer):
  def __init__(self, logfile='_logs/_logs_sttTrainer3.txt'):
    super().__init__(logfile=logfile, save_name_model='convnet/stt_trainer3.pt', encoding_fn=Data.letters_encoding,
                     metadata_file='_Data_metadata_letters_wav2vec.pk', block_type='dilated')
  
  def instanciate_model(self, emb_dim=50, d_model=512, n_heads=8, enc_d_ff=2048, dec_d_ff=1024, kernel_size=3, n_blocks=6):
    self.output_dim = len(self.data.idx_to_tokens)
    self.n_step_aheads = 1
    return Seq2Seq(self.output_dim, self.data.n_signal_feats, emb_dim, d_model, n_heads, enc_d_ff, dec_d_ff, kernel_size, n_blocks,
                   self.data.max_signal_len, self.data.max_source_len, dropout=0., n_step_aheads=self.n_step_aheads,
                   enc_block_type=self.block_type).to(self.device)


class STTTrainer4(STTTrainer):
  def __init__(self, logfile='_logs/_logs_sttTrainer4.txt'):
    super().__init__(logfile=logfile, save_name_model='convnet/stt_trainer4.pt', encoding_fn=Data.letters_encoding,
                     metadata_file='_Data_metadata_letters_wav2vec.pk', block_type='dilated', batch_size=32)
  
  def instanciate_model(self, emb_dim=50, d_model=512, n_heads=8, enc_d_ff=2048, dec_d_ff=1024, kernel_size=3, n_blocks=6):
    self.output_dim = len(self.data.idx_to_tokens)
    self.n_step_aheads = 1
    return Seq2Seq(self.output_dim, self.data.n_signal_feats, emb_dim, d_model, n_heads, enc_d_ff, dec_d_ff, kernel_size, n_blocks,
                   self.data.max_signal_len, self.data.max_source_len, dropout=0., n_step_aheads=self.n_step_aheads,
                   enc_block_type=self.block_type, dec_block_type=self.block_type).to(self.device)


class STTTrainer5(STTTrainer):
  def __init__(self, logfile='_logs/_logs_sttTrainer5.txt'):
    super().__init__(logfile=logfile, save_name_model='convnet/stt_trainer5.pt', encoding_fn=Data.letters_encoding, lr=1e-6,
                     metadata_file='_Data_metadata_letters_wav2vec.pk', block_type='dilated', batch_size=32, lr_scheduling=True)
  
  def instanciate_model(self, emb_dim=50, d_model=512, n_heads=8, enc_d_ff=2048, dec_d_ff=1024, kernel_size=3, n_blocks=6):
    self.output_dim = len(self.data.idx_to_tokens)
    self.n_step_aheads = 1
    return Seq2Seq(self.output_dim, self.data.n_signal_feats, emb_dim, d_model, n_heads, enc_d_ff, dec_d_ff, kernel_size, n_blocks,
                   self.data.max_signal_len, self.data.max_source_len, dropout=0., n_step_aheads=self.n_step_aheads,
                   enc_block_type=self.block_type, dec_block_type=self.block_type).to(self.device)


if __name__ == "__main__":
  ## SEEDING FOR REPRODUCIBILITY
  SEED = 42
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)

  # tt = TextTrainer()
  # tt.train()

  # stt_trainer = STTTrainer()
  # stt_trainer.train()

  # stt_trainer = STTTrainer(logfile='_logs/_logs_sttTrainerWise.txt', save_name_model='convnet/sttWise_trainer.pt', batch_size=8)
  # stt_trainer.train()

  # stt_review_trainer = STTReviewTrainer()
  # stt_review_trainer.train()

  rep = input('Start sttTrainer experiment? (y or n): ')
  if rep == 'y':
    experiments = {k.replace('STTTrainer', ''): v for k, v in locals().items() if re.search(r'STTTrainer\d+', k) is not None}
  
    rep = input(f'Which Experiment do you want to start? ({",".join(experiments.keys())}): ')
    exp = experiments[rep]()
    exp.train()
  
  rep = input('Start sttReviewTrainer experiment? (y or n): ')
  if rep == 'y':
    experiments = {k.replace('STTReviewTrainer', ''): v for k, v in locals().items() if re.search(r'STTReviewTrainer\d+', k) is not None}
  
    rep = input(f'Which Experiment do you want to start? ({",".join(experiments.keys())}): ')
    exp = experiments[rep]()
    exp.train()