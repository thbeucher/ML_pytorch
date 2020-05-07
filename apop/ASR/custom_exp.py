import os
import sys
import math
import torch
import random
import logging
import numpy as np
import torch.nn as nn

from tqdm import tqdm

sys.path.append(os.path.abspath(__file__).replace('ASR/custom_exp.py', ''))
import utils as u
import models.conv_seqseq as css

from data import Data
from convnet_trainer import ConvnetTrainer
from ngrams_experiments import multigrams_encoding


class PositionalEncoding(nn.Module):
  # from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
  def __init__(self, d_model, dropout=0.1, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:, :x.size(1)]
    return self.dropout(x)


def process_raw_signal(signal, sample_rate=16000, window_size=0.025, overlap_size=0.010):
  window = int(window_size * sample_rate)
  overlap = int(overlap_size * sample_rate)
  ## Slice signal into n chunks
  signal_sliced = [signal[i:i+window] for i in range(0, len(signal), window-overlap)]
  signal_sliced = [np.pad(signal_sliced[-1], (0, window - len(signal_sliced[-1])), mode='constant')
                    if len(ss) < 400 else ss for ss in signal_sliced]
  signal_sliced = np.array(signal_sliced)
  ## Hanning smoothing of chunks
  signal_sliced *= np.hanning(window)
  ## Extract spectrograms
  chunks_fft = np.fft.fft(signal_sliced)
  power_spectr = np.abs(chunks_fft) ** 2  # amplitude_spectr = np.abs(chunks_fft) | phase_spectr = np.angle(chunks_fft)
  ## log-compression
  scatering = np.log(1 + np.abs(power_spectr))
  gammatones = np.log(0.01 + np.abs(power_spectr))

  features = np.concatenate((power_spectr, scatering, gammatones), axis=1)
  return features


class AudioEmbedder(nn.Module):
  def __init__(self, input_size=1200, n_feats=120, kernel=3, stride=2, dropout=0.1, max_seq_len=3000):
    super().__init__()
    self.compressor_in = nn.Sequential(nn.Linear(input_size, n_feats), nn.ReLU(inplace=True))
    self.positional_encoding = PositionalEncoding(n_feats, dropout=dropout, max_len=max_seq_len)
    self.compressor = nn.Sequential(nn.Conv1d(n_feats, n_feats, kernel, stride=stride), nn.ReLU(inplace=True),
                                    nn.Conv1d(n_feats, n_feats, kernel, stride=stride), nn.ReLU(inplace=True))
  
  def forward(self, x):  # [batch_size, n_frames, input_size]
    out = self.compressor_in(x)  # [batch_size, n_frames, n_feats]
    out = self.positional_encoding(out)
    out = self.compressor(out.permute(0, 2, 1))  # [batch_size, n_feats, n_frames]
    return out.permute(0, 2, 1)  # [batch_size, n_frames, n_feats]


class TextEmbedder(nn.Module):
  def __init__(self, n_embeddings, emb_dim=120, max_seq_len=600, dropout=0.1):
    super().__init__()
    self.token_embedding = nn.Embedding(n_embeddings, emb_dim)
    self.positional_encoding = PositionalEncoding(emb_dim, dropout=dropout, max_len=max_seq_len)
  
  def forward(self, x):  # [batch_size, seq_len]
    out = self.token_embedding(x)  # [batch_size, seq_len, emb_dim]
    return self.positional_encoding(out)


class DecoderBlock(nn.Module):
  def __init__(self, emb_dim, n_heads, d_model, d_ff, dropout=0.1):
    super().__init__()
    self.query_proj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.ReLU(inplace=True))
    # self.key_proj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.ReLU(inplace=True))
    self.value_proj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.ReLU(inplace=True))

    self.attention_feed_forward = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=d_ff, dropout=dropout, activation='relu')

    self.backproj = nn.Sequential(nn.Linear(d_model, emb_dim), nn.ReLU(inplace=True))
  
  def forward(self, query, key, value, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
    query = self.query_proj(query).permute(1, 0, 2)
    # key = self.key_proj(key).permute(1, 0, 2)
    value = self.value_proj(value).permute(1, 0, 2)

    query_enriched = self.attention_feed_forward(query, value, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                                 memory_key_padding_mask=memory_key_padding_mask)
    
    return self.backproj(query_enriched).permute(1, 0, 2)


class Decoder(nn.Module):
  def __init__(self, n_embeddings, output_size, n_blocks=10, emb_dim=120, n_heads=8, d_model=512, d_ff=768, max_seq_len=600, dropout=0.1):
    super().__init__()
    self.token_embedder = TextEmbedder(n_embeddings, emb_dim=emb_dim, max_seq_len=max_seq_len, dropout=dropout)
    self.decoders = nn.ModuleList([DecoderBlock(emb_dim, n_heads, d_model, d_ff, dropout=dropout) for _ in range(n_blocks)])
    self.output_projection = nn.Linear(emb_dim, output_size)
  
  def _generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
  
  def forward(self, query, key, value, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
    tgt_mask = self._generate_square_subsequent_mask(query.shape[1]).to(query.device) if tgt_mask is None else tgt_mask
    query = self.token_embedder(query)
    for decoder in self.decoders:
      query = decoder(query, key, value, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask)
    return self.output_projection(query)


class Recognizer(nn.Module):
  def __init__(self, device=None, input_size=1200, emb_dim=120, kernel=3, stride=2, enc_dropout=0.25, enc_max_seq_len=3000,
               enc_hid_dim=256, enc_n_layers=10, output_size=4825, dec_n_layers=10, dec_n_heads=8, dec_d_model=512,
               dec_d_ff=768, dec_max_seq_len=600, dec_dropout=0.25):
    super().__init__()
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    audio_embedder = AudioEmbedder(input_size=input_size, n_feats=emb_dim, kernel=kernel, stride=stride, dropout=enc_dropout,
                                   max_seq_len=enc_max_seq_len)
    self.encoder = css.Encoder(emb_dim, enc_hid_dim, enc_n_layers, kernel, enc_dropout, self.device, embedder=audio_embedder)
    self.decoder = Decoder(output_size, output_size, n_blocks=dec_n_layers, emb_dim=emb_dim, n_heads=dec_n_heads, d_model=dec_d_model,
                           d_ff=dec_d_ff, max_seq_len=dec_max_seq_len, dropout=dec_dropout)
  
  def forward(self, enc_in, dec_in):
    encoder_conved, encoder_combined = self.encoder(enc_in)
    return self.decoder(dec_in, encoder_conved, encoder_combined)
  
  def greedy_decoding(self, enc_in, sos_idx, eos_idx, max_seq_len=100):
    encoder_conved, encoder_combined = self.encoder(enc_in)

    batch_size = enc_in.shape[0]

    dec_in = torch.LongTensor(batch_size, 1).fill_(sos_idx).to(self.device)

    finished = [False] * batch_size

    for _ in range(max_seq_len):
      output = self.decoder(dec_in, encoder_conved, encoder_combined)
      pred = output[:, -1, :].argmax(-1).unsqueeze(1)

      for idx in range(batch_size):
        if not finished[idx] and pred[idx].item() == eos_idx:
          finished[idx] = True

      dec_in = torch.cat((dec_in, pred), dim=1)

      if all(finished):
        break
    
    return dec_in[:, 1:]


class CustomExperiment1(ConvnetTrainer):
  def __init__(self, logfile='_logs/_logs_custom_experiment1.txt', save_name_model='convnet/custom_experiment1.pt',
               metadata_file='_Data_metadata_multigrams_customProcess.pk', encoding_fn=multigrams_encoding,
               slice_fn=process_raw_signal, scorer=Data.compute_scores, batch_size=32, lr=1e-5):
    convnet_config = {'emb_dim': 120, 'kernel': 3, 'stride': 2, 'enc_dropout': 0.25, 'dec_d_ff': 768, 'dec_dropout': 0.25,
                      'enc_hid_dim': 256, 'enc_n_layers': 10, 'dec_n_layers': 10, 'dec_n_heads': 8, 'dec_d_model': 512}
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, encoding_fn=encoding_fn,
                     slice_fn=slice_fn, scorer=scorer, batch_size=batch_size, convnet_config=convnet_config, lr=lr)
    self.criterion = u.CrossEntropyLoss(self.pad_idx)
  
  def instanciate_model(self, **kwargs):
    return Recognizer(input_size=kwargs['enc_input_dim'], output_size=kwargs['output_size'],
                      enc_max_seq_len=kwargs['enc_max_seq_len'], dec_max_seq_len=kwargs['dec_max_seq_len']).to(self.device)
  
  def train(self):
    print('Start Training...')
    eval_accuracy_memory = 0
    for epoch in tqdm(range(self.n_epochs)):
      epoch_loss, accs = self.train_pass(only_loss=False if epoch % self.eval_step == 0 else True)
      logging.info(f"Epoch {epoch} | train_loss = {epoch_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")
      eval_loss, accs = self.evaluation(only_loss=False if epoch % self.eval_step == 0 else True)
      logging.info(f"Epoch {epoch} | test_loss = {eval_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")

      oea = accs['preds_acc'] if 'preds_acc' in accs else accs.get('word_accuracy', None)

      if oea is not None and oea > eval_accuracy_memory:
        logging.info(f'Save model with eval_accuracy = {oea:.3f}')
        u.save_checkpoint(self.model, None, self.save_name_model)
        eval_accuracy_memory = oea
  
  @torch.no_grad()
  def evaluation(self, only_loss=True):
    losses, accs = 0, {}
    targets, predictions = [], []

    self.model.eval()

    for enc_in, dec_in in tqdm(self.test_data_loader):
      enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
      preds = self.model(enc_in, dec_in[:, :-1])

      losses += self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1), epsilon=self.smoothing_eps).item()
      
      if not only_loss:
        preds = self.model.greedy_decoding(enc_in, self.sos_idx, self.eos_idx, max_seq_len=dec_in.shape[1])
        targets += dec_in[:, 1:].tolist()
        predictions += preds.tolist()
    
    self.model.train()

    if not only_loss:
      accs = self.scorer(**{'targets': targets, 'predictions': predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.data.idx_to_tokens})

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
  

if __name__ == '__main__':
  ## SEEDING FOR REPRODUCIBILITY
  SEED = 42
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)

  ce1 = CustomExperiment1()
  ce1.train()