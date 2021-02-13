## signal -> Encoder -> classifier for speaker recognizer
##                   -> Decoder -> text recognition task
##                   -> count number of word
##                   -> word classification
##                   -> slice audio signal into n chunks (n = number of words)
##                   -> retrieve start and end position of words

## Experiments :
## on raw data window sliced | use pronouncing library for phonemes

## signal = [signal_len,]
## window-sliced signal = [n_frames, n_feats]
## std-threshold-selected window-sliced signal = [new_n_frames, n_feats]
## std-threshold-chunked window-sliced signal = [n_chunks, new_n_feats]

## 1. Encoder-Decoder speech-to-text, letters prediction, cross-entropy loss
## 2. Encoder-Decoder speech-to-text, letters prediction, Attention-Cross-entropy loss
## 3. Encoder-Decord speech-to-text, letters prediction, Attention-Cross-entropy loss, Audio-signal reduction with std-binary-mask

## 4. Encoder-Decoder speech-to-text, phonemes predictions then letters prediction, CE|ACE loss, signal reduction SBM
## 5. Encoder-Decoder speech-to-text, syllables predictions then letters predictio, CE|ACE loss, signal reduction SBM
## 6. Encoder-Decoder speech-to-text, syllables predictions then phonemes predictions then letters predictions

## 7. Encoder + MLP, speaker recognition
## 8. AutoEncoder, std-threshold-chunked window-sliced signal reconstruction task -> Sparse-Coding

## 9. speech-to-text task with sparse-code signal

## 10. Multi-Tasks -> Encoder + MLP for speaker recognition
##                            + Decoder for speech-to-text, letters predictions
##                            + Decoder for speech-to-text, phonemes predictions
##                            + Decoder for speech-to-text, syllables predictions

## 11. Language-Model like training on audio signals (ELECTRA?)
## 12. speech-to-text, increase number of training sample, train speaker per speaker

## sentence -> words -> phonemes -> syllables
## from g2p_en import G2p; import re
## g2p = G2p()
## sources = 'that had its source away back in the woods'
## phonemes = g2p(sources)  # ['DH', 'AE1', 'T', ' ', 'HH', 'AE1', 'D', ' ', 'IH1', 'T', 'S', ' ', 'S', 'AO1', 'R', 'S', ' ', 'AH0',
##                             'W', 'EY1', ' ', 'B', 'AE1', 'K', ' ', 'IH0', 'N', ' ', 'DH', 'AH0', ' ', 'W', 'UH1', 'D', 'Z']
## pgd = [0] + [i+1 for i, p in enumerate(phonemes) if p == ' '] + [len(phonemes)]  # phonemes_groups_delimiters
## pho_groups = [phonemes[pgd[i]:pgd[i+1]-1] for i in range(len(pgd) - 1)]
## syllables = [p for p in phonemes if re.search(r'\d+', p) is not None or p == ' ']
## # ['AE1', ' ', 'AE1', ' ', 'IH1', ' ', 'AO1', ' ', 'AH0', 'EY1', ' ', 'AE1', ' ', 'IH0', ' ', 'AH0', ' ', 'UH1']

## Wave signal Encoder -> Phonemes Decoder -> Letters Decoder -> Words Decoder
## Use image of texts
## Predictive Coding

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

sys.path.append(os.path.abspath(__file__).replace('ASR/multitasks_experiment.py', ''))
import utils as u
import models.conv_seqseq as css

from data import Data
from custom_exp import TextEmbedder
from convnet_trainer import ConvnetTrainer
from models.transformer.decoder import Decoder
from ngrams_experiments import multigrams_encoding
from optimizer import CosineAnnealingWarmUpRestarts
from models.transformer.transformer import Transformer
from models.transformer.embedder import PositionalEmbedder
from models.divers_models import Seq2Seq, Seq2SeqReview, TextEmbedder, ConvLayer


class RawEmbedder(nn.Module):
  '''
  Fourier Transform is usualy used for spectral features extraction, as it's a linear transformation we use a FF to simulates it
  then some Convolution to simulates various filters and maybe a log(relu()+eps), not forgetting positional encoding
  '''
  def __init__(self, input_dim, emb_dim, hid_dim, max_seq_len, dropout, device, reduce_dim=False,
               n_filters=80, hop_length=512, pooling=2, seq_kernel=3):
    super().__init__()
    self.device = device
    moving_len = hop_length // 2

    self.fourier = nn.Linear(input_dim, input_dim)
    self.filtering = nn.Conv2d(1, n_filters, (seq_kernel, hop_length), padding=(seq_kernel//2, moving_len), stride=(1, moving_len))
    self.non_linearity = nn.ReLU(inplace=True)
    self.low_pass_filter = nn.MaxPool1d(pooling)

    out_size = u.compute_out_conv(input_dim, kernel=hop_length, stride=moving_len, padding=moving_len) * n_filters // pooling
    self.projection = nn.Linear(out_size, emb_dim)
    self.positional_embedding = u.create_positional_embedding(max_seq_len, emb_dim, hid_dim)
  
  def forward(self, x):
    batch_size, seq_len, _ = x.shape

    out = self.fourier(x)
    out = self.non_linearity(self.filtering(out.unsqueeze(1)))
    out = self.low_pass_filter(out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1))
    out = self.projection(u.layer_normalization(out))

    index_x = torch.LongTensor(range(seq_len)).unsqueeze(0).repeat(batch_size, 1).to(self.device)
    pos_emb = self.positional_embedding(index_x).float()

    return out + pos_emb


# STATUS = FAILURE
class Experiment16(ConvnetTrainer):
  '''Convnet, letters prediction, adam, Attention-CrossEntropy, MultiHead, window-raw-slice with win=0.128, RawEmbedder'''
  def __init__(self, logfile='_logs/_logs_experiment16.txt', save_name_model='convnet/convnet_experiment16.pt', batch_size=8,
               metadata_file='_Data_metadata_letters_raw0128.pk', slice_fn=Data.overlapping_window_slicing_signal, multi_head=True):
    convnet_config = {'encoder_embedder': RawEmbedder}
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, slice_fn=slice_fn,
                     convnet_config=convnet_config, multi_head=multi_head, batch_size=batch_size)


# STATUS = FAILURE
class Experiment18(ConvnetTrainer):
  '''Convnet, letters prediction, adam, Attention-CrossEntropy, MultiHead, window-raw-slice with win=0.025, overlap 0.1, RawEmbedder'''
  def __init__(self, logfile='_logs/_logs_experiment18.txt', save_name_model='convnet/convnet_experiment18.pt', batch_size=8,
               metadata_file='_Data_metadata_letters_raw0025_001.pk', slice_fn=Data.overlapping_window_slicing_signal, multi_head=True,
               window_size=0.025, overlap_size=0.01):
    convnet_config = {'encoder_embedder': RawEmbedder}
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, slice_fn=slice_fn,
                     batch_size=batch_size, convnet_config=convnet_config, multi_head=multi_head, window_size=window_size,
                     overlap_size=overlap_size)


class Experiment19(ConvnetTrainer):
  '''Convnet, letters prediction, adam, CrossEntropy, MultiHead, window-raw-slice with win=0.025, no-overlap, RawEmbedder'''
  def __init__(self, logfile='_logs/_logs_experiment19.txt', save_name_model='convnet/convnet_experiment19.pt', batch_size=8,
               metadata_file='_Data_metadata_letters_raw0025.pk', multi_head=True, decay_factor=0):
    convnet_config = {'encoder_embedder': RawEmbedder}
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, decay_factor=decay_factor,
                     batch_size=batch_size, convnet_config=convnet_config, multi_head=multi_head)


class RawEmbedder2(nn.Module):
  def __init__(self, input_dim, emb_dim, hid_dim, max_seq_len, dropout, device, reduce_dim=False,
               n_filters=80, window=2048, hop_length=512, pooling=2):
    super().__init__()
    self.device = device

    self.conv = nn.Conv1d(1, n_filters, window, stride=hop_length)
    self.non_linearity = nn.ReLU(inplace=True)

    self.projection = nn.Linear(n_filters, emb_dim)
    self.normalization = nn.LayerNorm(emb_dim)
    self.positional_embedding = u.create_positional_embedding(max_seq_len, emb_dim, hid_dim)
  
  def forward(self, x):
    out = self.non_linearity(self.conv(x.unsqueeze(1)))
    out = self.normalization(self.projection(out.permute(0, 2, 1)))

    batch_size, seq_len, _ = out.shape
    index_x = torch.LongTensor(range(seq_len)).unsqueeze(0).repeat(batch_size, 1).to(self.device)
    pos_emb = self.positional_embedding(index_x).float()

    return out + pos_emb


## STATUS = stopped, epoch 160, seems promising but slower convergence than experiment21
class Experiment20(ConvnetTrainer):
  '''Convnet, letters prediction, adam, Attention-CrossEntropy, MultiHead, no-slicing, RawEmbedder2'''
  def __init__(self, logfile='_logs/_logs_experiment20.txt', save_name_model='convnet/convnet_experiment20.pt', batch_size=8,
               metadata_file='_Data_metadata_letters_raw0025.pk', multi_head=True, slice_fn=lambda x: x):
    convnet_config = {'encoder_embedder': RawEmbedder2}
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file,
                     batch_size=batch_size, convnet_config=convnet_config, multi_head=multi_head, slice_fn=slice_fn)


class ScateringFilter(nn.Module):
  def __init__(self, n_feats=80, kernel=400, stride=160, norm_pooling=2, kernel_pooling=2, non_linearity='l2_pooling'):
    super().__init__()
    self.n_feats = n_feats
    self.conv = nn.Conv1d(1, n_feats, kernel)

    if non_linearity == 'relu':
      self.non_linearity = nn.ReLU(inplace=True)
    else:
      self.non_linearity = nn.LPPool1d(norm_pooling, kernel_pooling)

    self.low_pass_filter = nn.Conv1d(n_feats, n_feats, kernel, stride=stride)
  
  def forward(self, x):  # x = [batch_size, signal_len]
    out = self.conv(x.unsqueeze(1))  # [batch_size, n_feats, new_signal_len]
    out = self.non_linearity(out)  # [batch_size, n_feats, new_signal_len/kernel_pooling]
    out = self.low_pass_filter(out)  # [batch_size, n_feats, new_len]
    return torch.log(1 + torch.abs(out))  # log-compression


class GammatonesFilter(nn.Module):
  def __init__(self, n_feats=40, kernel=400, kernel_pooling=400, stride=160, low_pass_filter='max_pool'):
    super().__init__()
    self.n_feats = n_feats
    self.conv = nn.Conv1d(1, n_feats, kernel)
    self.non_linearity = nn.ReLU(inplace=True)

    if low_pass_filter == 'sq_hanning':
      self.low_pass_filter = nn.Conv1d(n_feats, n_feats, kernel, stride=stride)
    else:
      self.low_pass_filter = nn.MaxPool1d(kernel_pooling)
  
  def forward(self, x):  # x = [batch_size, signal_len]
    out = self.non_linearity(self.conv(x.unsqueeze(1)))  # [batch_size, n_feats, new_signal_len] 
    out = self.low_pass_filter(out)  # [batch_size, n_feats, new_signal_len/kernel_pooling]
    return torch.log(0.01 + torch.abs(out))  # log-compression


class MixFilters(nn.Module):
  def __init__(self, n_feats_scatering=80, n_feats_gammatones=40, kernel=4, stride=2):
    super().__init__()
    self.scatering = ScateringFilter(n_feats=n_feats_scatering, non_linearity='relu')
    self.gammatones = GammatonesFilter(n_feats=n_feats_gammatones, low_pass_filter='sq_hanning')
    self.n_feats = n_feats_scatering + n_feats_gammatones 
    self.reducer = nn.Sequential(nn.Conv1d(self.n_feats, self.n_feats, kernel, stride=stride),
                                           nn.ReLU(),
                                           nn.Conv1d(self.n_feats, self.n_feats, kernel, stride=stride),
                                           nn.ReLU())
  
  def forward(self, x):  # x = [batch_size, signal_len]
    scatering_filters = self.scatering(x)
    gammatones_filters = self.gammatones(x)
    out = self.reducer(torch.cat((scatering_filters, gammatones_filters), axis=1))  # [batch_size, self.n_feats, seq_len]
    return out.permute(0, 2, 1)


class AudioEmbedder(nn.Module):
  def __init__(self, input_dim, emb_dim, hid_dim, max_seq_len, dropout, device, reduce_dim=False):
    super().__init__()
    self.device = device
    self.filters = MixFilters()

    self.projection = nn.Linear(self.filters.n_feats, emb_dim)
    self.normalization = nn.LayerNorm(emb_dim)
    self.positional_embedding = u.create_positional_embedding(max_seq_len, emb_dim, hid_dim)
  
  def forward(self, x):
    out = self.filters(x)
    out = self.normalization(self.projection(out))

    batch_size, seq_len, _ = out.shape
    index_x = torch.LongTensor(range(seq_len)).unsqueeze(0).repeat(batch_size, 1).to(self.device)
    pos_emb = self.positional_embedding(index_x).float()

    return out + pos_emb


## STATUS = stopped, best train_acc=0.978,test_acc=0.301 epoch 260, after test_acc decrease
class Experiment21(ConvnetTrainer):
  '''Convnet, letters prediction, adam, Attention-CrossEntropy, MultiHead, no-slicing, AudioEmbedder'''
  def __init__(self, logfile='_logs/_logs_experiment21.txt', save_name_model='convnet/convnet_experiment21.pt', batch_size=8,
               metadata_file='_Data_metadata_letters_mfcc0128.pk', multi_head=True, slice_fn=lambda x: x):
    convnet_config = {'encoder_embedder': AudioEmbedder, 'enc_layers': 4, 'dec_layers': 6}
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, batch_size=batch_size,
                     convnet_config=convnet_config, multi_head=multi_head, slice_fn=slice_fn)


class Experiment24(ConvnetTrainer):
  '''Convnet, letters prediction, adam, Attention-CrossEntropy, MultiHead, no-slicing, AudioEmbedder, scaling'''
  def __init__(self, logfile='_logs/_logs_experiment24.txt', save_name_model='convnet/convnet_experiment24.pt', batch_size=8,
               metadata_file='_Data_metadata_letters_mfcc0128.pk', multi_head=True, slice_fn=lambda x: x):
    convnet_config = {'encoder_embedder': AudioEmbedder, 'enc_layers': 4, 'dec_layers': 6}
    slice_fn = lambda x: (x - np.mean(x)) / (np.std(x) + 1e-15)
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, batch_size=batch_size,
                     convnet_config=convnet_config, multi_head=multi_head, slice_fn=slice_fn)


class TransformerExperiments(ConvnetTrainer):
  def __init__(self, logfile='_logs/_logs_experiment.txt', save_name_model='transformer/transformer_experiment.pt',
               d_keys_values=64, n_heads=4, d_model=256, d_ff=512, enc_layers=8, dec_layers=6, convnet_config={},
               dropout=0., reduce_dim=False, create_enc_mask=True, batch_size=32, **kwargs):
    default_config = {'d_keys': d_keys_values, 'd_values': d_keys_values, 'n_heads': n_heads, 'd_ff': d_ff, 'enc_layers': enc_layers,
                      'dec_layers': dec_layers, 'd_model': d_model, 'dropout': dropout, 'reduce_dim': False}
    convnet_config = {**default_config, **convnet_config}
    super().__init__(convnet_config=convnet_config, logfile=logfile, save_name_model=save_name_model, create_enc_mask=create_enc_mask,
                     batch_size=batch_size, **kwargs)
    self.criterion = u.CrossEntropyLoss(self.pad_idx)
    self.criterion.step = lambda x: x
  
  def instanciate_model(self, enc_max_seq_len=1400, enc_input_dim=400, d_model=256, dec_max_seq_len=600, d_ff=512, dropout=0.,
                        dec_input_dim=31, output_size=31, enc_layers=8, dec_layers=6, d_keys=64, d_values=64, n_heads=4,
                        reduce_dim=False, **kwargs):
    encoder_embedder = PositionalEmbedder(enc_max_seq_len, enc_input_dim, d_model, device=self.device, reduce_dim=reduce_dim)
    decoder_embedder = PositionalEmbedder(dec_max_seq_len, dec_input_dim, d_model, output_size=output_size, device=self.device)

    model = Transformer(enc_layers, dec_layers, d_model, d_keys, d_values, n_heads, d_ff, output_size,
                        encoder_embedder=encoder_embedder, decoder_embedder=decoder_embedder, dropout=dropout,
                        enc_max_seq_len=enc_max_seq_len, dec_max_seq_len=dec_max_seq_len, device=self.device,
                        encoder_reduce_dim=reduce_dim)
    return model.to(self.device)
  
  def train_pass(self, only_loss=True):
    losses, accs = 0, {}
    targets, predictions = [], []

    for enc_in, dec_in, pad_mask in tqdm(self.train_data_loader):
      enc_in, dec_in, pad_mask = enc_in.to(self.device), dec_in.to(self.device), pad_mask.to(self.device)
      preds = self.model(enc_in, dec_in[:, :-1], padding_mask=pad_mask)

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
  def evaluation(self, only_loss=True):
    losses, accs = 0, {}
    targets, predictions = [], []

    self.model.eval()

    for enc_in, dec_in, pad_mask in tqdm(self.test_data_loader):
      enc_in, dec_in, pad_mask = enc_in.to(self.device), dec_in.to(self.device), pad_mask.to(self.device)
      preds = self.model(enc_in, dec_in[:, :-1], padding_mask=pad_mask)

      losses += self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1), epsilon=self.smoothing_eps).item()
      
      if not only_loss:
        preds = self.model.greedy_decoding(enc_in, self.eos_idx, self.pad_idx, max_seq_len=dec_in.shape[1])
        targets += dec_in[:, 1:].tolist()
        predictions += preds.tolist()
    
    self.model.train()

    if not only_loss:
      accs = self.scorer(**{'targets': targets, 'predictions': predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.data.idx_to_tokens})

    return losses / len(self.test_data_loader), accs


# STATUS = FAILURE
class Experiment25(TransformerExperiments):
  '''Transformer letters prediction, adam, CrossEntropy loss, mfcc, n_fft=2048, hop_length=512'''
  def __init__(self, logfile='_logs/_logs_experiment25.txt', save_name_model='transformer/transformer_experiment25.pt',
               slice_fn=Data.mfcc_extraction, n_fft=2048, hop_length=512, scorer=Data.compute_scores,  batch_size=32,
               metadata_file='_Data_metadata_letters_mfcc0128.pk'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, slice_fn=slice_fn, batch_size=batch_size, n_fft=n_fft,
                     hop_length=hop_length, scorer=scorer, metadata_file=metadata_file)
    u.load_model(self.model, save_name_model, restore_only_similars=True)


class ConvnetFeedbackExperiments(ConvnetTrainer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
  
  def instanciate_model(self, enc_input_dim=400, enc_max_seq_len=1400, dec_input_dim=31, dec_max_seq_len=600, output_size=31,
                        enc_layers=10, dec_layers=10, enc_kernel_size=3, dec_kernel_size=3, enc_dropout=0.25, dec_dropout=0.25,
                        emb_dim=256, hid_dim=512, reduce_dim=False, pad_idx=2, score_fn=torch.softmax, multi_head=False, d_keys_values=64,
                        encoder_embedder=css.EncoderEmbedder, decoder_embedder=css.DecoderEmbedder):
    enc_embedder = encoder_embedder(enc_input_dim, emb_dim, hid_dim, enc_max_seq_len, enc_dropout, reduce_dim=reduce_dim)
    dec_embedder = decoder_embedder(dec_input_dim, emb_dim, dec_max_seq_len, dec_dropout)

    if self.relu:
      enc = css.EncoderRelu(emb_dim, hid_dim, enc_layers, enc_kernel_size, enc_dropout, embedder=enc_embedder)
      dec = css.DecoderRelu(output_size, emb_dim, hid_dim, dec_layers, dec_kernel_size, dec_dropout, pad_idx,
                            embedder=dec_embedder, score_fn=score_fn, multi_head=multi_head, d_keys_values=d_keys_values)
    else:
      enc = css.Encoder(emb_dim, hid_dim, enc_layers, enc_kernel_size, enc_dropout, embedder=enc_embedder)
      dec = css.Decoder(output_size, emb_dim, hid_dim, dec_layers, dec_kernel_size, dec_dropout, pad_idx,
                        embedder=dec_embedder, score_fn=score_fn, multi_head=multi_head, d_keys_values=d_keys_values)
    
    p_enc_embedder = encoder_embedder(dec_input_dim, emb_dim, hid_dim, dec_max_seq_len, enc_dropout)
    p_encoder = css.Encoder(emb_dim, hid_dim, 2, enc_kernel_size, enc_dropout, embedder=p_enc_embedder)
    p_decoder = css.DecoderFeedback(output_size, emb_dim, hid_dim, 2, dec_kernel_size, dec_dropout, pad_idx,
                                    embedder=dec_embedder, score_fn=score_fn, multi_head=multi_head, d_keys_values=d_keys_values)

    return css.Seq2SeqFeedback(enc, dec, p_encoder, p_decoder).to(self.device)

  def train_pass(self, only_loss=True):
    losses, accs = 0, {}
    targets, predictions = [], []

    for enc_in, dec_in in tqdm(self.train_data_loader):
      enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
      preds, att, f_preds, f_att = self.model(enc_in, dec_in[:, :-1])

      self.optimizer.zero_grad()

      current_loss = self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1), att, epsilon=self.smoothing_eps)
      current_loss += self.criterion(f_preds.reshape(-1, f_preds.shape[-1]), dec_in[:, 1:].reshape(-1), f_att, epsilon=self.smoothing_eps)

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
  def evaluation(self, only_loss=True):
    losses, accs = 0, {}
    targets, predictions = [], []

    self.model.eval()

    for enc_in, dec_in in tqdm(self.test_data_loader):
      enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
      preds, att, f_preds, f_att = self.model(enc_in, dec_in[:, :-1])

      losses += self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1), att, epsilon=self.smoothing_eps).item()
      losses += self.criterion(f_preds.reshape(-1, f_preds.shape[-1]), dec_in[:, 1:].reshape(-1), f_att, epsilon=self.smoothing_eps).item()
      
      if not only_loss:
        preds, _ = self.model.greedy_decoding(enc_in, self.sos_idx, self.eos_idx, max_seq_len=dec_in.shape[1])
        targets += dec_in[:, 1:].tolist()
        predictions += preds.tolist()
    
    self.model.train()

    if not only_loss:
      accs = self.scorer(**{'targets': targets, 'predictions': predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.data.idx_to_tokens})

    return losses / len(self.test_data_loader), accs


# STATUS = FAILURE
class Experiment26(ConvnetFeedbackExperiments):
  '''Convnet letters prediction, adam, Attention-CrossEntropy loss, mfcc, n_fft=2048, hop_length=512, MultiHead'''
  def __init__(self, logfile='_logs/_logs_experiment26.txt', save_name_model='convnet/convnet_experiment26.pt', batch_size=8,
               slice_fn=Data.mfcc_extraction, n_fft=2048, hop_length=512, scorer=Data.compute_scores, multi_head=True,
               metadata_file='_Data_metadata_letters_mfcc0128.pk', decay_factor=0):
    super().__init__(logfile=logfile, save_name_model=save_name_model, slice_fn=slice_fn, batch_size=batch_size, n_fft=n_fft,
                     hop_length=hop_length, scorer=scorer, multi_head=multi_head, metadata_file=metadata_file, decay_factor=decay_factor)
    u.load_model(self.model, save_name_model, restore_only_similars=True)


class SeqSeqConvnetTransformer(nn.Module):
  def __init__(self, encoder, decoder, device):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.device = device
  
  def forward(self, enc_in, dec_in, padding_mask=None):
    _, encoder_combined = self.encoder(enc_in)
    output = self.decoder(dec_in, encoder_combined, padding_mask=padding_mask)
    return output
  
  def greedy_decoding(self, enc_in, sos_idx, eos_idx, max_seq_len=100):
    _, encoder_combined = self.encoder(enc_in)

    batch_size = enc_in.shape[0]
    finished = [False] * batch_size
    dec_in = torch.LongTensor(batch_size, 1).fill_(sos_idx).to(self.device)
    
    for _ in range(max_seq_len):
      output = self.decoder(dec_in, encoder_combined, save=True, aggregate=True)
      pred = output[:, -1, :].argmax(-1).unsqueeze(1)

      for idx in range(batch_size):
        if not finished[idx] and pred[idx].item() == eos_idx:
          finished[idx] = True

      dec_in = torch.cat((dec_in, pred), dim=1)

      if all(finished):
        break
    
    return dec_in[:, 1:]


class ConvnetTransformerExperiments(TransformerExperiments):
  def __init__(self, batch_size=32, **kwargs):
    convnet_config = {'emb_dim': 256, 'hid_dim': 512, 'enc_dropout': 0.25, 'enc_layers': 10, 'd_model': 256, 'dec_layers': 10,
                      'd_keys_values': 64, 'n_heads': 4, 'd_ff': 512, 'dec_dropout': 0.25, 'scaling': True, 'dec_emb_dim': 80}
    super().__init__(convnet_config=convnet_config, batch_size=batch_size, **kwargs)
  
  def instanciate_model(self, enc_input_dim=400, enc_max_seq_len=1400, emb_dim=256, hid_dim=512, enc_dropout=0.25, enc_reduce_dim=False,
                        enc_layers=10, enc_kernel_size=3, dec_max_seq_len=600, dec_input_dim=31, d_model=256, output_size=31,
                        dec_layers=10, d_keys_values=64, n_heads=4, d_ff=512, dec_dropout=0.25, dec_reduce_dim=False, scaling=True,
                        encoder_embedder=css.EncoderEmbedder, dec_emb_dim=100, **kwargs):
    enc_embedder = encoder_embedder(enc_input_dim, emb_dim, hid_dim, enc_max_seq_len, enc_dropout, reduce_dim=enc_reduce_dim)
    enc = css.Encoder(emb_dim, hid_dim, enc_layers, enc_kernel_size, enc_dropout, embedder=enc_embedder)

    decoder_embedder = PositionalEmbedder(dec_max_seq_len, dec_emb_dim, d_model, output_size=output_size, device=self.device,
                                          dropout=dec_dropout)
    dec = Decoder(n_blocks=dec_layers, d_model=d_model, d_keys=d_keys_values, d_values=d_keys_values, n_heads=n_heads, d_ff=d_ff,
                  output_size=output_size, dropout=dec_dropout, device=self.device, embedder=decoder_embedder, reduce_dim=dec_reduce_dim,
                  max_seq_len=dec_max_seq_len, emb_dim=dec_emb_dim, scaling=scaling)
    
    return SeqSeqConvnetTransformer(enc, dec, self.device).to(self.device)


class Experiment27(ConvnetTransformerExperiments):
  '''Convnet letters prediction, adam, Attention-CrossEntropy loss, mfcc, n_fft=2048, hop_length=512, MultiHead'''
  def __init__(self, logfile='_logs/_logs_experiment27.txt', save_name_model='convnet/convnetTransformer_experiment27.pt',
               slice_fn=Data.mfcc_extraction, n_fft=2048, hop_length=512, scorer=Data.compute_scores, batch_size=32,
               metadata_file='_Data_metadata_letters_mfcc0128.pk'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, slice_fn=slice_fn, batch_size=batch_size, n_fft=n_fft,
                     hop_length=hop_length, scorer=scorer, metadata_file=metadata_file)
    init_transformer2(self.model, 256, 31)
    # u.load_model(self.model, 'convnet/convnet_experiment12_0475.pt', restore_only_similars=True)


def init_transformer1(model, n_layers, alpha=1):
  # https://www.aclweb.org/anthology/D19-1083.pdf
  for k, p in model.named_parameters():
    if p.dim() > 1 and 'decoder' in k:
      gamma = (6 / sum(p.shape)) ** 0.5
      val = gamma * alpha / (n_layers ** 0.5)
      nn.init.uniform_(p, a=-val, b=val)


def init_transformer2(model, emb_dim, vocab_size):
  # https://arxiv.org/pdf/1911.03179.pdf
  for k, p in model.named_parameters():
    if p.dim() > 1 and 'decoder' in k:
      if isinstance(p, nn.Linear):
        l = (1 / p.shape[-1]) ** 0.5
        nn.init.uniform_(p, a=-l, b=l)
      elif isinstance(p, nn.Embedding):
        e = (2 / (emb_dim + vocab_size)) ** 0.5
        nn.init.uniform_(p, a=-e, b=e)


class STTTrainer(object):
  def __init__(self, device=None, metadata_file='_Data_metadata_multigrams_wav2vec.pk', logfile='_logs/_logs_sttTrainer.txt',
               train_folder='../../../datasets/openslr/LibriSpeech/train-clean-100/', lr=1e-4, batch_size=8, scores_step=5,
               test_folder='../../../datasets/openslr/LibriSpeech/test-clean/', scorer=Data.compute_scores, eval_step=10,
               save_name_model='convnet/stt_trainer.pt', smoothing_eps=0.1, n_epochs=500, load_model=True,
               encoding_fn=multigrams_encoding, block_type='self_attn', lr_scheduling=False, config={}):
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

    self.config = {'emb_dim': 50, 'd_model': 256, 'n_heads': 4, 'd_ff': 512, 'kernel_size': 3, 'enc_n_blocks': 6,
                   'dec_n_blocks': 6, 'dropout': 0., 'enc_block_type': 'self_attn', 'dec_block_type': 'self_attn'}
    self.config = {**self.config, **config}
    self.model = self.instanciate_model(**self.config)

    u.dump_dict(self.config, 'STTTrainer PARAMETERS')
    logging.info(self.model)
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

  def instanciate_model(self, emb_dim=50, d_model=256, n_heads=4, d_ff=512, kernel_size=3, enc_n_blocks=6, dec_n_blocks=6, dropout=0.,
                        enc_block_type='self_attn', dec_block_type='self_attn', decoder_layer='conv_layer', **kwargs):
    self.output_dim = len(self.data.idx_to_tokens)
    self.n_step_aheads = 1
    return Seq2Seq(self.output_dim, self.data.n_signal_feats, emb_dim, d_model, n_heads, d_ff, d_ff, kernel_size, enc_n_blocks,
                   dec_n_blocks, self.data.max_signal_len, self.data.max_source_len, dropout=dropout, n_step_aheads=self.n_step_aheads,
                   enc_block_type=enc_block_type, dec_block_type=dec_block_type, decoder_layer=decoder_layer).to(self.device)

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
                   n_blocks, self.data.max_signal_len, self.data.max_source_len, dropout=0., n_step_aheads=self.n_step_aheads,
                   enc_block_type=self.block_type).to(self.device)


class STTTrainer4(STTTrainer):
  def __init__(self, logfile='_logs/_logs_sttTrainer4.txt'):
    super().__init__(logfile=logfile, save_name_model='convnet/stt_trainer4.pt', encoding_fn=Data.letters_encoding,
                     metadata_file='_Data_metadata_letters_wav2vec.pk', block_type='dilated', batch_size=32)
  
  def instanciate_model(self, emb_dim=50, d_model=512, n_heads=8, enc_d_ff=768, dec_d_ff=768, kernel_size=3, enc_n_blocks=6,
                        dec_n_blocks=6):
    self.output_dim = len(self.data.idx_to_tokens)
    self.n_step_aheads = 1
    return Seq2Seq(self.output_dim, self.data.n_signal_feats, emb_dim, d_model, n_heads, enc_d_ff, dec_d_ff, kernel_size, enc_n_blocks,
                   dec_n_blocks, self.data.max_signal_len, self.data.max_source_len, dropout=0., n_step_aheads=self.n_step_aheads,
                   enc_block_type=self.block_type, dec_block_type=self.block_type, decoder_layer='decoding_conv_layer').to(self.device)


class STTTrainer5(STTTrainer):
  def __init__(self, logfile='_logs/_logs_sttTrainer5.txt'):
    super().__init__(logfile=logfile, save_name_model='convnet/stt_trainer5.pt', encoding_fn=Data.letters_encoding, lr=1e-6,
                     metadata_file='_Data_metadata_letters_wav2vec.pk', block_type='dilated', batch_size=32, lr_scheduling=True)
  
  def instanciate_model(self, emb_dim=50, d_model=512, n_heads=8, enc_d_ff=768, dec_d_ff=768, kernel_size=3, enc_n_blocks=6,
                        dec_n_blocks=6):
    self.output_dim = len(self.data.idx_to_tokens)
    self.n_step_aheads = 1
    return Seq2Seq(self.output_dim, self.data.n_signal_feats, emb_dim, d_model, n_heads, enc_d_ff, dec_d_ff, kernel_size, enc_n_blocks,
                   dec_n_blocks, self.data.max_signal_len, self.data.max_source_len, dropout=0., n_step_aheads=self.n_step_aheads,
                   enc_block_type=self.block_type, dec_block_type=self.block_type).to(self.device)


class STTTrainer6(STTTrainer):
  def __init__(self, logfile='_logs/_logs_sttTrainer6.txt'):
    config = {'emb_dim': 50, 'd_model': 512, 'n_heads': 8, 'enc_d_ff': 768, 'dec_d_ff': 768, 'kernel_size': 3,
              'enc_n_blocks': 6, 'dec_n_blocks': 6, 'dropout': 0.25}
    super().__init__(logfile=logfile, save_name_model='convnet/stt_trainer6.pt', encoding_fn=Data.letters_encoding,
                     metadata_file='_Data_metadata_letters_wav2vec.pk', block_type='dilated', batch_size=32, config=config)
  
  def instanciate_model(self, emb_dim=50, d_model=512, n_heads=8, enc_d_ff=768, dec_d_ff=768, kernel_size=3, enc_n_blocks=6,
                        dec_n_blocks=6, dropout=0.25, **kwargs):
    self.output_dim = len(self.data.idx_to_tokens)
    self.n_step_aheads = 1
    return Seq2Seq(self.output_dim, self.data.n_signal_feats, emb_dim, d_model, n_heads, enc_d_ff, dec_d_ff, kernel_size, enc_n_blocks,
                   dec_n_blocks, self.data.max_signal_len, self.data.max_source_len, dropout=dropout, n_step_aheads=self.n_step_aheads,
                   enc_block_type=self.block_type, dec_block_type=self.block_type, decoder_layer='decoding_conv_layer').to(self.device)


class STTTrainer7(STTTrainer):
  def __init__(self, logfile='_logs/_logs_sttTrainer7.txt'):
    super().__init__(logfile=logfile, save_name_model='convnet/stt_trainer7.pt', encoding_fn=Data.letters_encoding,
                     metadata_file='_Data_metadata_letters_wav2vec.pk', config={'dropout': 0.25})


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


if __name__ == "__main__":
  ## SEEDING FOR REPRODUCIBILITY
  SEED = 42
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)
  
  rep = input('Start Experiment experiment? (y or n): ')
  if rep == 'y':
    experiments = {k.replace('Experiment', ''): v for k, v in locals().items() if re.search(r'Experiment\d+', k) is not None}
    rep = input(f'Which Experiment do you want to start? ({",".join(experiments.keys())}): ')
    exp = experiments[rep]()
    exp.train()

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