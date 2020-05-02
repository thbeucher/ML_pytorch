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

import os
import re
import sys
import json
import torch
import random
import logging
import numpy as np
import torch.optim as optim

from tqdm import tqdm

sys.path.append(os.path.abspath(__file__).replace('ASR/multitasks_experiment.py', ''))
import utils as u
import models.conv_seqseq as css

from optimizer import RAdam
from models.transformer.decoder import Decoder
from models.transformer.transformer import Transformer
from models.transformer.embedder import PositionalEmbedder


class RawEmbedder(torch.nn.Module):
  '''
  Fourier Transform is usualy used for spectral features extraction, as it's a linear transformation we use a FF to simulates it
  then some Convolution to simulates various filters and maybe a log(relu()+eps), not forgetting positional encoding
  '''
  def __init__(self, input_dim, emb_dim, hid_dim, max_seq_len, dropout, device, reduce_dim=False,
               n_filters=80, hop_length=512, pooling=2, seq_kernel=3):
    super().__init__()
    self.device = device
    moving_len = hop_length // 2

    self.fourier = torch.nn.Linear(input_dim, input_dim)
    self.filtering = torch.nn.Conv2d(1, n_filters, (seq_kernel, hop_length), padding=(seq_kernel//2, moving_len), stride=(1, moving_len))
    self.non_linearity = torch.nn.ReLU(inplace=True)
    self.low_pass_filter = torch.nn.MaxPool1d(pooling)

    out_size = u.compute_out_conv(input_dim, kernel=hop_length, stride=moving_len, padding=moving_len) * n_filters // pooling
    self.projection = torch.nn.Linear(out_size, emb_dim)
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
class Experiment16(ConvnetExperiments):
  '''Convnet, letters prediction, adam, Attention-CrossEntropy, MultiHead, window-raw-slice with win=0.128, RawEmbedder'''
  def __init__(self, logfile='_logs/_logs_experiment16.txt', save_name_model='convnet/convnet_experiment16.pt', batch_size=8,
               metadata_file='_Data_metadata_letters_raw0128.pk', slice_fn=Data.overlapping_window_slicing_signal, multi_head=True):
    convnet_config = {'encoder_embedder': RawEmbedder}
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, slice_fn=slice_fn,
                     convnet_config=convnet_config, multi_head=multi_head, batch_size=batch_size)


# STATUS = FAILURE
class Experiment18(ConvnetExperiments):
  '''Convnet, letters prediction, adam, Attention-CrossEntropy, MultiHead, window-raw-slice with win=0.025, overlap 0.1, RawEmbedder'''
  def __init__(self, logfile='_logs/_logs_experiment18.txt', save_name_model='convnet/convnet_experiment18.pt', batch_size=8,
               metadata_file='_Data_metadata_letters_raw0025_001.pk', slice_fn=Data.overlapping_window_slicing_signal, multi_head=True,
               window_size=0.025, overlap_size=0.01):
    convnet_config = {'encoder_embedder': RawEmbedder}
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, slice_fn=slice_fn,
                     batch_size=batch_size, convnet_config=convnet_config, multi_head=multi_head, window_size=window_size,
                     overlap_size=overlap_size)


class Experiment19(ConvnetExperiments):
  '''Convnet, letters prediction, adam, CrossEntropy, MultiHead, window-raw-slice with win=0.025, no-overlap, RawEmbedder'''
  def __init__(self, logfile='_logs/_logs_experiment19.txt', save_name_model='convnet/convnet_experiment19.pt', batch_size=8,
               metadata_file='_Data_metadata_letters_raw0025.pk', multi_head=True, decay_factor=0):
    convnet_config = {'encoder_embedder': RawEmbedder}
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, decay_factor=decay_factor,
                     batch_size=batch_size, convnet_config=convnet_config, multi_head=multi_head)


class RawEmbedder2(torch.nn.Module):
  def __init__(self, input_dim, emb_dim, hid_dim, max_seq_len, dropout, device, reduce_dim=False,
               n_filters=80, window=2048, hop_length=512, pooling=2):
    super().__init__()
    self.device = device

    self.conv = torch.nn.Conv1d(1, n_filters, window, stride=hop_length)
    self.non_linearity = torch.nn.ReLU(inplace=True)

    self.projection = torch.nn.Linear(n_filters, emb_dim)
    self.normalization = torch.nn.LayerNorm(emb_dim)
    self.positional_embedding = u.create_positional_embedding(max_seq_len, emb_dim, hid_dim)
  
  def forward(self, x):
    out = self.non_linearity(self.conv(x.unsqueeze(1)))
    out = self.normalization(self.projection(out.permute(0, 2, 1)))

    batch_size, seq_len, _ = out.shape
    index_x = torch.LongTensor(range(seq_len)).unsqueeze(0).repeat(batch_size, 1).to(self.device)
    pos_emb = self.positional_embedding(index_x).float()

    return out + pos_emb


## STATUS = stopped, epoch 160, seems promising but slower convergence than experiment21
class Experiment20(ConvnetExperiments):
  '''Convnet, letters prediction, adam, Attention-CrossEntropy, MultiHead, no-slicing, RawEmbedder2'''
  def __init__(self, logfile='_logs/_logs_experiment20.txt', save_name_model='convnet/convnet_experiment20.pt', batch_size=8,
               metadata_file='_Data_metadata_letters_raw0025.pk', multi_head=True, slice_fn=lambda x: x):
    convnet_config = {'encoder_embedder': RawEmbedder2}
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file,
                     batch_size=batch_size, convnet_config=convnet_config, multi_head=multi_head, slice_fn=slice_fn)


class ScateringFilter(torch.nn.Module):
  def __init__(self, n_feats=80, kernel=400, stride=160, norm_pooling=2, kernel_pooling=2, non_linearity='l2_pooling'):
    super().__init__()
    self.n_feats = n_feats
    self.conv = torch.nn.Conv1d(1, n_feats, kernel)

    if non_linearity == 'relu':
      self.non_linearity = torch.nn.ReLU(inplace=True)
    else:
      self.non_linearity = torch.nn.LPPool1d(norm_pooling, kernel_pooling)

    self.low_pass_filter = torch.nn.Conv1d(n_feats, n_feats, kernel, stride=stride)
  
  def forward(self, x):  # x = [batch_size, signal_len]
    out = self.conv(x.unsqueeze(1))  # [batch_size, n_feats, new_signal_len]
    out = self.non_linearity(out)  # [batch_size, n_feats, new_signal_len/kernel_pooling]
    out = self.low_pass_filter(out)  # [batch_size, n_feats, new_len]
    return torch.log(1 + torch.abs(out))  # log-compression


class GammatonesFilter(torch.nn.Module):
  def __init__(self, n_feats=40, kernel=400, kernel_pooling=400, stride=160, low_pass_filter='max_pool'):
    super().__init__()
    self.n_feats = n_feats
    self.conv = torch.nn.Conv1d(1, n_feats, kernel)
    self.non_linearity = torch.nn.ReLU(inplace=True)

    if low_pass_filter == 'sq_hanning':
      self.low_pass_filter = torch.nn.Conv1d(n_feats, n_feats, kernel, stride=stride)
    else:
      self.low_pass_filter = torch.nn.MaxPool1d(kernel_pooling)
  
  def forward(self, x):  # x = [batch_size, signal_len]
    out = self.non_linearity(self.conv(x.unsqueeze(1)))  # [batch_size, n_feats, new_signal_len] 
    out = self.low_pass_filter(out)  # [batch_size, n_feats, new_signal_len/kernel_pooling]
    return torch.log(0.01 + torch.abs(out))  # log-compression


class MixFilters(torch.nn.Module):
  def __init__(self, n_feats_scatering=80, n_feats_gammatones=40, kernel=4, stride=2):
    super().__init__()
    self.scatering = ScateringFilter(n_feats=n_feats_scatering, non_linearity='relu')
    self.gammatones = GammatonesFilter(n_feats=n_feats_gammatones, low_pass_filter='sq_hanning')
    self.n_feats = n_feats_scatering + n_feats_gammatones 
    self.reducer = torch.nn.Sequential(torch.nn.Conv1d(self.n_feats, self.n_feats, kernel, stride=stride),
                                       torch.nn.ReLU(),
                                       torch.nn.Conv1d(self.n_feats, self.n_feats, kernel, stride=stride),
                                       torch.nn.ReLU())
  
  def forward(self, x):  # x = [batch_size, signal_len]
    scatering_filters = self.scatering(x)
    gammatones_filters = self.gammatones(x)
    out = self.reducer(torch.cat((scatering_filters, gammatones_filters), axis=1))  # [batch_size, self.n_feats, seq_len]
    return out.permute(0, 2, 1)


class AudioEmbedder(torch.nn.Module):
  def __init__(self, input_dim, emb_dim, hid_dim, max_seq_len, dropout, device, reduce_dim=False):
    super().__init__()
    self.device = device
    self.filters = MixFilters()

    self.projection = torch.nn.Linear(self.filters.n_feats, emb_dim)
    self.normalization = torch.nn.LayerNorm(emb_dim)
    self.positional_embedding = u.create_positional_embedding(max_seq_len, emb_dim, hid_dim)
  
  def forward(self, x):
    out = self.filters(x)
    out = self.normalization(self.projection(out))

    batch_size, seq_len, _ = out.shape
    index_x = torch.LongTensor(range(seq_len)).unsqueeze(0).repeat(batch_size, 1).to(self.device)
    pos_emb = self.positional_embedding(index_x).float()

    return out + pos_emb


class Experiment21(ConvnetExperiments):
  '''Convnet, letters prediction, adam, Attention-CrossEntropy, MultiHead, no-slicing, AudioEmbedder'''
  def __init__(self, logfile='_logs/_logs_experiment21.txt', save_name_model='convnet/convnet_experiment21.pt', batch_size=8,
               metadata_file='_Data_metadata_letters_mfcc0128.pk', multi_head=True, slice_fn=lambda x: x):
    convnet_config = {'encoder_embedder': AudioEmbedder, 'enc_layers': 4, 'dec_layers': 6}
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, batch_size=batch_size,
                     convnet_config=convnet_config, multi_head=multi_head, slice_fn=slice_fn)


class Experiment24(ConvnetExperiments):
  '''Convnet, letters prediction, adam, Attention-CrossEntropy, MultiHead, no-slicing, AudioEmbedder, scaling'''
  def __init__(self, logfile='_logs/_logs_experiment24.txt', save_name_model='convnet/convnet_experiment24.pt', batch_size=8,
               metadata_file='_Data_metadata_letters_mfcc0128.pk', multi_head=True, slice_fn=lambda x: x):
    convnet_config = {'encoder_embedder': AudioEmbedder, 'enc_layers': 4, 'dec_layers': 6}
    slice_fn = lambda x: (x - np.mean(x)) / (np.std(x) + 1e-15)
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, batch_size=batch_size,
                     convnet_config=convnet_config, multi_head=multi_head, slice_fn=slice_fn)


class TransformerExperiments(ConvnetExperiments):
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


class ConvnetFeedbackExperiments(ConvnetExperiments):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
  
  def instanciate_model(self, enc_input_dim=400, enc_max_seq_len=1400, dec_input_dim=31, dec_max_seq_len=600, output_size=31,
                        enc_layers=10, dec_layers=10, enc_kernel_size=3, dec_kernel_size=3, enc_dropout=0.25, dec_dropout=0.25,
                        emb_dim=256, hid_dim=512, reduce_dim=False, pad_idx=2, score_fn=torch.softmax, multi_head=False, d_keys_values=64,
                        encoder_embedder=css.EncoderEmbedder, decoder_embedder=css.DecoderEmbedder):
    enc_embedder = encoder_embedder(enc_input_dim, emb_dim, hid_dim, enc_max_seq_len, enc_dropout, self.device, reduce_dim=reduce_dim)
    dec_embedder = decoder_embedder(dec_input_dim, emb_dim, dec_max_seq_len, dec_dropout, self.device)

    if self.relu:
      enc = css.EncoderRelu(emb_dim, hid_dim, enc_layers, enc_kernel_size, enc_dropout, self.device, embedder=enc_embedder)
      dec = css.DecoderRelu(output_size, emb_dim, hid_dim, dec_layers, dec_kernel_size, dec_dropout, pad_idx, self.device,
                            embedder=dec_embedder, score_fn=score_fn, multi_head=multi_head, d_keys_values=d_keys_values)
    else:
      enc = css.Encoder(emb_dim, hid_dim, enc_layers, enc_kernel_size, enc_dropout, self.device, embedder=enc_embedder)
      dec = css.Decoder(output_size, emb_dim, hid_dim, dec_layers, dec_kernel_size, dec_dropout, pad_idx, self.device,
                        embedder=dec_embedder, score_fn=score_fn, multi_head=multi_head, d_keys_values=d_keys_values)
    
    p_enc_embedder = encoder_embedder(dec_input_dim, emb_dim, hid_dim, dec_max_seq_len, enc_dropout, self.device)
    p_encoder = css.Encoder(emb_dim, hid_dim, 2, enc_kernel_size, enc_dropout, self.device, embedder=p_enc_embedder)
    p_decoder = css.DecoderFeedback(output_size, emb_dim, hid_dim, 2, dec_kernel_size, dec_dropout, pad_idx, self.device,
                                    embedder=dec_embedder, score_fn=score_fn, multi_head=multi_head, d_keys_values=d_keys_values)

    return css.Seq2SeqFeedback(enc, dec, p_encoder, p_decoder, self.device).to(self.device)

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


class Experiment26(ConvnetFeedbackExperiments):
  '''Convnet letters prediction, adam, Attention-CrossEntropy loss, mfcc, n_fft=2048, hop_length=512, MultiHead'''
  def __init__(self, logfile='_logs/_logs_experiment26.txt', save_name_model='convnet/convnet_experiment26.pt', batch_size=8,
               slice_fn=Data.mfcc_extraction, n_fft=2048, hop_length=512, scorer=Data.compute_scores, multi_head=True,
               metadata_file='_Data_metadata_letters_mfcc0128.pk', decay_factor=0):
    super().__init__(logfile=logfile, save_name_model=save_name_model, slice_fn=slice_fn, batch_size=batch_size, n_fft=n_fft,
                     hop_length=hop_length, scorer=scorer, multi_head=multi_head, metadata_file=metadata_file, decay_factor=decay_factor)
    u.load_model(self.model, save_name_model, restore_only_similars=True)


class SeqSeqConvnetTransformer(torch.nn.Module):
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
    enc_embedder = encoder_embedder(enc_input_dim, emb_dim, hid_dim, enc_max_seq_len, enc_dropout, self.device, reduce_dim=enc_reduce_dim)
    enc = css.Encoder(emb_dim, hid_dim, enc_layers, enc_kernel_size, enc_dropout, self.device, embedder=enc_embedder)

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
      torch.nn.init.uniform_(p, a=-val, b=val)


def init_transformer2(model, emb_dim, vocab_size):
  # https://arxiv.org/pdf/1911.03179.pdf
  for k, p in model.named_parameters():
    if p.dim() > 1 and 'decoder' in k:
      if isinstance(p, torch.nn.Linear):
        l = (1 / p.shape[-1]) ** 0.5
        torch.nn.init.uniform_(p, a=-l, b=l)
      elif isinstance(p, torch.nn.Embedding):
        e = (2 / (emb_dim + vocab_size)) ** 0.5
        torch.nn.init.uniform_(p, a=-e, b=e)


if __name__ == "__main__":
  ## SEEDING FOR REPRODUCIBILITY
  SEED = 42
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)

  experiments = {k.replace('Experiment', ''): v for k, v in locals().items() if re.search(r'Experiment\d+', k) is not None}
  
  rep = input('Which Experiment do you want to start? (1-28): ')
  exp = experiments[rep]()
  exp.train()