import os
import ast
import sys
import torch
import logging
import argparse
import threading
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import data as d

import utils as u
import optimizer as opt
import models.conv_seqseq as css


class DecoderFeedback(nn.Module):
  def __init__(self, output_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, pad_idx, device, embedder=None, max_seq_len=100):
    super().__init__()
    self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
    self.dropout = nn.Dropout(dropout)

    self.embedder = css.DecoderEmbedder(output_dim, emb_dim, max_seq_len, dropout, device) if embedder is None else embedder

    self.emb2hid = nn.Linear(emb_dim, hid_dim)
    self.decoders = nn.ModuleList([DecoderBlockFeedback(hid_dim, emb_dim, kernel_size, pad_idx, dropout, device) for _ in range(n_layers)])
    self.hid2emb = nn.Linear(hid_dim, emb_dim)

    self.out = nn.Linear(emb_dim, output_dim)
  
  def forward(self, x, encoder_conved, encoder_combined, pred_conved, pred_combined):
    '''
    Params:
      * x : [batch_size, seq_len]
      * encoder_conved : [batch_size, seq_len, emb_dim]
      * encoder_combined : [batch_size, seq_len, emb_dim]
      * pred_conved : [batch_size, seq_len, emb_dim]
      * pred_combined : [batch_size, seq_len, emb_dim]
    '''
    embedded = self.embedder(x)  # [batch_size, seq_len, emb_dim]
    conv_in = self.emb2hid(embedded)  # [batch_size, seq_len, hid_dim]
    conv_in = conv_in.permute(0, 2, 1)  # prepare for convolution layers

    for decoder in self.decoders:
      attention, conv_in = decoder(embedded, conv_in, encoder_conved, encoder_combined, pred_conved, pred_combined)

    conved = conv_in.permute(0, 2, 1)  # [batch_size, seq_len, hid_dim]
    conved = self.hid2emb(conved)  # [batch_size, seq_len, emb_dim]
    output = self.out(self.dropout(conved))  # [batch_size, seq_len, output_dim]
    return output, attention


class DecoderBlockFeedback(nn.Module):
  def __init__(self, hid_dim, emb_dim, kernel_size, pad_idx, dropout, device):
    super().__init__()
    self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
    self.dropout = nn.Dropout(dropout)

    self.kernel_size = kernel_size
    self.pad_idx = pad_idx
    self.device = device

    # nf_out = (nf_in + 2 * padding - dilation * (kernel_size -1) - 1) / stride + 1
    self.conv = nn.Conv1d(in_channels=hid_dim, out_channels=2 * hid_dim, kernel_size=kernel_size)

    self.attention = AttentionFeedback(hid_dim, emb_dim, device)
  
  def forward(self, embedded, conv_in, encoder_conved, encoder_combined, pred_conved, pred_combined):
    conv_in = self.dropout(conv_in)  # [batch_size, hid_dim, seq_len]
    padding = torch.zeros(conv_in.shape[0], conv_in.shape[1], self.kernel_size - 1).fill_(self.pad_idx).to(self.device)
    padded_conv_in = torch.cat((padding, conv_in), dim=2)  # [batch_size, hid_dim, seq_len + kernel_size - 1]
    conved = self.conv(padded_conv_in)  # [batch_size, 2 * hid_dim, seq_len]
    conved = F.glu(conved, dim=1)  # [batch_size, hid_dim, seq_len]
    attention, conved = self.attention(embedded, conved, encoder_conved, encoder_combined, pred_conved, pred_combined)
    conved = (conved + conv_in) * self.scale  # residual connection
    return attention, conved


class AttentionFeedback(nn.Module):
  def __init__(self, hid_dim, emb_dim, device):
    super().__init__()
    self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

    self.attention_hid2emb = nn.Linear(hid_dim, emb_dim)
    self.attention_emb2hid = nn.Linear(emb_dim, hid_dim)
    self.p_attention_emb2hid = nn.Linear(emb_dim, hid_dim)
  
  def forward(self, embedded, conved, encoder_conved, encoder_combined, pred_conved, pred_combined):
    '''
    Params:
      * embedded : [batch_size, dec_seq_len, emb_dim]
      * conved : [batch_size, hid_dim, dec_seq_len]
      * encoder_conved : [batch_size, enc_seq_len, emb_dim]
      * encoder_combined : [batch_size, enc_seq_len, emb_dim]
      * pred_conved : [batch_size, dec_seq_len, emb_dim]
      * pred_combined : [batch_size, dec_seq_len, emb_dim]
    '''
    conved_emb = self.attention_hid2emb(conved.permute(0, 2, 1))  # [batch_size, dec_seq_len, emb_dim]
    combined = (embedded + conved_emb) * self.scale

    # Alignment Attention of encoder input for decoding prediction
    energy = combined.matmul(encoder_conved.permute(0, 2, 1))  # [batch_size, dec_seq_len, enc_seq_len]
    attention = F.softmax(energy, dim=2)
    attented_encoding = attention.matmul(encoder_combined)  # [batch_size, dec_seq_len, emb_dim]
    attented_encoding = self.attention_emb2hid(attented_encoding)  # [batch_size, dec_seq_len, hid_dim]
    attented_combined = (conved + attented_encoding.permute(0, 2, 1)) * self.scale  # [batch_size, hid_dim, dec_seq_len]

    # Alignment Attention of decoder prediction for final decoding prediction
    p_energy = combined.matmul(pred_conved.permute(0, 2, 1))
    p_attention = F.softmax(p_energy, dim=2)
    p_attented_encoding = p_attention.matmul(pred_combined)
    p_attented_encoding = self.p_attention_emb2hid(p_attented_encoding)
    p_attented_encoding = (conved + p_attented_encoding.permute(0, 2, 1)) * self.scale

    return p_attention, attented_combined + p_attented_encoding


class Seq2SeqFeedback(nn.Module):
  def __init__(self, encoder, decoder, p_encoder, p_decoder, device):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.p_encoder = p_encoder
    self.p_decoder = p_decoder
    self.device = device
  
  def forward(self, enc_in, dec_in, warmup=False):
    encoder_conved, encoder_combined = self.encoder(enc_in)
    output, attention = self.decoder(dec_in, encoder_conved, encoder_combined)

    if warmup:
      return output, attention, None, None

    prediction = output.argmax(-1)

    pred_conved, pred_combined = self.p_encoder(prediction)
    # with torch.no_grad():
    #   pred = self.p_encoder(prediction)
      
    p_output, p_attention = self.p_decoder(dec_in, encoder_conved, encoder_combined, pred_conved, pred_combined)
    # p_output, p_attention = self.p_decoder(dec_in, encoder_conved, encoder_combined, pred, pred)
    
    return output, attention, p_output, p_attention
  
  def greedy_decoding(self, enc_in, sos_idx, eos_idx, max_seq_len=100, warmup=False):
    encoder_conved, encoder_combined = self.encoder(enc_in)

    batch_size = enc_in.shape[0]

    dec_in = torch.LongTensor(batch_size, 1).fill_(sos_idx).to(self.device)

    finished = [False] * batch_size

    for _ in range(max_seq_len):
      output, attention = self.decoder(dec_in, encoder_conved, encoder_combined)

      if not warmup:
        pred = output.argmax(-1)
        pred_conved, pred_combined = self.p_encoder(pred)
        # pred = self.p_encoder(pred)
        output, attention = self.p_decoder(dec_in, encoder_conved, encoder_combined, pred_conved, pred_combined)
        # output, attention = self.p_decoder(dec_in, encoder_conved, encoder_combined, pred, pred)

      pred = output[:, -1, :].argmax(-1).unsqueeze(1)

      for idx in range(batch_size):
        if not finished[idx] and pred[idx].item() == eos_idx:
          finished[idx] = True

      dec_in = torch.cat((dec_in, pred), dim=1)

      if all(finished):
        break
    
    return dec_in[:, 1:], attention


def compute_plot_scores_async(epoch, metadata, plotter):
  _, train_word_acc, train_sentence_acc, awer = metadata.SM.get_scores(metadata.SM.pred_labels,
                                                                       metadata.SM.true_labels,
                                                                       stop_idx=metadata.eos_idx,
                                                                       strategy='other')

  plotter.line_plot('word accuracy', 'train', 'Word Accuracy', epoch, train_word_acc)
  plotter.line_plot('sentence accuracy', 'train', 'Sentence Accuracy', epoch, train_sentence_acc)
  plotter.line_plot('word error rate', 'train', 'Word Error Rate', epoch, awer)

  logging.info(f'EPOCH {epoch} : Train Word accuracy = {train_word_acc:.3f} | Sentence accuracy = {train_sentence_acc:.3f}')


def instanciate_model(enc_input_dim=80, dec_input_dim=100, enc_max_seq_len=1100, dec_max_seq_len=600,
                      enc_layers=10, dec_layers=10, enc_kernel_size=3, dec_kernel_size=3,
                      enc_dropout=0.25, dec_dropout=0.25, emb_dim=256, hid_dim=512, output_size=31,
                      reduce_dim=False, device=None, pad_idx=2,
                      p_enc_layers=2, p_enc_kernel_size=3, p_enc_dropout=0.25,
                      p_dec_layers=6, p_dec_kernel_size=3, p_dec_dropout=0.25):
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if device is None else device

  enc_embedder = css.EncoderEmbedder(enc_input_dim, emb_dim, hid_dim, enc_max_seq_len, enc_dropout, device, reduce_dim=reduce_dim)
  dec_embedder = css.DecoderEmbedder(dec_input_dim, emb_dim, dec_max_seq_len, dec_dropout, device)

  enc = css.Encoder(emb_dim, hid_dim, enc_layers, enc_kernel_size, enc_dropout, device, embedder=enc_embedder)
  dec = css.Decoder(output_size, emb_dim, hid_dim, dec_layers, dec_kernel_size, dec_dropout, pad_idx, device, embedder=dec_embedder)

  p_enc_embedder = css.DecoderEmbedder(dec_input_dim, emb_dim, dec_max_seq_len, dec_dropout, device)
  p_dec_embedder = css.DecoderEmbedder(dec_input_dim, emb_dim, dec_max_seq_len, dec_dropout, device)

  p_enc = css.Encoder(emb_dim, hid_dim, p_enc_layers, p_enc_kernel_size, p_enc_dropout, device, embedder=p_enc_embedder)
  p_dec = DecoderFeedback(output_size, emb_dim, hid_dim, p_dec_layers, p_dec_kernel_size, p_dec_dropout, pad_idx, device,
                          embedder=p_dec_embedder)

  return Seq2SeqFeedback(enc, dec, p_enc, p_dec, device).to(device)
  # return css.Seq2SeqFeedback(enc, dec, dec_embedder, p_dec, device).to(device)


def train_pass(model, optimizer, metadata, settings, epoch):
  epoch_losses = 0
  warmup = epoch <= settings['warmup']

  for enc_in, dec_in in tqdm(metadata.train_data_loader):
    enc_in, dec_in = enc_in.to(metadata.device), dec_in.to(metadata.device)
    preds, att, f_preds, f_att = model(enc_in, dec_in[:, :-1], warmup=warmup)

    optimizer.zero_grad()

    current_loss = metadata.loss(preds.contiguous().view(-1, preds.shape[-1]), dec_in[:, 1:].contiguous().view(-1),
                                 att, epsilon=settings['smoothing_epsilon'])
    
    if not warmup:
      current_loss += metadata.loss(f_preds.contiguous().view(-1, f_preds.shape[-1]), dec_in[:, 1:].contiguous().view(-1),
                                    f_att, epsilon=settings['smoothing_epsilon'])
    
    if settings['l1_reg'] > 0:
      current_loss += u.l1_regularization(model, _lambda=settings['l1_reg'], device=metadata.device)

    current_loss.backward()

    if settings['clip_grad']:
      torch.nn.utils.clip_grad_norm_(model.parameters(), settings['clip_grad_val'])

    optimizer.step()

    if warmup:
      metadata.SM.partial_feed(dec_in[:, 1:].tolist(), preds.argmax(dim=-1).tolist())
    else:
      metadata.SM.partial_feed(dec_in[:, 1:].tolist(), f_preds.argmax(dim=-1).tolist())

    epoch_losses += current_loss.item()
  
  return epoch_losses / len(metadata.train_data_loader)


def eval_model(model, metadata, settings, epoch, only_loss=True):
  obj_plot_attn = {'eos': metadata.eos_idx, 'pad': metadata.pad_idx, 'i2v': metadata.idx_2_vocab}
  true_labels, pred_labels = [], []
  eval_losses, eval_word_acc, eval_sentence_acc = 0, 0, 0
  warmup = epoch <= settings['warmup']

  model.eval()

  with torch.no_grad():
    for enc_in, dec_in in tqdm(metadata.test_data_loader):
      enc_in, dec_in = enc_in.to(metadata.device), dec_in.to(metadata.device)
      eval_pred, att, f_eval_pred, f_att = model(enc_in, dec_in[:, :-1], warmup=warmup)

      if settings['plot_attention'] and eval_losses == 0:
        current_data = {'enc_in': enc_in[0].detach().cpu().numpy(),
                        'target': dec_in[0].detach().cpu().tolist(),
                        'attention': att[0].detach().cpu().numpy()}
        im = d.plot_attention({**obj_plot_attn, **current_data}, show=False, verbose=False, cbar=False)
        plotter.image_plot('attention', im)

      loss = metadata.loss(eval_pred.contiguous().view(-1, eval_pred.shape[-1]), dec_in[:, 1:].contiguous().view(-1),
                           att, epsilon=settings['smoothing_epsilon'])
      
      if not warmup:
        loss += metadata.loss(f_eval_pred.contiguous().view(-1, f_eval_pred.shape[-1]), dec_in[:, 1:].contiguous().view(-1),
                              f_att, epsilon=settings['smoothing_epsilon'])
      
      eval_losses += loss.item()
      
      if not only_loss:
        preds, _ = model.greedy_decoding(enc_in, metadata.sos_idx, metadata.eos_idx, max_seq_len=dec_in.shape[1], warmup=warmup)
        true_labels += dec_in[:, 1:].tolist()
        pred_labels += preds.tolist()
  
  if not only_loss:
    eval_word_acc, eval_sentence_acc, _ = u.model_scores(epoch, pred_labels, true_labels, metadata, plotter)
    logging.info(f'Epoch {epoch} - TEST word accuracy = {eval_word_acc:.4f} | sentence accuracy = {eval_sentence_acc:.4f}')
    
  model.train()

  eval_loss = eval_losses / len(metadata.test_data_loader)
  logging.info(f'Epoch {epoch} - TEST Loss = {eval_loss:.4f}')
  plotter.line_plot('loss', 'test', 'Loss', epoch, eval_loss)

  return eval_loss, eval_word_acc, eval_sentence_acc


def launch_experiment(settings):
  u.dump_dict(settings, 'CONVNET EXPERIMENT PARAMETERS')

  metadata = d.Metadata(train_folder=settings['train_folder'], test_folder=settings['test_folder'],
                        train_metadata=settings['train_metadata'], test_metadata=settings['test_metadata'],
                        ngram_metadata=settings['ngram_metadata'], vocab=settings['vocab'], decay_step=settings['decay_step'],
                        batch_size=settings['batch_size'], create_mask=settings['create_mask'], subset=settings['subset'],
                        percent=settings['percent'], size_limits=settings['size_limits'], loss=settings['loss'])

  model = instanciate_model(enc_input_dim=settings['enc_input_dim'], dec_input_dim=settings['dec_input_dim'],
                            enc_max_seq_len=settings['enc_max_seq_len'], dec_max_seq_len=settings['dec_max_seq_len'],
                            enc_layers=settings['enc_layers'], dec_layers=settings['dec_layers'],
                            enc_kernel_size=settings['enc_kernel_size'], dec_kernel_size=settings['dec_kernel_size'],
                            emb_dim=settings['emb_dim'], hid_dim=settings['hid_dim'],
                            enc_dropout=settings['enc_dropout'], dec_dropout=settings['dec_dropout'],
                            output_size=metadata.output_size, reduce_dim=settings['reduce_dim'], 
                            device=metadata.device, pad_idx=metadata.pad_idx,
                            p_enc_layers=settings['p_enc_layers'], p_enc_kernel_size=settings['p_enc_kernel_size'],
                            p_enc_dropout=settings['p_enc_dropout'], p_dec_layers=settings['p_dec_layers'],
                            p_dec_kernel_size=settings['p_dec_kernel_size'], p_dec_dropout=settings['p_dec_dropout'])

  logging.info(f'The model has {u.count_trainable_parameters(model):,} trainable parameters')

  if settings['weight_decay'] == 0:
    optimizer = optim.Adam(model.parameters(), lr=settings['lr'])
  else:
    optimizer = opt.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])

  if settings['load_model']:
    u.load_model(model, f"{settings['save_path']}convnet_feedback.pt", map_location=None, restore_only_similars=True)

  memory_word_acc = 0

  for epoch in tqdm(range(settings['max_epochs'])):
    epoch_losses = train_pass(model, optimizer, metadata, settings, epoch)

    if epoch % settings['train_acc_step'] == 0: 
      threading.Thread(target=compute_plot_scores_async, args=(epoch, metadata, plotter)).start()

    metadata.loss.step(epoch)  # annealing kld loss if args.loss = 'both'

    plotter.line_plot('loss', 'train', 'Loss', epoch, epoch_losses)

    metadata.SM.reset_feed()

    if epoch % settings['eval_step'] == 0:
      _, eval_word_acc, _ = eval_model(model, metadata, settings, epoch, only_loss=False)
    else:
      eval_model(model, metadata, settings, epoch)
    
    if eval_word_acc > memory_word_acc:
      memory_word_acc = eval_word_acc
      u.save_checkpoint(model, optimizer, settings['save_path'] + 'convnet.pt')


if __name__ == "__main__":
  # With raw encoding of audio file -> Train = [981, 400] | Test = [1399, 400] -> [max_seq_len, n_feats]
  # With mfcc encoding -> Train = [2453, 80] | Test = [3496, 80] -> by removing the longest 54 sequence, max_seq_len = 1700
  # With log_spectrogram encoding -> Train = [2451, 257] | Test = [3494, 257]
  settings = u.load_json('settings.json') if os.path.isfile('settings.json') else {}

  argparser = argparse.ArgumentParser(prog='convnet_experiments_feedback.py', description='ConvNet Experiments Feedback')
  argparser.add_argument('--train_folder', default='../../../datasets/openslr/LibriSpeech/train-clean-100/', type=str)
  argparser.add_argument('--test_folder', default='../../../datasets/openslr/LibriSpeech/test-clean/', type=str)
  argparser.add_argument('--train_metadata', default='metadata_train-clean-100.pk', type=str)
  argparser.add_argument('--test_metadata', default='metadata_test-clean.pk', type=str)
  argparser.add_argument('--ngram_metadata', default='metadata_custom_precoding.pk', type=str)
  argparser.add_argument('--vocab', default='unigram', type=str)
  argparser.add_argument('--decay_step', default=0.01, type=float)
  argparser.add_argument('--batch_size', default=32, type=int)
  argparser.add_argument('--subset', default=False, type=ast.literal_eval)
  argparser.add_argument('--percent', default=0.2, type=float)
  argparser.add_argument('--size_limits', default=False, type=ast.literal_eval)
  argparser.add_argument('--create_mask', default=False, type=ast.literal_eval)
  argparser.add_argument('--loss', default='attention', type=str)

  argparser.add_argument('--enc_input_dim', default=400, type=int)
  argparser.add_argument('--dec_input_dim', default=100, type=int)
  argparser.add_argument('--enc_max_seq_len', default=1400, type=int)
  argparser.add_argument('--dec_max_seq_len', default=600, type=int)
  argparser.add_argument('--enc_layers', default=10, type=int)
  argparser.add_argument('--dec_layers', default=8, type=int)
  argparser.add_argument('--p_enc_layers', default=2, type=int)
  argparser.add_argument('--p_dec_layers', default=6, type=int)
  argparser.add_argument('--enc_kernel_size', default=3, type=int)
  argparser.add_argument('--dec_kernel_size', default=3, type=int)
  argparser.add_argument('--p_enc_kernel_size', default=3, type=int)
  argparser.add_argument('--p_dec_kernel_size', default=3, type=int)
  argparser.add_argument('--emb_dim', default=256, type=int)
  argparser.add_argument('--hid_dim', default=512, type=int)
  argparser.add_argument('--enc_dropout', default=0.25, type=float)
  argparser.add_argument('--dec_dropout', default=0.25, type=float)
  argparser.add_argument('--p_enc_dropout', default=0.25, type=float)
  argparser.add_argument('--p_dec_dropout', default=0.25, type=float)
  argparser.add_argument('--reduce_dim', default=False, type=ast.literal_eval)

  argparser.add_argument('--lr', default=1e-4, type=float)
  argparser.add_argument('--smoothing_epsilon', default=0.1, type=float)
  argparser.add_argument('--save_path', default='convnet/', type=str)
  argparser.add_argument('--plot_attention', default=False, type=ast.literal_eval)
  argparser.add_argument('--eval_step', default=10, type=int)
  argparser.add_argument('--max_epochs', default=500, type=int)
  argparser.add_argument('--train_acc_step', default=5, type=int)
  argparser.add_argument('--load_model', default=False, type=ast.literal_eval)
  argparser.add_argument('--clip_grad', default=False, type=ast.literal_eval)
  argparser.add_argument('--weight_decay', default=0., type=float)  # L2 regularization -> 0.01
  argparser.add_argument('--l1_reg', default=0., type=float)  # L1 regularization -> 0.001
  argparser.add_argument('--warmup', default=100, type=int)
  argparser.add_argument('--clip_grad_val', default=0.1, type=float)

  argparser.add_argument('--logfile', default='_convnet_experiments_feedback_logs.txt', type=str)
  args = argparser.parse_args()

  # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  logging.basicConfig(filename=args.logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

  settings = u.populate_configuration(settings, vars(args))

  global plotter
  plotter = u.VisdomPlotter(env_name='ConvNet Experiments')

  if not os.path.isdir(settings['save_path']):
    os.makedirs(settings['save_path'])

  launch_experiment(settings)