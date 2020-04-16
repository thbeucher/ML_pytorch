import os
import ast
import sys
import torch
import logging
import argparse
import threading
import torch.optim as optim

from tqdm import tqdm

import data as d

import utils as u
import optimizer as opt
import models.conv_seqseq as css


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
  p_dec = css.DecoderFeedback(output_size, emb_dim, hid_dim, p_dec_layers, p_dec_kernel_size, p_dec_dropout, pad_idx, device,
                              embedder=p_dec_embedder)

  return css.Seq2SeqFeedback(enc, dec, p_enc, p_dec, device).to(device)


def train_pass(model, optimizer, metadata, settings):
  epoch_losses = 0

  for enc_in, dec_in in tqdm(metadata.train_data_loader):
    enc_in, dec_in = enc_in.to(metadata.device), dec_in.to(metadata.device)
    preds, att, f_preds, f_att = model(enc_in, dec_in[:, :-1])

    optimizer.zero_grad()

    current_loss = metadata.loss(preds.contiguous().view(-1, preds.shape[-1]), dec_in[:, 1:].contiguous().view(-1),
                                 att, epsilon=settings['smoothing_epsilon'])
    
    current_loss += metadata.loss(f_preds.contiguous().view(-1, f_preds.shape[-1]), dec_in[:, 1:].contiguous().view(-1),
                                  f_att, epsilon=settings['smoothing_epsilon'])

    current_loss.backward()

    if settings['clip_grad']:
      torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

    optimizer.step()

    metadata.SM.partial_feed(dec_in[:, 1:].tolist(), f_preds.argmax(dim=-1).tolist())

    epoch_losses += current_loss.item()
  
  return epoch_losses / len(metadata.train_data_loader)


def eval_model(model, metadata, settings, epoch, only_loss=True):
  obj_plot_attn = {'eos': metadata.eos_idx, 'pad': metadata.pad_idx, 'i2v': metadata.idx_2_vocab}
  true_labels, pred_labels = [], []
  eval_losses, eval_word_acc, eval_sentence_acc = 0, 0, 0

  model.eval()

  with torch.no_grad():
    for enc_in, dec_in in tqdm(metadata.test_data_loader):
      enc_in, dec_in = enc_in.to(metadata.device), dec_in.to(metadata.device)
      eval_pred, att, f_eval_pred, f_att = model(enc_in, dec_in[:, :-1])

      if settings['plot_attention'] and eval_losses == 0:
        current_data = {'enc_in': enc_in[0].detach().cpu().numpy(),
                        'target': dec_in[0].detach().cpu().tolist(),
                        'attention': att[0].detach().cpu().numpy()}
        im = d.plot_attention({**obj_plot_attn, **current_data}, show=False, verbose=False, cbar=False)
        plotter.image_plot('attention', im)

      loss = metadata.loss(eval_pred.contiguous().view(-1, eval_pred.shape[-1]), dec_in[:, 1:].contiguous().view(-1),
                           att, epsilon=settings['smoothing_epsilon'])
      loss += metadata.loss(f_eval_pred.contiguous().view(-1, f_eval_pred.shape[-1]), dec_in[:, 1:].contiguous().view(-1),
                            f_att, epsilon=settings['smoothing_epsilon'])
      
      eval_losses += loss.item()
      
      if not only_loss:
        preds, _ = model.greedy_decoding(enc_in, metadata.sos_idx, metadata.eos_idx, max_seq_len=dec_in.shape[1])
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

  optimizer = optim.Adam(model.parameters(), lr=settings['lr'])
  # optimizer = opt.AdamW(model.parameters(), lr=settings['lr'])

  if settings['load_model']:
    u.load_model(model, f"{settings['save_path']}convnet_feedback.pt", map_location=None, restore_only_similars=True)

  memory_word_acc = 0

  for epoch in tqdm(range(settings['max_epochs'])):
    epoch_losses = train_pass(model, optimizer, metadata, settings)

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