import os
import ast
import sys
import torch
import logging
import argparse

from tqdm import tqdm

import data as d

sys.path.append(os.path.abspath(__file__).replace('ASR/transformer_pretraining.py', ''))
import utils as u
import optimizer as opt
from models.transformer.transformer import Transformer
from models.transformer.embedder import PositionalEmbedder


def train_model(model, metadata, max_epochs=500, train_score_step=10, eval_step=50, save_path=''):
  optimizer = opt.RAdam(model.parameters(), lr=settings['lr'])
  loss = u.CrossEntropyLoss(metadata.pad_idx)

  val_word_acc, val_word_acc_memory, val_sentence_acc = 0, 0, 0

  for epoch in tqdm(range(max_epochs)):
    for _, dec_in, _ in metadata.train_data_loader:
      dec_in = dec_in.to(metadata.device)

      preds = model(dec_in, dec_in)

      optimizer.zero_grad()

      current_loss = loss(preds.view(-1, preds.shape[-1]), dec_in.view(-1))
      current_loss.backward()

      optimizer.step()

      metadata.SM.partial_feed(dec_in.tolist(), preds.argmax(dim=-1).tolist())

    
    if epoch % train_score_step == 0:
      _, word_acc, sentence_acc, _ = metadata.SM.get_scores(None, None, stop_idx=metadata.eos_idx, from_feed=True)
      logging.info(f'Epoch {epoch} - Training word accuracy = {word_acc:.3f} | sentence accuracy = {sentence_acc:.3f}')
      plotter.line_plot('train word accuracy', 'train', 'Pretrainer Word Accuracy', epoch, word_acc)

    metadata.SM.reset_feed()
    
    if epoch % eval_step == 0:
      val_word_acc, val_sentence_acc = eval_model(model, metadata)

      if val_word_acc > val_word_acc_memory:
        u.save_checkpoint(model, optimizer, save_path + 'parrot_transformer.pt')
        logging.info(f'Save model with validation word accuracy = {val_word_acc} | sentence accuracy = {val_sentence_acc}')
        val_word_acc_memory = val_word_acc


def eval_model(model, metadata):
  model.eval()

  with torch.no_grad():
    for _, dec_in, _ in metadata.test_data_loader:
      dec_in = dec_in.to(metadata.device)

      preds = model.greedy_decoding(dec_in, metadata.eos_idx, metadata.pad_idx, max_seq_len=dec_in.shape[1])

      metadata.SM.partial_feed(dec_in.tolist(), preds.tolist())
  
  _, word_acc, sentence_acc, _ = metadata.SM.get_scores(None, None, stop_idx=metadata.eos_idx, from_feed=True)
  
  model.train()
  metadata.SM.reset_feed()
  
  return word_acc, sentence_acc


def pretrain_transformer_parrot(settings):
  metadata = d.Metadata(train_folder=settings['train_folder'], test_folder=settings['test_folder'],
                        train_metadata=settings['train_metadata'], test_metadata=settings['test_metadata'],
                        ngram_metadata=settings['ngram_metadata'],
                        vocab=settings['vocab'], decay_step=settings['decay_step'],
                        subset=settings['subset'], percent=settings['percent'],
                        batch_size=settings['batch_size'], size_limits=settings['size_limits'],
                        create_mask=settings['create_mask'], loss=settings['loss'])
  
  encoder_embedder = PositionalEmbedder(settings['max_dec_in_seq_len'], settings['decoder_embedding_dim'], settings['d_model'],
                                        scaling=settings['scaling'], reduce_dim=settings['decoder_reduce_dim'],
                                        dropout=settings['dropout'], device=metadata.device, output_size=metadata.output_size)
  decoder_embedder = PositionalEmbedder(settings['max_dec_in_seq_len'], settings['decoder_embedding_dim'], settings['d_model'],
                                        scaling=settings['scaling'], reduce_dim=settings['decoder_reduce_dim'],
                                        dropout=settings['dropout'], device=metadata.device, output_size=metadata.output_size)
  
  model = Transformer(settings['n_encoder_blocks'],
                      settings['n_decoder_blocks'],
                      settings['d_model'],
                      settings['d_keys'],
                      settings['d_values'],
                      settings['n_heads'],
                      settings['d_ff'],
                      metadata.output_size,
                      encoder_embedder=encoder_embedder,
                      decoder_embedder=decoder_embedder,
                      encoder_embedding_dim=settings['encoder_embedding_dim'],
                      decoder_embedding_dim=settings['decoder_embedding_dim'],
                      max_enc_in_seq_len=settings['max_enc_in_seq_len'],
                      max_dec_in_seq_len=settings['max_dec_in_seq_len'],
                      encoder_reduce_dim=True,
                      decoder_reduce_dim=True,
                      apply_softmax=False,
                      scaling=settings['scaling'],
                      pad_idx=metadata.pad_idx,
                      dropout=settings['dropout'],
                      device=metadata.device).to(metadata.device)
  
  train_model(model, metadata, max_epochs=settings['max_epochs'], train_score_step=settings['train_score_step'],
              eval_step=settings['eval_step'], save_path=settings['save_path'])


if __name__ == "__main__":
  settings = u.load_json('settings.json') if os.path.isfile('settings.json') else {}

  argparser = argparse.ArgumentParser(prog='transformer_pretraining.py', description='Pretrain Transformer')
  argparser.add_argument('--train_folder', default='../../datasets/openslr/LibriSpeech/train-clean-100/', type=str)
  argparser.add_argument('--test_folder', default='../../datasets/openslr/LibriSpeech/test-clean/', type=str)
  argparser.add_argument('--train_metadata', default='metadata_train-clean-100.pk', type=str)
  argparser.add_argument('--test_metadata', default='metadata_test-clean.pk', type=str)
  argparser.add_argument('--ngram_metadata', default='metadata_custom_precoding.pk', type=str)
  argparser.add_argument('--vocab', default='unigram', type=str)
  argparser.add_argument('--decay_step', default=0.01, type=float)
  argparser.add_argument('--batch_size', default=32, type=int)
  argparser.add_argument('--subset', default=False, type=ast.literal_eval)
  argparser.add_argument('--percent', default=0.2, type=float)
  argparser.add_argument('--size_limits', default=False, type=ast.literal_eval)
  argparser.add_argument('--create_mask', default=True, type=ast.literal_eval)
  argparser.add_argument('--loss', default='attention', type=str)

  argparser.add_argument('--n_encoder_blocks', default=6, type=int)
  argparser.add_argument('--n_decoder_blocks', default=6, type=int)
  argparser.add_argument('--d_model', default=512, type=int)
  argparser.add_argument('--d_keys', default=64, type=int)
  argparser.add_argument('--d_values', default=64, type=int)
  argparser.add_argument('--n_heads', default=8, type=int)
  argparser.add_argument('--d_ff', default=2048, type=int)
  argparser.add_argument('--dropout', default=0., type=float)
  argparser.add_argument('--encoder_embedding_dim', default=80, type=int)
  argparser.add_argument('--decoder_embedding_dim', default=100, type=int)
  argparser.add_argument('--encoder_reduce_dim', default=False, type=ast.literal_eval)
  argparser.add_argument('--decoder_reduce_dim', default=False, type=ast.literal_eval)
  argparser.add_argument('--max_enc_in_seq_len', default=900, type=int)
  argparser.add_argument('--max_dec_in_seq_len', default=600, type=int)
  argparser.add_argument('--scaling', default=True, type=ast.literal_eval)

  argparser.add_argument('--lr', default=1e-4, type=float)
  argparser.add_argument('--save_path', default='pretraining/', type=str)
  argparser.add_argument('--eval_step', default=50, type=int)
  argparser.add_argument('--train_score_step', default=10, type=int)
  argparser.add_argument('--max_epochs', default=500, type=int)
  args = argparser.parse_args()

  logging.basicConfig(stream=sys.stdout, level=logging.INFO)

  settings = u.populate_configuration(settings, vars(args))

  global plotter
  plotter = u.VisdomPlotter(env_name='Pretrainer Plots')

  if not os.path.isdir(settings['save_path']):
    os.mkdir(settings['save_path'])

  rep = input('Start Transformer Parrot Pretraining? (y or n): ')
  if rep == 'y':
    pretrain_transformer_parrot(settings)