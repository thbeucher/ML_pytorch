import os
import ast
import torch
import logging
import argparse

import data as d
import utils as u
import optimizer as opt

from tqdm import tqdm
from models.transformer.transformer import Transformer
from models.transformer.embedder import PositionalEmbedder


def instanciate_model(settings, metadata):
  encoder_embedder = u.PositionalEmbedder(settings['enc_max_seq_len'], settings['encoder_embedding_dim'], settings['d_model'])
  decoder_embedder = u.PositionalEmbedder(settings['dec_max_seq_len'], settings['decoder_embedding_dim'], settings['d_model'],
                                          output_size=metadata.output_size)

  model = Transformer(settings['n_encoder_blocks'], settings['n_decoder_blocks'], settings['d_model'], settings['d_keys'],
                      settings['d_values'], settings['n_heads'], settings['d_ff'], metadata.output_size,
                      encoder_embedder=encoder_embedder, decoder_embedder=decoder_embedder, dropout=settings['dropout'],
                      enc_max_seq_len=settings['enc_max_seq_len'], dec_max_seq_len=settings['dec_max_seq_len'])

  logging.info(f'The model has {u.count_trainable_parameters(model):,} trainable parameters')
  
  return model.to(metadata.device)


def train_pass(model, optimizer, metadata, settings):
  epoch_losses = 0

  for enc_in, dec_in, padding_mask in tqdm(metadata.train_data_loader):
    enc_in, dec_in, padding_mask = enc_in.to(metadata.device), dec_in.to(metadata.device), padding_mask.to(metadata.device)
    preds = model(enc_in, dec_in[:, :-1], padding_mask=padding_mask)

    optimizer.zero_grad()

    current_loss = metadata.loss(preds.contiguous().view(-1, preds.shape[-1]), dec_in[:, 1:].contiguous().view(-1),
                                 epsilon=settings['smoothing_epsilon'])

    current_loss.backward()

    optimizer.step()

    metadata.SM.partial_feed(dec_in[:, 1:].tolist(), preds.argmax(dim=-1).tolist())

    epoch_losses += current_loss.item()
  
  return epoch_losses / len(metadata.train_data_loader)


def eval_model(model, metadata, settings, epoch, only_loss=True):
  true_labels, pred_labels = [], []
  eval_losses, eval_word_acc, eval_sentence_acc = 0, 0, 0

  model.eval()

  with torch.no_grad():
    for enc_in, dec_in, _ in tqdm(metadata.test_data_loader):
      enc_in, dec_in = enc_in.to(metadata.device), dec_in.to(metadata.device)
      eval_pred = model(enc_in, dec_in[:, :-1])

      eval_losses += metadata.loss(eval_pred.contiguous().view(-1, eval_pred.shape[-1]), dec_in[:, 1:].contiguous().view(-1),
                                   epsilon=settings['smoothing_epsilon']).item()
      
      if not only_loss:
        preds, _ = model.greedy_decoding(enc_in, metadata.sos_idx, metadata.eos_idx, max_seq_len=dec_in.shape[1])
        true_labels += dec_in[:, 1:].tolist()
        pred_labels += preds.tolist()
  
  if not only_loss:
    eval_word_acc, eval_sentence_acc, _ = u.model_scores(epoch, pred_labels, true_labels, metadata)
    logging.info(f'Epoch {epoch} - TEST word accuracy = {eval_word_acc:.4f} | sentence accuracy = {eval_sentence_acc:.4f}')
    
  model.train()

  eval_loss = eval_losses / len(metadata.test_data_loader)
  logging.info(f'Epoch {epoch} - TEST Loss = {eval_loss:.4f}')
  plotter.line_plot('loss', 'test', 'Loss', epoch, eval_loss)

  return eval_loss, eval_word_acc, eval_sentence_acc


def launch_experiment(settings):
  u.dump_dict(settings, 'TRANSFORMER EXPERIMENT PARAMETERS')

  metadata = d.Metadata(train_folder=settings['train_folder'], test_folder=settings['test_folder'],
                        train_metadata=settings['train_metadata'], test_metadata=settings['test_metadata'],
                        ngram_metadata=settings['ngram_metadata'], vocab=settings['vocab'], decay_step=settings['decay_step'],
                        batch_size=settings['batch_size'], create_mask=settings['create_mask'], subset=settings['subset'],
                        percent=settings['percent'], size_limits=settings['size_limits'], loss=settings['loss'])
  
  model = instanciate_model(settings, metadata)

  optimizer = opt.RAdam(model.parameters(), lr=settings['lr'])

  memory_word_acc = 0

  for epoch in tqdm(range(settings['max_epochs'])):
    epoch_losses = train_pass(model, optimizer, metadata, settings)

    metadata.loss.step(epoch)  # annealing kld loss if args.loss = 'both'

    plotter.line_plot('loss', 'train', 'Loss', epoch, epoch_losses)

    metadata.SM.reset_feed()

    if epoch % settings['eval_step'] == 0:
      _, eval_word_acc, _ = eval_model(model, metadata, settings, epoch, only_loss=False)
    else:
      eval_model(model, metadata, settings, epoch)
    
    if eval_word_acc > memory_word_acc:
      memory_word_acc = eval_word_acc
      u.save_checkpoint(model, optimizer, settings['save_path'] + 'transformer.pt')


if __name__ == "__main__":
  settings = u.load_json('settings.json') if os.path.isfile('settings.json') else {}

  argparser = argparse.ArgumentParser(prog='transformer_experiments.py', description='Transformer Experiments')
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
  argparser.add_argument('--create_mask', default=True, type=ast.literal_eval)
  argparser.add_argument('--loss', default='cross_entropy', type=str)

  argparser.add_argument('--d_model', default=256, type=int)
  argparser.add_argument('--d_keys', default=64, type=int)
  argparser.add_argument('--d_values', default=64, type=int)
  argparser.add_argument('--enc_max_seq_len', default=1500, type=int)
  argparser.add_argument('--dec_max_seq_len', default=600, type=int)
  argparser.add_argument('--encoder_embedding_dim', default=80, type=int)
  argparser.add_argument('--decoder_embedding_dim', default=100, type=int)
  argparser.add_argument('--n_encoder_blocks', default=8, type=int)
  argparser.add_argument('--n_decoder_blocks', default=4, type=int)
  argparser.add_argument('--n_heads', default=4, type=int)
  argparser.add_argument('--d_ff', default=2048, type=int)
  argparser.add_argument('--dropout', default=0.1, type=float)

  argparser.add_argument('--lr', default=1e-4, type=float)
  argparser.add_argument('--smoothing_epsilon', default=0.1, type=float)
  argparser.add_argument('--save_path', default='transformer/', type=str)
  argparser.add_argument('--plot_attention', default=False, type=ast.literal_eval)
  argparser.add_argument('--eval_step', default=10, type=int)
  argparser.add_argument('--max_epochs', default=500, type=int)

  argparser.add_argument('--logfile', default='_transformer_experiments_logs.txt', type=str)
  args = argparser.parse_args()

  # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  logging.basicConfig(filename=args.logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

  settings = u.populate_configuration(settings, vars(args))

  global plotter
  plotter = u.VisdomPlotter(env_name='ConvNet Experiments')

  if not os.path.isdir(settings['save_path']):
    os.makedirs(settings['save_path'])

  launch_experiment(settings)