import os
import sys
import json
import torch
import logging
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

sys.path.append(os.path.abspath(__file__).replace('ASR/data.py', ''))
import utils as u
import models.conv_seqseq as css

from data import Data


class ConvnetTrainer(object):
  def __init__(self, device=None, logfile='_logs/_logs_experiment.txt', save_name_model='convnet/convnet_experiment.pt', readers=[],
               metadata_file='_Data_metadata_letters.pk', dump_config=True, encoding_fn=Data.letters_encoding, score_fn=F.softmax,
               list_files_fn=Data.get_openslr_files, process_file_fn=Data.read_and_slice_signal, signal_type='window-sliced',
               slice_fn=Data.window_slicing_signal, multi_head=False, d_keys_values=64, lr=1e-4, smoothing_eps=0.1, n_epochs=500,
               batch_size=32, decay_factor=1, decay_step=0.01, create_enc_mask=False, eval_step=10, scorer=Data.compute_accuracy,
               convnet_config={}, relu=False, pin_memory=True, train_folder='../../../datasets/openslr/LibriSpeech/train-clean-100/',
               test_folder='../../../datasets/openslr/LibriSpeech/test-clean/', **kwargs):
    '''
    Params:
      * device (optional) : torch.device
      * logfile (optional) : str, filename for logs dumping
      * save_name_model (optional) : str, filename of model saving
      * readers (optional) : list of str
      * metadata_file (optional) : str, filename where metadata from Data are saved
      * dump_config (optional) : bool, True to dump convnet configuration to logging file
      * encoding_fn (optional) : function, handle text encoding
      * score_fn (optional) : function, handle energy computation for attention mechanism
      * list_files_fn (optional) : function, handles files list retrieval
      * process_file_fn (optional) : function, reads and process audio files
      * signal_type (optional) : str
      * slice_fn (optional) : function, handles audio raw signal framing
      * multi_head (optional) : bool, True to use a MultiHeadAttention mechanism
      * d_keys_values (optional) : int, key/values dimension for the multihead-attention
      * lr (optional) : float, learning rate passed to the optimizer
      * smoothing_eps (optional) : float, Label-Smoothing epsilon
      * n_epochs (optional) : int
      * batch_size (optional) : int
      * decay_factor (optional) : int, 0 for cross-entropy loss only, 1 for Attention-CrossEntropy loss
      * decay_step (optional) : float, decreasing step of Attention loss
      * create_enc_mask (optional) : bool
      * eval_step (optional) : int, computes accuracies on test set when (epoch % eval_step == 0)
      * scorer (optional) : function, computes training and testing metrics
      * convnet_config (optional) : dict
      * relu (optional) : bool, True to use ReLU version of ConvEncoder&Decoder
      * pin_memory (optional) : bool, passed to DataLoader
      * train_folder (optional) : str
      * test_folder (optional) : str
      * kwargs (optional) : arguments passed to process_file_fn
    '''
    # [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
    logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.logfile = logfile
    self.save_name_model = save_name_model
    self.readers = readers
    self.metadata_file = metadata_file
    self.dump_config = dump_config
    self.encoding_fn = encoding_fn
    self.score_fn = score_fn
    self.list_files_fn = list_files_fn
    self.process_file_fn = process_file_fn
    self.signal_type = signal_type
    self.slice_fn = slice_fn
    self.multi_head = multi_head
    self.d_keys_values = d_keys_values
    self.lr = lr
    self.smoothing_eps = smoothing_eps
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.decay_factor = decay_factor
    self.decay_step = decay_step
    self.create_enc_mask = create_enc_mask
    self.eval_step = eval_step
    self.scorer = scorer
    self.relu = relu
    self.pin_memory = pin_memory
    self.train_folder = train_folder
    self.test_folder = test_folder
    self.process_file_fn_args = {**kwargs, **{'slice_fn': slice_fn}}

    self.set_data()
    self.sos_idx = self.data.tokens_to_idx['<sos>']
    self.eos_idx = self.data.tokens_to_idx['<eos>']
    self.pad_idx = self.data.tokens_to_idx['<pad>']

    self.convnet_config = {'enc_input_dim': self.data.n_signal_feats, 'enc_max_seq_len': self.data.max_signal_len,
                           'dec_input_dim': len(self.data.idx_to_tokens), 'dec_max_seq_len': self.data.max_source_len,
                           'output_size': len(self.data.idx_to_tokens), 'pad_idx': self.pad_idx, 'score_fn': score_fn,
                           'enc_layers': 10, 'dec_layers': 10, 'enc_kernel_size': 3, 'dec_kernel_size': 3, 'enc_dropout': 0.25,
                           'dec_dropout': 0.25, 'emb_dim': 256, 'hid_dim': 512, 'reduce_dim': False,
                           'multi_head': multi_head, 'd_keys_values': d_keys_values}
    self.convnet_config = {**self.convnet_config, **convnet_config}
    self.model = self.instanciate_model(**self.convnet_config)

    if dump_config:
      u.dump_dict(self.convnet_config, 'ENCODER-DECODER PARAMETERS')
      logging.info(f'The model has {u.count_trainable_parameters(self.model):,} trainable parameters')

    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.criterion = u.AttentionLoss(self.pad_idx, self.device, decay_step=decay_step, decay_factor=decay_factor)

    self.set_data_loader()
  
  def set_data(self):
    self.data = Data()

    if not os.path.isfile(self.metadata_file):
      self.data.set_audio_metadata(self.train_folder, self.test_folder, list_files_fn=self.list_files_fn,
                                   process_file_fn=self.process_file_fn, **self.process_file_fn_args)
      self.data.process_all_transcripts(self.train_folder, self.test_folder, encoding_fn=self.encoding_fn)
      self.data.save_metadata(save_name=self.metadata_file)
    else:
      self.data.load_metadata(save_name=self.metadata_file)
  
  def set_data_loader(self):
    self.train_data_loader = self.data.get_dataset_generator(batch_size=self.batch_size, pad_idx=self.pad_idx,
                                                             signal_type=self.signal_type, create_enc_mask=self.create_enc_mask,
                                                             readers=self.readers, pin_memory=self.pin_memory,
                                                             process_file_fn=self.process_file_fn, **self.process_file_fn_args)
    self.test_data_loader = self.data.get_dataset_generator(train=False, batch_size=self.batch_size, pad_idx=self.pad_idx,
                                                            pin_memory=self.pin_memory, signal_type=self.signal_type, shuffle=False,
                                                            create_enc_mask=self.create_enc_mask, readers=self.readers,
                                                            process_file_fn=self.process_file_fn, **self.process_file_fn_args)
  
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

    return css.Seq2Seq(enc, dec, self.device).to(self.device)

  def train(self):
    print('Start Training...')
    eval_accuracy_memory = 0
    for epoch in tqdm(range(self.n_epochs)):
      epoch_loss, accs = self.train_pass(only_loss=False if epoch % self.eval_step == 0 else True)
      logging.info(f"Epoch {epoch} | train_loss = {epoch_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")
      eval_loss, accs = self.evaluation(only_loss=False if epoch % self.eval_step == 0 else True)
      logging.info(f"Epoch {epoch} | test_loss = {eval_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")

      oea = accs['preds_acc'] if 'preds_acc' in accs else accs.get('word_accuracy', None)

      self.criterion.step(999 if self.decay_factor == 0 else epoch)

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
      preds, att = self.model(enc_in, dec_in[:, :-1])

      losses += self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1), att, epsilon=self.smoothing_eps).item()
      
      if not only_loss:
        preds, _ = self.model.greedy_decoding(enc_in, self.sos_idx, self.eos_idx, max_seq_len=dec_in.shape[1])
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
      preds, att = self.model(enc_in, dec_in[:, :-1])

      self.optimizer.zero_grad()

      current_loss = self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1), att, epsilon=self.smoothing_eps)

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
  def dump_predictions(self, save_name='_convnet_preds_results.json'):
    u.load_model(self.model, self.save_name_model, restore_only_similars=True)
    self.model.eval()

    targets, predictions = [], []
    for enc_in, dec_in in tqdm(self.test_data_loader):
      enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
      preds, _ = self.model.greedy_decoding(enc_in, self.sos_idx, self.eos_idx, max_seq_len=dec_in.shape[1])
      targets += dec_in[:, 1:].tolist()
      predictions += preds.tolist()
    
    targets_sentences = Data.reconstruct_sources(targets, self.data.idx_to_tokens, self.eos_idx, joiner='')
    predictions_sentences = Data.reconstruct_sources(predictions, self.data.idx_to_tokens, self.eos_idx, joiner='')

    with open(save_name, 'w') as f:
      json.dump([{'target': t, 'prediction': p} for t, p in zip(targets_sentences, predictions_sentences)], f)
