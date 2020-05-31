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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(__file__).replace('ASR/multitasks_experiments.py', ''))
import utils as u

from data import Data
from ctc_experiments import CTCTrainer
from models.final_net import Decoder, Encoder, EncoderMultiHeadObjective
from models.final_net_configs import get_decoder_config, get_encoder_config


class TextDataset(Dataset):
  def __init__(self, ids_to_encodedsources, pad_idx, mask_idx, mask_percent=0.15, sort_by_target_len=True, seed=None):
    self.ids_to_encodedsources = ids_to_encodedsources
    self.pad_idx = pad_idx
    self.mask_idx = mask_idx
    self.mask_percent = mask_percent
    self.seed = seed
    self.identities = list(sorted(ids_to_encodedsources.keys()))

    if sort_by_target_len:
      self.identities = TextDataset._sort_by_targets_len(self.identities, ids_to_encodedsources)
  
  @staticmethod
  def _sort_by_targets_len(ids, ids2es):
    return list(map(lambda x: x[0], sorted([(i, len(ids2es[i])) for i in ids], key=lambda x: x[1])))
  
  def __len__(self):
    return len(self.identities)
  
  def __getitem__(self, idx):
    if self.seed is not None:
      random.seed(self.seed + idx)

    input_source = torch.LongTensor(self.ids_to_encodedsources[self.identities[idx]])
    target = input_source.clone()

    mask = [True if i < self.mask_percent * (len(input_source)-2) else False for i in range(len(input_source)-2)]
    random.shuffle(mask)
    mask = [False] + mask + [False]

    input_source[mask] = self.mask_idx  # hide some tokens but not <sos> & <eos> tokens
    target[(~np.array(mask)).tolist()] = self.pad_idx  # ignore not-masked tokens

    return input_source, target


class TextCollator(object):
  def __init__(self, pad_idx):
    self.pad_idx = pad_idx
  
  def __call__(self, batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad_idx)
    targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
    return inputs, targets


class MaskedLanguageModelTrainer(object):
  def __init__(self, device=None, logfile='_logs/_logs_masked_language_model.txt', metadata_file='_Data_metadata_letters.pk',
               train_folder='../../../datasets/openslr/LibriSpeech/train-clean-100/', encoding_fn=Data.letters_encoding,
               test_folder='../../../datasets/openslr/LibriSpeech/test-clean/', batch_size=32, lr=1e-4, smoothing_eps=0.1,
               save_name_model='convnet/masked_language_model.pt', n_epochs=5000, load_model=True):
    logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.metadata_file = metadata_file
    self.train_folder = train_folder
    self.test_folder = test_folder
    self.encoding_fn = encoding_fn
    self.batch_size = batch_size
    self.save_name_model = save_name_model
    self.smoothing_eps = smoothing_eps
    self.n_epochs = n_epochs

    self.set_data()
    self.set_data_loader()

    self.model = self.instanciate_model()
    logging.info(self.model)
    logging.info(f'The model has {u.count_trainable_parameters(self.model):,} trainable parameters')

    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.criterion = u.CrossEntropyLoss(self.data.tokens_to_idx['<pad>'])

    if load_model:
      u.load_model(self.model, self.save_name_model, restore_only_similars=True)
  
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
    train_dataset = TextDataset(self.data.ids_to_encodedsources_train, self.data.tokens_to_idx['<pad>'], self.data.tokens_to_idx['<mask>'])
    test_dataset = TextDataset(self.data.ids_to_encodedsources_test, self.data.tokens_to_idx['<pad>'], self.data.tokens_to_idx['<mask>'])
    collator = TextCollator(self.data.tokens_to_idx['<pad>'])
    self.train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collator,
                                        pin_memory=True, shuffle=True)
    self.test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collator, pin_memory=True)
  
  def instanciate_model(self):
    return Decoder(config='css_decoder', embed_x=True, metadata_file=self.metadata_file).to(self.device)
  
  def train(self):
    print('Start Training...')
    eval_accuracy_memory = 0
    for epoch in tqdm(range(self.n_epochs)):
      epoch_loss, accuracy = self.train_pass()
      logging.info(f"Epoch {epoch} | train_loss = {epoch_loss:.3f} | train_accuracy = {accuracy:.3f}")
      eval_loss, eval_accuracy = self.evaluation()
      logging.info(f"Epoch {epoch} | test_loss = {eval_loss:.3f} | test_accuracy = {eval_accuracy:.3f}")

      if eval_accuracy is not None and eval_accuracy > eval_accuracy_memory:
        logging.info(f'Save model with eval_accuracy = {eval_accuracy:.3f}')
        u.save_checkpoint(self.model, None, self.save_name_model)
        eval_accuracy_memory = eval_accuracy
  
  def train_pass(self):
    losses = 0
    all_targets, predictions = [], []

    for inputs, targets in tqdm(self.train_data_loader):
      inputs, targets = inputs.to(self.device), targets.to(self.device)
      preds = self.model(inputs, inputs)  # [batch_size, seq_len, output_dim]

      self.optimizer.zero_grad()
      current_loss = self.criterion(preds.reshape(-1, preds.shape[-1]), targets.reshape(-1), epsilon=self.smoothing_eps)
      current_loss.backward()
      self.optimizer.step()

      all_targets += targets[targets != self.data.tokens_to_idx['<pad>']].tolist()
      predictions += preds.argmax(dim=-1)[targets != self.data.tokens_to_idx['<pad>']].tolist()

      losses += current_loss.item()

    accuracy = (np.array(all_targets) == np.array(predictions)).sum() / len(all_targets)
    
    return losses / len(self.train_data_loader), accuracy
  
  @torch.no_grad()
  def evaluation(self):
    losses = 0
    all_targets, predictions = [], []

    self.model.eval()

    for inputs, targets in tqdm(self.test_data_loader):
      inputs, targets = inputs.to(self.device), targets.to(self.device)
      preds = self.model(inputs, inputs)  # [batch_size, seq_len, output_dim]

      losses += self.criterion(preds.reshape(-1, preds.shape[-1]), targets.reshape(-1), epsilon=self.smoothing_eps).item()

      all_targets += targets[targets != self.data.tokens_to_idx['<pad>']].tolist()
      predictions += preds.argmax(dim=-1)[targets != self.data.tokens_to_idx['<pad>']].tolist()
    
    self.model.train()

    accuracy = (np.array(all_targets) == np.array(predictions)).sum() / len(all_targets)

    return losses / len(self.test_data_loader), accuracy


class AudioReaderDataset(Dataset):
  def __init__(self, ids_to_filenames, ids_to_encodedReader):
    self.ids_to_audiofilefeatures = {i: f.replace('.flac', '.features.npy') for i, f in ids_to_filenames.items()}
    self.ids_to_encodedReader = ids_to_encodedReader
    self.identities = list(sorted(ids_to_filenames.keys()))
  
  def __len__(self):
    return len(self.identities)
  
  def __getitem__(self, idx):
    input_ = torch.tensor(np.load(self.ids_to_audiofilefeatures[self.identities[idx]]))
    target = self.ids_to_encodedReader[self.identities[idx]]
    return input_, target


class AudioReaderCollator(object):
  def __init__(self, audio_pad):
    self.audio_pad = audio_pad

  def __call__(self, batch):
    inputs, targets = zip(*batch)
    inputs_batch = pad_sequence(inputs, batch_first=True, padding_value=self.audio_pad).float()
    targets_batch = torch.LongTensor(targets)
    return inputs_batch, targets_batch


class ReaderRecognitionTrainer(object):
  def __init__(self, device=None, logfile='_logs/_logs_reader_recognition.txt', metadata_file='_Data_metadata_letters_wav2vec.pk',
               train_folder='../../../datasets/openslr/LibriSpeech/train-clean-100/', n_epochs=500, load_model=True,
               test_folder='../../../datasets/openslr/LibriSpeech/test-clean/', batch_size=32, lr=1e-4, smoothing_eps=0.,
               save_name_model='convnet/reader_recognizer.pt', list_files_fn=Data.get_openslr_files,
               process_file_fn=Data.read_and_slice_signal):
    logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.metadata_file = metadata_file
    self.train_folder = train_folder
    self.test_folder = test_folder
    self.list_files_fn = list_files_fn
    self.process_file_fn = process_file_fn
    self.batch_size = batch_size
    self.save_name_model = save_name_model
    self.smoothing_eps = smoothing_eps
    self.n_epochs = n_epochs

    self.set_data()
    self.prepare_data()
    self.set_data_loader()

    self.model = self.instanciate_model()
    logging.info(self.model)
    logging.info(f'The model has {u.count_trainable_parameters(self.model):,} trainable parameters')

    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.criterion = u.CrossEntropyLoss(-1)

    if load_model:
      u.load_model(self.model, self.save_name_model, restore_only_similars=True)
  
  def set_data(self):
    self.data = Data()

    if not os.path.isfile(self.metadata_file):
      self.data.set_audio_metadata(self.train_folder, self.test_folder, list_files_fn=self.list_files_fn,
                                   process_file_fn=self.process_file_fn)  # add Wav2VecModel
      self.data.save_metadata(save_name=self.metadata_file)
    else:
      self.data.load_metadata(save_name=self.metadata_file)
  
  def prepare_data(self):
    readers_ids_filenames = [(k.split('/')[-1].split('-')[0], k, v) for k, v in self.data.ids_to_audiofile_train.items()]
    readers = [el[0] for el in readers_ids_filenames]
    train_ids, test_ids = train_test_split(readers_ids_filenames, test_size=0.1, shuffle=True, stratify=readers)
    self.ids_to_fnames_train = {el[1]: el[2] for el in train_ids}
    self.ids_to_fnames_test = {el[1]: el[2] for el in test_ids}

    self.readers = sorted(list(set(readers)))
    self.ids_to_encodedReader_train = {k: self.readers.index(k.split('-')[0]) for k in self.ids_to_fnames_train}
    self.ids_to_encodedReader_test = {k: self.readers.index(k.split('-')[0]) for k in self.ids_to_fnames_test}
  
  def set_data_loader(self):
    train_dataset = AudioReaderDataset(self.ids_to_fnames_train, self.ids_to_encodedReader_train)
    test_dataset = AudioReaderDataset(self.ids_to_fnames_test, self.ids_to_encodedReader_test)
    collator = AudioReaderCollator(0)
    self.train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collator,
                                        pin_memory=True, shuffle=True)
    self.test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collator, pin_memory=True)
  
  def instanciate_model(self):
    return Encoder(config=get_encoder_config(config='rnn_base'), output_size=len(self.readers), one_pred=True).to(self.device)
  
  def train(self):
    print('Start Training...')
    eval_accuracy_memory = 0
    for epoch in tqdm(range(self.n_epochs)):
      epoch_loss, accuracy = self.train_pass()
      logging.info(f"Epoch {epoch} | train_loss = {epoch_loss:.3f} | train_accuracy = {accuracy:.3f}")
      eval_loss, eval_accuracy = self.evaluation()
      logging.info(f"Epoch {epoch} | test_loss = {eval_loss:.3f} | test_accuracy = {eval_accuracy:.3f}")

      if eval_accuracy is not None and eval_accuracy > eval_accuracy_memory:
        logging.info(f'Save model with eval_accuracy = {eval_accuracy:.3f}')
        u.save_checkpoint(self.model, None, self.save_name_model)
        eval_accuracy_memory = eval_accuracy
  
  def train_pass(self):
    losses = 0
    all_targets, predictions = [], []

    for inputs, targets in tqdm(self.train_data_loader):
      inputs, targets = inputs.to(self.device), targets.to(self.device)
      preds = self.model(inputs)  # [batch_size, output_dim]

      self.optimizer.zero_grad()
      current_loss = self.criterion(preds, targets, epsilon=self.smoothing_eps)
      current_loss.backward()
      self.optimizer.step()

      all_targets += targets.tolist()
      predictions += preds.argmax(dim=-1).tolist()

      losses += current_loss.item()

    accuracy = (np.array(all_targets) == np.array(predictions)).sum() / len(all_targets)
    
    return losses / len(self.train_data_loader), accuracy
  
  @torch.no_grad()
  def evaluation(self):
    losses = 0
    all_targets, predictions = [], []

    self.model.eval()

    for inputs, targets in tqdm(self.test_data_loader):
      inputs, targets = inputs.to(self.device), targets.to(self.device)
      preds = self.model(inputs)  # [batch_size, output_dim]

      losses += self.criterion(preds, targets, epsilon=self.smoothing_eps).item()

      all_targets += targets.tolist()
      predictions += preds.argmax(dim=-1).tolist()
    
    self.model.train()

    accuracy = (np.array(all_targets) == np.array(predictions)).sum() / len(all_targets)

    return losses / len(self.test_data_loader), accuracy


class CTCAttentionDataset(Dataset):
  def __init__(self, ids_to_audiofile, ids_to_encodedsources_ctc, ids_to_encodedsources_attn, sort_by_target_len=True):
    self.ids_to_audiofilefeatures = {i: f.replace('.flac', '.features.npy') for i, f in ids_to_audiofile.items()}
    self.ids_to_encodedsources_ctc = ids_to_encodedsources_ctc
    self.ids_to_encodedsources_attn = ids_to_encodedsources_attn
    self.identities = list(sorted(ids_to_encodedsources_ctc.keys()))

    if sort_by_target_len:
      self.identities = CTCAttentionDataset._sort_by_targets_len(self.identities, ids_to_encodedsources_ctc)
  
  @staticmethod
  def _sort_by_targets_len(ids, ids2es):
    return list(map(lambda x: x[0], sorted([(i, len(ids2es[i])) for i in ids], key=lambda x: x[1])))
  
  def __len__(self):
    return len(self.identities)
  
  def __getitem__(self, idx):
    input_ = torch.tensor(np.load(self.ids_to_audiofilefeatures[self.identities[idx]]))
    ctc_target = torch.LongTensor(self.ids_to_encodedsources_ctc[self.identities[idx]])
    attn_target = torch.LongTensor(self.ids_to_encodedsources_attn[self.identities[idx]])
    input_len = len(input_)
    target_len = len(ctc_target)
    return input_, ctc_target, input_len, target_len, attn_target


class CTCAttentionCollator(object):
  def __init__(self, audio_pad, ctc_text_pad, attn_text_pad):
    self.audio_pad = audio_pad
    self.ctc_text_pad = ctc_text_pad
    self.attn_text_pad = attn_text_pad

  def __call__(self, batch):
    inputs, ctc_targets, input_lens, target_lens, attn_targets = zip(*batch)
    inputs_batch = pad_sequence(inputs, batch_first=True, padding_value=self.audio_pad).float()
    ctc_targets_batch = pad_sequence(ctc_targets, batch_first=True, padding_value=self.ctc_text_pad)
    attn_targets_batch = pad_sequence(attn_targets, batch_first=True, padding_value=self.attn_text_pad)
    input_lens = torch.LongTensor(input_lens)
    target_lens = torch.LongTensor(target_lens)
    return inputs_batch, ctc_targets_batch, input_lens, target_lens, attn_targets_batch


class CTCAttentionTrainer(object):
  def __init__(self, device=None, logfile='_logs/_logs_CTCAttn.txt', metadata_file='_Data_metadata_letters_wav2vec.pk',
               batch_size=32, lr=1e-4, load_model=True, n_epochs=500, save_name_model='convnet/ctc_attn.pt', config={},
               lambda_ctc=0.2, lambda_attn=0.8, smoothing_eps=0.1):
    logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.batch_size = batch_size
    self.n_epochs = n_epochs
    self.save_name_model = save_name_model
    self.metadata_file = metadata_file
    self.lambda_ctc = lambda_ctc
    self.lambda_attn = lambda_attn
    self.smoothing_eps = smoothing_eps

    self.set_data()
    self.set_data_loader()

    self.model = self.instanciate_model(**config)
    self.model = nn.DataParallel(self.model)
    
    logging.info(self.model)
    logging.info(f'The model has {u.count_trainable_parameters(self.model):,} trainable parameters')

    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.ctc_criterion = nn.CTCLoss()
    self.attn_criterion = u.CrossEntropyLoss(self.data.tokens_to_idx['<pad>'])

    if load_model:
      u.load_model(self.model.module.decoder, 'convnet/masked_language_model.pt', restore_only_similars=True)
      u.load_model(self.model, self.save_name_model, restore_only_similars=True)
  
  def set_data(self):
    self.data = Data()
    self.data.load_metadata(save_name=self.metadata_file)

    self.idx_to_tokens = ['<blank>'] + self.data.idx_to_tokens[3:]
    self.tokens_to_idx = {t: i for i, t in enumerate(self.idx_to_tokens)}

    self.ids_to_encodedsources_train = {k: (np.array(v[1:-1])-2).tolist() for k, v in self.data.ids_to_encodedsources_train.items()}
    self.ids_to_encodedsources_test = {k: (np.array(v[1:-1])-2).tolist() for k, v in self.data.ids_to_encodedsources_test.items()}
  
  def set_data_loader(self):
    train_dataset = CTCAttentionDataset(self.data.ids_to_audiofile_train, self.ids_to_encodedsources_train,
                                        self.data.ids_to_encodedsources_train)
    test_dataset = CTCAttentionDataset(self.data.ids_to_audiofile_test, self.ids_to_encodedsources_test,
                                       self.data.ids_to_encodedsources_test)

    collator = CTCAttentionCollator(0, 0, self.data.tokens_to_idx['<pad>'])

    self.train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collator,
                                        shuffle=True, pin_memory=True)
    self.test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collator, pin_memory=True)
  
  def instanciate_model(self, **kwargs):
    enc_config = kwargs.get('encoder_config', 'base')
    decoder_config = kwargs.get('decoder_config', 'css_decoder')
    return EncoderMultiHeadObjective(encoder_config=get_encoder_config(config=enc_config), output_size=len(self.idx_to_tokens),
                                     decoder_config=decoder_config, metadata_file=self.metadata_file).to(self.device)
  
  def train(self):
    print('Start Training...')
    eval_accuracy_memory = 0
    for epoch in tqdm(range(self.n_epochs)):
      epoch_loss, accs = self.train_pass()
      logging.info(f"Epoch {epoch} | train_loss = {epoch_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")
      eval_loss, accs = self.evaluation()
      logging.info(f"Epoch {epoch} | test_loss = {eval_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")

      oea = accs.get('attn_word_accuracy', None)

      if oea is not None and oea > eval_accuracy_memory:
        logging.info(f'Save model with eval_accuracy = {oea:.3f}')
        u.save_checkpoint(self.model, None, self.save_name_model)
        eval_accuracy_memory = oea

  def train_pass(self):
    losses, accs = 0, {}
    ctc_all_targets, ctc_all_preds = [], []
    attn_all_targets, attn_all_preds = [], []

    for inputs, ctc_targets, input_lens, target_lens, attn_targets in tqdm(self.train_data_loader):
      input_lens = u.compute_out_conv(u.compute_out_conv(input_lens, kernel=3, stride=2, padding=1, dilation=1),
                                      kernel=3, stride=2, padding=1, dilation=1)

      inputs, ctc_targets, attn_targets = inputs.to(self.device), ctc_targets.to(self.device), attn_targets.to(self.device)
      input_lens, target_lens = input_lens.to(self.device), target_lens.to(self.device)
      
      ctc_preds, attn_pred = self.model(inputs, attn_targets[:, :-1])  # [batch_size, seq_len, output_dim]

      self.optimizer.zero_grad()
      ctc_loss = self.ctc_criterion(ctc_preds.permute(1, 0, 2).log_softmax(-1), ctc_targets, input_lens, target_lens)
      attn_loss = self.attn_criterion(attn_pred.reshape(-1, attn_pred.shape[-1]), attn_targets[:, 1:].reshape(-1),
                                      epsilon=self.smoothing_eps)
      current_loss = self.lambda_ctc * ctc_loss + self.lambda_attn * attn_loss
      current_loss.backward()
      self.optimizer.step()

      ctc_all_targets += ctc_targets.tolist()
      ctc_all_preds += ctc_preds.argmax(dim=-1).tolist()

      attn_all_targets += attn_targets[:, 1:].tolist()
      attn_all_preds += attn_pred.argmax(dim=-1).tolist()

      losses += current_loss.item()

    ctc_accs = CTCTrainer.scorer(ctc_all_targets, ctc_all_preds, self.idx_to_tokens, self.tokens_to_idx)
    attn_accs = Data.compute_scores(targets=attn_all_targets, predictions=attn_all_preds, eos_idx=self.data.tokens_to_idx['<eos>'],
                                    idx_to_tokens=self.data.idx_to_tokens)
    accs = {f'ctc_{k}': v for k, v in ctc_accs.items()}
    accs = {**accs, **{f'attn_{k}': v for k, v in attn_accs.items()}}
    
    return losses / len(self.train_data_loader), accs
  
  @torch.no_grad()
  def evaluation(self):
    losses, accs = 0, {}
    ctc_all_targets, ctc_all_preds = [], []
    attn_all_targets, attn_all_preds = [], []

    self.model.eval()

    for inputs, ctc_targets, input_lens, target_lens, attn_targets in tqdm(self.test_data_loader):
      input_lens = u.compute_out_conv(u.compute_out_conv(input_lens, kernel=3, stride=2, padding=1, dilation=1),
                                      kernel=3, stride=2, padding=1, dilation=1)

      inputs, ctc_targets, attn_targets = inputs.to(self.device), ctc_targets.to(self.device), attn_targets.to(self.device)
      input_lens, target_lens = input_lens.to(self.device), target_lens.to(self.device)
      
      ctc_preds, attn_pred = self.model(inputs, attn_targets[:, :-1])  # [batch_size, seq_len, output_dim]

      ctc_loss = self.ctc_criterion(ctc_preds.permute(1, 0, 2).log_softmax(-1), ctc_targets, input_lens, target_lens)
      attn_loss = self.attn_criterion(attn_pred.reshape(-1, attn_pred.shape[-1]), attn_targets[:, 1:].reshape(-1),
                                      epsilon=self.smoothing_eps)
      current_loss = self.lambda_ctc * ctc_loss + self.lambda_attn * attn_loss

      ctc_all_targets += ctc_targets.tolist()
      ctc_all_preds += ctc_preds.argmax(dim=-1).tolist()

      attn_all_targets += attn_targets[:, 1:].tolist()
      attn_all_preds += attn_pred.argmax(dim=-1).tolist()

      losses += current_loss.item()
    
    self.model.train()

    ctc_accs = CTCTrainer.scorer(ctc_all_targets, ctc_all_preds, self.idx_to_tokens, self.tokens_to_idx)
    attn_accs = Data.compute_scores(targets=attn_all_targets, predictions=attn_all_preds, eos_idx=self.data.tokens_to_idx['<eos>'],
                                    idx_to_tokens=self.data.idx_to_tokens)
    accs = {f'ctc_{k}': v for k, v in ctc_accs.items()}
    accs = {**accs, **{f'attn_{k}': v for k, v in attn_accs.items()}}
    
    return losses / len(self.test_data_loader), accs


class AttentionOP(nn.Module):
  def __init__(self, n_heads, max_source_len, max_signal_len, output_size, d_model, bias=True):
    super().__init__()
    self.max_source_len = max_source_len
    self.n_heads = n_heads
    self.bias = bias
    self.encoder = Encoder(config=get_encoder_config(config='base'), input_proj='base')
    self.w_soft = nn.Parameter(torch.Tensor(n_heads * max_source_len, max_signal_len))
    nn.init.xavier_uniform_(self.w_soft)
    self.w_proj = nn.Parameter(torch.Tensor(output_size, d_model * n_heads))
    nn.init.xavier_uniform_(self.w_proj)
    if self.bias:
      self.b_proj = nn.Parameter(torch.Tensor(output_size))
      nn.init.constant_(self.b_proj, 0.)
  
  def forward(self, x, t=None):  # x = [batch_size, signal_len, n_feats] | t = batch_source_len
    t = self.max_source_len if t is None else t
    x = self.encoder(x)  # [batch_size, signal_len, n_feats]
    x = self.w_soft[:t*self.n_heads, :x.size(1)].softmax(-1).matmul(x)  # [batch_size, t * n_heads, n_feats]
    x = x.reshape(x.size(0), t, -1).matmul(self.w_proj.t())  # [batch_size, t, output_size]
    if self.bias:
      x = x + self.b_proj
    return x


class AttentionDataset(Dataset):
  def __init__(self, ids_to_audiofile, ids_to_encodedsources, sort_by_target_len=True):
    self.ids_to_audiofilefeatures = {i: f.replace('.flac', '.features.npy') for i, f in ids_to_audiofile.items()}
    self.ids_to_encodedsources = ids_to_encodedsources
    self.identities = list(sorted(ids_to_encodedsources.keys()))

    if sort_by_target_len:
      self.identities = AttentionDataset._sort_by_targets_len(self.identities, ids_to_encodedsources)
  
  @staticmethod
  def _sort_by_targets_len(ids, ids2es):
    return list(map(lambda x: x[0], sorted([(i, len(ids2es[i])) for i in ids], key=lambda x: x[1])))
  
  def __len__(self):
    return len(self.identities)
  
  def __getitem__(self, idx):
    input_ = torch.tensor(np.load(self.ids_to_audiofilefeatures[self.identities[idx]]))
    target = torch.LongTensor(self.ids_to_encodedsources[self.identities[idx]])
    return input_, target


class AttentionCollator(object):
  def __init__(self, audio_pad, text_pad):
    self.audio_pad = audio_pad
    self.text_pad = text_pad

  def __call__(self, batch):
    inputs, targets = zip(*batch)
    inputs_batch = pad_sequence(inputs, batch_first=True, padding_value=self.audio_pad).float()
    targets_batch = pad_sequence(targets, batch_first=True, padding_value=self.text_pad)
    return inputs_batch, targets_batch


class AttentionTrainer(object):
  def __init__(self, device=None, logfile='_logs/_logs_Attn.txt', metadata_file='_Data_metadata_letters_wav2vec.pk',
               batch_size=64, lr=1e-4, load_model=True, n_epochs=500, save_name_model='convnet/attn.pt', config={}):
    logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.batch_size = batch_size
    self.n_epochs = n_epochs
    self.save_name_model = save_name_model
    self.metadata_file = metadata_file
    self.smoothing_eps = 0.1

    self.set_data()
    self.set_data_loader()

    self.model = self.instanciate_model(**config)
    self.model = nn.DataParallel(self.model)

    logging.info(self.model)
    logging.info(f'The model has {u.count_trainable_parameters(self.model):,} trainable parameters')

    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.criterion = u.CrossEntropyLoss(self.tokens_to_idx['<pad>'])

    if load_model:
      u.load_model(self.model, self.save_name_model, restore_only_similars=True)
      # u.load_model(self.model.module.encoder, 'convnet/ctc_conv_dilated12.pt', restore_only_similars=True)
  
  def set_data(self):
    self.data = Data()
    self.data.load_metadata(save_name=self.metadata_file)

    self.idx_to_tokens = self.data.idx_to_tokens[1:]
    self.tokens_to_idx = {t: i for i, t in enumerate(self.idx_to_tokens)}

    self.ids_to_encodedsources_train = {k: (np.array(v[1:])-1).tolist() for k, v in self.data.ids_to_encodedsources_train.items()}
    self.ids_to_encodedsources_test = {k: (np.array(v[1:])-1).tolist() for k, v in self.data.ids_to_encodedsources_test.items()}
  
  def set_data_loader(self):
    train_dataset = AttentionDataset(self.data.ids_to_audiofile_train, self.ids_to_encodedsources_train)
    test_dataset = AttentionDataset(self.data.ids_to_audiofile_test, self.ids_to_encodedsources_test)

    collator = AttentionCollator(0, self.tokens_to_idx['<pad>'])

    self.train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collator,
                                        shuffle=True, pin_memory=True)
    self.test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collator, pin_memory=True)
  
  def instanciate_model(self, **kwargs):
    return AttentionOP(kwargs.get('n_heads', 8), self.data.max_source_len, self.data.max_signal_len,
                       len(self.idx_to_tokens), kwargs.get('d_model', 512)).to(self.device)
  
  def train(self):
    print('Start Training...')
    eval_accuracy_memory = 0
    for epoch in tqdm(range(self.n_epochs)):
      epoch_loss, accs = self.train_pass()
      logging.info(f"Epoch {epoch} | train_loss = {epoch_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")
      eval_loss, accs = self.evaluation()
      logging.info(f"Epoch {epoch} | test_loss = {eval_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")

      oea = accs.get('word_accuracy', None)

      if oea is not None and oea > eval_accuracy_memory:
        logging.info(f'Save model with eval_accuracy = {oea:.3f}')
        u.save_checkpoint(self.model, None, self.save_name_model)
        eval_accuracy_memory = oea
  
  def train_pass(self):
    losses, accs = 0, {}
    all_targets, all_preds = [], []

    for inputs, targets in tqdm(self.train_data_loader):
      inputs, targets = inputs.to(self.device), targets.to(self.device)
      
      preds = self.model(inputs, t=targets.size(1))  # [batch_size, seq_len, output_dim]

      self.optimizer.zero_grad()
      current_loss = self.criterion(preds.reshape(-1, preds.shape[-1]), targets.reshape(-1), epsilon=self.smoothing_eps)
      current_loss.backward()
      self.optimizer.step()

      all_targets += targets.tolist()
      all_preds += preds.argmax(dim=-1).tolist()

      losses += current_loss.item()

    accs = Data.compute_scores(targets=all_targets, predictions=all_preds, eos_idx=self.tokens_to_idx['<eos>'],
                               idx_to_tokens=self.idx_to_tokens)
    
    return losses / len(self.train_data_loader), accs
  
  @torch.no_grad()
  def evaluation(self):
    losses, accs = 0, {}
    all_targets, all_preds = [], []

    self.model.eval()

    for inputs, targets in tqdm(self.test_data_loader):
      inputs, targets = inputs.to(self.device), targets.to(self.device)
      
      preds = self.model(inputs, t=targets.size(1))  # [batch_size, seq_len, output_dim]

      losses = self.criterion(preds.reshape(-1, preds.shape[-1]), targets.reshape(-1), epsilon=self.smoothing_eps).item()

      all_targets += targets.tolist()
      all_preds += preds.argmax(dim=-1).tolist()
    
    self.model.train()

    accs = Data.compute_scores(targets=all_targets, predictions=all_preds, eos_idx=self.tokens_to_idx['<eos>'],
                               idx_to_tokens=self.idx_to_tokens)
    
    return losses / len(self.test_data_loader), accs


if __name__ == "__main__":
  ## SEEDING FOR REPRODUCIBILITY
  SEED = 42
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)

  rep = input('Train Masked Language Model? (y or n): ')
  if rep == 'y':
    mlm_trainer = MaskedLanguageModelTrainer()
    mlm_trainer.train()

  rep = input('Train Reader Recognizer? (y or n): ')
  if rep == 'y':
    reader_recog = ReaderRecognitionTrainer()
    reader_recog.train()
  
  rep = input('Train CTC Attention Model? (y or n): ')
  if rep == 'y':
    ctc_attn = CTCAttentionTrainer()
    ctc_attn.train()
  
  rep = input('Train Attention Model? (y or n): ')
  if rep == 'y':
    attn = AttentionTrainer()
    attn.train()