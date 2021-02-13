import os
import re
import sys
import json
import torch
import random
import logging
import numpy as np
import pickle as pk
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(__file__).replace('ASR/multitasks_experiments.py', ''))
import utils as u

from data import Data
from ctc_experiments import CTCTrainer
from models.transformer.decoder import TransformerDecoder
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


class AudioVisualDataset(Dataset):
  def __init__(self, ids_to_captionsImagesAudios):
    self.preprocess_img = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    self.ids_to_captionsImagesAudios = ids_to_captionsImagesAudios
    self._sort_by_targets_len()
  
  def __len__(self):
    return len(self.identities)
  
  def _sort_by_targets_len(self):
    ids_lenCaption = [(k, len(v[0])) for k, v in self.ids_to_captionsImagesAudios.items()]
    self.identities = list(map(lambda x: x[0], sorted(ids_lenCaption, key=lambda x: x[1])))
  
  def __getitem__(self, idx):
    caption, image_file, audio_file, audio_len = self.ids_to_captionsImagesAudios[self.identities[idx]]
    signal, _ = Data.read_audio_file(audio_file)
    target = torch.LongTensor(caption)
    target_len = len(target)
    audio = torch.Tensor(signal)
    image = self.preprocess_img(Image.open(image_file))  # [3, 224, 224]
    return audio, image, target, target_len, audio_len


class AudioVisualCollator(object):
  def __init__(self, audio_pad, text_pad):
    self.audio_pad = audio_pad
    self.text_pad = text_pad
  
  def __call__(self, batch):
    audios, images, targets, targets_lens, audios_lens = zip(*batch)
    audios_batch = pad_sequence(audios, batch_first=True, padding_value=self.audio_pad).float()
    images_batch = torch.stack(images)
    targets_batch = pad_sequence(targets, batch_first=True, padding_value=self.text_pad)
    targets_lens = torch.LongTensor(targets_lens)
    audios_lens = torch.LongTensor(audios_lens)
    return audios_batch, images_batch, targets_batch, targets_lens, audios_lens


class AudioVisualModel(nn.Module):
  def __init__(self, output_size, **kwargs):
    super().__init__()
    self.visual_backbone = resnet50(pretrained=True)
    self.visual_backbone.avgpool = nn.Identity()
    self.visual_backbone.fc = nn.Identity()
    self.visual_backbone_proj = nn.Linear(2048, 512)
    self.audio_backbone = Encoder(config=get_encoder_config(config='conv_attention'), input_proj='base', wav2vec_frontend=True)
    self.transcriptor = TransformerDecoder(kwargs.get('n_blocks', 4), kwargs.get('d_model', 512), kwargs.get('d_keys', 64),
                                           kwargs.get('d_values', 64), kwargs.get('n_heads', 8), kwargs.get('d_ff', 1024),
                                           dropout=kwargs.get('dropout', 0.25))
    self.output_proj = nn.Linear(kwargs.get('d_model', 512), output_size)
  
  def forward(self, img, audio):
    for i, (_, layer) in enumerate(self.visual_backbone.named_children()):
      visual_output = layer(img) if i == 0 else layer(visual_output)
    visual_output = self.visual_backbone_proj(visual_output.reshape(visual_output.size(0), visual_output.size(1), -1).permute(0, 2, 1))
    audio_output = self.audio_backbone(audio)
    out = self.transcriptor(audio_output, visual_output, futur_masking=False)
    return self.output_proj(out)


class AudioVisualTrainer(object):
  '''
  From the COCO dataset we retrieve natural image and their annotations,
    we use tacotron2+waveglow to get audio signal from the image captions (see process_coco.py)
    We creates a network that will accept an image and an audio signal and ask it to produce the transcription
    By associating an Image with the sound, we hope that the network will better learn how to distinguish tokens
    from spoken language and be able to reproduce more faithfully the correct transcription
    We will have a Visual-Backbone e.g. a ResNet50 and a Audio-Backbone, the two will join to a network that will
    produce the transcription
    If we see this step as a powerful pretraining for Speech-to-Text task, we can then use the Audio-Backbone and
    fine-tune it on openSLR dataset, see if it helps to improve performance

    => To get input_len as required by CTC loss
    import os;import sys;import torch;import pickle as pk;from tqdm import tqdm;from data import Data
    sys.path.append('../')
    from models.final_net import Encoder;from models.final_net_configs import get_encoder_config
    save_folder = 'train2014_wav_lens/'
    if not os.path.isdir(save_folder):
      os.makedirs(save_folder)
    net = Encoder(config=get_encoder_config(config='conv_attention'), input_proj='base', wav2vec_frontend=True)
    folder = '../../../datasets/coco/train2014_wav/'
    files = os.listdir(folder)
    for f in tqdm(files):
      signal, _ = Data.read_audio_file(os.path.join(folder, f))
      out = net(torch.Tensor(signal).reshape(1, -1))
      with open(os.path.join(save_folder, f.replace('.wav', '.pk')), 'wb') as f:
        pk.dump(out.shape[1], f)
    
    => To clean captions
    import json;import text_cleaner as tc (.py file in work/datasets/coco/)
    with open('annotations/captions_train2014.json', 'r') as f:
      data = json.load(f)
    sources = [el['caption'] for el in data['annotations']]
    sources_cleaned = [tc.english_cleaners(s) for s in sources]
    letters = " 'abcdefghijklmnopqrstuvwxyz"
    sources_cleaned = [''.join([l for l in s if l in letters]) for s in sources_cleaned]
    sources_cleaned = [s.strip() for s in sources_cleaned]
    new_annotations = []
    for i, a in enumerate(data['annotations']):
      a['caption'] = sources_cleaned[i]
      new_annotations.append(a)
    data['annotations'] = new_annotations
    with open('annotations/captions_train2014_clean.json', 'w') as f:
      json.dump(data, f)
  '''
  def __init__(self, device=None, logfile='_logs/_audio_visual_trainer_logs.txt', save_name_model='convnet/audio_visual_model.pt',
               captions_file='../../../datasets/coco/annotations/captions_train2014_clean.json', batch_size=32, lr=1e-4,
               images_folder='../../../datasets/coco/train2014/', audios_folder='../../../datasets/coco/train2014_wav/',
               audioLens_folder='../../../datasets/coco/train2014_wav_lens/', lr_scheduling=True, eval_step=1, n_epochs=1000,
               encoding_fn=Data.letters_encoding, metadata_file='_audio_visual_metadata.pk', **kwargs):
    logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.captions_file = captions_file
    self.images_folder = images_folder
    self.audios_folder = audios_folder
    self.audioLens_folder = audioLens_folder
    self.encoding_fn = encoding_fn
    self.metadata_file = metadata_file
    self.batch_size = batch_size
    self.n_epochs = n_epochs
    self.eval_step = eval_step
    self.save_name_model = save_name_model
    self.lr_scheduling = lr_scheduling

    self.set_data()
    self.set_data_loader()

    self.model = self.instanciate_model()
    self.model = nn.DataParallel(self.model)

    u.dump_dict({'batch_size': batch_size, 'lr': lr, 'lr_scheduling': lr_scheduling}, 'CTCModel Hyperparameters')
    logging.info(self.model)
    logging.info(f'The model has {u.count_trainable_parameters(self.model):,} trainable parameters')

    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.criterion = nn.CTCLoss(zero_infinity=True)

    if lr_scheduling:
      patience, min_lr, threshold = kwargs.get('patience', 75), kwargs.get('min_lr', 1e-5), kwargs.get('lr_threshold', 0.003)
      self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=patience, verbose=True,
                                                                min_lr=min_lr, threshold_mode='abs', threshold=threshold)

  def set_data(self):
    if os.path.isfile(self.metadata_file):
      with open(self.metadata_file, 'rb') as f:
        self.idx_to_tokens, self.tokens_to_idx, self.train_data, self.test_data = pk.load(f)
    else:
      with open(self.captions_file, 'r') as f:
        captions = json.load(f)
      ids_captions = [(f"{capt['image_id']}-{capt['id']}", capt['caption']) for capt in captions['annotations']]
      captions_encoded, idx_to_tokens, tokens_to_idx = self.encoding_fn(list(map(lambda x: x[1], ids_captions)),
                                                                        add_sos_eos_pad_tokens=False)
      captions_encoded, self.idx_to_tokens, self.tokens_to_idx = Data.add_blank_token(encodedsources=captions_encoded,
                                                                                      idx_to_tokens=idx_to_tokens,
                                                                                      tokens_to_idx=tokens_to_idx)
      ids_to_caption = {ids_captions[i][0]: ce for i, ce in enumerate(captions_encoded)}

      ids_to_images = {re.search(r'_0+(\d+).', f).group(1): f for f in os.listdir(self.images_folder)}

      ids_to_audioLens = {f.split('.')[0]: pk.load(open(os.path.join(self.audioLens_folder, f), 'rb'))
                            for f in os.listdir(self.audioLens_folder)}

      data = {f.split('.')[0]: (ids_to_caption[f.split('.')[0]],
                                os.path.join(self.images_folder, ids_to_images[f.split('-')[0]]),
                                os.path.join(self.audios_folder, f),
                                ids_to_audioLens[f.split('.')[0]])
                                  for f in os.listdir(self.audios_folder)}

      train_ids, test_ids = train_test_split(list(data.keys()), test_size=0.1, shuffle=True)
      self.train_data = {i: data[i] for i in train_ids}
      self.test_data = {i: data[i] for i in test_ids}

      with open(self.metadata_file, 'wb') as f:
        pk.dump([self.idx_to_tokens, self.tokens_to_idx, self.train_data, self.test_data], f)
  
  def set_data_loader(self):
    train_dataset = AudioVisualDataset(self.train_data)
    test_dataset = AudioVisualDataset(self.test_data)

    collator = AudioVisualCollator(0, 0)

    self.train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=8, collate_fn=collator,
                                        shuffle=True, pin_memory=True)
    self.test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=8, collate_fn=collator,
                                       shuffle=True, pin_memory=True)
  
  def instanciate_model(self):
    return AudioVisualModel(len(self.idx_to_tokens)).to(self.device)
  
  def train(self):
    print('Start Training...')
    eval_accuracy_memory = 0
    for epoch in tqdm(range(self.n_epochs)):
      epoch_loss, accs = self.train_pass(only_loss=epoch % self.eval_step != 0)
      logging.info(f"Epoch {epoch} | train_loss = {epoch_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")
      eval_loss, accs = self.evaluation(only_loss=epoch % self.eval_step != 0)
      logging.info(f"Epoch {epoch} | test_loss = {eval_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")

      oea = accs.get('word_accuracy', None)

      if self.lr_scheduling and oea is not None:
        self.lr_scheduler.step(oea)

      if oea is not None and oea > eval_accuracy_memory:
        logging.info(f'Save model with eval_accuracy = {oea:.3f}')
        u.save_checkpoint(self.model, None, self.save_name_model)
        eval_accuracy_memory = oea
  
  def train_pass(self, only_loss=True):
    losses, accs = 0, {}
    all_targets, all_preds = [], []

    for audios, images, targets, targets_lens, audios_lens in tqdm(self.train_data_loader):
      audios, images, targets = audios.to(self.device), images.to(self.device), targets.to(self.device)
      targets_lens, audios_lens = targets_lens.to(self.device), audios_lens.to(self.device)
      
      preds = self.model(images, audios)  # [batch_size, seq_len, output_dim]

      self.optimizer.zero_grad()
      current_loss = self.criterion(preds.permute(1, 0, 2).log_softmax(-1), targets, audios_lens, targets_lens)
      current_loss.backward()
      self.optimizer.step()

      all_targets += targets.tolist()
      all_preds += preds.argmax(dim=-1).tolist()

      losses += current_loss.item()
    
    if not only_loss:
      accs = Data.ctc_scorer(all_targets, all_preds, self.idx_to_tokens, self.tokens_to_idx)
    
    return losses / len(self.train_data_loader), accs
  
  @torch.no_grad()
  def evaluation(self, only_loss=True):
    losses, accs = 0, {}
    all_targets, all_preds = [], []

    self.model.eval()

    for audios, images, targets, targets_lens, audios_lens in tqdm(self.test_data_loader):
      audios, images, targets = audios.to(self.device), images.to(self.device), targets.to(self.device)
      targets_lens, audios_lens = targets_lens.to(self.device), audios_lens.to(self.device)

      preds = self.model(images, audios)  # [batch_size, seq_len, output_dim]

      losses += self.criterion(preds.permute(1, 0, 2).log_softmax(-1), targets, audios_lens, targets_lens).item()

      all_targets += targets.tolist()
      all_preds += preds.argmax(dim=-1).tolist()
    
    self.model.train()

    if not only_loss:
      accs = Data.ctc_scorer(all_targets, all_preds, self.idx_to_tokens, self.tokens_to_idx)

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
  
  rep = input('Train AudioVisualModel? (y or n): ')
  if rep == 'y':
    avt = AudioVisualTrainer()
    avt.train()