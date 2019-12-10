import io
import os
import sys
import torch
import librosa
import argparse
import numpy as np
import pickle as pk
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from scipy.signal import stft
from collections import defaultdict, namedtuple
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
# from python_speech_features.base import logfbank

try:
  import utils as u
except:
  sys.path.append(os.path.abspath(__file__).replace('ASR/data.py', ''))
  import utils as u


def get_list_files(folder):
  '''
  Specifically designed for openSLR tree format.
  
  Params:
    * folder : str
  '''
  dataset_filelist = {'audio': [], 'features': [], 'transcript': []}

  for fname in os.listdir(folder):
    if os.path.isdir(folder + fname):
      for f2name in os.listdir(folder + fname):
        if os.path.isdir(f'{folder}{fname}/{f2name}'):
          for f3name in os.listdir(f'{folder}{fname}/{f2name}'):
            filename = f'{folder}{fname}/{f2name}/{f3name}'
            if '.flac' in f3name:
              dataset_filelist['audio'].append(filename)
            elif '.npy' in f3name:
              dataset_filelist['features'].append(filename)
            elif '.txt' in f3name:
              dataset_filelist['transcript'].append(filename)

  return dataset_filelist


def preprocess_audio_data(folder, dct_type=2, extract_type='mfcc', win_size=0.025):
  '''
  Preprocess audio data from given folder.
  It will calls get_list_files to retrieves all .flac files from given folder then
  extract informations from audio data (mfcc/log_spectrogram/filterbank) to finally
  save it in a new file with the same name but the extension .features.npy

  Params:
    * folder : str
    * dct_type (optional) : int
    * extract_type (optional) : str, possible values = mfcc/log_spectrogram/filterbank/raw
    * win_size (optional) : float, default to 0.025ms
  '''
  print(f'Preprocess audio files in folder: {folder}')
  extract_types = ['mfcc', 'log_spectrogram', 'filterbank', 'raw']

  assert extract_type in extract_types, f'extract_type must be one of the following: {extract_types}'

  print(f'extract_type: {extract_type}')

  filelist = get_list_files(folder)

  for filename in tqdm(filelist['audio']):
    signal, sample_rate = librosa.load(filename, sr=16000)
    hop_length = int(0.010 * sample_rate)
    n_fft = int(win_size * sample_rate)

    if extract_type == 'mfcc':
      # by default librosa mfcc have -> n_fft = 2048 | hop_length = 512
      # default window_size = n_fft / sample_rate = 0.128 ms
      # default stride = hop_length / sample_rate = 0.032 ms
      # to compute mfccs with a 25ms window and shifted every 10ms
      # add args hop_length=int(0.010*sample_rate), n_fft=int(0.025*sample_rate)
      # features = librosa.feature.mfcc(y=signal, sr=sample_rate, dct_type=dct_type, n_mfcc=80).T
      features = librosa.feature.mfcc(y=signal, sr=sample_rate, dct_type=dct_type, n_mfcc=80,
                                      hop_length=hop_length, n_fft=n_fft).T
    elif extract_type == 'log_spectrogram':
      features = log_spectrogram(signal, sample_rate).T
    elif extract_type == 'filterbank':
      #  80-dimensional log-Mel filterbank features, computed with a 25ms window and shifted every 10ms
      # features = logfbank(signal, nfilt=80, samplerate=16000, winlen=0.025, winstep=0.01)
      features = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=80, n_fft=n_fft, hop_length=hop_length)
      features = np.log(features + 1e-6)
    else:
      # raw cutting
      window = int(win_size * sample_rate)
      features = np.split(signal, list(range(window, len(signal), window)))
      features[-1] = np.concatenate((features[-1], np.zeros(window - len(features[-1]))), 0)
      features = np.array(features)

    np.save(filename.replace('.flac', '.features.npy'), features)


def log_spectrogram(wav, sample_rate):
  freqs, times, spec = stft(wav, sample_rate, nperseg=400, noverlap=240, nfft=512, padded=False, boundary=None)
  return np.log(np.abs(spec)+1e-10)


def load_metadata(path_to_file):
  '''path_to_file : str'''
  Token = namedtuple('Token', ['index', 'word'])

  with open(path_to_file, 'rb') as f:
    metadata = pk.load(f)

  metadata['SOS'] = Token(**metadata['SOS'])
  metadata['EOS'] = Token(**metadata['EOS'])
  metadata['PAD'] = Token(**metadata['PAD'])

  return metadata


def normalize_and_save(folder, level=0):
  '''
  Feature are normalized via mean subtraction and variance normalization on the speaker basis.
  Will loads the .features.npy files from given folder then normalize the data to finally save it inplace.

  Params:
    * level : 0, 1 or None, 0 to scale on feature level, 1 to scale on time level, None for global scale
  '''
  print('Normalize data in folder: {}'.format(folder))
  filelist = get_list_files(folder)

  files_per_speaker = defaultdict(list)

  for filename in filelist['features']:
    files_per_speaker[filename.split('/')[-3]].append(filename)

  for files in tqdm(files_per_speaker.values()):
    features = []

    for filename in files:
      features.append(np.load(filename))

    speaker_features = np.concatenate(features)

    speaker_mean = np.mean(speaker_features, axis=0)
    speaker_std = np.std(speaker_features, axis=0)

    normalized_features = [(x - speaker_mean) / speaker_std for x in features]
    
    for filename, data in zip(files, normalized_features):
      np.save(filename, data)


def get_transcript(transcript_files):
  '''
  Parses .txt files to retrieves the transcripts

  Params:
    * transcript_files : list of str
  '''
  transcripts = {}

  for filename in transcript_files:
    with open(filename, 'r') as f:
      transcript = f.read().splitlines()

    ids_sentences = [line.split(' ', 1) for line in transcript]
    new_transcripts = {id_s: sentence for id_s, sentence in ids_sentences}
    transcripts = {**transcripts, **new_transcripts}

  return transcripts


def get_documents(folder):
  filelist = get_list_files(folder)
  transcripts = get_transcript(filelist['transcript'])
  documents = list(transcripts.values())
  return documents


def get_metadata(documents):
  Token = namedtuple('Token', ['index', 'word'])
  SOS = Token(0, '<sos>')
  EOS = Token(1, '<eos>')
  PAD = Token(2, '<pad>')
  
  letters = [l for d in documents for l in d.lower()]
  vocabulary = list(sorted(set(letters)))
  idx_2_letter = [SOS.word, EOS.word, PAD.word] + vocabulary
  letter_2_idx = {w: i for i, w in enumerate(idx_2_letter)}

  return SOS, EOS, PAD, idx_2_letter, letter_2_idx


def preprocess_text_data(folder, save=True, pad_docs=False, save_name=None):
  '''
  Reads transcripts from .txt files in given folder then creates
  following variables:
    * SOS : namedtuple
    * EOS : namedtuple
    * PAD : namedtuple
    * idx_2_letter : list
    * letter_2_idx : dict
    * id_2_doc : dict

  Params:
    * folder : str
    * save (optional) : Boolean, default to True
    * pad_docs (optional) : Boolean, default to False
    * save_name (optional) : str, default to None
  '''
  filelist = get_list_files(folder)
  transcripts = get_transcript(filelist['transcript'])
  documents = list(transcripts.values())

  SOS, EOS, PAD, idx_2_letter, letter_2_idx = get_metadata(documents)

  documents = [[SOS.word] + list(d.lower()) + [EOS.word] for d in documents]

  max_len_doc = max(map(len, documents))

  if pad_docs:
    for d in documents:
        d.extend([PAD.word] * (max_len_doc - len(d)))
  
  documents = [[letter_2_idx[l] for l in d] for d in documents]

  id_2_doc = {identity: doc for identity, doc in zip(transcripts, documents)}

  if not save:
    return SOS, EOS, PAD, idx_2_letter, letter_2_idx, id_2_doc

  metadata = {'SOS': SOS._asdict(), 'EOS': EOS._asdict(), 'PAD': PAD._asdict(),
              'idx_2_letter': idx_2_letter, 'letter_2_idx': letter_2_idx, 'id_2_doc': id_2_doc}

  if save_name is None:
    save_name = f"metadata_{folder.split('/')[-2]}.pk"

  with open(save_name, 'wb') as f:
    pk.dump(metadata, f)


def dataset_generator(folder, batch_size=32, metadata=None):
  '''
  Custom generator that reads data from folder then yield batches

  Params:
    * folder : str
    * batch_size (optional) : int, default to 32
    * metadata (optional) : dict, default to None
  '''
  filelist = get_list_files(folder)

  save_name = '{}_{}.pk'.format('metadata', folder.split('/')[-2])

  if metadata is None:
    metadata = load_metadata(save_name)

  batch = [[], []]
  num_els = len(filelist['features'])

  for i, filename in enumerate(filelist['features']):
    identity = filename.split('/')[-1].replace('.features.npy', '')

    encoder_input = torch.tensor(np.load(filename))
    decoder_input = metadata['id_2_doc'][identity]

    batch[0].append(encoder_input)
    batch[1].append(decoder_input)

    if len(batch[0]) == batch_size or i == num_els - 1:
      encoder_input_batch = pad_sequence(batch[0], batch_first=True, padding_value=metadata['PAD'].index)
      decoder_input_batch = torch.LongTensor(batch[1])
      padding_mask_batch = u.create_padding_mask(encoder_input_batch, metadata['PAD'].index)

      yield encoder_input_batch, decoder_input_batch, padding_mask_batch
      batch = [[], []]


class CustomDataset(Dataset):
  '''
  Currently, if you set size_limits to True, check value of enc_seq_len_max, argument of check_size_limits function 
  '''
  def __init__(self, folder, metadata=None, save_metadata='metadata.pk', ordering=False, size_limits=False):
    '''
    Params:
      * folder : str, folder where the data are
      * metadata (optional) : dict
      * save_metadata (optional) : str, path to save retrieves metadata when arg metadata is None
      * ordering (optional) : Boolean, default to False, orders data by length if True
      * size_limits (optional) : Boolean, default to False, removes data where length > size_limits if True
    '''
    filelist = get_list_files(folder)
    self.metadata = load_metadata(save_metadata) if metadata is None else metadata
    self.filelist = self.check_size_limits(filelist) if size_limits else filelist
    self.filelist = self.ordering_data(self.filelist, self.metadata) if ordering else self.filelist
  
  def ordering_data(self, filelist, metadata):
    '''
    Orders data by length

    Params:
      * filelist : list
      * metadata : dict

    Returns:
      * list
    '''
    identities = [f.split('/')[-1].replace('.features.npy', '') for f in filelist['features']]
    docs = [metadata['id_2_doc'][fid] for fid in identities]
    doc_lens = [len(d[:d.index(metadata['EOS'].index)]) for d in docs]
    data = list(zip(filelist['features'], doc_lens))
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    ordered_filelist_feats = list(map(lambda x: x[0], sorted_data))
    filelist['features'] = ordered_filelist_feats
    return filelist
  
  def check_size_limits(self, filelist, enc_seq_len_max=1600, filename='_tmp_check_size_limits.pk'):
    '''
    Removes too big documents.

    Params:
      * filelist : list
      * enc_seq_len_max (optional) : int, default to 1600
      * filename (optional) : str
    
    Returns:
      * list
    '''
    if os.path.isfile(filename):
      with open(filename, 'rb') as f:
        new_features = pk.load(f)
    else:
      new_features = []

      for f in filelist['features']:
        data = np.load(f)

        if data.shape[0] < enc_seq_len_max:
          new_features.append(f)
      
      with open(filename, 'wb') as f:
        pk.dump(new_features, f)

    filelist['features'] = new_features

    return filelist
  
  def __len__(self):
    return len(self.filelist['features'])
  
  def __getitem__(self, idx):
    filename = self.filelist['features'][idx]
    identity = filename.split('/')[-1].replace('.features.npy', '')
    encoder_input = torch.tensor(np.load(filename))
    decoder_input = torch.LongTensor(self.metadata['id_2_doc'][identity])

    return encoder_input, decoder_input


class PrecodingDataset(Dataset):
  '''
  pytorch Dataset module for data using CustomPrecoding (ie data prepared as n-grams instead of chars)
  '''
  def __init__(self, training=True, filename='metadata_custom_precoding.pk', create_mask=True):
    '''
    Params:
      * training (optional) : Boolean, default to True, retrieves training data if True else test data
      * filename (optional) : str, filename where metadata where saved or will saved
      * create_mask (optional) : Boolean, default to True
    '''
    self.cp = CustomPrecoding(None, None, filename=filename, create_mask=create_mask)

    if training:
      self.dataset_size = len(self.cp.id_2_ed_train)
      self.identities, self.outputs = zip(*list(self.cp.id_2_ed_train.items()))
      self.inputs = [self.cp.id_2_filepath_train[identity] for identity in self.identities]
    else:
      self.dataset_size = len(self.cp.id_2_ed_test)
      self.identities, self.outputs = zip(*list(self.cp.id_2_ed_test.items()))
      self.inputs = [self.cp.id_2_filepath_test[identity] for identity in self.identities]
  
  def __len__(self):
    return self.dataset_size

  def __getitem__(self, idx):
    encoder_input = torch.tensor(np.load(self.inputs[idx]))
    decoder_input = self.outputs[idx]
    return encoder_input, decoder_input


class CustomCollator(object):
  '''
  To use with torch DataLoader module, will pad batch data prepared by Dataset Module 
  '''
  def __init__(self, metadata=None, save_metadata='metadata.pk', create_mask=True):
    '''
    Params:
      * metadata (optional) : dict, will load metadata from save_metadata if not provided
      * save_metadata (optional) : str, path to metadata file to load
      * create_mask (optional) : Boolean, default to True
    '''
    metadata = load_metadata(save_metadata) if metadata is None else metadata
    self.pad_idx = metadata['PAD'].index
    self.create_mask = create_mask

  def __call__(self, batch):
    encoder_inputs, decoder_inputs = zip(*batch)
    encoder_input_batch = pad_sequence(encoder_inputs, batch_first=True, padding_value=self.pad_idx)
    decoder_input_batch = pad_sequence(decoder_inputs, batch_first=True, padding_value=self.pad_idx)

    if self.create_mask:
      padding_mask_batch = u.create_padding_mask(encoder_input_batch, self.pad_idx)
      return encoder_input_batch, decoder_input_batch, padding_mask_batch

    return encoder_input_batch, decoder_input_batch


def get_dataset_generator(folder, metadata, batch_size, num_workers=4, ordering=False, shuffle=True, subset=False,
                          percent=0.2, size_limits=False, create_mask=True):
  '''
  Params:
    * folder : str, folder where data are
    * metadata : dict
    * batch_size : int
    * num_workers (optional) : int
    * ordering (optional) : Boolean, default to False
    * shuffle (optional) : Boolean, default to True
    * subset (optional) : Boolean, default to False
    * percent (optional) : float, default to 0.2
    * size_limits (optional) : Boolean, default to False
    * create_mask (optional) : Boolean, default to True
  
  Returns:
    * generator, instance of torch.utils.data.DataLoader
  '''
  custom_dataset = CustomDataset(folder, metadata=metadata, ordering=ordering, size_limits=size_limits)

  if subset:
    custom_dataset = u.extract_subset(custom_dataset, percent=percent)
    
  custom_collator = CustomCollator(metadata=metadata, create_mask=create_mask)
  
  return DataLoader(custom_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collator, shuffle=shuffle)


def get_dataset_generator_CP(training=True, batch_size=32, num_workers=4, shuffle=True, subset=False, percent=0.2,
                             filename='metadata_custom_precoding.pk', create_mask=True):
  '''
  Retrieves generator on precoded data (n-grams instead of chars)

  Params:
    * training : Boolean, default to True, load training data if True else test data
    * batch_size : int, default to 32
    * num_workers (optional) : int, default to 4
    * shuffle (optional) : Boolean, default to True
    * subset (optional) : Boolean, default to False
    * percent (optional) : float, default to 0.2
    * filename (optional) : str, name of file where metadata are saved
    * create_mask (optional) : Boolean, default to True
  
  Returns:
    * generator, instance of torch.utils.data.DataLoader
  '''
  precoding_dataset = PrecodingDataset(training=training, filename=filename, create_mask=create_mask)

  custom_collator = precoding_dataset.cp.custom_collator

  if subset:
    precoding_dataset = u.extract_subset(precoding_dataset, percent=percent)
  
  return DataLoader(precoding_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collator, shuffle=shuffle)


class CustomPrecoding(object):
  def __init__(self, train_folder, test_folder, sos='<sos>', eos='<eos>', pad='<pad>', max_ngrams=3,
               filename='metadata_custom_precoding.pk', create_mask=True):
    '''
    Followings variables will be available after instanciation:
      * idx_2_vocab
      * vocab_2_idx
      * id_2_ed_train
      * id_2_ed_test
      * id_2_filepath_train
      * id_2_filepath_test

    Params:
      * train_folder : str
      * test_folder : str
      * sos (optional) : str
      * eos (optional) : str
      * pad (optional) : str
      * max_ngrams (optional) : int, maximum n-grams to retrieve from words, if n=3 it will retrieve trigrams and bigrams
      * filename (optional) : str, name of the file to retrieves metadata from if train_folder & test_folder not provided
      * create_mask (optional) : Boolean, default to True
    '''
    self.sos, self.eos, self.pad = sos, eos, pad
    self.max_ngrams = max_ngrams
    self.create_mask = create_mask

    if train_folder is None and test_folder is None:
      self.load_metadata(filename=filename)
    else:
      train_filelist = get_list_files(train_folder)
      test_filelist = get_list_files(test_folder)

      self.id_2_filepath_train = {f.split('/')[-1].replace('.features.npy', ''): f for f in train_filelist['features']}
      self.id_2_filepath_test = {f.split('/')[-1].replace('.features.npy', ''): f for f in test_filelist['features']}

      self.train_transcripts = {k: v.lower() for k, v in get_transcript(train_filelist['transcript']).items()}
      self.test_transcripts = {k: v.lower() for k, v in get_transcript(test_filelist['transcript']).items()}

      self.build_vocabulary()
  
  def build_vocabulary(self, size_per_ngrams={3: 270, 2: 180}):
    '''
    Params:
      size_per_ngrams (optional) : dict, keys correspond to a n-gram and values are the number of n-grams to retained
    '''
    train_documents = list(self.train_transcripts.values())
    test_documents = list(self.test_transcripts.values())

    words = [w for d in train_documents + test_documents for w in d.split(' ')]

    freq_ngrams, letters = u.process_ngrams(words, max_ngrams=self.max_ngrams)

    self.idx_2_vocab = []
    for ngram, size in size_per_ngrams.items():
      self.idx_2_vocab += sorted(list(map(lambda x: x[0], freq_ngrams[ngram].most_common(size))))

    self.idx_2_vocab += sorted(letters)
    self.idx_2_vocab += [' ', self.sos, self.eos, self.pad]

    self.vocab_2_idx = {v: i for i, v in enumerate(self.idx_2_vocab)}
  
  def encode_train_test_documents(self):
    # id = identity, ed = encoded document
    self.id_2_ed_train = {k: encode_document(d, self.vocab_2_idx, sos=self.sos, eos=self.eos)
                          for k, d in tqdm(self.train_transcripts.items())}
    self.id_2_ed_test = {k: encode_document(d, self.vocab_2_idx, sos=self.sos, eos=self.eos)
                         for k, d in tqdm(self.test_transcripts.items())}
  
  def decode_document(self, doc):
    return ''.join([self.idx_2_vocab[i] for i in doc])
  
  def decode_documents(self, docs):
    return [self.decode_document(d) for d in tqdm(docs)]
  
  def save_metadata(self, filename='metadata_custom_precoding.pk'):
    metadata = {'idx_2_vocab': self.idx_2_vocab,
                'vocab_2_idx': self.vocab_2_idx,
                'id_2_ed_train': self.id_2_ed_train,
                'id_2_ed_test': self.id_2_ed_test,
                'id_2_filepath_train': self.id_2_filepath_train,
                'id_2_filepath_test': self.id_2_filepath_test}

    with open(filename, 'wb') as f:
      pk.dump(metadata, f)
  
  def load_metadata(self, filename='metadata_custom_precoding.pk'):
    if os.path.isfile(filename):
      with open(filename, 'rb') as f:
        metadata = pk.load(f)

      self.idx_2_vocab = metadata['idx_2_vocab']
      self.vocab_2_idx = metadata['vocab_2_idx']
      self.id_2_ed_train = metadata['id_2_ed_train']
      self.id_2_ed_test = metadata['id_2_ed_test']
      self.id_2_filepath_train = metadata['id_2_filepath_train']
      self.id_2_filepath_test = metadata['id_2_filepath_test']

      print('Metadata succesfully loaded.')
    else:
      print(f'File {filename} not found.')
  
  def custom_collator(self, batch):
    encoder_inputs, decoder_inputs = zip(*batch)
    encoder_input_batch = pad_sequence(encoder_inputs, batch_first=True, padding_value=self.vocab_2_idx[self.pad])
    decoder_input_batch = torch.LongTensor(u.pad_documents(decoder_inputs, self.vocab_2_idx[self.pad]))

    if self.create_mask:
      padding_mask_batch = u.create_padding_mask(encoder_input_batch, self.vocab_2_idx[self.pad])
      return encoder_input_batch, decoder_input_batch, padding_mask_batch

    return encoder_input_batch, decoder_input_batch


def preprocess_text_data_custom_precoding(settings, save_path='metadata_custom_precoding.pk'):
  '''
  Preprocess text data from given train_folder & test_folder as n-grams then save
  metadata created in save_path.

  Params:
    * settings : dict, must contains following keys: train_folder, test_folder
    * save_path (optional) : str
  '''
  cpc = CustomPrecoding(settings['train_folder'], settings['test_folder'])

  if os.path.isfile(save_path):
    rep = input(f'File {save_path} already exist, do you want to override it? (y or n): ')

    if rep == 'y':
      cpc.load_metadata()
  else:
    cpc.encode_train_test_documents()
    cpc.save_metadata()


def encode_document(doc, vocab_2_idx, sos='<sos>', eos='<eos>'):
  '''
  Encodes a document (string) based on the given mapping (vocab_2_idx)

  Params:
    * doc : string, document to encode, it suppose that the token separator is a space
    * vocab_2_idx : dictionary, string to index
    * sos (optional) : string, Start Of Sentence token
    * eos (optional) : string, End Of Sentence token
  
  Returns:
    * doc_encoded : list of int
  '''
  doc_encoded = []
  for w in doc.split(' '):
    # handle trigrams
    encoded = u.process_word(w, vocab_2_idx)
    # handle bigrams
    for i, wp in enumerate(encoded):
      if not wp[1]:
        encoded[i] = u.process_word(wp[0], vocab_2_idx, n=2)
    
    encoded = u.flat_list(encoded)
    # handle unigrams
    encoded = [[(c, True) for c in wp[0]] if not wp[1] else wp for wp in encoded]
    encoded = u.flat_list(encoded)

    doc_encoded += [vocab_2_idx[wp] for wp, _ in encoded]
    doc_encoded.append(vocab_2_idx[' '])

  doc_encoded = [vocab_2_idx[sos]] + doc_encoded[:-1] + [vocab_2_idx[eos]]
  return doc_encoded


def encode_documents(docs, vocab_2_idx, sos='<sos>', eos='<eos>'):
  return [encode_document(d, vocab_2_idx, sos=sos, eos=eos) for d in tqdm(docs)]


class Metadata(object):
  '''
  Handles all metadata for ASR project
  '''
  def __init__(self, train_folder='datasets/openslr/LibriSpeech/train-clean-100/',
                     test_folder='datasets/openslr/LibriSpeech/test-clean/',
                     train_metadata='transformer/metadata_train-clean-100.pk',
                     test_metadata='transformer/metadata_test-clean.pk',
                     ngram_metadata='transformer/metadata_custom_precoding.pk',
                     vocab='unigram',
                     decay_step=0.01,
                     subset=False,
                     percent=0.2,
                     batch_size=32,
                     size_limits=False,
                     create_mask=True,
                     loss='attention'):
    '''
    The followings variable are available after instanciate this class:
      * train_folder                      * device
      * test_folder                       * SM (score master)
      * train_metadata                    * loss
      * test_metadata                     * idx_2_vocab
      * n_gram_metadata                   * output_size
      * pad_idx                           * sos_idx
      * eos_idx
      * metadata                          * test_metadata (only if vocab == unigram)
      * train_data_loader                 * test_data_loader
    '''
    self.train_folder = train_folder
    self.test_folder = test_folder
    self.train_metadata = train_metadata
    self.test_metadata = test_metadata
    self.ngram_metadata = ngram_metadata
    self.vocab = vocab
    self.decay_step = decay_step
    self.subset = subset
    self.percent = percent
    self.batch_size = batch_size
    self.size_limits = size_limits
    self.create_mask = create_mask

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.load_metadata_vars()

    self.SM = u.ScoresMaster(idx_2_letter=self.idx_2_vocab, pad_idx=self.pad_idx)

    if loss == 'attention':
      self.loss = u.AttentionLoss(self.pad_idx, self.device, decay_step=self.decay_step)
    else:
      self.loss = u.CrossEntropyLoss(self.pad_idx)

    self.load_dataloader()
  
  def load_metadata_vars(self):
    if self.vocab == 'unigram':
      self.metadata = load_metadata(self.train_metadata)
      self.test_metadata = load_metadata(self.test_metadata)
    else:
      self.metadata = u.load_pickle(self.ngram_metadata)
    
    self.idx_2_vocab = self.metadata.get('idx_2_letter', self.metadata.get('idx_2_vocab', None))
    self.output_size = len(self.idx_2_vocab)

    self.pad_idx = self.metadata['PAD'].index if self.vocab == 'unigram' else self.metadata['vocab_2_idx']['<pad>']
    self.sos_idx = self.metadata['SOS'].index if self.vocab == 'unigram' else self.metadata['vocab_2_idx']['<sos>']
    self.eos_idx = self.metadata['EOS'].index if self.vocab == 'unigram' else self.metadata['vocab_2_idx']['<eos>']
  
  def load_dataloader(self):
    if self.vocab == 'unigram':
      self.train_data_loader = get_dataset_generator(self.train_folder,
                                                     self.metadata,
                                                     self.batch_size,
                                                     subset=self.subset,
                                                     percent=self.percent,
                                                     size_limits=self.size_limits,
                                                     create_mask=self.create_mask)
      self.test_data_loader = get_dataset_generator(self.test_folder,
                                                    self.test_metadata,
                                                    self.batch_size,
                                                    subset=self.subset,
                                                    create_mask=self.create_mask)
    else:
      self.train_data_loader = get_dataset_generator_CP(batch_size=self.batch_size,
                                                        subset=self.subset,
                                                        percent=self.percent,
                                                        filename=self.ngram_metadata,
                                                        create_mask=self.create_mask)
      self.test_data_loader = get_dataset_generator_CP(training=False,
                                                       batch_size=self.batch_size,
                                                       subset=self.subset,
                                                       filename=self.ngram_metadata,
                                                       create_mask=self.create_mask)


def plot_attention(data, show=True, cbar=True, verbose=True):
  '''
  Plots attention

  Params:
    * data : dictionary that must contain following keys:
                - pad, eos, target, enc_in, i2v, attention
  '''
  att = data['attention']
  n_feats = data['enc_in'].shape[-1]

  enc_in_pad = np.sum(data['enc_in'] == np.ones(n_feats) * 2, axis=-1).tolist()
  enc_in_seq_len = enc_in_pad.index(n_feats) if n_feats in enc_in_pad else len(enc_in_pad)

  dec_in = [el for el in data['target'][1:] if el != data['pad'] and el != data['eos']]
  dec_in_seq_len = len(dec_in)

  v2i = {v: i for i, v in enumerate(data['i2v'])}
  space_idx = [i for i, el in enumerate(dec_in) if el == v2i[' ']]
  
  if len(space_idx) == 0:
    return
    
  sentence0 = ''.join([data['i2v'][el] for el in dec_in])

  if verbose:
    print('target sentence -> ', sentence0)
    print('number of words -> ', len(sentence0.split()))

  att = data['attention'][:dec_in_seq_len, :enc_in_seq_len]

  att_word = [att[:space_idx[0]].mean(0)]
  att_word += [att[space_idx[i]+1:space_idx[i+1]].mean(0) for i in range(len(space_idx) - 1)]
  att_word += [att[space_idx[-1]+1:].mean(0)]
  att_word = np.stack(att_word)

  ax = sns.heatmap(att_word, yticklabels=sentence0.split(), cbar=cbar)#, linewidths=0.005)
  plt.xlabel('Audio signal')
  plt.ylabel('Target sentence')
  plt.title('Attention heatmap')
  plt.tight_layout()

  if show:
    plt.show()
  else:
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = np.asarray(Image.open(buf))
    return np.transpose(im, (2, 0, 1))


if __name__ == '__main__':
  argparser = argparse.ArgumentParser(prog='data.py', description='data utils')
  argparser.add_argument('--dct_type', default=2, type=int)
  argparser.add_argument('--extract_type', default='mfcc', type=str)
  argparser.add_argument('--train_folder', default='../../datasets/openslr/LibriSpeech/train-clean-100/', type=str)
  argparser.add_argument('--test_folder', default='../../datasets/openslr/LibriSpeech/test-clean/', type=str)
  args = argparser.parse_args()

  rep = input('Preprocess audio data? (y or n): ')
  if rep == 'y':
    preprocess_audio_data(args.train_folder, dct_type=args.dct_type, extract_type=args.extract_type)
    preprocess_audio_data(args.test_folder, dct_type=args.dct_type, extract_type=args.extract_type)
  
  rep = input('Normalize audio data? (y or n): ')
  if rep == 'y':
    normalize_and_save(args.train_folder)
    normalize_and_save(args.test_folder)
  
  rep = input('Preprocess text data (default)? (y or n): ')
  if rep == 'y':
    preprocess_text_data(args.train_folder)
    preprocess_text_data(args.test_folder)
  
  rep = input('Preprocess text data (Custom Precoding)? (y or n): ')
  if rep == 'y':
    preprocess_text_data_custom_precoding(settings)
