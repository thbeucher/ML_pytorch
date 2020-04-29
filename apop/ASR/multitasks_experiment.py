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

import os
import re
import sys
import json
import torch
import random
import logging
import librosa
import numpy as np
import pickle as pk
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from g2p_en import G2p
from scipy.signal import stft
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Subset

sys.path.append(os.path.abspath(__file__).replace('ASR/multitasks_experiment.py', ''))
import utils as u
import models.naive_conv as nc
import models.conv_seqseq as css

from optimizer import RAdam
from models.transformer.transformer import Transformer
from models.transformer.embedder import PositionalEmbedder

class CustomDataset(Dataset):
  def __init__(self, ids_to_audiofile, ids_to_encodedsources, signal_type='window-sliced', readers=[], process_file_fn=None, **kwargs):
    '''
    Params:
      * ids_to_audiofile : dict
      * ids_to_encodedsources : dict
      * signal_type (optional) : str
      * readers (optional) : list
      * process_file_fn (optional) : function
      * kwargs (optional) : arguments passed to process_file_fn
    '''
    self.ids_to_audiofile = ids_to_audiofile
    self.ids_to_encodedsources = ids_to_encodedsources
    self.signal_type = signal_type

    self.process_file_fn = process_file_fn = Data.read_and_slice_signal if process_file_fn is None else process_file_fn
    self.process_file_fn_args = kwargs

    self.identities = list(sorted(ids_to_audiofile.keys()))

    if len(readers) > 0:
      self.identities = [i for i in self.identities if i.split('-')[0] in readers]

  def __len__(self):
    return len(self.identities)
  
  def __getitem__(self, idx):
    identity = self.identities[idx]

    signal = self.process_file_fn(self.ids_to_audiofile[identity], **self.process_file_fn_args)

    if self.signal_type == 'std-threshold-selected':
      signal = Data.get_std_threshold_selected_signal(signal)

    encoder_input = torch.tensor(signal)
    decoder_input = torch.LongTensor(self.ids_to_encodedsources[identity])
    return encoder_input, decoder_input


class CustomCollator(object):
  def __init__(self, max_signal_len, max_source_len, enc_pad_val, dec_pad_val, create_enc_mask=False):
    '''
    Params:
      * max_signal_len : int
      * max_source_len : int
      * enc_pad_val : scalar
      * dec_pad_val : scalar
      * create_enc_mask (optional) : bool
    '''
    self.max_signal_len = max_signal_len
    self.max_source_len = max_source_len
    self.enc_pad_val = enc_pad_val
    self.dec_pad_val = dec_pad_val
    self.create_enc_mask = create_enc_mask

  def __call__(self, batch):
    encoder_inputs, decoder_inputs = zip(*batch)
    encoder_input_batch = pad_sequence(encoder_inputs, batch_first=True, padding_value=self.enc_pad_val).float()
    decoder_input_batch = pad_sequence(decoder_inputs, batch_first=True, padding_value=self.dec_pad_val)

    if self.create_enc_mask:
      padding_mask_batch = u.create_padding_mask(encoder_input_batch, self.enc_pad_val)
      return encoder_input_batch, decoder_input_batch, padding_mask_batch

    return encoder_input_batch, decoder_input_batch


class Data(object):
  '''
  This class is designed to handle data & metadata required to perform all experiments
  + every metrics computation required
  + human readable dumping of predictions
  '''
  def __init__(self):
    self.vars_to_save = ['ids_to_audiofile_train', 'ids_to_audiofile_test', 'max_signal_len',
                         'max_source_len', 'ids_to_transcript_train', 'ids_to_transcript_test',
                         'ids_to_encodedsources_train', 'ids_to_encodedsources_test',
                         'idx_to_tokens', 'tokens_to_idx', 'n_signal_feats']

    self.ids_to_audiofile_train = {}
    self.ids_to_audiofile_test = {}
    self.max_signal_len = 0
    self.n_signal_feats = 0
    
    self.max_source_len = 0
    self.ids_to_transcript_train = {}
    self.ids_to_transcript_test = {}
    self.ids_to_encodedsources_train = {}
    self.ids_to_encodedsources_train = {}
    self.idx_to_tokens = []
    self.tokens_to_idx = {}
  
  def save_metadata(self, save_name='_Data_metadata.pk'):
    metadata = {name: getattr(self, name) for name in self.vars_to_save}
    with open(save_name, 'wb') as f:
      pk.dump(metadata, f)
  
  def load_metadata(self, save_name='_Data_metadata.pk'):
    with open(save_name, 'rb') as f:
      metadata = pk.load(f)
    for k, v in metadata.items():
      setattr(self, k, v)
  
  @staticmethod
  def get_openslr_files(folder):
    '''
    Retrieves files in a openSLR folder (look for .flac|.txt|.npy files)

    Params:
      * folder : str

    Returns:
      * dataset_filelist : dict, {'audio': [path_file1.flac, ...],
                                  'features': [path_file2.npy, ...],
                                  'transcript': [path_file3.txt, ...]}
    '''
    dataset_filelist = {'audio': [], 'features': [], 'transcript': []}

    for fname in os.listdir(folder):
      full_path_fname = os.path.join(folder, fname)
      if os.path.isdir(full_path_fname):
        for f2name in os.listdir(full_path_fname):
          full_path_f2name = os.path.join(full_path_fname, f2name)
          if os.path.isdir(full_path_f2name):
            for f3name in os.listdir(full_path_f2name):
              filename = os.path.join(full_path_f2name, f3name)
              if '.flac' in f3name:
                dataset_filelist['audio'].append(filename)
              elif '.npy' in f3name:
                dataset_filelist['features'].append(filename)
              elif '.txt' in f3name:
                dataset_filelist['transcript'].append(filename)

    return dataset_filelist
  
  @staticmethod
  def read_audio_file(filename, sample_rate=16000):
    '''
    Uses librosa libray to read audio file
    
    Params:
      * filename : str
      * sample_rate (optional) : int
    
    Returns:
      signal, sample_rate : np.ndarray, int
    '''
    return librosa.load(filename, sr=sample_rate)
  
  @staticmethod
  def window_slicing_signal(signal, sample_rate=16000, window_size=0.025):
    '''
    Slices signal into n chunks

    Params:
      * signal : np.ndarray
      * sample_rate (optional) : int
      * window_size (optional) : float
    
    Returns:
      * features : np.ndarray, shape = [math.ceil(len(signal)/(window_size*sample_rate)), window_size*sample_rate]
    '''
    window = int(window_size * sample_rate)
    features = np.split(signal, list(range(window, len(signal), window)))
    features[-1] = np.concatenate((features[-1], np.zeros(window - len(features[-1]))), 0)
    return np.array(features)
  
  @staticmethod
  def overlapping_window_slicing_signal(signal, sample_rate=16000, window_size=0.128, overlap_size=0.032):
    window = int(window_size * sample_rate)
    overlap = int(overlap_size * sample_rate)
    windowed_signal = [signal[i:i+window] for i in range(0, len(signal), window - overlap)]
    windowed_signal = [np.pad(s, (0, window - len(s)), mode='constant') for s in windowed_signal]
    return np.array(windowed_signal)
  
  @staticmethod
  def filterbank_extraction(signal, sample_rate=16000, n_mels=80, n_fft=None, hop_length=None, window_size=0.025):
    '''
    Extracts filter-bank features from signal

    Params:
      * signal : np.ndarray
      * sample_rate (optional) : int
      * n_mels (optional) : int
      * n_fft (optional) : int
      * hop_lenght (optional) : int

    Returns:
      * features : np.ndarray
    '''
    hop_length = int(0.010 * sample_rate) if hop_length is None else hop_length
    n_fft = int(window_size * sample_rate) if n_fft is None else n_fft

    features = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    return np.log(features + 1e-6).T
  
  @staticmethod
  def log_spectrogram_extraction(signal, sample_rate=16000, nperseg=400, noverlap=240, nfft=512, padded=False, boundary=None):
    '''
    Extracts log-spectrogram features from signal

    Params:
      * signal : np.ndarray
      * sample_rate (optional) : int
      * nperseg (optional) : int
      * noverlap (optional) : int
      * nfft (optional) : int
      * padded (optional) : bool
      * boundary (optional) : str

    Returns:
      * features : np.ndarray
    '''
    freqs, times, spec = stft(signal, sample_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft, padded=padded, boundary=boundary)
    return np.log(np.abs(spec)+1e-10).T
  
  @staticmethod
  def mfcc_extraction(signal, sample_rate=16000, n_mfcc=80, hop_length=None, n_fft=None, window_size=0.025, dct_type=2):
    '''
    Extracts mfccs features from signal

    By default librosa mfcc have:
      n_fft = 2048
      hop_length = 512
      window_size = n_fft / sample_rate = 0.128 ms
      stride = hop_length / sample_rate = 0.032 ms
    to compute mfccs with a 25ms window and shifted every 10ms leave hop_length and n_fft args to None

    Params:
      * signal : np.ndarray
      * sample_rate (optional) : int
      * n_mfcc (optional) : int
      * hop_length (optional) : int
      * n_fft (optional) : int
      * window_size (optional) : float
      * dct_type (optional) : int

    Returns:
      * features : np.ndarray
    '''
    hop_length = int(0.010 * sample_rate) if hop_length is None else hop_length
    n_fft = int(window_size * sample_rate) if n_fft is None else n_fft

    return librosa.feature.mfcc(y=signal, sr=sample_rate, dct_type=dct_type, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft).T
  
  @staticmethod
  def read_and_slice_signal(filename, slice_fn=None, **kwargs):
    '''
    Reads audio signal from given file then slice it into n chunks

    Params:
      * filename : str
      * slice_fn (optional) : function, used to slice the signal
      * sample_rate (optional) : int
      * window_size (optional) : float

    Returns:
      * features : np.ndarray, shape = [math.ceil(len(signal)/(window_size*sample_rate)), window_size*sample_rate]
    '''
    slice_fn = Data.window_slicing_signal if slice_fn is None else slice_fn

    signal, sample_rate = Data.read_audio_file(filename)
    return slice_fn(signal, **kwargs)
  
  @staticmethod
  def get_std_threshold_selected_signal(signal, threshold=0.01):
    '''
    Computes standard-deviation on last dimension of given signal then filter out frames based on given threshold

    Params:
      * signal : np.ndarray, shape = [n_frames, n_feats]
      * threshold (optional) : float
    
    Returns:
      * std_threshold_selected_signal : np.ndarray, shape = [new_n_frames, n_feats]
    '''
    stds = np.std(signal, axis=-1)
    return signal[stds>threshold]
  
  @staticmethod
  def get_std_threshold_chunked_signal(signal, threshold=0.01, padding=True, pad_val=0):
    '''
    Divides signal into n chunks

    Params:
      * signal : np.ndarray, shape = [n_frames, n_feats]
      * threshold (optional) : float
      * padding (optional) : bool
      * pad_val (optional) : number

    Returns:
      * chunked_signal : list of np.ndarray if padding == True else np.ndarray, shape = [n_chunks, new_n_feats]
    '''
    stds = np.std(signal, axis=-1)
    chunks_mask = np.zeros(stds.shape[0])
    chunks_mask[stds>threshold] = 1

    chunks_limits = [i for i in range(1, len(chunks_mask)) if chunks_mask[i] != chunks_mask[i-1]]

    chunked_signal = []
    for i in range(0, len(chunks_limits), 2):
      if i+1 >= len(chunks_limits):
        chunked_signal.append(signal[chunks_limits[i]:].flatten())
      else:
        chunked_signal.append(signal[chunks_limits[i]:chunks_limits[i+1]].flatten())
    
    if padding:
      max_len = max(map(len, chunked_signal))
      chunked_signal = np.array([np.pad(chunk, (0, max_len - len(chunk)), mode='constant') for chunk in chunked_signal])
    
    return chunked_signal
  
  @staticmethod
  def read_openslr_transcript_file(filename):
    '''
    Reads and parse transcript file from openSLR format

    Params:
      * filename : str
    
    Returns:
      ids, transcripts : tuple of str (identities), tuple of str (sources)
    '''
    with open(filename, 'r') as f:
      transcripts = f.read().splitlines()
    return zip(*[t.split(' ', 1) for t in transcripts])

  def get_transcripts(self, folder, list_files_fn=None, parse_fn=None, var_name='ids_to_transcript'):
    '''
    Reads transcripts files from given folder then creates dictionary mapping
    transcript identity to corresponding source

    self.var_name is created

    Params:
      * folder : str
      * list_files_fn (optional) : function
      * parse_fn (optional) : function
      * var_name (optional) : str
    '''
    list_files_fn = Data.get_openslr_files if list_files_fn is None else list_files_fn
    parse_fn = Data.read_openslr_transcript_file if parse_fn is None else parse_fn

    ids_to_transcript = {}
    for filename in tqdm(list_files_fn(folder)['transcript']):
      ids, transcripts = parse_fn(filename)
      ids_to_transcript = {**ids_to_transcript, **dict(zip(ids, transcripts))}
    
    setattr(self, var_name, ids_to_transcript)
  
  @staticmethod
  def pad_list(mylist, max_len, pad_val):
    '''
    Params:
      * mylist : list of list
      * max_len : int
      * pad_val : str|int|float
    
    Returns:
      * padded_mylist : list of list where all sublists have same size
    '''
    for el in mylist:
      el.extend([pad_val] * (max_len - len(el)))
    return mylist
  
  @staticmethod
  def words_encoding(sources, sos_tok='<sos>', eos_tok='<eos>', pad_tok='<pad>', idx_to_words=None, words_to_idx=None):
    sources = [s.lower() for s in sources]

    if idx_to_words is None or words_to_idx is None:
      words = list(sorted(set([w for s in sources for w in s.split(' ')])))
      idx_to_words = [sos_tok, eos_tok, pad_tok] + words
      words_to_idx = {l: i for i, l in enumerate(idx_to_words)}

    sources_encoded = [[words_to_idx[sos_tok]] + [words_to_idx[w] for w in s.split(' ')] + [words_to_idx[eos_tok]] for s in tqdm(sources)]
    return sources_encoded, idx_to_words, words_to_idx

  @staticmethod
  def letters_encoding(sources, sos_tok='<sos>', eos_tok='<eos>', pad_tok='<pad>', idx_to_letters=None, letters_to_idx=None):
    '''
    Encodes given sources into numerical vectors

    Params:
      * sources : list of str
      * sos_tok (optional) : str
      * eos_tok (optional) : str
      * pad_tok (optional) : str
      * idx_to_letters (optional) : list of str
      * letters_to_idx (optional) : dict

    Returns:
      sources_encoded, idx_to_letters, letters_to_idx : list of list of int, list of str, dict
    '''
    sources = [s.lower() for s in sources]

    if idx_to_letters is None or letters_to_idx is None:
      letters = list(sorted(set([l for s in sources for l in s])))
      idx_to_letters = [sos_tok, eos_tok, pad_tok] + letters
      letters_to_idx = {l: i for i, l in enumerate(idx_to_letters)}

    sources_encoded = [[letters_to_idx[sos_tok]] + [letters_to_idx[l] for l in s] + [letters_to_idx[eos_tok]] for s in tqdm(sources)]
    return sources_encoded, idx_to_letters, letters_to_idx
  
  @staticmethod
  def phonemes_encoding(sources, sos_tok='<sos>', eos_tok='<eos>', pad_tok='<pad>', idx_to_phonemes=None, phonemes_to_idx=None):
    '''
    Encodes given sources into numerical vectors

    Params:
      * sources : list of str
      * sos_tok (optional) : str
      * eos_tok (optional) : str
      * pad_tok (optional) : str
      * idx_to_letters (optional) : list of str
      * letters_to_idx (optional) : dict

    Returns:
      sources_encoded, idx_to_phonemes, phonemes_to_idx : list of list of int, list of str, dict
    '''
    g2p = G2p()
    converted_sources = [g2p(s.lower()) for s in tqdm(sources)]

    if idx_to_phonemes is None or phonemes_to_idx is None:
      phonemes = list(sorted(set([p for s in converted_sources for p in s])))
      idx_to_phonemes = [sos_tok, eos_tok, pad_tok] + phonemes
      phonemes_to_idx = {p: i for i, p in enumerate(idx_to_phonemes)}
    
    sources_encoded = [[phonemes_to_idx[sos_tok]] + [phonemes_to_idx[p] for p in s] + [phonemes_to_idx[eos_tok]] for s in converted_sources]
    return sources_encoded, idx_to_phonemes, phonemes_to_idx
  
  @staticmethod
  def syllables_encoding(sources, sos_tok='<sos>', eos_tok='<eos>', pad_tok='<pad>', idx_to_syllables=None, syllables_to_idx=None):
    '''
    Encodes given sources into numerical vectors

    Params:
      * sources : list of str
      * sos_tok (optional) : str
      * eos_tok (optional) : str
      * pad_tok (optional) : str
      * idx_to_letters (optional) : list of str
      * letters_to_idx (optional) : dict

    Returns:
      sources_encoded, idx_to_letters, letters_to_idx : list of list of int, list of str, dict
    '''
    g2p = G2p()
    conv_sources = [g2p(s.lower()) for s in tqdm(sources)]
    conv_sources = [[p for p in s if re.search(r'\d+', p) is not None or p == ' '] for s in tqdm(conv_sources)]

    if idx_to_syllables is None or syllables_to_idx is None:
      syllables = list(sorted(set([s for cs in conv_sources for s in cs])))
      idx_to_syllables = [sos_tok, eos_tok, pad_tok] + syllables
      syllables_to_idx = {s: i for i, s in enumerate(idx_to_syllables)}
    
    sources_encoded = [[syllables_to_idx[sos_tok]] + [syllables_to_idx[s] for s in cs] + [syllables_to_idx[eos_tok]] for cs in conv_sources]
    return sources_encoded, idx_to_syllables, syllables_to_idx
  
  def process_all_transcripts(self, train_folder, test_folder, encoding_fn=None, pad_tok='<pad>'):
    '''
    Default transcripts processing

    self.max_source_len is created
    self.idx_to_tokens is created
    self.tokens_to_idx is created
    self.ids_to_transcript_train is created
    self.ids_to_transcript_test is created
    self.ids_to_encodedsources_train is created
    self.ids_to_encodedsources_test is created

    Params:
      * train_folder : str
      * test_folder : str
      * encoding_fn (optional) : function
      * pad_tok (optional) : str
    '''
    encoding_fn = Data.letters_encoding if encoding_fn is None else encoding_fn

    print('Processing training & testing transcripts files...')
    self.get_transcripts(train_folder, var_name='ids_to_transcript_train')
    ids_train, sources_train = zip(*[(k, v) for k, v in self.ids_to_transcript_train.items()])

    self.get_transcripts(test_folder, var_name='ids_to_transcript_test')
    ids_test, sources_test = zip(*[(k, v) for k, v in self.ids_to_transcript_test.items()])

    sources_encoded, self.idx_to_tokens, self.tokens_to_idx = encoding_fn(sources_train + sources_test)

    self.max_source_len = max(map(len, sources_encoded))
    # sources_encoded = Data.pad_list(sources_encoded, self.max_source_len, self.tokens_to_idx[pad_tok])

    self.ids_to_encodedsources_train = {ids_train[i]: s for i, s in enumerate(sources_encoded[:len(sources_train)])}
    self.ids_to_encodedsources_test = {ids_test[i]: s for i, s in enumerate(sources_encoded[len(sources_train):])}
  
  @staticmethod
  def get_readers(ids_to_something):
    return list(sorted(set([r.split('-')[0] for r in ids_to_something.keys()])))

  @staticmethod
  def extract_subset(dataset, percent=0.2):
    '''
    Params:
      * dataset : torch.utils.data.Dataset
      * percent : float, between 0 and 1
    '''
    num_samples = int(percent * len(dataset))
    subset = random.sample(list(range(len(dataset))), num_samples)
    return Subset(dataset, subset)
  
  @staticmethod
  def get_seq_len_n_feat(folder, list_files_fn=None, process_file_fn=None, slice_fn=None, **kwargs):
    list_files_fn = Data.get_openslr_files if list_files_fn is None else list_files_fn
    process_file_fn = Data.read_and_slice_signal if process_file_fn is None else process_file_fn
    slice_fn = Data.window_slicing_signal if slice_fn is None else slice_fn

    max_seq_len = 0
    for filename in tqdm(list_files_fn(folder)['audio']):
      features = process_file_fn(filename, slice_fn=slice_fn, **kwargs)
      seq_len, n_feats = features.shape
      max_seq_len = max(max_seq_len, seq_len)
    
    return max_seq_len, n_feats

  def set_audio_metadata(self, train_folder, test_folder, list_files_fn=None, process_file_fn=None, slice_fn=None, **kwargs):
    '''
    Updates following variables:
      - self.ids_to_audiofile_train, self.ids_to_audiofile_test, self.max_signal_len, self.n_signal_feats

    Params:
      * train_folder : str
      * test_folder : str
      * list_files_fn (optional) : function
      * process_file_fn (optional) : function
      * slice_fn (optional) : function
      * kwargs (optional) : args passed to process_file_fn
    '''
    list_files_fn = Data.get_openslr_files if list_files_fn is None else list_files_fn
    process_file_fn = Data.read_and_slice_signal if process_file_fn is None else process_file_fn
    slice_fn = Data.window_slicing_signal if slice_fn is None else slice_fn

    self.ids_to_audiofile_train = {f.split('/')[-1].split('.')[0]: f for f in list_files_fn(train_folder)['audio']}
    self.ids_to_audiofile_test = {f.split('/')[-1].split('.')[0]: f for f in list_files_fn(test_folder)['audio']}

    for filename in tqdm(list(self.ids_to_audiofile_train.values()) + list(self.ids_to_audiofile_test.values())):
      features = process_file_fn(filename, slice_fn=slice_fn, **kwargs)
      seq_len, self.n_signal_feats = features.shape
      self.max_signal_len = max(self.max_signal_len, seq_len)
  
  def get_dataset_generator(self, train=True, batch_size=32, num_workers=4, shuffle=True, pin_memory=True, subset=False, percent=0.2,
                            pad_idx=2, signal_type='window-sliced', create_enc_mask=False, readers=[], process_file_fn=None, **kwargs):
    '''
    Params:
      * train (optional) : bool, True to return training generator, False for testing generator
      * batch_size (optional) : int
      * num_workers (optional) : int
      * shuffle (optional) : bool
      * pin_memory (optional) : bool
      * subset (optional) : bool
      * percent (optional) : float
      * pad_tok (optional) : str
      * signal_type (optional) : str
      * create_enc_mask (optional) : bool
      * readers (optional) : list
      * process_file_fn (optional) : function
      * kwargs : arguments passed to process_file_fn

    Returns:
      * DataLoader : torch.utils.data.DataLoader
    '''
    process_file_fn = Data.read_and_slice_signal if process_file_fn is None else process_file_fn
    ids_to_audiofile = self.ids_to_audiofile_train if train else self.ids_to_audiofile_test
    ids_to_sources = self.ids_to_encodedsources_train if train else self.ids_to_encodedsources_test

    custom_dataset = CustomDataset(ids_to_audiofile, ids_to_sources, signal_type=signal_type, readers=readers,
                                   process_file_fn=process_file_fn, **kwargs)
    
    if subset:
      custom_dataset = Data.extract_subset(custom_dataset, percent=percent)
      
    custom_collator = CustomCollator(self.max_signal_len, self.max_source_len, 0, pad_idx, create_enc_mask=create_enc_mask)
    
    return DataLoader(custom_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collator, shuffle=shuffle,
                      pin_memory=pin_memory)

  @staticmethod
  def reconstruct_sources(sources_encoded, idx_to_tokens, eos_idx, joiner=''):
    '''
    Params:
      * sources_encoded : list of list of int
      * idx_to_tokens : list of str
    
    Returns:
      * sources : list of str
    '''
    return [joiner.join([idx_to_tokens[idx] for idx in s[1:s.index(eos_idx) if eos_idx in s else -1]]) for s in sources_encoded]
  
  @staticmethod
  def compute_accuracy(targets=[], predictions=[], eos_idx=1, **kwargs):
    targets = [np.array(l[:l.index(eos_idx) + 1 if eos_idx in l else None]) for l in targets]
    predictions = [np.array(l[:l.index(eos_idx) + 1 if eos_idx in l else None]) for l in predictions]

    acc_per_preds = []
    n_correct_all, n_total_all = 0, 0

    for t, p in zip(targets, predictions):
      t_len = len(t)
      p = np.pad(p, (0, t_len - len(p)), mode='constant', constant_values=-1) if len(p) < t_len else p[:t_len]
      compare = t == p
      n_correct = sum(compare)
      acc_per_preds.append(n_correct / t_len)
      n_correct_all += n_correct
      n_total_all += t_len
    
    return {'mean_preds_acc': np.mean(acc_per_preds), 'preds_acc': n_correct_all / n_total_all}
  
  @staticmethod
  def compute_scores(targets=[], predictions=[], eos_idx=1, idx_to_tokens={}, joiner='', strategy='other', **kwargs):
    '''
    Params:
      * targets : list of string
      * predictions : list of string
      * eos_idx : int
      * idx_to_tokens : dict
      * joiner (optional) : str
      * strategy (optional) : str

    Returns:
      * character_accuracy : float
      * word_accuracy : float
      * sentence_accuracy : float
      * mwer : float, mean word error rate
    '''
    targets_sentences = Data.reconstruct_sources(targets, idx_to_tokens, eos_idx, joiner=joiner)
    predictions_sentences = Data.reconstruct_sources(predictions, idx_to_tokens, eos_idx, joiner=joiner)

    count_correct_sentences = 0
    count_correct_words, count_words = 0, 0
    count_correct_characters, count_characters = 0, 0
    wers = []

    for target, pred in zip(targets_sentences, predictions_sentences):
      count_characters += len(target)
      count_correct_characters += sum([1 for t, p in zip(target, pred) if t == p])

      if strategy == 'align':
        space_idxs = [0] + [i for i, c in enumerate(target) if c == ' '] + [len(target)]
        is_words_correct = [
          target[space_idxs[i] + 1 : space_idxs[i+1]] == pred[space_idxs[i] + 1 : space_idxs[i+1]]
          for i in range(len(space_idxs) - 1)
        ]
        count_words += len(is_words_correct)
        count_correct_words += sum(is_words_correct)
      else:
        target_word_list = target.split(' ')
        pred_word_list = pred.split(' ')

        count_words += len(target_word_list)
        count_correct_words += sum([1 for tw, pw in zip(target_word_list, pred_word_list) if tw == pw])

      if target == pred:
        count_correct_sentences += 1
      
      wers.append(u.compute_WER(target, pred))
    
    character_accuracy = count_correct_characters / count_characters
    word_accuracy = count_correct_words / count_words
    sentence_accuracy = count_correct_sentences / len(targets_sentences)

    wer = np.mean(wers)

    return {'character_accuracy': character_accuracy, 'sentence_accuracy': sentence_accuracy, 'wer': wer, 'word_accuracy': word_accuracy}


class ConvnetExperiments(object):
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
                                                            pin_memory=self.pin_memory, signal_type=self.signal_type,
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


## STATUS = FAILURE
class Experiment1(ConvnetExperiments):
  '''Encoder-Decoder Convnet for syllables prediction, adam optimizer, CrossEntropy loss, window-sliced'''
  def __init__(self, logfile='_logs/_logs_experiment1.txt', save_name_model='convnet/convnet_experiment1.pt',
               encoding_fn=Data.syllables_encoding, decay_factor=0, metadata_file='_Data_metadata_syllables.pk'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, encoding_fn=encoding_fn, decay_factor=decay_factor,
                     metadata_file=metadata_file)


## STATUS = FAILURE
class Experiment2(ConvnetExperiments):
  '''Encoder-Decoder Convnet for syllables prediction, adam optimizer, Attention-CrossEntropy loss, window-sliced'''
  def __init__(self, logfile='_logs/_logs_experiment2.txt', save_name_model='convnet/convnet_experiment2.pt',
               encoding_fn=Data.syllables_encoding, metadata_file='_Data_metadata_syllables.pk'):
    super().__init__(save_name_model=save_name_model, logfile=logfile, encoding_fn=encoding_fn, metadata_file=metadata_file)


## STATUS = FAILURE
class Experiment3(ConvnetExperiments):
  '''Encoder-Decoder Convnet for syllables prediction, adam optimizer, CrossEntropy loss, std-threshold-selected'''
  def __init__(self, logfile='_logs/_logs_experiment3.txt', decay_factor=0, save_name_model='convnet/convnet_experiment3.pt',
               signal_type='std-threshold-selected', encoding_fn=Data.syllables_encoding, metadata_file='_Data_metadata_syllables.pk'):
    super().__init__(save_name_model=save_name_model, logfile=logfile, signal_type=signal_type, decay_factor=decay_factor,
                     encoding_fn=encoding_fn, metadata_file=metadata_file)


## STATUS = FAILURE
class Experiment5(ConvnetExperiments):
  '''Encoder-Decoder Convnet for phonemes prediction, adam optimizer, CrossEntropy loss, window-sliced'''
  def __init__(self, logfile='_logs/_logs_experiment5.txt', decay_factor=0, encoding_fn=Data.phonemes_encoding,
               save_name_model='convnet/convnet_experiment5.pt', metadata_file='_Data_metadata_phonemes.pk'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, encoding_fn=encoding_fn, metadata_file=metadata_file,
                     decay_factor=decay_factor)


## STATUS = FAILURE
class Experiment6(ConvnetExperiments):
  '''Encoder-Decoder Convnet for syllables prediction, adam optimizer, CrossEntropy loss, window-sliced, sigmoid score_fn'''
  def __init__(self, logfile='_logs/_logs_experiment6.txt', decay_factor=0, save_name_model='convnet/convnet_experiment6.pt',
               encoding_fn=Data.syllables_encoding, metadata_file='_Data_metadata_syllables.pk', score_fn=u.sigmoid_energy):
    super().__init__(logfile=logfile, save_name_model=save_name_model, score_fn=score_fn, decay_factor=decay_factor,
                     encoding_fn=encoding_fn, metadata_file=metadata_file)


## STATUS = FAILURE
class Experiment7(ConvnetExperiments):
  '''Encoder-Decoder Convnet for syllables prediction, adam optimizer, CrossEntropy loss, window-sliced, MultiHeadAttention'''
  def __init__(self, logfile='_logs/_logs_experiment7.txt', decay_factor=0, save_name_model='convnet/convnet_experiment7.pt',
               encoding_fn=Data.syllables_encoding, metadata_file='_Data_metadata_syllables.pk', multi_head=True, d_keys_values=64):
    super().__init__(logfile=logfile, save_name_model=save_name_model, multi_head=multi_head, d_keys_values=d_keys_values,
                     decay_factor=decay_factor, encoding_fn=encoding_fn, metadata_file=metadata_file)


# STATUS = FAILURE
class Experiment8(ConvnetExperiments):
  '''Encoder-Decoder Convnet for syllables prediction, adam optimizer, Attention-CrossEntropy loss, window-sliced, MultiHeadAttention'''
  def __init__(self, logfile='_logs/_logs_experiment8.txt', save_name_model='convnet/convnet_experiment8.pt',
               encoding_fn=Data.syllables_encoding, metadata_file='_Data_metadata_syllables.pk', multi_head=True, d_keys_values=64):
    super().__init__(logfile=logfile, save_name_model=save_name_model, multi_head=multi_head, d_keys_values=d_keys_values,
                     encoding_fn=encoding_fn, metadata_file=metadata_file)


## STATUS = FAILURE
class Experiment9(ConvnetExperiments):
  '''Encoder-Decoder Convnet for syllables prediction, adam optimizer, Attention-CrossEntropy loss, window-sliced=0.05'''
  def __init__(self, logfile='_logs/_logs_experiment9.txt', save_name_model='convnet/convnet_experiment9.pt',
               encoding_fn=Data.syllables_encoding, metadata_file='_Data_metadata_syllables.pk', window_size=0.05):
    super().__init__(logfile=logfile, save_name_model=save_name_model, encoding_fn=encoding_fn, metadata_file=metadata_file,
                     window_size=window_size)


## STATUS = GOOD, but slower convergence than experiment12&14
class Experiment10(ConvnetExperiments):
  '''Encoder-Decoder Convnet for letters prediction, adam optimizer, Attention-CrossEntropy loss, mfcc, n_fft=2048, hop_length=512'''
  def __init__(self, logfile='_logs/_logs_experiment10.txt', save_name_model='convnet/convnet_experiment10.pt', batch_size=8,
               slice_fn=Data.mfcc_extraction, n_fft=2048, hop_length=512, scorer=Data.compute_scores,
               metadata_file='_Data_metadata_letters_mfcc0128.pk'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, slice_fn=slice_fn, batch_size=batch_size,
                     n_fft=n_fft, hop_length=hop_length, scorer=scorer, metadata_file=metadata_file)
    

class Experiment11(object):
  '''Conv-ConvTranspose to creates compressed representation by reconstruction task'''
  def __init__(self, device=None, logfile='_logs/_logs_experiment11.txt', save_name_model='convnet/naive_convnet_experiment11.pt',
               batch_size=32, lr=1e-4, metadata_file='_Data_metadata_letters_raw0025.pk', n_epochs=500,
               train_folder='../../../datasets/openslr/LibriSpeech/train-clean-100/',
               test_folder='../../../datasets/openslr/LibriSpeech/test-clean/'):
    logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.save_name_model = save_name_model
    self.batch_size = batch_size
    self.lr = lr
    self.metadata_file = metadata_file
    self.n_epochs = n_epochs
    self.train_folder = train_folder
    self.test_folder = test_folder

    self.set_data()

    self.model = nc.NaiveConvED(n_feats=10).to(self.device)
    logging.info(f'The model has {u.count_trainable_parameters(self.model):,} trainable parameters')

    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.criterion = torch.nn.MSELoss(reduction='sum')

    self.train_data_loader = self.data.get_dataset_generator(batch_size=batch_size)
    self.test_data_loader = self.data.get_dataset_generator(train=False, batch_size=batch_size)
    
  def set_data(self):
    self.data = Data()

    if not os.path.isfile(self.metadata_file):
      self.data.set_audio_metadata(self.train_folder, self.test_folder)
      self.data.process_all_transcripts(self.train_folder, self.test_folder)
      self.data.save_metadata(save_name=self.metadata_file)
    else:
      self.data.load_metadata(save_name=self.metadata_file)
  
  def train(self):
    print('Start Training...')
    eval_loss_memory = None
    for epoch in tqdm(range(self.n_epochs)):
      epoch_loss = self.train_pass()
      logging.info(f"Epoch {epoch} | train_loss = {epoch_loss:.3f}")
      eval_loss = self.evaluation()
      logging.info(f"Epoch = {epoch} | test_loss = {eval_loss:.3f}")

      if eval_loss_memory is None or eval_loss < eval_loss_memory:
        u.save_checkpoint(self.model, None, self.save_name_model)
        eval_loss_memory = eval_loss
  
  @torch.no_grad()
  def evaluation(self):
    losses = 0

    self.model.eval()

    for enc_in, _ in tqdm(self.test_data_loader):
      enc_in = enc_in.to(self.device)

      if enc_in.shape[1] % 2 == 1:
        enc_in = enc_in[:, :-1]

      preds = self.model(enc_in)

      losses += self.criterion(preds, enc_in).item()
    
    self.model.train()

    return losses / len(self.test_data_loader)
  
  def train_pass(self):
    losses = 0

    for enc_in, _ in tqdm(self.train_data_loader):
      enc_in = enc_in.to(self.device)

      if enc_in.shape[1] % 2 == 1:
        enc_in = enc_in[:, :-1]

      preds = self.model(enc_in)

      self.optimizer.zero_grad()

      current_loss = self.criterion(preds, enc_in)

      current_loss.backward()

      self.optimizer.step()

      losses += current_loss.item()
    
    return losses / len(self.train_data_loader)


class Experiment12(ConvnetExperiments):
  '''Convnet letters prediction, adam, Attention-CrossEntropy loss, mfcc, n_fft=2048, hop_length=512, MultiHead'''
  def __init__(self, logfile='_logs/_logs_experiment12.txt', save_name_model='convnet/convnet_experiment12.pt', batch_size=32,
               slice_fn=Data.mfcc_extraction, n_fft=2048, hop_length=512, scorer=Data.compute_scores, multi_head=True,
               metadata_file='_Data_metadata_letters_mfcc0128.pk', lr=1e-5):
    super().__init__(logfile=logfile, save_name_model=save_name_model, slice_fn=slice_fn, batch_size=batch_size, lr=lr,
                     n_fft=n_fft, hop_length=hop_length, scorer=scorer, multi_head=multi_head, metadata_file=metadata_file)
    u.load_model(self.model, save_name_model, restore_only_similars=True)


## STATUS = extremely slow convergence, wait only till epoch 100
class Experiment13(ConvnetExperiments):
  '''Convnet letters prediction, adam, Attention-CrossEntropy loss, window-raw-sliced'''
  def __init__(self, logfile='_logs/_logs_experiment13.txt', save_name_model='convnet/convnet_experiment13.pt', batch_size=8,
               scorer=Data.compute_scores, metadata_file='_Data_metadata_letters_raw0025.pk'):
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=batch_size, scorer=scorer, metadata_file=metadata_file)


class Experiment14(ConvnetExperiments):
  '''Convnet letters prediction, adam, Attention-CrossEntropy loss, mfcc, n_fft=2048, hop_length=512, MultiHead, ReLU'''
  def __init__(self, logfile='_logs/_logs_experiment14.txt', save_name_model='convnet/convnet_experiment14.pt', batch_size=8,
               slice_fn=Data.mfcc_extraction, n_fft=2048, hop_length=512, scorer=Data.compute_scores, multi_head=True,
               metadata_file='_Data_metadata_letters_mfcc0128.pk', relu=True):
    super().__init__(logfile=logfile, save_name_model=save_name_model, slice_fn=slice_fn, batch_size=batch_size, relu=relu,
                     n_fft=n_fft, hop_length=hop_length, scorer=scorer, multi_head=multi_head, metadata_file=metadata_file)


class EncoderEmbedderNaiveConv(torch.nn.Module):
  def __init__(self, enc_input_dim, emb_dim, hid_dim, enc_max_seq_len, enc_dropout, device, reduce_dim=False,
               n_feats=10, in_size=400, out_size=80):
    super().__init__()
    self.naive_encoder = nc.NaiveConvEncoder(n_feats=n_feats, in_size=in_size, out_size=out_size)
    self.projection = torch.nn.Linear(out_size, emb_dim)
    self.relu = torch.nn.ReLU(inplace=True)
  
  def forward(self, x):  # x = [batch_size, seq_len, in_size]
    out, _ = self.naive_encoder(x.unsqueeze(1))  # out = [batch_size, 1, new_seq_len, out_size]
    out = self.relu(self.projection(out.squeeze(1)))  ## out = [batch_size, new_seq_len, emb_dim]
    return out


# STATUS = STOPPED, epoch 90, flat testing accuracy curve
class Experiment15(ConvnetExperiments):  # epoch 160 of experiment11 NaiveConvEncoder
  '''Convnet letters prediction, adam, Attention-CrossEntropy loss, window-raw-sliced, EncoderEmbedder = NaiveConvEncoder'''
  def __init__(self, logfile='_logs/_logs_experiment15.txt', save_name_model='convnet/convnet_experiment15.pt', batch_size=8,
               metadata_file='_Data_metadata_letters_raw0025.pk'):
    convnet_config = {'encoder_embedder': EncoderEmbedderNaiveConv, 'enc_layers': 2}
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=batch_size, metadata_file=metadata_file,
                    convnet_config=convnet_config)
    nc.load_model_from_NaiveConvED(self.model.encoder.embedder.naive_encoder, 'convnet/naive_convnet_experiment11.pt')


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


class Experiment17(ConvnetExperiments):
  '''Convnet, letters prediction, adam, Attention-CrossEntropy, MultiHead, window-raw-slice with win=0.128'''
  def __init__(self, logfile='_logs/_logs_experiment17.txt', save_name_model='convnet/convnet_experiment17.pt', batch_size=8,
               metadata_file='_Data_metadata_letters_raw0128.pk', slice_fn=Data.overlapping_window_slicing_signal, multi_head=True):
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, slice_fn=slice_fn,
                     batch_size=batch_size, multi_head=multi_head)


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


class Experiment22(ConvnetExperiments):
  '''Convnet phonemes prediction, adam, Attention-CrossEntropy loss, mfcc, n_fft=2048, hop_length=512, MultiHead'''
  def __init__(self, logfile='_logs/_logs_experiment22.txt', save_name_model='convnet/convnet_experiment22.pt', batch_size=8,
               slice_fn=Data.mfcc_extraction, n_fft=2048, hop_length=512, scorer=Data.compute_scores, multi_head=True,
               metadata_file='_Data_metadata_phonemes_mfcc0128.pk', encoding_fn=Data.phonemes_encoding):
    super().__init__(logfile=logfile, save_name_model=save_name_model, slice_fn=slice_fn, batch_size=batch_size, encoding_fn=encoding_fn,
                     n_fft=n_fft, hop_length=hop_length, scorer=scorer, multi_head=multi_head, metadata_file=metadata_file)


class Experiment23(ConvnetExperiments):
  '''Convnet words prediction, adam, Attention-CrossEntropy loss, mfcc, n_fft=2048, hop_length=512, MultiHead'''
  def __init__(self, logfile='_logs/_logs_experiment23.txt', save_name_model='convnet/convnet_experiment23.pt', batch_size=8,
               slice_fn=Data.mfcc_extraction, n_fft=2048, hop_length=512, scorer=Data.compute_scores, multi_head=True,
               metadata_file='_Data_metadata_words_mfcc0128.pk', encoding_fn=Data.words_encoding):
    super().__init__(logfile=logfile, save_name_model=save_name_model, slice_fn=slice_fn, batch_size=batch_size, encoding_fn=encoding_fn,
                     n_fft=n_fft, hop_length=hop_length, scorer=scorer, multi_head=multi_head, metadata_file=metadata_file)


if __name__ == "__main__":
  ## SEEDING FOR REPRODUCIBILITY
  SEED = 42
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)

  experiments = {k.replace('Experiment', ''): v for k, v in locals().items() if re.search(r'Experiment\d+', k) is not None}
  
  rep = input('Which Experiment do you want to start? (1-23): ')
  exp = experiments[rep]()
  exp.train()


## DEPRECATED OR UNUSED FN

# class Experiment1(object):
#   '''Encoder-Decoder Convnet for syllables prediction, adam optimizer, CrossEntropy loss, window-sliced'''
#   def __init__(self, device=None, logfile='_logs/_logs_experiment1.txt', lr=1e-4, smoothing_eps=0.1, dump_config=True, decay_factor=0,
#                save_name_model='convnet/convnet_experiment1.pt', encoding_fn=Data.syllables_encoding,
#                metadata_file='_Data_metadata_syllables.pk', score_fn=torch.softmax, signal_type='window-sliced',
#                multi_head=False, d_keys_values=64):
#     logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#     self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
#     self.smoothing_eps = smoothing_eps
#     self.decay_factor = decay_factor
#     self.save_name_model = save_name_model

#     self.data = data_routine(encoding_fn=encoding_fn, metadata_file=metadata_file, process_audio=False)

#     self.sos_idx = self.data.tokens_to_idx['<sos>']
#     self.eos_idx = self.data.tokens_to_idx['<eos>']
#     self.pad_idx = self.data.tokens_to_idx['<pad>']

#     self.convnet_config = {'enc_input_dim': self.data.n_signal_feats, 'enc_max_seq_len': self.data.max_signal_len,
#                            'dec_input_dim': len(self.data.idx_to_tokens), 'dec_max_seq_len': self.data.max_source_len,
#                            'output_size': len(self.data.idx_to_tokens), 'pad_idx': self.pad_idx, 'score_fn': score_fn,
#                            'enc_layers': 10, 'dec_layers': 10, 'enc_kernel_size': 3, 'dec_kernel_size': 3, 'enc_dropout': 0.25,
#                            'dec_dropout': 0.25, 'emb_dim': 256, 'hid_dim': 512, 'reduce_dim': False,
#                            'multi_head': multi_head, 'd_keys_values': d_keys_values}
#     self.model = self.convnet_instanciation(**self.convnet_config)

#     if dump_config:
#       u.dump_dict(self.convnet_config, 'ENCODER-DECODER PARAMETERS')
#       logging.info(f'The model has {u.count_trainable_parameters(self.model):,} trainable parameters')

#     self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

#     self.criterion = u.AttentionLoss(self.pad_idx, self.device, decay_step=0.01, decay_factor=self.decay_factor)

#     self.train_data_loader = self.data.get_dataset_generator(signal_type=signal_type)
#     self.test_data_loader = self.data.get_dataset_generator(train=False, signal_type=signal_type)
  
#   def convnet_instanciation(self, enc_input_dim=400, enc_max_seq_len=1400, dec_input_dim=31, dec_max_seq_len=600, output_size=31,
#                             enc_layers=10, dec_layers=10, enc_kernel_size=3, dec_kernel_size=3, enc_dropout=0.25,
#                             dec_dropout=0.25, emb_dim=256, hid_dim=512, reduce_dim=False, pad_idx=2, score_fn=torch.softmax,
#                             multi_head=False, d_keys_values=64):
#     enc_embedder = css.EncoderEmbedder(enc_input_dim, emb_dim, hid_dim, enc_max_seq_len, enc_dropout, self.device, reduce_dim=reduce_dim)
#     dec_embedder = css.DecoderEmbedder(dec_input_dim, emb_dim, dec_max_seq_len, dec_dropout, self.device)

#     enc = css.Encoder(emb_dim, hid_dim, enc_layers, enc_kernel_size, enc_dropout, self.device, embedder=enc_embedder)
#     dec = css.Decoder(output_size, emb_dim, hid_dim, dec_layers, dec_kernel_size, dec_dropout, pad_idx, self.device,
#                       embedder=dec_embedder, score_fn=score_fn, multi_head=multi_head, d_keys_values=d_keys_values)

#     return css.Seq2Seq(enc, dec, self.device).to(self.device)
  
#   def train_pass(self):
#     losses = 0
#     targets, predictions = [], []

#     for enc_in, dec_in in tqdm(self.train_data_loader):
#       enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
#       preds, att = self.model(enc_in, dec_in[:, :-1])

#       self.optimizer.zero_grad()

#       current_loss = self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1),
#                                     att, epsilon=self.smoothing_eps)

#       current_loss.backward()

#       self.optimizer.step()

#       targets += dec_in[:, 1:].tolist()
#       predictions += preds.argmax(dim=-1).tolist()

#       losses += current_loss.item()
    
#     mean_preds_acc, overall_acc = Data.compute_accuracy(targets, predictions, self.eos_idx)
    
#     return losses / len(self.train_data_loader), mean_preds_acc, overall_acc
  
#   def train(self, n_epochs=500, eval_step=5):
#     print('Start Training...')
#     eval_accuracy_memory = 0
#     for epoch in tqdm(range(n_epochs)):
#       epoch_loss, mpta, ota = self.train_pass()
#       logging.info(f'Epoch {epoch} | train_loss = {epoch_loss:.3f} | mean_preds_train_acc = {mpta} | overall_train_acc = {ota}')
#       eval_loss, mpea, oea = self.evaluation(only_loss=False if epoch % eval_step == 0 else True)
#       logging.info(f'Epoch {epoch} | test_loss = {eval_loss:.3f} | mean_preds_eval_acc = {mpea} | overall_eval_acc = {oea}')

#       self.criterion.step(200 if self.decay_factor == 0 else epoch)

#       if oea is not None and oea > eval_accuracy_memory:
#         u.save_checkpoint(self.model, None, self.save_name_model)
  
#   @torch.no_grad()
#   def evaluation(self, only_loss=True):
#     losses, mean_preds_acc, overall_acc = 0, None, None
#     targets, predictions = [], []

#     self.model.eval()

#     for enc_in, dec_in in tqdm(self.test_data_loader):
#       enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
#       preds, att = self.model(enc_in, dec_in[:, :-1])

#       losses += self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1),
#                                att, epsilon=self.smoothing_eps).item()
      
#       if not only_loss:
#         preds, _ = self.model.greedy_decoding(enc_in, self.sos_idx, self.eos_idx, max_seq_len=dec_in.shape[1])
#         targets += dec_in[:, 1:].tolist()
#         predictions += preds.tolist()
    
#     self.model.train()

#     if not only_loss:
#       mean_preds_acc, overall_acc = Data.compute_accuracy(targets, predictions, self.eos_idx)

#     return losses / len(self.test_data_loader), mean_preds_acc, overall_acc

# def get_dataset_generator(self, train=True, batch_size=32, num_workers=4, shuffle=True, subset=False, percent=0.2,
#                           pad_tok='<pad>', device=None, signal_type='window-sliced', create_enc_mask=False, readers=[],
#                           process_audio_on_the_fly=False, list_files_fn=None, folder=None, process_file_fn=None, **kwargs):
#   list_files_fn = Data.get_openslr_files if list_files_fn is None else list_files_fn
#   process_file_fn = Data.read_and_slice_signal if process_file_fn is None else process_file_fn

#   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
#   folder = '../../../datasets/openslr/LibriSpeech/train-clean-100/' if train else '../../../datasets/openslr/LibriSpeech/test-clean/'

#   if process_audio_on_the_fly:
#     ids_to_audiofile = {f.split('/')[-1].split('.')[0]: f for f in list_files_fn(folder)['audio']}
#   else:
#     ids_to_audiofile = self.ids_to_audiofile_train if train else self.ids_to_audiofile_test
  
#   ids_to_sources = self.ids_to_encodedsources_train if train else self.ids_to_encodedsources_test

#   custom_dataset = CustomDataset(ids_to_audiofile, ids_to_sources, signal_type=signal_type, readers=readers,
#                                   process_audio_on_the_fly=process_audio_on_the_fly, process_file_fn=process_file_fn, **kwargs)

#   if subset:
#     custom_dataset = Data.extract_subset(custom_dataset, percent=percent)
    
#   custom_collator = CustomCollator(self.max_signal_len, self.max_source_len, 0, self.tokens_to_idx[pad_tok],
#                                     create_enc_mask=create_enc_mask)
  
#   return DataLoader(custom_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collator, shuffle=shuffle)

# def data_routine(train_folder = '../../../datasets/openslr/LibriSpeech/train-clean-100/',
#                  test_folder = '../../../datasets/openslr/LibriSpeech/test-clean/',
#                  metadata_file='_Data_metadata.pk', encoding_fn=None, slice_fn=None, process_audio=True):
#   data = Data()
    
#   if not os.path.isfile(metadata_file):
#     data.process_all_audio_files(train_folder, test_folder, slice_fn=slice_fn, process=process_audio)  # Extract features from audio files
#     data.process_all_transcripts(train_folder, test_folder, encoding_fn=encoding_fn)  # Prepare transcripts data

#     data.save_metadata(save_name=metadata_file)
#   else:
#     data.load_metadata(save_name=metadata_file)
  
#   return data

# class Experiment9(Experiment1):
#   '''Encoder-Decoder Convnet for syllables prediction, adam optimizer, Attention-CrossEntropy loss, window-sliced=0.05'''
#   def __init__(self, logfile='_logs/_logs_experiment9.txt', device=None, decay_factor=1, save_name_model='convnet/convnet_experiment9.pt',
#                window_size=0.05, score_fn=torch.softmax, multi_head=False, d_keys_values=64):
#     super().__init__(save_name_model=save_name_model, decay_factor=decay_factor, logfile=logfile, dump_config=False)
#     # [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]

#     train_folder = '../../../datasets/openslr/LibriSpeech/train-clean-100/'
#     test_folder = '../../../datasets/openslr/LibriSpeech/test-clean/'
#     tmp_data = '_tmp_data_experiment9.pk'

#     if not os.path.isfile(tmp_data):
#       train_sl, n_feats = Data.get_seq_len_n_feat(train_folder, window_size=window_size)
#       test_sl, n_feats = Data.get_seq_len_n_feat(test_folder, window_size=window_size)
#       max_audio_seq_len = max(train_sl, test_sl)

#       with open(tmp_data, 'wb') as f:
#         pk.dump([max_audio_seq_len, n_feats], f)
#     else:
#       with open(tmp_data, 'rb') as f:
#         max_audio_seq_len, n_feats = pk.load(f)

#     self.convnet_config = {'enc_input_dim': n_feats, 'enc_max_seq_len': max_audio_seq_len,
#                            'dec_input_dim': len(self.data.idx_to_tokens), 'dec_max_seq_len': self.data.max_source_len,
#                            'output_size': len(self.data.idx_to_tokens), 'pad_idx': self.pad_idx, 'score_fn': score_fn,
#                            'enc_layers': 10, 'dec_layers': 10, 'enc_kernel_size': 3, 'dec_kernel_size': 3, 'enc_dropout': 0.25,
#                            'dec_dropout': 0.25, 'emb_dim': 256, 'hid_dim': 512, 'reduce_dim': False,
#                            'multi_head': multi_head, 'd_keys_values': d_keys_values}
#     self.model = self.convnet_instanciation(**self.convnet_config)

#     u.dump_dict(self.convnet_config, 'ENCODER-DECODER PARAMETERS')
#     logging.info(f'The model has {u.count_trainable_parameters(self.model):,} trainable parameters')
    
#     self.train_data_loader = self.data.get_dataset_generator(process_audio_on_the_fly=True, window_size=window_size)
#     self.test_data_loader = self.data.get_dataset_generator(train=False, process_audio_on_the_fly=True, window_size=window_size)

# def process_all_audio_files(self, train_folder, test_folder, slice_fn=None, process=True):
#   '''
#   Default audio files processing

#   Params:
#     * train_folder : str
#     * test_folder : str
#   '''
#   print('Processing training audio files...')
#   self.process_audio_files(train_folder, var_name='ids_to_audiofile_train', slice_fn=slice_fn, process=process)
#   print('Processing testing audio files...')
#   self.process_audio_files(test_folder, var_name='ids_to_audiofile_test', slice_fn=slice_fn, process=process)

# def process_audio_files(self, folder, list_files_fn=None, process_file_fn=None, slice_fn=None,
#                         file_ext='.flac', var_name='ids_to_audiofile', process=True):
#   '''
#   Process audio files and save features in file {audio_filename}.features.npy

#   self.var_name is created
#   self.max_signal_len is updated
#   self.n_signal_feats is updated

#   Params:
#     * folder : str
#     * list_files_fn (optional) : function, used to get list of audio files
#     * process_file_fn (optional) : function, used to read and process audio files
#     * file_ext (optional) : str
#   '''
#   list_files_fn = Data.get_openslr_files if list_files_fn is None else list_files_fn
#   process_file_fn = Data.read_and_slice_signal if process_file_fn is None else process_file_fn
#   slice_fn = Data.window_slicing_signal if slice_fn is None else slice_fn

#   ids_to_audiofile = {}
#   for filename in tqdm(list_files_fn(folder)['audio']):
#     features_filename = filename.replace(file_ext, '.features.npy')

#     if process:
#       features = process_file_fn(filename, slice_fn=slice_fn)
#       np.save(features_filename, features)
#     else:
#       features = np.load(features_filename)

#     ids_to_audiofile[filename.split('/')[-1].replace(file_ext, '')] = features_filename
#     self.max_signal_len = max(self.max_signal_len, features.shape[0])
#     self.n_signal_feats = features.shape[-1]

#   setattr(self, var_name, ids_to_audiofile)

# class CustomDataset(Dataset):
#   def __init__(self, ids_to_audiofile, ids_to_encodedsources, signal_type='window-sliced', readers=[],
#                process_audio_on_the_fly=False, process_file_fn=None, **kwargs):
#     '''
#     Params:
#       * process_audio_on_the_fly : bool, if True, process_file_fn must be given
#       * kwargs : used to pass argument to process_file_fn
#     '''
#     self.ids_to_audiofile = ids_to_audiofile
#     self.ids_to_encodedsources = ids_to_encodedsources
#     self.signal_type = signal_type
#     self.process_audio_on_the_fly = process_audio_on_the_fly
#     self.process_file_fn = process_file_fn
#     self.process_file_fn_args = kwargs

#     self.identities = list(sorted(ids_to_audiofile.keys()))

#     if len(readers) > 0:
#       self.identities = [i for i in self.identities if i.split('-')[0] in readers]

#   def __len__(self):
#     return len(self.identities)
  
#   def __getitem__(self, idx):
#     identity = self.identities[idx]

#     if self.process_audio_on_the_fly:
#       signal = self.process_file_fn(self.ids_to_audiofile[identity], **self.process_file_fn_args)
#     else:
#       signal = np.load(self.ids_to_audiofile[identity])

#     if self.signal_type == 'std-threshold-selected':
#       signal = Data.get_std_threshold_selected_signal(signal)

#     encoder_input = torch.tensor(signal)
#     decoder_input = torch.LongTensor(self.ids_to_encodedsources[identity])
#     return encoder_input, decoder_input


# ## STATUS = FAILURE
# class Experiment4(object):
#   '''Encoder-Decoder Transformer for syllables prediction, Radam optimizer, CrossEntropy loss, window-sliced'''
#   def __init__(self, device=None, logfile='_logs/_logs_experiment4.txt', lr=1e-4, smoothing_eps=0.1, dump_config=True,
#                save_name_model='transformer/transformer_experiment4.pt', enc_reduce_dim=True):
#     logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#     self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
#     self.smoothing_eps = smoothing_eps
#     self.save_name_model = save_name_model
#     self.enc_reduce_dim = enc_reduce_dim

#     self.data = data_routine(encoding_fn=Data.syllables_encoding, metadata_file='_Data_metadata_syllables.pk', process_audio=False)

#     self.sos_idx = self.data.tokens_to_idx['<sos>']
#     self.eos_idx = self.data.tokens_to_idx['<eos>']
#     self.pad_idx = self.data.tokens_to_idx['<pad>']
#     self.identities_train = list(sorted(set([i.split('-')[0] for i in self.data.ids_to_audiofile_train])))

#     self.transformer_config = {'encoder_embedding_dim': self.data.n_signal_feats, 'enc_max_seq_len': self.data.max_signal_len,
#                                'decoder_embedding_dim': len(self.data.idx_to_tokens), 'dec_max_seq_len': self.data.max_source_len,
#                                'output_size': len(self.data.idx_to_tokens), 'n_encoder_blocks': 8, 'n_decoder_blocks': 6,
#                                'd_model': 384, 'd_keys': 64, 'd_values': 64, 'n_heads': 6, 'd_ff': 512, 'dropout': 0.}
#     self.model = self.transformer_instanciation(**self.transformer_config)

#     if dump_config:
#       u.dump_dict(self.transformer_config, 'ENCODER-DECODER PARAMETERS')
#       logging.info(f'The model has {u.count_trainable_parameters(self.model):,} trainable parameters')

#     self.optimizer = RAdam(self.model.parameters(), lr=lr)
#     self.criterion = u.CrossEntropyLoss(self.pad_idx)

#     self.train_data_loader = self.data.get_dataset_generator(create_enc_mask=True)
#     self.test_data_loader = self.data.get_dataset_generator(train=False, create_enc_mask=True)
  
#   def transformer_instanciation(self, enc_max_seq_len=1400, encoder_embedding_dim=400, d_model=256, dec_max_seq_len=600,
#                                 decoder_embedding_dim=31, output_size=31, n_encoder_blocks=8, n_decoder_blocks=6,
#                                 d_keys=64, d_values=64, n_heads=4, d_ff=512, dropout=0.):
#     encoder_embedder = PositionalEmbedder(enc_max_seq_len, encoder_embedding_dim, d_model, device=self.device,
#                                           reduce_dim=self.enc_reduce_dim)
#     decoder_embedder = PositionalEmbedder(dec_max_seq_len, decoder_embedding_dim, d_model, output_size=output_size, device=self.device)

#     model = Transformer(n_encoder_blocks, n_decoder_blocks, d_model, d_keys, d_values, n_heads, d_ff, output_size,
#                         encoder_embedder=encoder_embedder, decoder_embedder=decoder_embedder, dropout=dropout,
#                         enc_max_seq_len=enc_max_seq_len, dec_max_seq_len=dec_max_seq_len, device=self.device,
#                         encoder_reduce_dim=self.enc_reduce_dim)
    
#     return model.to(self.device)
  
#   def train_pass(self, data_loader):
#     losses = 0
#     targets, predictions = [], []

#     for enc_in, dec_in, pad_mask in tqdm(data_loader):
#       enc_in, dec_in, pad_mask = enc_in.to(self.device), dec_in.to(self.device), pad_mask.to(self.device)
#       preds = self.model(enc_in, dec_in[:, :-1], padding_mask=pad_mask)

#       self.optimizer.zero_grad()

#       current_loss = self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1), epsilon=self.smoothing_eps)

#       current_loss.backward()

#       self.optimizer.step()

#       targets += dec_in[:, 1:].tolist()
#       predictions += preds.argmax(dim=-1).tolist()

#       losses += current_loss.item()
    
#     mean_preds_acc, overall_acc = Data.compute_accuracy(targets, predictions, self.eos_idx)
    
#     return losses / len(data_loader), mean_preds_acc, overall_acc
  
#   def train_pass_accumulate(self, data_loader, accumulate_step=10):
#     losses = 0
#     targets, predictions = [], []
#     self.optimizer.zero_grad()

#     for i, (enc_in, dec_in, pad_mask) in enumerate(tqdm(data_loader)):
#       enc_in, dec_in, pad_mask = enc_in.to(self.device), dec_in.to(self.device), pad_mask.to(self.device)
#       preds = self.model(enc_in, dec_in[:, :-1], padding_mask=pad_mask)

#       current_loss = self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1), epsilon=self.smoothing_eps)

#       current_loss.backward()

#       targets += dec_in[:, 1:].tolist()
#       predictions += preds.argmax(dim=-1).tolist()

#       losses += current_loss.item()

#       if accumulate_step > 0:
#         if i % accumulate_step == 0:
#           self.optimizer.step()
#           self.optimizer.zero_grad()
    
#     self.optimizer.step()

#     mean_preds_acc, overall_acc = Data.compute_accuracy(targets, predictions, self.eos_idx)
    
#     return losses / len(data_loader), mean_preds_acc, overall_acc
  
#   def train_progressively(self, step=50, n_epochs=500, eval_step=10):
#     print('Start Training...')
#     eval_accuracy_memory = 0
#     current_identities = [self.identities_train.pop(0)]
#     current_data_loader = self.data.get_dataset_generator(create_enc_mask=True, readers=current_identities)

#     for epoch in tqdm(range(n_epochs)):
#       if epoch > 0 and epoch % step == 0 and len(self.identities_train) > 0:
#         current_identities.append(self.identities_train.pop(0))
#         current_data_loader = self.data.get_dataset_generator(create_enc_mask=True, readers=current_identities)

#       # epoch_loss, mpta, ota = self.train_pass_accumulate(current_data_loader)
#       epoch_loss, mpta, ota = self.train_pass(current_data_loader)
#       logging.info(f'Epoch {epoch} | train_loss = {epoch_loss:.3f} | mean_preds_train_acc = {mpta} | overall_train_acc = {ota}')
#       eval_loss, mpea, oea = self.evaluation(only_loss=False if epoch % eval_step == 0 else True)
#       logging.info(f'Epoch {epoch} | test_loss = {eval_loss:.3f} | mean_preds_eval_acc = {mpea} | overall_eval_acc = {oea}')

#       if oea is not None and oea > eval_accuracy_memory:
#         u.save_checkpoint(self.model, None, self.save_name_model)
  
#   def train(self, n_epochs=500, eval_step=2):
#     print('Start Training...')
#     eval_accuracy_memory = 0
#     for epoch in tqdm(range(n_epochs)):
#       # epoch_loss, mpta, ota = self.train_pass(self.train_data_loader)
#       epoch_loss, mpta, ota = self.train_pass_accumulate(self.train_data_loader)
#       logging.info(f'Epoch {epoch} | train_loss = {epoch_loss:.3f} | mean_preds_train_acc = {mpta} | overall_train_acc = {ota}')
#       eval_loss, mpea, oea = self.evaluation(only_loss=False if epoch % eval_step == 0 else True)
#       logging.info(f'Epoch {epoch} | test_loss = {eval_loss:.3f} | mean_preds_eval_acc = {mpea} | overall_eval_acc = {oea}')

#       if oea is not None and oea > eval_accuracy_memory:
#         u.save_checkpoint(self.model, None, self.save_name_model)

#   @torch.no_grad()
#   def evaluation(self, only_loss=True):
#     losses, mean_preds_acc, overall_acc = 0, None, None
#     targets, predictions = [], []

#     self.model.eval()

#     for enc_in, dec_in, pad_mask in tqdm(self.test_data_loader):
#       enc_in, dec_in, pad_mask = enc_in.to(self.device), dec_in.to(self.device), pad_mask.to(self.device)
#       preds = self.model(enc_in, dec_in[:, :-1], padding_mask=pad_mask)

#       losses += self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1), epsilon=self.smoothing_eps).item()
      
#       if not only_loss:
#         preds = self.model.greedy_decoding(enc_in, self.eos_idx, self.pad_idx, max_seq_len=dec_in.shape[1])
#         targets += dec_in[:, 1:].tolist()
#         predictions += preds.tolist()
    
#     self.model.train()

#     if not only_loss:
#       mean_preds_acc, overall_acc = Data.compute_accuracy(targets, predictions, self.eos_idx)

#     return losses / len(self.test_data_loader), mean_preds_acc, overall_acc