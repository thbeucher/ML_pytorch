import os
import re
import sys
import torch
import random
import librosa
import numpy as np
import pickle as pk
import soundfile as sf

from tqdm import tqdm
from g2p_en import G2p
from scipy.signal import stft
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Subset

sys.path.append(os.path.abspath(__file__).replace('ASR/data.py', ''))
import utils as u


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

    self.mess_with_targets = kwargs.get('mess_with_targets', False)
    self.vocab_size = kwargs.get('vocab_size', max([i for s in ids_to_encodedsources.values() for i in s]))
    self.mess_prob = kwargs.get('mess_prob', 0.15)

    self.identities = list(sorted(ids_to_audiofile.keys()))

    if len(readers) > 0:
      self.identities = [i for i in self.identities if i.split('-')[0] in readers]
  
  def _mess_with_targets(self, target):
    mask = np.random.choice([True, False], size=len(target), p=[self.mess_prob, 1-self.mess_prob])
    new_idxs = np.random.choice(list(range(self.vocab_size)), size=len(target))
    target[mask] = new_idxs[mask]
    return target

  def __len__(self):
    return len(self.identities)
  
  def __getitem__(self, idx):
    identity = self.identities[idx]

    signal = self.process_file_fn(self.ids_to_audiofile[identity], **self.process_file_fn_args)

    if self.signal_type == 'std-threshold-selected':
      signal = Data.get_std_threshold_selected_signal(signal)

    encoder_input = torch.Tensor(signal) if isinstance(signal, np.ndarray) else signal

    decoder_input = self.ids_to_encodedsources[identity]
    if self.mess_with_targets:
      decoder_input = self._mess_with_targets(np.array(decoder_input))
    decoder_input = torch.LongTensor(decoder_input)

    return encoder_input, decoder_input


class CustomCollator(object):
  def __init__(self, enc_pad_val, dec_pad_val, create_enc_mask=False):
    '''
    Params:
      * enc_pad_val : scalar
      * dec_pad_val : scalar
      * create_enc_mask (optional) : bool
    '''
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
    if sample_rate == 16000:
      return sf.read(filename)
    else:
      return librosa.load(filename, sr=sample_rate)
  
  @staticmethod
  def window_slicing_signal(signal, sample_rate=16000, window_size=0.025, **kwargs):
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
  def overlapping_window_slicing_signal(signal, sample_rate=16000, window_size=0.128, overlap_size=0.032, **kwargs):
    window = int(window_size * sample_rate)
    overlap = int(overlap_size * sample_rate)
    windowed_signal = [signal[i:i+window] for i in range(0, len(signal), window - overlap)]
    windowed_signal = [np.pad(s, (0, window - len(s)), mode='constant') for s in windowed_signal]
    return np.array(windowed_signal)
  
  @staticmethod
  def filterbank_extraction(signal, sample_rate=16000, n_mels=80, n_fft=None, hop_length=None, window_size=0.025, **kwargs):
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
  def log_spectrogram_extraction(signal, sample_rate=16000, nperseg=400, noverlap=240, nfft=512, padded=False, boundary=None, **kwargs):
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
  def mfcc_extraction(signal, sample_rate=16000, n_mfcc=80, hop_length=None, n_fft=None, window_size=0.025, dct_type=2, **kwargs):
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
  def wav2vec_extraction(signal, wav2vec_model=None, filename='', save_features=False, **kwargs):
    if save_features and os.path.isfile(filename.replace('.flac', '.features.npy')):
      c = np.load(filename.replace('.flac', '.features.npy'))
      return c

    assert wav2vec_model is not None, 'wav2vec_model need to be given'

    with torch.no_grad():
      z = wav2vec_model.feature_extractor(torch.Tensor(signal).reshape(1, -1))
      c = wav2vec_model.feature_aggregator(z)
      c = c.squeeze(0).T
    
    if save_features:
      np.save(filename.replace('.flac', '.features.npy'), c.numpy())

    return c

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
    return slice_fn(signal, filename=filename, **kwargs)
  
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
  def ngrams_encoding(sources, sos_tok='<sos>', eos_tok='<eos>', pad_tok='<pad>', n=2, idx_to_ngrams=None, ngrams_to_idx=None):
    sources = [s.lower() for s in sources]

    if idx_to_ngrams is None or ngrams_to_idx is None:
      ngrams = list(sorted(set([s[i:i+n] for s in sources for i in range(0, len(s), n)])))
      idx_to_ngrams = [sos_tok, eos_tok, pad_tok] + ngrams
      ngrams_to_idx = {l: i for i, l in enumerate(idx_to_ngrams)}

    sources_encoded = [[ngrams_to_idx[sos_tok]] + [ngrams_to_idx[s[i:i+n]] for i in range(0, len(s), n)] + [ngrams_to_idx[eos_tok]]
                        for s in tqdm(sources)]
    return sources_encoded, idx_to_ngrams, ngrams_to_idx

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
  
  def process_all_transcripts(self, train_folder, test_folder, encoding_fn=None, padding=False, pad_tok='<pad>'):
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
      * padding (optional) : bool
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

    if padding:
      sources_encoded = Data.pad_list(sources_encoded, self.max_source_len, self.tokens_to_idx[pad_tok])

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
      
    custom_collator = CustomCollator(0, pad_idx, create_enc_mask=create_enc_mask)
    
    return DataLoader(custom_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collator, shuffle=shuffle,
                      pin_memory=pin_memory)

  @staticmethod
  def reconstruct_sources(sources_encoded, idx_to_tokens, eos_idx, joiner='', start=0):
    '''
    Params:
      * sources_encoded : list of list of int
      * idx_to_tokens : list of str
    
    Returns:
      * sources : list of str
    '''
    return [joiner.join([idx_to_tokens[idx] for idx in s[start:s.index(eos_idx) if eos_idx in s else -1]]) for s in sources_encoded]
  
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
  def compute_scores(targets=[], predictions=[], eos_idx=1, idx_to_tokens={}, joiner='', strategy='other', rec=True, **kwargs):
    '''
    Params:
      * targets : list of string
      * predictions : list of string
      * eos_idx : int
      * idx_to_tokens : dict
      * joiner (optional) : str
      * strategy (optional) : str
      * rec (optional) : bool

    Returns:
      * character_accuracy : float
      * word_accuracy : float
      * sentence_accuracy : float
      * mwer : float, mean word error rate
    '''
    if rec:
      targets = Data.reconstruct_sources(targets, idx_to_tokens, eos_idx, joiner=joiner)
      predictions = Data.reconstruct_sources(predictions, idx_to_tokens, eos_idx, joiner=joiner)

    count_correct_sentences = 0
    count_correct_words, count_words = 0, 0
    count_correct_characters, count_characters = 0, 0
    wers = []

    for target, pred in zip(targets, predictions):
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
    sentence_accuracy = count_correct_sentences / len(targets)

    wer = np.mean(wers)

    return {'character_accuracy': character_accuracy, 'sentence_accuracy': sentence_accuracy, 'wer': wer, 'word_accuracy': word_accuracy}
