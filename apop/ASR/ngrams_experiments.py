## Raw Waveform or mfcc -> Encoder -> Encoded Representation
## Recurrent decoding receiving previous prediction
## Decoder encode previous context then attentive use of encoded representation from the Encoder
## then choose between 4 actions
## action 1 = letter prediction, 31 choices
## action 2 = bigram prediction, 600 choices
## action 3 = trigram prediction, 1500 choices
## action 4 = word prediction, 3K choices
## RL tasks, learn the best policy, at each step, is it better to predict a letter, a bigram, a trigram or a word?
## Reward function must be equal to let free to choose the best solution
## 1 good letter = 1pt | 1 good bigram = 2pts | 1 good trigram = 3pts | 1 good word = word-len pts
## at the end of the episode (sentence reconstruction), the same amount of reward has been given whatever policy choosed
## Pretraining to stabilize RL learning by alternate between the 4 tasks and using CrossEntropy training

## Try oracle training, encode using strategy: words then trigrams then bigrams then letters
import os
import re
import sys
import torch
import random
import logging
import numpy as np
import pickle as pk
import torch.optim as optim

from tqdm import tqdm
from itertools import chain
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from fairseq.models.wav2vec import Wav2VecModel
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(__file__).replace('ASR/ngrams_experiments.py', ''))
import utils as u
import models.conv_seqseq as css

from data import Data
from convnet_trainer import ConvnetTrainer
from models.transformer.encoder import TransformerEncoder
from models.transformer.attention import MultiHeadAttention


def get_ngrams(sources, n_words=3000, n_trigrams=1500):
  # bigrams, trigrams and sorted by occurences
  print('Retrieve letters, bigrams, trigrams, words.')
  letters = list(sorted(set([l for s in sources for l in s])))
  bigrams = [s[i:i+2] for s in sources for i in range(len(s)-2+1)]
  trigrams = [s[i:i+3] for s in sources for i in range(len(s)-3+1)]
  words = [w for s in sources for w in s.split()]
  print(f'letters = {len(letters)} | bigrams = {len(set(bigrams))} | trigrams = {len(set(trigrams))} | words = {len(set(words))}')

  words = sorted([w for w, c in Counter(words).most_common(n_words+50) if w not in letters])[:n_words]
  print('filtering of bigrams...')
  bigrams = sorted([(k, v) for k, v in Counter(bigrams).items()], key=lambda x: x[1], reverse=True)
  median_occ = np.median([el[1] for el in bigrams])
  bigrams = [b for b, n in bigrams if n >= median_occ and b not in words]

  print('filtering of trigrams...')
  trigrams = sorted([(k, v) for k, v in Counter(trigrams).items()], key=lambda x: x[1], reverse=True)
  median_occ = np.median([el[1] for el in trigrams])
  trigrams = [t for t, n in trigrams if n >= median_occ and t not in words][:n_trigrams]
  print(f'letters = {len(letters)} | bigrams = {len(set(bigrams))} | trigrams = {len(set(trigrams))} | words = {len(set(words))}')

  return letters, bigrams, trigrams, words


def encode_one_ngram(word, ngrams):
  for t, i in ngrams.items():
    if t in word:
      k = word.index(t)
      tmp = [word[:k]] + [(t, i)] + [word[k+3:]]
      return [el for el in tmp if el != '']


def encode_ngrams(encoded, ngrams):
  encoded_tmp = []
  for i in range(len(encoded)):
    if not isinstance(encoded[i], tuple):
      w = [encoded[i]]
      while not all([True if isinstance(el, tuple) else False for el in w]):
        tmp = []
        for p in w:
          if isinstance(p, tuple):
            tmp.append(p)
          elif len(p) < 3:
            tmp.append((p, None))
          else:
            match_ngram = encode_one_ngram(p, ngrams)
            if match_ngram is not None:
              tmp += match_ngram
            else:
              tmp.append((p, None))
        w = tmp
      encoded_tmp += [el[0] if el[1] is None else el for el in w]
    else:
      encoded_tmp.append(encoded[i])
  return encoded_tmp


def encode_letters(encoded, letters):
  encoded_tmp = []
  for el in encoded:
    if isinstance(el, tuple):
      encoded_tmp.append(el)
    else:
      encoded_tmp += [(l, letters[l]) for l in el]
  return encoded_tmp


def ngrams_encoding(source, letters, bigrams, trigrams, vocab_words):
  # strategy = words then trigrams then bigrams then letters
  words = [e for el in [[w] + [' '] for w in source.split()] for e in el][:-1]
  # encode space
  encoded = [(w, letters[' ']) if w == ' ' else w for w in words]
  # encode words present in vocab_words
  encoded = [(w, vocab_words[w]) if not isinstance(w, tuple) and w in vocab_words else w for w in encoded]
  # encode trigrams present in given trigrams
  encoded = encode_ngrams(encoded, trigrams)
  # encode bigrams present in given bigrams
  encoded = encode_ngrams(encoded, bigrams)
  # encode letters
  encoded_tmp = []
  for el in encoded:
    if isinstance(el, tuple):
      encoded_tmp.append(el)
    else:
      encoded_tmp += [(l, letters[l]) for l in el]
  encoded = encoded_tmp
  return [i for _, i in encoded]


def encode_words_letters(source, letters, words_to_idx):
  words = [e for el in [[w] + [' '] for w in source.split()] for e in el][:-1]
  # encode space
  encoded = [(w, letters[' ']) if w == ' ' else w for w in words]
  # encode words present in vocab_words
  encoded = [(w, words_to_idx[w]) if not isinstance(w, tuple) and w in words_to_idx else w for w in encoded]
  # encode letters
  encoded_tmp = []
  for el in encoded:
    if isinstance(el, tuple):
      encoded_tmp.append(el)
    else:
      encoded_tmp += [(l, letters[l]) for l in el]
  encoded = encoded_tmp
  return [i for _, i in encoded]


def words_letters_encoding(sources, sos_tok='<sos>', eos_tok='<eos>', pad_tok='<pad>', n_words=5000):
  sources = [s.lower() for s in sources]

  letters = list(sorted(set([l for s in sources for l in s])))
  letters = [sos_tok, eos_tok, pad_tok] + letters

  words = [w for s in sources for w in s.split()]
  words = sorted([w for w, c in Counter(words).most_common(n_words+50) if w not in letters])[:n_words]

  letters_to_idx = {l: i for i, l in enumerate(letters)}
  words_to_idx = {w: i + len(letters) for i, w in enumerate(words)}

  print('words & letters encoding...')
  sources_encoded = [encode_words_letters(s, letters_to_idx, words_to_idx) for s in tqdm(sources)]

  wordsletters_to_idx = {**letters_to_idx, **words_to_idx}
  idx_to_wordsletters = [k for k in wordsletters_to_idx]

  sources_encoded = [[wordsletters_to_idx[sos_tok]] + se + [wordsletters_to_idx[eos_tok]] for se in sources_encoded]
  
  return sources_encoded, idx_to_wordsletters, wordsletters_to_idx


def multigrams_encoding(sources, sos_tok='<sos>', eos_tok='<eos>', pad_tok='<pad>'):
  sources = [s.lower() for s in sources]

  letters, bigrams, trigrams, words = get_ngrams(sources)
  letters = [sos_tok, eos_tok, pad_tok] + letters
  print(f'=> letters = {len(letters)} | bigrams = {len(bigrams)} | trigrams = {len(trigrams)} | words = {len(words)}')

  letters_to_idx = {l: i for i, l in enumerate(letters)}
  bigrams_to_idx = {b: i + len(letters) for i, b in enumerate(bigrams)}
  trigrams_to_idx = {t: i + len(letters) + len(bigrams) for i, t in enumerate(trigrams)}
  words_to_idx = {w: i + len(letters) + len(bigrams) + len(trigrams) for i, w in enumerate(words)}

  print('multigrams encoding...')
  sources_encoded = [ngrams_encoding(s, letters_to_idx, bigrams_to_idx, trigrams_to_idx, words_to_idx) for s in tqdm(sources)]

  multigrams_to_idx = {**letters_to_idx, **bigrams_to_idx, **trigrams_to_idx, **words_to_idx}
  idx_to_multigrams = [k for k in multigrams_to_idx]

  sources_encoded = [[multigrams_to_idx[sos_tok]] + se + [multigrams_to_idx[eos_tok]] for se in sources_encoded]
  
  return sources_encoded, idx_to_multigrams, multigrams_to_idx


def multi_encoding_ngrams(ids, sources, sos_tok='<sos>', eos_tok='<eos>', pad_tok='<pad>'):
  sources = [s.lower() for s in sources]

  sources_multigrams, idx_to_multigrams, multigrams_to_idx = multigrams_encoding(sources)

  sos_idx, eos_idx, pad_idx = multigrams_to_idx[sos_tok], multigrams_to_idx[eos_tok], multigrams_to_idx[pad_tok]

  sources_letters = [[sos_idx] + [multigrams_to_idx[l] for l in s] + [eos_idx] for s in sources]

  sources_words = [[e for el in [[w] + [' '] for w in s.split()] for e in el][:-1] for s in sources]
  sources_spaces = [[(w, multigrams_to_idx[' ']) if w == ' ' else w for w in words] for words in sources_words]

  print('encode known bigrams...')
  sources_bigrams = [encode_ngrams(s, {k: v for k, v in multigrams_to_idx.items() if len(k) == 2}) for s in tqdm(sources_spaces)]
  print('encode leaved letters...')
  sources_bigrams = [list(map(lambda x: x[1], encode_letters(s, multigrams_to_idx))) for s in tqdm(sources_bigrams)]

  print('encode known trigrams...')
  sources_trigrams = [encode_ngrams(s, {k: v for k, v in multigrams_to_idx.items() if len(k) == 3}) for s in tqdm(sources_spaces)]
  print('encode leaved letters...')
  sources_trigrams = [list(map(lambda x: x[1], encode_letters(s, multigrams_to_idx))) for s in tqdm(sources_trigrams)]
  
  ids_to_encoded = {f'{ids[i]}_multigrams': s for i, s in enumerate(sources_multigrams)}
  ids_to_encoded = {**ids_to_encoded, **{f'{ids[i]}_letters': s for i, s in enumerate(sources_letters)}}
  ids_to_encoded = {**ids_to_encoded, **{f'{ids[i]}_bigrams': s for i, s in enumerate(sources_bigrams)}}
  ids_to_encoded = {**ids_to_encoded, **{f'{ids[i]}_trigrams': s for i, s in enumerate(sources_trigrams)}}

  return ids_to_encoded, idx_to_multigrams, multigrams_to_idx


def set_metadata(data, train_folder, test_folder):
  data.get_transcripts(train_folder, var_name='ids_to_transcript_train')
  data.get_transcripts(test_folder, var_name='ids_to_transcript_test')
  ids_train, sources_train = zip(*[(k, v) for k, v in data.ids_to_transcript_train.items()])
  ids_test, sources_test = zip(*[(k, v) for k, v in data.ids_to_transcript_test.items()])

  ids_to_encoded, idx_to_tokens, tokens_to_idx = multi_encoding_ngrams(ids_train + ids_test, sources_train + sources_test)
  data.idx_to_tokens = idx_to_tokens
  data.tokens_to_idx = tokens_to_idx
  data.max_source_len = max(map(len, ids_to_encoded.values()))

  print('Create ids_to_encodedsources_train and ids_to_encodedsources_test variables...')
  data.ids_to_encodedsources_train = {k: v for k, v in tqdm(ids_to_encoded.items()) if k.split('_')[0] in ids_train}
  data.ids_to_encodedsources_test = {k.split('_')[0]: v for k, v in tqdm(ids_to_encoded.items()) if k.split('_')[0] in ids_test}

  print('Create ids_to_audiofile_train and ids_to_audiofile_train variables...')
  data.ids_to_audiofile_train = {k: data.ids_to_audiofile_train[k.split('_')[0]]
                                  for k in tqdm(ids_to_encoded) if k.split('_')[0] in ids_train}
  data.ids_to_audiofile_test = {k.split('_')[0]: data.ids_to_audiofile_test[k.split('_')[0]]
                                  for k in tqdm(ids_to_encoded) if k.split('_')[0] in ids_test}


def analyze():
  # keys = ids_to_audiofile_train, ids_to_audiofile_test, max_signal_len, max_source_len, ids_to_transcript_train,
  # ids_to_transcript_test, ids_to_encodedsources_train, ids_to_encodedsources_test, idx_to_tokens, tokens_to_idx, n_signal_feats
  with open('divers/_Data_metadata_letters_raw0025.pk', 'rb') as f:
    data = pk.load(f)
  
  sources = [s.lower() for s in data['ids_to_transcript_train'].values()] + [s.lower() for s in data['ids_to_transcript_test'].values()]

  sources_encoded, idx_to_multigrams, multigrams_to_idx = multigrams_encoding(sources)
  lens = [len(el) for el in sources_encoded]
  print(f'min = {min(lens)} | max = {max(lens)} | mean = {np.mean(lens)}')


## STATUS = 500 epochs finished, best train_acc = 0.925 | test_acc = 0.511 (epoch 500)
class NgramsTrainer1(ConvnetTrainer):
  def __init__(self, logfile='_logs/_logs_multigrams1.txt', save_name_model='convnet/ngrams_convnet_experiment.pt',
               metadata_file='_Data_metadata_multigrams_mfcc0128.pk', encoding_fn=multigrams_encoding, multi_head=True,
               slice_fn=Data.mfcc_extraction, n_fft=2048, hop_length=512, scorer=Data.compute_scores, batch_size=32):
    convnet_config = {'emb_dim': 384, 'hid_dim': 512}
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, encoding_fn=encoding_fn,
                     multi_head=multi_head, slice_fn=slice_fn, n_fft=n_fft, hop_length=hop_length, scorer=scorer,
                     batch_size=batch_size, convnet_config=convnet_config)


## STATUS = stopped, start diverging at epoch 170, best train_acc = 0.893 | test_acc = 0.565 (epoch 140)
class NgramsTrainer2(ConvnetTrainer):
  def __init__(self, logfile='_logs/_logs_ngramsEXP2.txt', save_name_model='convnet/ngrams_convnet_experiment2.pt',
               metadata_file='_Data_metadata_multigrams_wav2vec.pk', encoding_fn=multigrams_encoding, multi_head=True,
               slice_fn=Data.wav2vec_extraction, scorer=Data.compute_scores, batch_size=8, save_features=True):
    convnet_config = {'emb_dim': 384, 'hid_dim': 512}
    cp = torch.load('wav2vec_large.pt')
    wav2vec_model = Wav2VecModel.build_model(cp['args'], task=None)
    wav2vec_model.load_state_dict(cp['model'])
    wav2vec_model.eval()
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, encoding_fn=encoding_fn,
                     multi_head=multi_head, slice_fn=slice_fn, scorer=scorer, batch_size=batch_size, convnet_config=convnet_config,
                     wav2vec_model=wav2vec_model, save_features=save_features, lr=1e-5, decay_factor=0)
    u.load_model(self.model, self.save_name_model, restore_only_similars=True)


class NgramsTrainer3(ConvnetTrainer):
  def __init__(self, logfile='_logs/_logs_multigrams3.txt', save_name_model='convnet/ngrams_convnet_experiment3.pt',
               metadata_file='_Data_metadata_multigrams_mfcc0128.pk', encoding_fn=multigrams_encoding, multi_head=True,
               slice_fn=Data.mfcc_extraction, n_fft=2048, hop_length=512, scorer=Data.compute_scores, batch_size=32,
               mess_with_targets=True):
    convnet_config = {'emb_dim': 384, 'hid_dim': 512}
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, encoding_fn=encoding_fn,
                     multi_head=multi_head, slice_fn=slice_fn, n_fft=n_fft, hop_length=hop_length, scorer=scorer,
                     batch_size=batch_size, convnet_config=convnet_config, mess_with_targets=mess_with_targets)
    u.load_model(self.model, 'convnet/ngrams_convnet_experiment_051.pt', restore_only_similars=True)


class NgramsTrainer4(ConvnetTrainer):
  def __init__(self, logfile='_logs/_logs_multigrams4.txt', save_name_model='convnet/ngrams_convnet_experiment4.pt',
               metadata_file='_Data_metadata_multiEncoding_mfcc0128.pk', encoding_fn=multigrams_encoding, multi_head=True,
               slice_fn=Data.mfcc_extraction, n_fft=2048, hop_length=512, scorer=Data.compute_scores, batch_size=24):
    convnet_config = {'emb_dim': 384, 'hid_dim': 512}
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, encoding_fn=encoding_fn,
                     multi_head=multi_head, slice_fn=slice_fn, n_fft=n_fft, hop_length=hop_length, scorer=scorer,
                     batch_size=batch_size, convnet_config=convnet_config)
  
  def set_data(self):
    self.data = Data()

    if not os.path.isfile(self.metadata_file):
      self.data.set_audio_metadata(self.train_folder, self.test_folder, list_files_fn=self.list_files_fn,
                                   process_file_fn=self.process_file_fn, **self.process_file_fn_args)
      set_metadata(self.data, self.train_folder, self.test_folder)
      self.data.save_metadata(save_name=self.metadata_file)
    else:
      self.data.load_metadata(save_name=self.metadata_file)


class NgramsTrainer5(ConvnetTrainer):
  def __init__(self, logfile='_logs/_logs_ngramsEXP5.txt', save_name_model='convnet/ngrams_convnet_experiment5.pt',
               metadata_file='_Data_metadata_multiEncoding_wav2vec.pk', encoding_fn=multigrams_encoding, multi_head=True,
               slice_fn=Data.wav2vec_extraction, scorer=Data.compute_scores, batch_size=8, save_features=True):
    convnet_config = {'emb_dim': 384, 'hid_dim': 512}
    cp = torch.load('wav2vec_large.pt')
    wav2vec_model = Wav2VecModel.build_model(cp['args'], task=None)
    wav2vec_model.load_state_dict(cp['model'])
    wav2vec_model.eval()
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, encoding_fn=encoding_fn,
                     multi_head=multi_head, slice_fn=slice_fn, scorer=scorer, batch_size=batch_size, convnet_config=convnet_config,
                     wav2vec_model=wav2vec_model, save_features=save_features)
  
  def set_data(self):
    self.data = Data()

    if not os.path.isfile(self.metadata_file):
      self.data.set_audio_metadata(self.train_folder, self.test_folder, list_files_fn=self.list_files_fn,
                                   process_file_fn=self.process_file_fn, **self.process_file_fn_args)
      set_metadata(self.data, self.train_folder, self.test_folder)
      self.data.save_metadata(save_name=self.metadata_file)
    else:
      self.data.load_metadata(save_name=self.metadata_file)


class NgramsTrainer6(ConvnetTrainer):
  def __init__(self, logfile='_logs/_logs_multigrams6.txt', save_name_model='convnet/ngrams_convnet_experiment6.pt',
               metadata_file='_Data_metadata_multigrams_mfcc0128.pk', encoding_fn=multigrams_encoding, multi_head=True,
               slice_fn=Data.mfcc_extraction, n_fft=2048, hop_length=512, scorer=Data.compute_scores, batch_size=32):
    convnet_config = {'emb_dim': 384, 'hid_dim': 512}
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, encoding_fn=encoding_fn,
                     multi_head=multi_head, slice_fn=slice_fn, n_fft=n_fft, hop_length=hop_length, scorer=scorer,
                     batch_size=batch_size, convnet_config=convnet_config)
    u.load_model(self.model, 'convnet/ngrams_convnet_experiment6.pt', restore_only_similars=True)
    self.train_pass = self.beam_decoding_training


class NgramsTrainer7(ConvnetTrainer):
  def __init__(self, logfile='_logs/_logs_ngramsEXP7.txt', save_name_model='convnet/ngrams_convnet_experiment7.pt',
               metadata_file='_Data_metadata_multigrams_wav2vec.pk', encoding_fn=multigrams_encoding, multi_head=True,
               slice_fn=Data.wav2vec_extraction, scorer=Data.compute_scores, batch_size=8, save_features=True):
    convnet_config = {'emb_dim': 384, 'hid_dim': 512}
    cp = torch.load('wav2vec_large.pt')
    wav2vec_model = Wav2VecModel.build_model(cp['args'], task=None)
    wav2vec_model.load_state_dict(cp['model'])
    wav2vec_model.eval()
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, encoding_fn=encoding_fn,
                     multi_head=multi_head, slice_fn=slice_fn, scorer=scorer, batch_size=batch_size, convnet_config=convnet_config,
                     wav2vec_model=wav2vec_model, save_features=save_features, lr=1e-5)
    u.load_model(self.model, 'convnet/ngrams_convnet_experiment2.pt', restore_only_similars=True)
    self.train_pass = self.beam_decoding_training


class NgramsTrainer8(ConvnetTrainer):
  def __init__(self, logfile='_logs/_logs_ngramsEXP8.txt', save_name_model='convnet/ngrams_convnet_experiment8.pt',
               metadata_file='_Data_metadata_wordsLetters_wav2vec.pk', encoding_fn=words_letters_encoding, multi_head=True,
               slice_fn=Data.wav2vec_extraction, scorer=Data.compute_scores, batch_size=8, save_features=True):
    convnet_config = {'emb_dim': 384, 'hid_dim': 512}
    cp = torch.load('wav2vec_large.pt')
    wav2vec_model = Wav2VecModel.build_model(cp['args'], task=None)
    wav2vec_model.load_state_dict(cp['model'])
    wav2vec_model.eval()
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, encoding_fn=encoding_fn,
                     multi_head=multi_head, slice_fn=slice_fn, scorer=scorer, batch_size=batch_size, convnet_config=convnet_config,
                     wav2vec_model=wav2vec_model, save_features=save_features, lr=1e-4, decay_factor=0)
    u.load_model(self.model, self.save_name_model, restore_only_similars=True)


class CustomDataset(Dataset):
  def __init__(self, ids_to_audiofile, ids_to_transcript, sos_tok='<sos>', eos_tok='<eos>', process_file_fn=None, **kwargs):
    self.ids_to_audiofile = ids_to_audiofile
    self.ids_to_transcript = ids_to_transcript
    self.sos_tok = sos_tok
    self.eos_tok = eos_tok

    self.process_file_fn = process_file_fn = Data.read_and_slice_signal if process_file_fn is None else process_file_fn
    self.process_file_fn_args = kwargs

    self.identities = list(sorted(ids_to_audiofile.keys()))

  def __len__(self):
    return len(self.identities)
  
  def __getitem__(self, idx):
    identity = self.identities[idx]

    signal = self.process_file_fn(self.ids_to_audiofile[identity], **self.process_file_fn_args)

    encoder_input = torch.Tensor(signal)
    decoder_input = self.ids_to_transcript[identity].lower()
    decoder_input = self.sos_tok + decoder_input + self.eos_tok

    return encoder_input, decoder_input


class CustomCollator(object):
  def __init__(self, enc_pad_val):
    self.enc_pad_val = enc_pad_val

  def __call__(self, batch):
    encoder_inputs, decoder_inputs = zip(*batch)

    encoder_input_batch = pad_sequence(encoder_inputs, batch_first=True, padding_value=self.enc_pad_val).float()

    return encoder_input_batch, decoder_inputs


class MultigramsGame(object):
  def __init__(self, device=None, logfile='_multigramsGame_logs.txt', metadata_file='_Data_metadata_multigrams_mfcc0128.pk',
               train_folder='../../../datasets/openslr/LibriSpeech/train-clean-100/', process_file_fn=Data.read_and_slice_signal,
               test_folder='../../../datasets/openslr/LibriSpeech/test-clean/', encoding_fn=multigrams_encoding, batch_size=32,
               create_enc_mask=False, list_files_fn=Data.get_openslr_files, n_epochs=500, lr=1e-4, smoothing_eps=0., eval_step=1,
               save_model_path='ngrams_experiments/', subset=True, subset_percent=0.05):
    logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

    self.metadata_file = metadata_file
    self.train_folder = train_folder
    self.test_folder = test_folder
    self.list_files_fn = list_files_fn
    self.process_file_fn = process_file_fn
    self.encoding_fn = encoding_fn
    self.process_file_fn_args = {'slice_fn': Data.mfcc_extraction, 'n_fft': 2048, 'hop_length': 512}
    self.batch_size = batch_size
    self.create_enc_mask = create_enc_mask
    self.n_epochs = n_epochs
    self.smoothing_eps = smoothing_eps
    self.eval_step = eval_step
    self.save_model_path = save_model_path

    self.set_data()
    self.eos_tok = '<eos>'
    self.sos_idx = self.data.tokens_to_idx['<sos>']
    self.eos_idx = self.data.tokens_to_idx['<eos>']
    self.pad_idx = self.data.tokens_to_idx['<pad>']

    self.max_source_len = max(max(map(len, self.data.ids_to_transcript_train.values())),
                              max(map(len, self.data.ids_to_transcript_test.values()))) + 10

    self.set_data_loader(subset=subset, subset_percent=subset_percent)

    n_letters, n_bigrams, n_trigrams, n_words = 31, 294, 1500, 3000
    self.predictors_idx = [0, n_letters, n_letters + n_bigrams, n_letters + n_bigrams + n_trigrams]
    self.action_space = 4
    self.letters = self.data.idx_to_tokens[:n_letters]
    self.bigrams = self.data.idx_to_tokens[n_letters:n_letters+n_bigrams]
    self.trigrams = self.data.idx_to_tokens[n_letters+n_bigrams:n_letters+n_bigrams+n_trigrams]
    self.words = self.data.idx_to_tokens[n_letters+n_bigrams+n_trigrams:]

    model_config = {'enc_input_dim': self.data.n_signal_feats, 'enc_max_seq_len': self.data.max_signal_len, 'enc_dropout': 0.25,
                    'reduce_dim': False, 'emb_dim': 256, 'hid_dim': 512, 'enc_layers': 10, 'enc_kernel_size': 3, 'dec_n_blocks': 4,
                    'dec_input_dim': len(self.data.idx_to_tokens), 'dec_max_seq_len': self.max_source_len, 'dec_dropout': 0.25,
                    'd_model': 256, 'd_keys': 64, 'd_values': 64, 'n_heads': 4, 'attn_dropout': 0.1, 'out_proj': 384,
                    'n_letters': n_letters, 'n_bigrams': n_bigrams, 'n_trigrams': n_trigrams, 'n_words': n_words, 'action_space': 4,
                    'proj_n_blocks': 4, 'proj_dropout': 0.25}
    self.instanciate_model(**model_config)

    n_params = sum(p.numel() for layer in [self.encoder, self.dec_embedder, self.self_attn_dec_in, self.enc_in_dec_in_attention,
                                           self.dec_out_attn_proj, self.proj_self_attn, self.predictors, self.action_layer]
                                for p in layer.parameters() if p.requires_grad)
    logging.info(f'The model has {n_params:,} trainable parameters')

    self.optimizer = optim.Adam(self.parameters, lr=lr)
    self.criterion = u.CrossEntropyLoss(self.pad_idx)

    if not os.path.isdir(save_model_path):
      os.makedirs(save_model_path)

    self.load_model()
  
  def set_data(self):
    self.data = Data()

    if not os.path.isfile(self.metadata_file):
      self.data.set_audio_metadata(self.train_folder, self.test_folder, list_files_fn=self.list_files_fn,
                                   process_file_fn=self.process_file_fn, **self.process_file_fn_args)
      self.data.process_all_transcripts(self.train_folder, self.test_folder, encoding_fn=self.encoding_fn)
      self.data.save_metadata(save_name=self.metadata_file)
    else:
      self.data.load_metadata(save_name=self.metadata_file)
  
  def set_data_loader(self, num_workers=4, shuffle=True, subset=False, subset_percent=0.2):
    custom_collator = CustomCollator(0)

    train_custom_dataset = CustomDataset(self.data.ids_to_audiofile_train, self.data.ids_to_transcript_train,
                                         process_file_fn=self.process_file_fn, **self.process_file_fn_args)
    test_custom_dataset = CustomDataset(self.data.ids_to_audiofile_test, self.data.ids_to_transcript_test,
                                         process_file_fn=self.process_file_fn, **self.process_file_fn_args)
    
    if subset:
      train_custom_dataset = Data.extract_subset(train_custom_dataset, percent=subset_percent)
      test_custom_dataset = Data.extract_subset(test_custom_dataset, percent=subset_percent)
    
    self.train_data_loader = DataLoader(train_custom_dataset, batch_size=self.batch_size, num_workers=num_workers, shuffle=shuffle,
                                        collate_fn=custom_collator)
    self.test_data_loader = DataLoader(test_custom_dataset, batch_size=self.batch_size, num_workers=num_workers, shuffle=shuffle,
                                       collate_fn=custom_collator)

  def instanciate_model(self, enc_input_dim=400, emb_dim=256, hid_dim=512, enc_max_seq_len=1400, enc_dropout=0.25, reduce_dim=False,
                        enc_layers=10, enc_kernel_size=3, dec_input_dim=4825, dec_max_seq_len=600, dec_dropout=0.25, dec_n_blocks=4,
                        d_ff=512, d_model=256, d_keys=64, d_values=64, n_heads=4, attn_dropout=0.1, out_proj=384, n_letters=31,
                        n_bigrams=294, n_trigrams=1500, n_words=3000, action_space=4, dec_out_attn_proj_inner=2048,
                        proj_n_blocks=4, proj_dropout=0.25):
    enc_embedder = css.EncoderEmbedder(enc_input_dim, emb_dim, hid_dim, enc_max_seq_len, enc_dropout, self.device, reduce_dim=reduce_dim)
    self.encoder = css.Encoder(emb_dim, hid_dim, enc_layers, enc_kernel_size, enc_dropout,
                               self.device, embedder=enc_embedder).to(self.device)

    self.dec_embedder = css.DecoderEmbedder(dec_input_dim, emb_dim, dec_max_seq_len, dec_dropout, self.device).to(self.device)
    self.self_attn_dec_in = TransformerEncoder(dec_n_blocks, d_model, d_keys, d_values, n_heads, d_ff,
                                               dropout=dec_dropout, act_fn='relu', block_type='standard').to(self.device)
    self.enc_in_dec_in_attention = MultiHeadAttention(d_model, d_keys, d_values, n_heads, dropout=attn_dropout).to(self.device)
    self.dec_out_attn_proj = torch.nn.Sequential(torch.nn.Linear(emb_dim + d_model, out_proj), torch.nn.ReLU(inplace=True)).to(self.device)

    self.proj_self_attn = TransformerEncoder(proj_n_blocks, out_proj, d_keys, d_values, out_proj//d_keys, d_ff,
                                             dropout=proj_dropout, act_fn='relu', block_type='standard').to(self.device)

    self.predictors = torch.nn.ModuleList([torch.nn.Linear(out_proj, n_letters),  # letters
                                           torch.nn.Linear(out_proj, n_bigrams),  # bigrams
                                           torch.nn.Linear(out_proj, n_trigrams),  # trigrams
                                           torch.nn.Linear(out_proj, n_words)]).to(self.device)  # words
    # action_space = 4, choose between predicting letter, bigram, trigram or word
    self.action_layer = torch.nn.Linear(out_proj, action_space).to(self.device)

    self.parameters = chain(*([self.encoder.parameters()] + [self.dec_embedder.parameters()] + [self.self_attn_dec_in.parameters()]
                               + [self.enc_in_dec_in_attention.parameters()] + [self.dec_out_attn_proj.parameters()]
                               + [self.predictors.parameters()] + [self.action_layer.parameters()]))
  
  def save_model(self):
    u.save_checkpoint(self.encoder, None, self.save_model_path + 'encoder.pt')
    u.save_checkpoint(self.dec_embedder, None, self.save_model_path + 'dec_embedder.pt')
    u.save_checkpoint(self.self_attn_dec_in, None, self.save_model_path + 'self_attn_dec_in.pt')
    u.save_checkpoint(self.enc_in_dec_in_attention, None, self.save_model_path + 'enc_in_dec_in_attention.pt')
    u.save_checkpoint(self.dec_out_attn_proj, None, self.save_model_path + 'dec_out_attn_proj.pt')
    u.save_checkpoint(self.predictors, None, self.save_model_path + 'predictors.pt')
    u.save_checkpoint(self.action_layer, None, self.save_model_path + 'action_layer.pt')
    u.save_checkpoint(self.proj_self_attn, None, self.save_model_path + 'proj_self_attn.pt')
  
  def load_model(self):
    u.load_model(self.encoder, self.save_model_path + 'encoder.pt', restore_only_similars=True)
    u.load_model(self.dec_embedder, self.save_model_path + 'dec_embedder.pt', restore_only_similars=True)
    u.load_model(self.self_attn_dec_in, self.save_model_path + 'self_attn_dec_in.pt', restore_only_similars=True)
    u.load_model(self.enc_in_dec_in_attention, self.save_model_path + 'enc_in_dec_in_attention.pt', restore_only_similars=True)
    u.load_model(self.dec_out_attn_proj, self.save_model_path + 'dec_out_attn_proj.pt', restore_only_similars=True)
    u.load_model(self.predictors, self.save_model_path + 'predictors.pt', restore_only_similars=True)
    u.load_model(self.action_layer, self.save_model_path + 'action_layer.pt', restore_only_similars=True)
    u.load_model(self.proj_self_attn, self.save_model_path + 'proj_self_attn.pt', restore_only_similars=True)
  
  def _train_mode(self):
    self.encoder.train()
    self.dec_embedder.train()
    self.self_attn_dec_in.train()
    self.enc_in_dec_in_attention.train()
    self.dec_out_attn_proj.train()
    self.predictors.train()
    self.action_layer.train()
    self.proj_self_attn.train()
  
  def _eval_mode(self):
    self.encoder.eval()
    self.dec_embedder.eval()
    self.self_attn_dec_in.eval()
    self.enc_in_dec_in_attention.eval()
    self.dec_out_attn_proj.eval()
    self.predictors.eval()
    self.action_layer.eval()
    self.proj_self_attn.eval()
  
  def train(self):
    print('Start Training...')
    eval_accuracy_memory = 0
    for epoch in tqdm(range(self.n_epochs)):
      compute_scores = False if epoch % self.eval_step == 0 else True

      # epoch_action_loss, epoch_ce_loss, accuracies
      eal, ecl, accs = self.multigrams_pass(self.train_data_loader, only_loss=compute_scores)
      logging.info(f"Epoch {epoch} | train_loss = {eal:.3f}-{ecl:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")

      self._eval_mode()
      with torch.no_grad():
        # eval_action_loss, eval_ce_loss, accuracies
        eal, ecl, accs = self.multigrams_pass(self.test_data_loader, only_loss=compute_scores, training=False, dump_sample=False)
        logging.info(f"Epoch {epoch} | test_loss = {eal:.3f}-{ecl:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")
      self._train_mode()

      oea = accs.get('word_accuracy', None)

      if oea is not None and oea > eval_accuracy_memory:
        logging.info(f'Save model with eval_accuracy = {oea:.3f}')
        self.save_model()
        eval_accuracy_memory = oea
  
  def get_action(self, state, enc_out_conved, enc_out_combined):
    # state = [batch_size, current_seq_len], enc_out_conved = [batch_size, enc_seq_len, emb_dim]
    dec_in = self.dec_embedder(state)  # [batch_size, current_seq_len, emb_dim]
    dec_out = self.self_attn_dec_in(dec_in)  # [batch_size, current_seq_len, emb_dim]
    attention = self.enc_in_dec_in_attention(dec_out, enc_out_conved, enc_out_combined)  # [batch_size, current_seq_len, d_model=emb_dim]

    dec_out = self.dec_out_attn_proj(torch.cat((dec_out, attention), axis=-1))  # [batch_size, current_seq_len, out_proj]
    dec_out = self.proj_self_attn(dec_out)  # [batch_size, current_seq_len, out_proj]
    dec_out = dec_out[:, -1]  # [batch_size, out_proj]

    action_probs = self.action_layer(dec_out).softmax(-1)  # [batch_size, action_space]
    return action_probs, dec_out
  
  def step(self, actions, dec_out, target_texts, text_states, ends):
    current_lens = list(map(len, text_states))
    rewards = torch.Tensor([0]).repeat(len(ends))
    next_states = torch.zeros(len(ends)).long()
    cross_entropy_losses = 0

    for i in range(len(ends)):
      if ends[i]:
        continue

      output = self.predictors[actions[i]](dec_out[i:i+1])  # [1, predictor_out_size]
      output_idx = output.argmax() + self.predictors_idx[actions[i]]

      pred_token = self.data.idx_to_tokens[output_idx]
      target_token = target_texts[i][current_lens[i]:current_lens[i] + len(pred_token)]
      
      rewards[i] = float(sum(np.array(list(pred_token)) == np.array(list(target_token)))) if len(pred_token) == len(target_token) else -1

      next_states[i] = output_idx
      text_states[i] += pred_token

      if pred_token == self.eos_tok or len(text_states[i]) >= len(target_texts[i]):
        ends[i] = True
        rewards[i] += 10 if text_states[i] == target_texts[i] else -1
      
      ## Compute CrossEntropy loss if possible
      if actions[i] == 0:
        good_rep = self.data.tokens_to_idx[target_texts[i][current_lens[i]]] if target_texts[i][current_lens[i]] in self.letters else None
      elif actions[i] == 1:
        if target_texts[i][current_lens[i]:current_lens[i]+2] in self.bigrams:
          good_rep = self.data.tokens_to_idx[target_texts[i][current_lens[i]:current_lens[i]+2]]
        else:
          good_rep = None
      elif actions[i] == 2:
        if target_texts[i][current_lens[i]:current_lens[i]+3] in self.trigrams:
          good_rep = self.data.tokens_to_idx[target_texts[i][current_lens[i]:current_lens[i]+3]]
        else:
          good_rep = None
      else:
        if ' ' in target_texts[i][current_lens[i]:]:
          next_word = target_texts[i][current_lens[i]:target_texts[i][current_lens[i]:].index(' ')]
        elif self.eos_tok in target_texts[i][current_lens[i]:]:
          next_word = target_texts[i][current_lens[i]:target_texts[i][current_lens[i]:].index(self.eos_tok)]
        else:
          next_word = None
        good_rep = self.data.tokens_to_idx[next_word] if next_word in self.words else None
      
      if good_rep is not None:
        good_rep -= self.predictors_idx[actions[i]]
        cross_entropy_losses += self.criterion(output, torch.LongTensor([good_rep]).to(self.device), epsilon=self.smoothing_eps)
    
    return next_states, rewards, text_states, ends, cross_entropy_losses
  
  def multigrams_pass(self, data_loader, only_loss=True, training=True, dump_sample=False):
    action_losses, ce_losses, accs = 0, 0, {}
    for enc_in, target_texts in tqdm(data_loader):  # enc_in = [batch_size, enc_seq_len, enc_input_dim]
      current_batch_size = enc_in.shape[0]
      enc_out_conved, enc_out_combined = self.encoder(enc_in.to(self.device))  # [batch_size, enc_seq_len, emb_dim]

      states = torch.LongTensor([self.data.tokens_to_idx['<sos>']]).repeat(current_batch_size, self.max_source_len).to(self.device)
      text_states = ['<sos>'] * current_batch_size
      ends = torch.zeros(current_batch_size).bool().to(self.device)
      rewards = torch.zeros(current_batch_size, self.max_source_len).to(self.device)

      for i in range(self.max_source_len - 1):
        current_state = states[:, :i+1].clone()
        action_probs, dec_out = self.get_action(current_state, enc_out_conved, enc_out_combined)
        action = torch.multinomial(action_probs, 1)  # [batch_size, 1]

        state, reward, text_states, ends, cross_entropy_losses = self.step(action, dec_out, target_texts, text_states, ends)

        rewards[:, i] = reward
        states[:, i+1] = state

        current_rewards = rewards.sum(-1)
        action_loss = -(current_rewards * torch.log(action_probs[torch.arange(current_batch_size), action.reshape(-1)]) * ends).mean()

        if training:
          self.optimizer.zero_grad()
          if not isinstance(cross_entropy_losses, int):
            cross_entropy_losses.backward(retain_graph=True)

          action_loss.backward(retain_graph=True)
          self.optimizer.step()

        action_losses += action_loss.item()
        ce_losses += cross_entropy_losses if isinstance(cross_entropy_losses, int) else cross_entropy_losses.item()

        if ends.sum() == current_batch_size:
          break
    
    if not only_loss:
      accs = Data.compute_scores(targets=target_texts, predictions=text_states, rec=False)
    
    if dump_sample:
      to_dump = ['\n'] + [f'Target: {t}\nPrediction: {p}\n' for t, p in zip(target_texts[:5], text_states[:5])]
      logging.info('\n'.join(to_dump))
    
    return action_losses, ce_losses, accs
  
  def oracle_train(self):
    pass


if __name__ == "__main__":
  ## SEEDING FOR REPRODUCIBILITY
  SEED = 42
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)

  experiments = {k.replace('NgramsTrainer', ''): v for k, v in locals().items() if re.search(r'NgramsTrainer\d+', k) is not None}
  
  rep = input(f'Which Experiment do you want to start? ({",".join(experiments.keys())}): ')
  exp = experiments[rep]()
  exp.train()

  # analyze()

  # mg = MultigramsGame()
  # mg.train()