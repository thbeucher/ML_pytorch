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
import numpy as np
import pickle as pk

from tqdm import tqdm
from collections import Counter

sys.path.append(os.path.abspath(__file__).replace('ASR/ngrams_experiments.py', ''))
import utils as u

from data import Data
from convnet_trainer import ConvnetTrainer


def get_ngrams(sources, save_data='_ngrams_data.pk', n_words=3000, n_trigrams=1500):
  # bigrams, trigrams and sorted by occurences
  if not os.path.isfile(save_data):
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

    with open(save_data, 'wb') as f:
      pk.dump([letters, bigrams, trigrams, words], f)
  else:
    with open(save_data, 'rb') as f:
      letters, bigrams, trigrams, words = pk.load(f)
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


def multigrams_encoding(sources, sos_tok='<sos>', eos_tok='<eos>', pad_tok='<pad>'):
  sources = [s.lower() for s in sources]

  letters, bigrams, trigrams, words = get_ngrams(sources)
  letters = [sos_tok, eos_tok, pad_tok] + letters
  print(f'=> letters = {len(letters)} | bigrams = {len(bigrams)} | trigrams = {len(trigrams)} | words = {len(words)}')

  letters_to_idx = {l: i for i, l in enumerate(letters)}
  bigrams_to_idx = {b: i + len(letters) for i, b in enumerate(bigrams)}
  trigrams_to_idx = {t: i + len(letters) + len(bigrams) for i, t in enumerate(trigrams)}
  words_to_idx = {w: i + len(letters) + len(bigrams) + len(trigrams) for i, w in enumerate(words)}

  sources_encoded = [ngrams_encoding(s, letters_to_idx, bigrams_to_idx, trigrams_to_idx, words_to_idx) for s in tqdm(sources)]

  multigrams_to_idx = {**letters_to_idx, **bigrams_to_idx, **trigrams_to_idx, **words_to_idx}
  idx_to_multigrams = [k for k in multigrams_to_idx]

  sources_encoded = [[multigrams_to_idx[sos_tok]] + se + [multigrams_to_idx[eos_tok]] for se in sources_encoded]
  
  return sources_encoded, idx_to_multigrams, multigrams_to_idx


def analyze():
  # keys = ids_to_audiofile_train, ids_to_audiofile_test, max_signal_len, max_source_len, ids_to_transcript_train,
  # ids_to_transcript_test, ids_to_encodedsources_train, ids_to_encodedsources_test, idx_to_tokens, tokens_to_idx, n_signal_feats
  with open('divers/_Data_metadata_letters_raw0025.pk', 'rb') as f:
    data = pk.load(f)
  
  sources = [s.lower() for s in data['ids_to_transcript_train'].values()] + [s.lower() for s in data['ids_to_transcript_test'].values()]

  sources_encoded, idx_to_multigrams, multigrams_to_idx = multigrams_encoding(sources)
  lens = [len(el) for el in sources_encoded]
  print(f'min = {min(lens)} | max = {max(lens)} | mean = {np.mean(lens)}')


class NgramsTrainer(ConvnetTrainer):
  def __init__(self, logfile='_ngrams_experiment_logs.txt', save_name_model='convnet/ngrams_convnet_experiment.pt',
               metadata_file='_Data_metadata_multigrams_mfcc0128.pk', encoding_fn=multigrams_encoding, multi_head=True,
               slice_fn=Data.mfcc_extraction, n_fft=2048, hop_length=512, scorer=Data.compute_scores, batch_size=32):
    convnet_config = {'emb_dim': 512, 'hid_dim': 1024}
    super().__init__(logfile=logfile, save_name_model=save_name_model, metadata_file=metadata_file, encoding_fn=encoding_fn,
                     multi_head=multi_head, slice_fn=slice_fn, n_fft=n_fft, hop_length=hop_length, scorer=scorer,
                     batch_size=batch_size, convnet_config=convnet_config)
  
  def compute_loss(self, preds, dec_in):
    loss = 0
    for i, di in enumerate(dec_in):
      if di < 31:
        loss += self.criterion(preds[i][:,:31], di, epsilon=self.smoothing_eps)
      elif di < len(self.data.idx_to_tokens[:-4500]):
        loss += self.criterion(preds[i][:,31:-4500], di-31, epsilon=self.smoothing_eps)
      elif di < len(self.data.idx_to_tokens[:-3000]):
        loss += self.criterion(preds[i][:,-4500:-3000], di-31-294, epsilon=self.smoothing_eps)
      else:
        loss += self.criterion(preds[i][:,-3000:], di-31-294-1500, epsilon=self.smoothing_eps)
    return loss / sum(preds.shape[:2])


if __name__ == "__main__":
  ## SEEDING FOR REPRODUCIBILITY
  SEED = 42
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)

  # analyze()
  nt = NgramsTrainer()
  nt.train()