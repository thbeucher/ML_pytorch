import os
import re
import ast
import json
import math
import torch
import random
import logging
import requests
import itertools
import numpy as np
import pickle as pk
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from visdom import Visdom
from tabulate import tabulate
from PIL import Image, ImageDraw, ImageFont
from collections import Counter, OrderedDict


class GELU(nn.Module):
  def __init__(self, inplace=True):
    super().__init__()

  def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def gelu(x):
  return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def populate_configuration(base_configuration, to_add):
  '''
  Adds configuration keys from to_add to base_configuration if it doesn't exist

  Params:
    * base_configuration : dictionary
    * to_add : dictionary

  Returns
    * base_configuration : dictionary
  '''
  for k, v in to_add.items():
    if k not in base_configuration:
      base_configuration[k] = v
  return base_configuration


def create_positional_embedding(max_seq_len, embedding_size, d_model):
  '''
  Positional Embedding from https://arxiv.org/abs/1706.03762

  Params:
    * max_seq_len : int
    * embedding_size : int
    * d_model : int
  
  Returns:
    * torch.nn.Embedding instance
  '''
  emb_matrix = (
    [
      np.sin(np.array(range(max_seq_len), dtype=np.float32) / (10000 ** (i / d_model))),
      np.cos(np.array(range(max_seq_len), dtype=np.float32) / (10000 ** (i / d_model)))
    ]
    for i in range(0, embedding_size, 2)
  )
  emb_matrix = np.stack(list(itertools.chain(*emb_matrix))).T

  # if max_seq_len is an odd number, than the last entry of the embedding matrix has to be removed again
  if emb_matrix.shape[0] > max_seq_len:
    emb_matrix = emb_matrix[:-1]

  return nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix))


def right_shift_sequence(x, zero_range=1e-22):
  '''
  Params:
    * x : torch.tensor, shape = (batch_size, sequence_len, d_model)
    * zero_range (optional) : float, defaul to 1e-22
  
  Returns:
    * torch.tensor
  '''
  batch_size, _, d_model = x.shape
  return torch.cat(
    [
      x.new(batch_size, 1, d_model).uniform_(-zero_range, zero_range),
      x[:, :-1, :]
    ],
    dim=1
  )


def create_padding_mask(x, pad_idx):
  '''
  Params:
    * x : torch.tensor, shape = (batch, seq_len) or (batch, seq_len, num_feat)
    * pad_idx : int
  
  Returns:
    * torch.tensor
  '''
  if x.dim() == 3:
    x = x[:, :, 0]
  return (x == pad_idx).unsqueeze(1).unsqueeze(2)


def create_futur_mask(x):
  '''
  Creates a mask to hide the next tokens during Decoder training of the Transformer

  Params:
    * x : torch.tensor, shape = (batch, seq_len)
  
  Returns:
    * torch.tensor
  '''
  batch_size, seq_len = x.size(0), x.size(1)
  return torch.triu(torch.ones(batch_size, seq_len, seq_len), diagonal=1) == 0


def save_checkpoint(model, optimizer, save_path):
  '''
  Params:
    * model : torch.nn.Module
    * optimizer : torch.nn.Module
    * save_path : string
  '''
  torch.save({
              'model_state_dict': model.state_dict() if model is not None else None,
              'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None
              }, save_path)


def load_model(model, save_name, optimizer=None, restore_only_model=False, restore_only_similars=False, map_location='cpu', verbose=True):
  '''
  Params:
    * model : torch.nn.Module
    * save_name : str
    * optimizer (optional) : torch.nn.Module
    * restore_only_model (optional) : Boolean, default to False
    * restore_only_similars (optional) : Boolean, default to False
    * map_location (optional) : str, default to 'cpu'
    * verbose (optional) : Boolean, default to True
  '''
  if os.path.isfile(save_name):
    if verbose:
      print(f'Restoring weights from {save_name} ...')

    checkpoint = torch.load(save_name, map_location=map_location)

    # handle if model saved with nn.DataParallel and loaded without
    if 'module.' in list(checkpoint['model_state_dict'].keys())[0] and not 'module.' in list(model.state_dict().keys())[0]:
      checkpoint['model_state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    
    # load data from saved model only if name and shape correspond
    if restore_only_similars:
      current_keys = list(model.state_dict().keys())
      current_shapes = {k: v.shape for k, v in model.state_dict().items()}
      n_tensors_in_checkpoint = len(checkpoint['model_state_dict'])
      checkpoint['model_state_dict'] = {k: v for k, v in checkpoint['model_state_dict'].items()
                                              if k in current_keys and v.shape == current_shapes[k]}
      
      if verbose:
        print(f"LOADING MODEL - {len(checkpoint['model_state_dict'])}/{n_tensors_in_checkpoint} tensors loaded from checkpoint")

      checkpoint['model_state_dict'] = {**model.state_dict(), **checkpoint['model_state_dict']}
    
    if not restore_only_model and optimizer is not None:
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.load_state_dict(checkpoint['model_state_dict'])
  else:
    if verbose:
      print(f'File {save_name} doesnt exist, cannot load data from it...')


def initialize_network(model, distribution=nn.init.xavier_uniform_):
  '''
  Params:
    * model : torch.nn.Module
    * distribution (optional) : distribution to use for the model parameters initialization
  '''
  for p in model.parameters():
    if p.dim() > 1:
      distribution(p)


def layer_normalization(x, epsilon=1e-15):
  '''
  Basic Layer Normalization

  Params:
    * x : torch.tensor
    * epsilon (optional) : float
  
  Returns:
    * torch.tensor
  '''
  mean = torch.mean(x, dim=-1, keepdim=True)
  std = torch.std(x, dim=-1, keepdim=True)
  
  return (x - mean) / (std + epsilon)


def load_json(path):
  '''path : str'''
  with open(path, 'r') as f:
    data = json.load(f)
  return data


def load_pickle(path):
  '''path : str'''
  with open(path, 'rb') as f:
    data = pk.load(f)
  return data


def get_human_readable(size, precision=2):
  '''
  Returns interpretable string of object size
  use to parse number of bytes given by my_tensor.element_size() * my_tensor.nelement() into TB, GB, MB, KB, B

  Params:
    * size
    * precision (optional) : int
  
  Returns:
    * str
  '''
  suffixes=['B','KB','MB','GB','TB']
  suffixIndex = 0
  while size > 1024:
      suffixIndex += 1 #increment the index of the suffix
      size = size/1024.0 #apply the division
  return "%.*f %s"%(precision,size,suffixes[suffixIndex])


def server_is_running(port, hostname):
  '''
  Checks if given server is currently running or not

  Params:
    * port : int
    * hostname : str
  
  Returns:
    * is_running : Boolean
  '''
  server_addr = f'http://{hostname}:{port}'

  try:
    requests.get(server_addr)
    is_running = True
  except:
    is_running = False
  
  print(f'Server {server_addr} is running: {is_running}')

  return is_running


class VisdomPlotter(object):
  def __init__(self, env_name='main', port=80, hostname='localhost'):
    '''
    Params:
      * env_name : str
      * port : int
      * hostname : str
    '''
    self.server_is_running = server_is_running(port, hostname)
    self.viz = Visdom(port=port) if self.server_is_running else None
    self.env = env_name
    self.plots = {}

  def line_plot(self, var_name, split_name, title_name, x, y, x_label='Epochs'):
    '''
    Params:
      * var_name : variable name (e.g. loss, acc)
      * split_name : split name (e.g. train, val)
      * title_name : titles of the graph (e.g. Classification Accuracy)
      * x : x axis value (e.g. epoch number)
      * y : y axis value (e.g. epoch loss)
    '''
    if not self.server_is_running:
      return

    if var_name not in self.plots:
      self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
        legend=[split_name],
        title=title_name,
        xlabel=x_label,
        ylabel=var_name
      ))
    else:
      self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
  
  def heatmap_plot(self, var_name, split_name, title_name, data, x_label='Epochs'):
    '''
    Params:
      * var_name : str
      * split_name : str
      * title_name : str
      * data
      * x_label (optional) : str
    '''
    if not self.server_is_running and data is not None:
      return

    labels = list(map(lambda x: x[0], data[0]))
    y = np.array([list(map(lambda x: x[1], line)) for line in data]).T
    ax = sns.heatmap(y, yticklabels=labels, cbar=False)

    plt.tight_layout()
    plt.title(title_name)
    plt.xlabel(x_label)
    plt.ylabel('layers')

    self.viz.matplot(plt, win=var_name, env=self.env)
  
  def matplot_plot(self, var_name, data):
    '''
    Params:
      * var_name : str
      * data
    '''
    if not self.server_is_running and data is not None:
      return
    self.viz.matplot(data, win=var_name, env=self.env)
  
  def image_plot(self, var_name, image):
    '''
    Params:
      * var_name : str
      * image
    '''
    if not self.server_is_running or image is None:
      return
    self.viz.image(image, win=var_name, env=self.env)


def compute_WER(r, h):
  '''
  Computes Word Error Rate, similar to Levenshtein distance on word level

  WER = (D + I + S) / N

  D = number of deletions
  I = number of insertions
  S = number of substitutions
  N = number of words in the reference

  Params:
    * r : string, reference string
    * f : string, hypothese string
  '''
  rw, hw = r.split(), h.split()  # reference_words, hypothese_words

  M = np.zeros((len(rw) + 1, len(hw) + 1))
  M[0] = np.arange(len(M[0]))
  M[:, 0] = np.arange(len(M[:, 0]))

  costs = np.array([[0 if rw[i] == hw[j] else 1 for j in range(len(hw))] for i in range(len(rw))])

  for i in range(1, len(rw) + 1):
    for j in range(1, len(hw) + 1):
      deletion = M[i - 1][j] + 1
      insertion = M[i][j - 1] + 1
      substitution = M[i - 1][j - 1] + costs[i - 1][j - 1]
      M[i][j] = min(deletion, insertion, substitution)
  
  return M[len(rw)][len(hw)] / len(rw)


def retrieve_ngrams(words, n=2, uniq=True):
  '''
  Retrieves n-grams of given words

  Example:
    3-grams of word hello -> [hel, ell, llo]
  
  Params:
    * words : list of string
    * n : int
    * uniq : boolean, True to get uniq n-grams list
  
  Returns:
    * list of string
  '''
  ngrams = []
  for word in words:
    word_ngrams = [word[i:i+n] for i in range(len(word)-n+1)]
    ngrams += word_ngrams
  return list(set(ngrams)) if uniq else ngrams


def process_ngrams(words, max_ngrams=3):
  '''
  Retrieves n-grams from given words and compute frequency statistics on it

  Params:
    * words : list of string
    * max_ngrams : int, maximum n-grams to retrieve from words, if n=3 it will retrieve trigrams and bigrams
  
  Returns:
    * freq_ngrams : dictionary where (key = given n-grams) and (value = Counter object)
    * letters : list of string
  '''
  ngrams = {}
  for n in range(max_ngrams, 1, -1):
    ngrams[n] = retrieve_ngrams(words, n=n, uniq=False)
  letters = list(set([l for w in words for l in w]))

  freq_ngrams = {n: Counter(g) for n, g in ngrams.items()}

  return freq_ngrams, letters


def remove_match_doublons(matchs):
  '''
  Removes overlapping doublons

  Params:
    * matchs : list like [(ngram, start_idx, end_idx), ...]

  Returns:
    * keeped_matchs
  '''
  ordered_matchs = sorted(matchs, key=lambda x: x[1])
  keeped_matchs = [ordered_matchs[0]]

  for i in range(1, len(ordered_matchs)):
    if ordered_matchs[i][1] >= ordered_matchs[i-1][2]:
      keeped_matchs.append(ordered_matchs[i])

  return keeped_matchs


def split_process(word, matchs):
  '''
  Decomposes given word into list of tuple where each tuple contain a substring and a boolean
  that inform is this substring were in the given matchs list

  Params:
    * word : string
    * matchs : list like [(ngram, start_idx, end_idx), ...]
  
  Returns:
    * out : list like [(ngram, boolean), ...] where boolean inform if ngram were in matchs list
  '''
  matchs = remove_match_doublons(matchs)
  word_piece = ''
  out = []
  start_idxs = list(map(lambda x: x[1], matchs))

  i = 0
  while True:
    if i in start_idxs:
      if word_piece != '':
        out.append((word_piece, False))
        word_piece = ''

      end_idx = matchs[start_idxs.index(i)][2]
      out.append((matchs[start_idxs.index(i)][0], True))
      start_idxs[start_idxs.index(i)] = False
      i = end_idx

      if sum(start_idxs) == 0:
        word_piece = word[end_idx:]
        break
    else:
      word_piece += word[i]
      i += 1

  if word_piece != '':
    out.append((word_piece, False))

  return out


def flat_list(mylist):
  '''
  Flats given list

  Example:
    mylist = [hi, [my, name]] -> [hi, my , name]

  Params:
    * mylist : list
  
  Returns:
    * list flattened
  '''
  flattened = []
  for el in mylist:
    if isinstance(el, list):
      flattened += el
    else:
      flattened.append(el)
  return flattened


def process_word(word, vocab_2_idx, n=3):
  '''
  Decomposes a word into substring to reveals n-grams matchs (n-grams to match are retrieve from the vocabulary)
  It creates a list of tuple were each tuple are like (ngram(string), boolean)
  if n=3 the ngrams of each tuple will be trigrams

  Params:
    * word : string
    * vocab_2_idx : dictionary like {ngram(string): index(int), ...}
    * n (optional) : int
  
  Returns:
    * list of tuple
  '''
  ngrams = [el for el in vocab_2_idx if len(el) == n and el in word]

  if len(ngrams) == 0:
    return [(word, False)]

  matchs = [(ngram, match.start(), match.end()) for ngram in ngrams for match in re.finditer(ngram, word)]

  return split_process(word, matchs)


def pad_documents(docs, pad_idx):
  '''
  Pads documents with pad_idx to retrieves a list of equal sublists

  Params:
    * docs : list of list of int
    * pad_idx : int
  
  Returns:
    * padded list of list
  '''
  max_len_doc = max([len(d) for d in docs])
  for d in docs:
    d.extend([pad_idx] * (max_len_doc - len(d)))
  return docs


def count_trainable_parameters(model):
  '''model : torch.nn.Module'''
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_imported_libs(dirname='.'):
  '''
  Reads .py files from given directory and extract a list of imported libraries

  Params:
    * dirname (optional) : str

  Returns:
    * list
  '''
  def get_imports(filename):
    with open(filename, 'r') as f:
      lines = f.read().splitlines()
    ilines = [l for l in lines if 'import ' in l and '=' not in l]

    return [l.split()[1] for l in ilines]

  libs = []
  for filename in os.listdir(dirname):
    if os.path.isdir(filename):
      for file_dir in os.listdir(filename):
        if file_dir[-3:] == '.py':
          libs += get_imports(os.path.join(filename, file_dir))
    elif '.py' in filename:
      libs += get_imports(filename)
  
  return list(set([i.split('.')[0] if '.' in i else i for i in list(set(libs))]))


def gaussian(x, mu, sigma):
  return np.exp(-(x - mu) ** 2/(2 * (sigma ** 2)))


def dump_argparser_parameters(args):
  params = [p for p in dir(args) if not '__' in p and not '_get' in p]
  max_size = max(map(len, params))

  logging.info('PARAMETERS:')
  for p in params:
    logging.info(f"{p + ' ' + '-' * ((max_size + 1) - len(p)) + '>:'} {getattr(args, p, '')}")


def dump_dict(dict_to_dump, dump_title):
  '''
  Params:
    * dict_to_dump : dict
    * dump_title : str
  '''
  max_size = max(map(len, dict_to_dump.keys()))
  
  logging.info(f'{dump_title} : ')
  for k, v in dict_to_dump.items():
    logging.info(f"{k + ' ' + '-' * ((max_size + 1) - len(k)) + '>:'} {v}")


def extract_subset(dataset, percent=0.2):
  '''
  Parameters:
    * dataset : torch.utils.data.Dataset
    * percent : float, between 0 and 1
  '''
  num_samples = int(percent * len(dataset))
  subset = random.sample(list(range(len(dataset))), num_samples)
  return torch.utils.data.Subset(dataset, subset)


class CrossEntropyLoss(nn.Module):
  '''
  Cross-Entropy loss that implements Label-Smoothing
  '''
  def __init__(self, pad_idx):
    '''
    Params:
      * pad_idx : int
    '''
    super().__init__()
    self.pad_idx = pad_idx
    self.cross_entropy = nn.CrossEntropyLoss(ignore_index=pad_idx)
  
  def _compute_cross_entropy(self, outputs, targets, epsilon=0.1):
    n_class = outputs.shape[-1]

    one_hot = torch.zeros_like(outputs).scatter(1, targets.view(-1, 1), 1)
    one_hot = one_hot * (1 - epsilon) + (1 - one_hot) * epsilon / (n_class - 1)
    log_prb = torch.nn.functional.log_softmax(outputs, dim=1)

    non_pad_mask = targets.ne(self.pad_idx)

    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.masked_select(non_pad_mask).mean()

    return loss
  
  def forward(self, outputs, targets, epsilon=0.):
    '''
    Params:
      * outputs : torch.tensor
      * targets : torch.tensor
      * epsilon (optional) : float, default to 0.1, use for Label-Smoothing
    '''
    if epsilon > 0:
      return self._compute_cross_entropy(outputs, targets, epsilon=epsilon)
    else:
      return self.cross_entropy(outputs, targets)


class AttentionLoss(nn.Module):
  '''
  Loss usefull for Neural Network that use Attention Mechanism as it forces the network
  to learn to pay attention in a diagonal way which empirically show to speed up the training
  '''
  def __init__(self, pad_idx, device=None, decay_step=0.01, decay_factor=1):
    '''
    Params:
      * pad_idx : int
      * device (optional) : torch.device, defaul to None
      * decay_step (optional) : float, default to 0.01
    '''
    super().__init__()
    self.pad_idx = pad_idx
    self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.kld = nn.KLDivLoss(reduction='sum')

    self.decay_factor = decay_factor
    self.decay_step = decay_step

    self.cross_entropy = CrossEntropyLoss(pad_idx)
  
  def step(self, epoch):
    self.decay_factor = max(1 - epoch * self.decay_step, 0)
  
  @staticmethod
  def _get_attention_target(attention, lambda_=10):
    batch, dec_seq_len, enc_seq_len = attention.shape

    line_base = [gaussian(i, enc_seq_len, lambda_) for i in range(0, 2 * enc_seq_len)]

    max_idx = line_base.index(max(line_base))
    max_starts = [int(l) for l in np.linspace(0, enc_seq_len - 1, dec_seq_len)]

    target = np.array([line_base[max_idx - i:max_idx - i + enc_seq_len] for i in max_starts])

    return torch.from_numpy(target).unsqueeze(0).repeat(batch, 1, 1).float()
  
  def forward(self, outputs, targets, attention, loss='both', float_epsilon=1e-9, epsilon=0.):
    '''
    Params:
      * outputs : torch.tensor
      * targets : torch.tensor
      * attention : torch.tensor
      * loss (optional) : str, default to 'both'
      * float_epsilon (optional) : float, default to 1e-9
      * epsilon (optional) : float, default to 0.1, use for Label-Smoothing
    '''
    attention = attention + float_epsilon
    target_attn = self._get_attention_target(attention).to(self.device) + float_epsilon

    kld_loss = self.kld(attention.log(), target_attn)

    if loss == 'kld':
      return kld_loss
    
    cross_entropy = self.cross_entropy(outputs, targets, epsilon=epsilon)
    current_loss = cross_entropy * (1 + self.decay_factor * kld_loss)
    
    return current_loss


class ScoresMaster(object):
  def __init__(self, idx_2_letter=None, pad_idx=None, joiner=''):
    self.idx_2_letter = idx_2_letter
    self.pad_idx = pad_idx
    self.joiner = joiner

    self.true_labels = []
    self.pred_labels = []
  
  def reset_feed(self):
    self.true_labels = []
    self.pred_labels = []
  
  def partial_feed(self, true_labels, pred_labels):
    self.true_labels += true_labels
    self.pred_labels += pred_labels
  
  @classmethod
  def remove_queue(cls, labels, stop_idx=-1):
    return [l[:l.index(stop_idx) + 1 if stop_idx in l else None] for l in labels]
  
  def get_scores(self, pred_labels, true_labels, idx_2_letter=None, pad_idx=None, stop_idx=-1, from_feed=False, strategy='align'):
    if from_feed:
      pred_labels = self.pred_labels
      true_labels = self.true_labels

    pred_labels = self.remove_queue(pred_labels, stop_idx=stop_idx)
    true_labels = self.remove_queue(true_labels, stop_idx=stop_idx)

    idx_2_letter = idx_2_letter if idx_2_letter is not None else self.idx_2_letter
    pad_idx = pad_idx if pad_idx is not None else self.pad_idx

    assert idx_2_letter is not None, 'Must provide idx_2_letter'
    assert pad_idx is not None, 'Must provide pad_idx'

    true_sentences, pred_sentences = self.reconstruct_sentences(true_labels, pred_labels, idx_2_letter, pad_idx, joiner=self.joiner)
    l_acc, w_acc, s_acc, awer = self.compute_scores(true_sentences, pred_sentences, strategy=strategy)

    return l_acc, w_acc, s_acc, awer
  
  @classmethod
  def compute_scores(cls, true_sentences, pred_sentences, strategy='align'):
    '''

    Params:
      * true_sentences: list of string
      * pred_sentences: list of string

    Returns:
      * character_accuracy: float
      * word_accuracy: float
      * sentence_accuracy: float
      * wer: float, word error rate
    '''
    count_correct_sentences = 0
    count_correct_words, count_words = 0, 0
    count_correct_characters, count_characters = 0, 0
    wers = []

    for true, pred in zip(true_sentences, pred_sentences):
      count_characters += len(true)
      count_correct_characters += sum([1 for t, p in zip(true, pred) if t == p])

      if strategy == 'align':
        space_idxs = [0] + [i for i, c in enumerate(true) if c == ' '] + [len(true)]
        is_words_correct = [
          true[space_idxs[i] + 1 : space_idxs[i+1]] == pred[space_idxs[i] + 1 : space_idxs[i+1]]
          for i in range(len(space_idxs) - 1)
        ]
        count_words += len(is_words_correct)
        count_correct_words += sum(is_words_correct)
      else:
        true_word_list = true.split(' ')
        pred_word_list = pred.split(' ')

        count_words += len(true_word_list)
        count_correct_words += sum([1 for tw, pw in zip(true_word_list, pred_word_list) if tw == pw])

      if true == pred:
        count_correct_sentences += 1
      
      wers.append(compute_WER(true, pred))
    
    character_accuracy = count_correct_characters / count_characters
    word_accuracy = count_correct_words / count_words
    sentence_accuracy = count_correct_sentences / len(true_sentences)

    wer = np.mean(wers)

    return character_accuracy, word_accuracy, sentence_accuracy, wer
  
  @classmethod
  def reconstruct_sentences(cls, true_labels, pred_labels, idx_2_letter, pad_idx, joiner=''):
    '''
    Params:
      * true_labels: list of list of int
      * pred_labels: list of list of int

    Returns:
      * true_sentences: list of string
      * pred_sentences: list of string
    '''
    true_sentences, pred_sentences = zip(*[
      (
        joiner.join([idx_2_letter[idx] for idx in true[:true.index(pad_idx) if pad_idx in true else None]]),
        joiner.join([idx_2_letter[idx] for idx in pred[:pred.index(pad_idx) if pad_idx in pred else None]])
      )
      for true, pred in zip(true_labels, pred_labels)
    ])
    return true_sentences, pred_sentences


def add_args_from_settings(argparser, settings_file='settings.json'):
  '''
  Params:
    * argparser
    * settings_file (optional) : str
  '''
  settings = load_json(settings_file)

  converter = {'int': int, 'float': float, 'str': str, 'ast.literal_eval': ast.literal_eval}

  for arg, info in settings.items():
    argparser.add_argument(f'--{arg}', default=converter[info['type']](info['default']), type=converter[info['type']])


def model_scores(epoch, pred_labels, true_labels, metadata, plotter, partial_dump=True, n_dumps=50, visdom_plotting=True, logging_dump=True):
  pred_labels = metadata.SM.remove_queue(pred_labels, stop_idx=metadata.eos_idx)
  true_labels = metadata.SM.remove_queue(true_labels, stop_idx=metadata.eos_idx)

  true_sentences, pred_sentences = metadata.SM.reconstruct_sentences(true_labels, pred_labels, metadata.idx_2_vocab, metadata.pad_idx)

  _, eval_word_acc, eval_sentence_acc, awer = metadata.SM.compute_scores(true_sentences, pred_sentences, strategy='other')

  if visdom_plotting:
    plotter.line_plot('word accuracy', 'test', 'Word Accuracy', epoch, eval_word_acc)
    plotter.line_plot('sentence accuracy', 'test', 'Sentence Accuracy', epoch, eval_sentence_acc)
    plotter.line_plot('word error rate', 'test', 'Word Error Rate', epoch, awer)

  if logging_dump:
    if partial_dump:
      to_dump = [f'\nNEW EPOCH {epoch}:\n'] + [f'TRUE: {t}\nPRED: {p}\n' for t, p in zip(true_sentences[:n_dumps], pred_sentences[:n_dumps])]
    else:
      to_dump = [f'\nNEW EPOCH {epoch}:\n'] + [f'TRUE: {t}\nPRED: {p}\n' for t, p in zip(true_sentences, pred_sentences)]

    logging.info('\n'.join(to_dump))
    logging.info(f'TEST Word accuracy = {eval_word_acc:.3f} | Sentence accuracy = {eval_sentence_acc:.3f}')
    logging.info(f'TEST WER = {awer:.3f}')

  return eval_word_acc, eval_sentence_acc, awer


def text_to_image(text='Hello World', fname='hello_world.png', size=12, color=(255, 255, 0), bg='blue'):
  fnt = ImageFont.truetype('../../../Downloads/arial.ttf', size)

  image = Image.new(mode='RGB', size=(size * len(text) // 2 + size, 2 * size), color=bg)
  draw = ImageDraw.Draw(image)

  draw.text((size // 2, size // 2), text, font=fnt, fill=color)

  image.save(fname)


class AnalyseModelSize(object):
  '''Modified from torchsummary library'''
  def __init__(self, model, input_size, batch_size=-1, device='cuda'):
    self.model = model
    self.input_size = input_size
    self.batch_size = batch_size
    self.device = device

  def register_hook(self, module):
    def hook(module, input, output):
      class_name = str(module.__class__).split(".")[-1].split("'")[0]
      module_idx = len(self.summary)

      m_key = "%s-%i" % (class_name, module_idx + 1)
      self.summary[m_key] = OrderedDict()
      self.summary[m_key]["input_shape"] = list(input[0].size())
      self.summary[m_key]["input_shape"][0] = self.batch_size
      if isinstance(output, (list, tuple)):
        self.summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
      else:
        self.summary[m_key]["output_shape"] = list(output.size())
        self.summary[m_key]["output_shape"][0] = self.batch_size

      params = 0
      if hasattr(module, "weight") and hasattr(module.weight, "size"):
        params += torch.prod(torch.LongTensor(list(module.weight.size())))
        self.summary[m_key]["trainable"] = module.weight.requires_grad
      if hasattr(module, "bias") and hasattr(module.bias, "size"):
        params += torch.prod(torch.LongTensor(list(module.bias.size())))
      self.summary[m_key]["nb_params"] = params
    
    if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == self.model)):
      self.hooks.append(module.register_forward_hook(hook))
  
  def analyse(self):
    device = self.device.lower()
    assert device in ["cuda", "cpu"], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
      dtype = torch.cuda.FloatTensor
    else:
      dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(self.input_size, tuple):
      self.input_size = [self.input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in self.input_size]
    # print(type(x[0]))

    # create properties
    self.summary = OrderedDict()
    self.hooks = []

    # register hook
    self.model.apply(self.register_hook)

    # make a forward pass
    # print(x.shape)
    self.model(*x)

    # remove these hooks
    for h in self.hooks:
      h.remove()

    total_params = 0
    total_output = 0
    trainable_params = 0

    for layer in self.summary:
      # input_shape, output_shape, trainable, nb_params
      total_params += self.summary[layer]["nb_params"]
      total_output += np.prod(self.summary[layer]["output_shape"])

      if "trainable" in self.summary[layer]:
        if self.summary[layer]["trainable"] == True:
          trainable_params += self.summary[layer]["nb_params"]
    
    # Gave in MB
    total_input_size = abs(np.prod(self.input_size) * self.batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    results = {'total_params': total_params, 'trainable_params': trainable_params,
               'non_trainable_params': total_params - trainable_params, 'input_size': total_input_size,
               'forward_backward_size': total_output_size, 'params_size': total_params_size, 'total_size': total_size}
    
    return results


def save_tmp(fname, savetype='pk'):
  '''
  Decorator to use in order to save in a file the data returned by the function decorated.
  To avoid computing each time the function is called.

  Usage:
    @save_tmp('_my_temporary_file.pk')
    def my_function():
      data = compute_my_data()
      return data
  '''
  def decorator(function):
    def inner(*args, **kwargs):
      if os.path.isfile(fname):
        if savetype == 'pk':
          with open(fname, 'rb') as f:
            to_return = pk.load(f)
        else:
          with open(fname, 'r') as f:
            to_return = json.load(f)
      else:
        to_return = function(*args, **kwargs)

        if savetype == 'pk':
          with open(fname, 'wb') as f:
            pk.dump(to_return, f)
        else:
          with open(fname, 'w') as f:
            json.dump(to_return, f)
      
      return to_return
    return inner
  return decorator


def l1_regularization(model, _lambda=1e-3, device=None):
  device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu') if device is None else device
  L1_reg = torch.tensor(0., requires_grad=True).to(device)
  for name, param in model.named_parameters():
    if 'weight' in name:
      L1_reg = L1_reg + torch.norm(param, 1)

  return _lambda * L1_reg


def board_printing(data):
  '''
  Prints a results board like this:
    +----+-------------------+---------+
    |    | name              |    f1   |
    |----+-------------------+---------+
    |  0 | mon-premier-bot_6 |   0.773 |
    |  1 | vox_13            |   0.685 |
    |  2 | amel_13           |   0.81  |
    |  3 | artebot_18        |   0.831 |
    |  4 | anass-a_14        |   0.739 |
    |  5 | zambot_14         |   0.705 |
    +----+-------------------+---------+
  
  Params:
    * data : dict like {key1: [], key2: [], ...}
  '''
  df = pd.DataFrame.from_dict(data)
  print(tabulate(df, headers='keys', tablefmt='psql'))


def sigmoid_energy(mytensor, dim=-1):
  return mytensor.sigmoid() / mytensor.sigmoid().sum(dim, keepdim=True)


def compute_out_conv(size_in, kernel=3, stride=1, padding=0, dilation=1):
  return (size_in + 2 * padding - dilation * (kernel - 1) -1) // stride + 1