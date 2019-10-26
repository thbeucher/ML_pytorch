import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(__file__).replace('models/transformer/embedder.py', ''))
import utils as u


class PositionalEmbedder(nn.Module):
  '''
  2 usages:
    * add a positional information to an embedding -> leave output_size to 0
    * embed given sequence of words/chars then add a positional information -> provide output_size = vocabulary_size
  '''
  def __init__(self, max_seq_len, embedding_size, d_model, output_size=0, reduce_dim=False, scaling=True, dropout=0.1, device=None):
    '''
    Params:
      * max_seq_len : int
      * embedding_size : int
      * d_model : int
    '''
    super().__init__()
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.max_seq_len = max_seq_len
    self.embedding_size = embedding_size
    self.d_model = d_model
    self.reduce_dim = reduce_dim
    self.output_size = output_size
    self.scaling = scaling

    self.positional_embedding = u.create_positional_embedding(max_seq_len, embedding_size, d_model)

    if output_size > 0:
      self.word_embedder = nn.Embedding(output_size, embedding_size)

    if reduce_dim:
      self.reducer = nn.Sequential(nn.Conv2d(1, 1, 3, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(1, 1, 3, stride=2),
                                   nn.ReLU())
    
    self.dropout = nn.Dropout(dropout)
    self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(self.device) if self.scaling else 1
  
  def forward(self, x):
    ''' x : torch.tensor, shape = (batch, seq_len, n_feats) or (batch, seq_len) '''
    if self.reduce_dim:
      x = self.reducer(x.unsqueeze(1)).squeeze(1)

    batch_size, seq_len = x.shape[:2]

    assert seq_len <= self.max_seq_len, f'Tensor sequence len ({seq_len}) cannot be bigger than max_seq_len ({self.max_seq_len})'

    index_x = torch.LongTensor(range(seq_len)).unsqueeze(0).repeat(batch_size, 1).to(self.device)
    pos_emb = self.positional_embedding(index_x)

    if self.output_size > 0:
      x = self.word_embedder(x)
    
    if pos_emb.shape[-1] == x.shape[-1]:
      return self.dropout((x * self.scale) + pos_emb)
    else:
      return self.dropout((x * self.scale) + pos_emb[:, :, :-1])