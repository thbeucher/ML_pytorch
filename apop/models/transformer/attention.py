import torch
import numpy as np
import torch.nn as nn

from math import sqrt


class ScaledDotProductAttention(nn.Module):
  def __init__(self, scaling_factor, dropout=0.1):
    super().__init__()
    self.scaling_factor = scaling_factor  # correspond to sqrt(d_keys)
    self.dropout = nn.Dropout(dropout)
    self.softmax = nn.Softmax(dim=-1)

    self.keys = None
    self.attention = None
    self.energy = None

  def forward(self, queries, keys, values, mask=None, save=False, aggregate=False):
    '''
    Params:
      * queries : torch tensor of shape (batch_size, seq_len, d_model) or (batch_size, n_heads, seq_len, d_keys)
      * keys : torch tensor of shape (batch_size, seq_len, d_model) or (batch_size, n_heads, seq_len, d_keys)
      * values : torch tensor of shape (batch_size, seq_len, d_model) or (batch_size, n_heads, seq_len, d_values)
      * mask (optional) : torch tensor of shape (batch_size, seq_len, seq_len) or (batch_size, n_heads, seq_len, seq_len)
      * save (optional) : boolean, set to True only for decoding phase
      * aggregate (optional) : boolean, set to True only for decoding phase
    
    save and aggregate can be set to True to speed up decoding phase (do not use it during training)
    
    Return:
      * torch tensor of shape (batch_size, seq_len, d_model) or (bathc_size, n_heads, seq_len, d_values)
    '''
    if save:
      if self.keys is None:
        self.keys = keys.transpose(-2, -1)
    else:
      self.keys = keys.transpose(-2, -1)

    if aggregate:
      if self.attention is None:
        self.attention = queries.matmul(self.keys)
      else:
        attention = queries.narrow(-2, -1, 1).matmul(self.keys)
        self.attention = torch.cat([self.attention, attention], dim=-2)
    else:
      attention = queries.matmul(self.keys) / self.scaling_factor

    if mask is not None:
      attention = attention.masked_fill(mask, -1e10)

    self.energy = self.softmax(attention)
    attention = self.dropout(self.energy)
    return attention.matmul(values)
  
  def reset_memory(self):
    self.keys = None
    self.attention = None


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, d_keys, d_values, n_heads, dropout=0.1):
    super().__init__()
    self.d_model = d_model
    self.d_keys = d_keys
    self.d_values = d_values
    self.n_heads = n_heads

    self.attention = ScaledDotProductAttention(sqrt(d_keys), dropout=dropout)

    self.query_projection = nn.Linear(d_model, n_heads * d_keys, bias=False)
    self.key_projection = nn.Linear(d_model, n_heads * d_keys, bias=False)
    self.value_projection = nn.Linear(d_model, n_heads * d_values, bias=False)

    self.output_projection = nn.Linear(n_heads * d_values, d_model, bias=False)

    self.K, self.V = None, None
    self.reset_parameters()
  
  def forward(self, queries, keys, values, mask=None, save=False, aggregate=False):
    '''
    Params:
      * queries : torch tensor of shape (batch, seq_len_query, d_model)
      * keys : torch tensor of shape (batch, seq_len_key, d_model)
      * values : torch tensor of shape (batch, seq_len_value, d_model)
      * mask (optional) : torch tensor of shape (batch, seq_len_query, seq_len_key)
      * save (optional) : boolean, set to True only for decoding phase
      * aggregate (optional) : boolean, set to True only for decoding phase
    
    save and aggregate can be set to True to speed up decoding phase (do not use it during training)
    '''
    batch, query_seq_len, _ = queries.shape
    _, key_seq_len, _ = keys.shape
    _, value_seq_len, _ = values.shape
    # perform linear operation on full queries, keys and values
    # then split into n_heads then rearange to compute attention on each head
    # (batch, seq_len, d_model) -> (batch, seq_len, d_k*n_heads) -> (batch, n_heads, seq_len, d_k)
    Q = self.query_projection(queries).view(batch, query_seq_len, self.n_heads, self.d_keys).permute(0, 2, 1, 3)

    if save:
      if self.K is None:
        self.K = self.key_projection(keys).view(batch, key_seq_len, self.n_heads, self.d_keys).permute(0, 2, 1, 3)
        self.V = self.value_projection(values).view(batch, value_seq_len, self.n_heads, self.d_values).permute(0, 2, 1, 3)
    else:
      self.K = self.key_projection(keys).view(batch, key_seq_len, self.n_heads, self.d_keys).permute(0, 2, 1, 3)
      self.V = self.value_projection(values).view(batch, value_seq_len, self.n_heads, self.d_values).permute(0, 2, 1, 3)

    output = self.attention(Q, self.K, self.V, mask=mask, save=save, aggregate=False)

    # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, n_heads, d_k) -> (batch, seq_len, d_model)
    output = output.permute(0, 2, 1, 3).contiguous().view(batch, query_seq_len, self.d_model)

    return self.output_projection(output)
  
  def reset_memory(self):
    self.K, self.V = None, None
    self.attention.reset_memory()
  
  def reset_parameters(self):  # default linear init is kaiming_uniform_
    nn.init.xavier_uniform_(self.query_projection.weight)
    nn.init.xavier_uniform_(self.key_projection.weight)
    nn.init.xavier_uniform_(self.value_projection.weight)
