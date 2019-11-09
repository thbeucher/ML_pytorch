import torch
import utils as u
import torch.nn as nn

from attention import MultiHeadAttention


class GatedEncoderBlock(nn.Module):
  '''Modify Architecture according to paper https://arxiv.org/pdf/1910.06764.pdf'''
  def __init__(self, d_model, d_keys, d_values, n_heads, d_ff, dropout=0.1, act_fn='relu'):
    super().__init__()
    self.attention_head = MultiHeadAttention(d_model, d_keys, d_values, n_heads, dropout=dropout)

    self.feed_forward = nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.ReLU(inplace=True),
        nn.Linear(d_ff, d_model),
    )

    self.layer_norm1 = nn.LayerNorm(d_model)
    self.layer_norm2 = nn.LayerNorm(d_model)

    self.relu = nn.ReLU(inplace=True)

    self.linear_gate1 = nn.Linear(d_model, d_model)
    self.linear_gate2 = nn.Linear(d_model, d_model)

    self.dropout = nn.Dropout(dropout)
  
  def _gru_like_gating(self, x, y):
    pass
  
  def forward(self, x, padding_mask=None):
    # normalize before multi-head attention
    x_norm = self.layer_norm1(x)
    # compute attention
    attention = self.attention_head(x_norm, x_norm, x_norm, mask=padding_mask)
    # apply gated output connection
    x = x + torch.sigmoid(self.linear_gate1(x)) * self.relu(attention)
    # normalize before feed-forward
    x_norm = self.layer_norm2(x)
    # apply feed-forward
    pos = self.feed_forward(x_norm)
    # apply gated output connection
    x = x + torch.sigmoid(self.linear_gate2(x)) * self.relu(pos)
    return x


class EncoderBlock(nn.Module):
  def __init__(self, d_model, d_keys, d_values, n_heads, d_ff, dropout=0.1, act_fn='relu'):
    super().__init__()
    self.attention_head = MultiHeadAttention(d_model, d_keys, d_values, n_heads, dropout=dropout)
    self.feed_forward = nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.ReLU(),
        nn.Linear(d_ff, d_model),
    )
    self.layer_norm1 = nn.LayerNorm(d_model)
    self.layer_norm2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)
        
  def forward(self, x, padding_mask=None):
    attention = self.attention_head(x, x, x, mask=padding_mask)
    # Apply normalization and residual connection
    x = self.layer_norm1(x + self.dropout(attention))
    # Apply position-wise feedforward network
    pos = self.feed_forward(x)
    # Apply normalization and residual connection
    x = self.layer_norm2(x + self.dropout(pos))
    return x


class TransformerEncoder(nn.Module):
  def __init__(self, n_blocks, d_model, d_keys, d_values, n_heads, d_ff, dropout=0.1, act_fn='relu', block_type='standard'):
    super().__init__()
    block = {'standard': EncoderBlock, 'gated': GatedEncoderBlock}
    self.encoders = nn.ModuleList([block[block_type](d_model, d_keys, d_values, n_heads, d_ff, dropout=dropout, act_fn=act_fn)
                                    for _ in range(n_blocks)])
    
  def forward(self, x, padding_mask=None):
    for encoder in self.encoders:
      x = encoder(x, padding_mask=padding_mask)
    return x
