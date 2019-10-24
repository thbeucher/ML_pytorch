import utils as u
import torch.nn as nn

from attention import MultiHeadAttention


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
  def __init__(self, n_blocks, d_model, d_keys, d_values, n_heads, d_ff, dropout=0.1, act_fn='relu'):
    super().__init__()
    self.encoders = nn.ModuleList([EncoderBlock(d_model, d_keys, d_values, n_heads, d_ff, dropout=dropout, act_fn=act_fn)
                                    for _ in range(n_blocks)])
    
  def forward(self, x, padding_mask=None):
    for encoder in self.encoders:
      x = encoder(x, padding_mask=padding_mask)
    return x
