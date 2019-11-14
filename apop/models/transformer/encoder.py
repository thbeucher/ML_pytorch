import torch
import utils as u
import torch.nn as nn

from attention import MultiHeadAttention


class GatedEncoderBlock(nn.Module):
  '''Modify Architecture according to paper https://arxiv.org/pdf/1910.06764.pdf'''
  def __init__(self, d_model, d_keys, d_values, n_heads, d_ff, dropout=0.1, act_fn='relu', mode='output'):
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

    if mode == 'gru':
      self.reset_att_w = nn.Linear(d_model, d_model, bias=False)
      self.reset_x_w = nn.Linear(d_model, d_model, bias=False)

      self.update_att_w = nn.Linear(d_model, d_model, bias=False)
      self.update_x = nn.Linear(d_model, d_model)

      self.memory_att_w = nn.Linear(d_model, d_model, bias=False)
      self.memory_rx = nn.Linear(d_model, d_model, bias=False)

      self.reset_att_w2 = nn.Linear(d_model, d_model, bias=False)
      self.reset_x_w2 = nn.Linear(d_model, d_model, bias=False)

      self.update_att_w2 = nn.Linear(d_model, d_model, bias=False)
      self.update_x2 = nn.Linear(d_model, d_model)

      self.memory_att_w2 = nn.Linear(d_model, d_model, bias=False)
      self.memory_rx2 = nn.Linear(d_model, d_model, bias=False)

      self.gate_pos = {0: {'reset_att': self.reset_att_w, 'reset_x': self.reset_x_w,
                           'update_att': self.update_att_w, 'update_x': self.update_x,
                           'memory_att': self.memory_att_w, 'memory_rx': self.memory_rx},
                       1: {'reset_att': self.reset_att_w2, 'reset_x': self.reset_x_w2,
                           'update_att': self.update_att_w2, 'update_x': self.update_x2,
                           'memory_att': self.memory_att_w2, 'memory_rx': self.memory_rx2}}
      self.gate = self._gru_like_gating
    elif mode == 'sigmoid_tanh':
      self.att_gate1 = nn.Linear(d_model, d_model)
      self.att_tanh1 = nn.Linear(d_model, d_model, bias=False)

      self.att_gate2 = nn.Linear(d_model, d_model)
      self.att_tanh2 = nn.Linear(d_model, d_model, bias=False)

      self.gate_pos = {0: {'att_s': self.att_gate1, 'att_t': self.att_tanh1}, 
                       1: {'att_s': self.att_gate2, 'att_t': self.att_tanh2}}
      self.gate = self._gated_sigmoid_tanh
    else:  # mode = output or highway
      self.x_gate1 = nn.Linear(d_model, d_model)
      self.x_gate2 = nn.Linear(d_model, d_model)

      self.gate_pos = {0: self.x_gate1, 1: self.x_gate2}
      self.gate = self._gated_output_connection

    self.dropout = nn.Dropout(dropout)
  
  def _gru_like_gating(self, x, y, n=0):
    reset_gate = torch.sigmoid(self.gate_pos[n]['reset_att'](y) + self.gate_pos[n]['reset_x'](x))
    update_gate = torch.sigmoid(self.gate_pos[n]['update_att'](y) + self.gate_pos[n]['update_x'](x))
    memory_gate = torch.tanh(self.gate_pos[n]['memory_att'](y) + self.gate_pos[n]['memory_rx'](reset_gate * x))
    return (1 - update_gate) * x + update_gate * memory_gate
  
  def _gated_output_connection(self, x, y, n=0):
    return x + torch.sigmoid(self.gate_pos[n](x)) * y
  
  def _gated_highway(self, x, y, n=0):
    sigmoid_x = torch.sigmoid(self.gate_pos[n](x))
    return sigmoid_x * x + (1 - sigmoid_x) * y
  
  def _gated_sigmoid_tanh(self, x, y, n=0):
    return x + torch.sigmoid(self.gate_pos[n]['att_s'](y)) * torch.tanh(self.gate_pos[n]['att_t'](y))
  
  def forward(self, x, padding_mask=None):
    # normalize before multi-head attention
    x_norm = self.layer_norm1(x)
    # compute attention
    attention = self.attention_head(x_norm, x_norm, x_norm, mask=padding_mask)
    # apply gated output connection
    x = self.gate(x, self.relu(attention))
    # normalize before feed-forward
    x_norm = self.layer_norm2(x)
    # apply feed-forward
    pos = self.feed_forward(x_norm)
    # apply gated output connection
    x = self.gate(x, self.relu(pos), n=1)
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
