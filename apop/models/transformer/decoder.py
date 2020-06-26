import os
import sys
import torch
import utils as u
import torch.nn as nn

from embedder import PositionalEmbedder
from attention import MultiHeadAttention

sys.path.append(os.path.abspath(__file__).replace('models/transformer/decoder.py', ''))
import utils as u


class DecoderBlock(nn.Module):
  def __init__(self, d_model, d_keys, d_values, n_heads, d_ff, dropout=0.1):
    super().__init__()
    self.masked_attention_head = MultiHeadAttention(d_model, d_keys, d_values, n_heads, dropout=dropout)
    self.attention_head = MultiHeadAttention(d_model, d_keys, d_values, n_heads, dropout=dropout)
    self.feed_forward = nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.ReLU(),
        nn.Linear(d_ff, d_model),
    )
    self.layer_norm1 = nn.LayerNorm(d_model)
    self.layer_norm2 = nn.LayerNorm(d_model)
    self.layer_norm3 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, enc_out, padding_mask=None, save=False, aggregate=False, futur_masking=True):
    '''
    Params:
      * x : torch tensor
      * enc_out : torch tensor
      * padding_mask : byte torch tensor, shape = (batch_size, enc_out_seq_len, enc_out_seq_len)
      * save (optional) : boolean, set to True only for decoding phase
      * aggregate (optional) : boolean, set to True only for decoding phase
    
    save and aggregate can be set to True to speed up decoding phase (do not use it during training)
    '''
    # creates triangular mask to hide the futur
    futur_padding_mask = u.create_futur_mask(x).to(x.device).unsqueeze(1) if futur_masking else None
    # Apply attention to inputs
    att = self.masked_attention_head(x, x, x, mask=futur_padding_mask, aggregate=aggregate)
    x = self.layer_norm1(x + self.dropout(att))
    # Apply attention to the encoder outputs and outputs of the previous layer
    att = self.attention_head(queries=x, keys=enc_out, values=enc_out, mask=padding_mask, save=save, aggregate=aggregate)
    x = self.layer_norm2(x + self.dropout(att))
    # Apply position-wise feedforward network
    pos = self.feed_forward(x)
    x = self.layer_norm3(x + self.dropout(pos))
    return x
  
  def reset_memory(self):
    self.attention_head.reset_memory()


class TransformerDecoder(nn.Module):
  def __init__(self, n_blocks, d_model, d_keys, d_values, n_heads, d_ff, dropout=0.1):
    super().__init__()
    self.decoders = nn.ModuleList([DecoderBlock(d_model, d_keys, d_values, n_heads, d_ff, dropout=dropout)
                                    for _ in range(n_blocks)])
        
  def forward(self, x, enc_out, padding_mask=None, save=False, aggregate=False, futur_masking=True):
    for decoder in self.decoders:
      x = decoder(x, enc_out, padding_mask=padding_mask, save=save, aggregate=aggregate, futur_masking=futur_masking)
    return x
  
  def reset_memory(self):
    [decoder.reset_memory() for decoder in self.decoders]


class Decoder(nn.Module):
  def __init__(self, n_blocks=6, d_model=512, d_keys=64, d_values=64, n_heads=8, d_ff=1024, output_size=31, dropout=0.,
               device=None, embedder=None, reduce_dim=False, max_seq_len=600, emb_dim=256, scaling=True):
    super().__init__()
    self.embedder = embedder
    if embedder is None:
      self.embedder = PositionalEmbedder(max_seq_len, emb_dim, d_model, scaling=scaling, device=device,
                                         reduce_dim=reduce_dim, dropout=dropout, output_size=output_size)
    self.input_projection = nn.Linear(emb_dim, d_model)
    self.decoders = nn.ModuleList([DecoderBlock(d_model, d_keys, d_values, n_heads, d_ff, dropout=dropout)
                                    for _ in range(n_blocks)])
    self.output_projection = nn.Linear(d_model, output_size)
        
  def forward(self, x, enc_out, padding_mask=None, save=False, aggregate=False):
    x = self.embedder(x)
    x = self.input_projection(x)
    for decoder in self.decoders:
      x = decoder(x, enc_out, padding_mask=padding_mask, save=save, aggregate=aggregate)
    return self.output_projection(x)
  
  def reset_memory(self):
    [decoder.reset_memory() for decoder in self.decoders]
