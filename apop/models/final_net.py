import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(__file__).replace('final_net.py', ''))
from conv_seqseq import Decoder as CSSDecoder
from transformer.decoder import TransformerDecoder
from transformer.attention import MultiHeadAttention
from final_net_configs import get_encoder_config, get_decoder_config, get_input_proj_layer


class PositionalEncoding(nn.Module):
  # from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
  def __init__(self, d_model, dropout=0.1, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:, :x.size(1)]
    return self.dropout(x)


class TextEmbedder(nn.Module):
  def __init__(self, n_embeddings, emb_dim=120, max_seq_len=600, dropout=0.1):
    super().__init__()
    self.token_embedding = nn.Embedding(n_embeddings, emb_dim)
    self.positional_encoding = PositionalEncoding(emb_dim, dropout=dropout, max_len=max_seq_len)
  
  def forward(self, x):  # [batch_size, seq_len]
    out = self.token_embedding(x)  # [batch_size, seq_len, emb_dim]
    return self.positional_encoding(out)


class ConvBlock(nn.Module):
  def __init__(self, in_chan, out_chan, kernel=3, stride=1, pad=1, dil=1, dropout=0., groups=1, k=1, **kwargs):
    super().__init__()
    assert k in [1, 2], 'Handle only k = 1 or 2'
    self.conv = nn.Sequential(nn.Conv1d(in_chan, out_chan, kernel, stride=stride, padding=pad, dilation=dil, groups=groups),
                              nn.BatchNorm1d(out_chan),
                              nn.ReLU(inplace=True) if k == 1 else nn.GLU(dim=1),
                              nn.Dropout(dropout))
  
  def forward(self, x):  # [batch_size, in_chan, seq_len]
    return self.conv(x)  # [batch_size, out_chan, seq_len] or [batch_size, out_chan // 2, seq_len] if k == 2


class SeparableConvBlock(nn.Module):
  def __init__(self, in_chan, out_chan, kernel=3, stride=1, pad=1, dil=1, dropout=0., k=1, **kwargs):
    super().__init__()
    assert k in [1, 2], 'Handle only k = 1 or 2'
    self.conv = nn.Sequential(nn.Conv1d(in_chan, k * in_chan, kernel, stride=stride, padding=pad, dilation=dil, groups=in_chan),
                              nn.BatchNorm1d(k * in_chan),
                              nn.ReLU(inplace=True) if k == 1 else nn.GLU(dim=1),
                              nn.Dropout(dropout),
                              nn.Conv1d(k * in_chan, out_chan, 1),
                              nn.BatchNorm1d(out_chan),
                              nn.ReLU(inplace=True),
                              nn.Dropout(dropout))
  
  def forward(self, x):  # [batch_size, in_chan, seq_len]
    return self.conv(x)  # [batch_size, out_chan, seq_len]


class AttentionConvBlock(nn.Module):
  def __init__(self, in_chan, n_heads=8, kernel=5, dropout=0., pad=2, bias=True, **kwargs):
    super().__init__()
    assert in_chan // n_heads * n_heads == in_chan, 'in_chan must be evenly divisible by n_heads'
    self.n_heads = n_heads
    self.dropout = dropout
    self.pad = pad
    self.bias = None

    self.weight = nn.Parameter(torch.Tensor(n_heads, 1, kernel))
    nn.init.xavier_uniform_(self.weight)

    if bias:
      self.bias = nn.Parameter(torch.Tensor(in_chan))
      nn.init.constant_(self.bias, 0.)
  
  def forward(self, x):  # [batch_size, in_chan, seq_len]
    in_ = x.reshape(-1, self.n_heads, x.size(2))
    weight = F.dropout(F.softmax(self.weight, dim=-1), self.dropout, training=self.training)
    out = F.conv1d(in_, weight, padding=self.pad, groups=self.n_heads).reshape(x.shape)

    if self.bias is not None:
      out = out + self.bias.view(1, -1, 1)
    return out


class ConvAttentionConvBlock(nn.Module):
  def __init__(self, in_chan, out_chan, kernel_conv=3, stride_conv=1, pad_conv=1, dil_conv=1, dropout_conv=0., groups=1, k=1,
               n_heads=8, kernel_attn=5, dropout_attn=0., pad_attn=2, bias=True):
    super().__init__()
    self.conv = ConvBlock(in_chan, out_chan, kernel=kernel_conv, stride=stride_conv, pad=pad_conv, dil=dil_conv, dropout=dropout_conv,
                          groups=groups, k=k)
    self.attn_conv = AttentionConvBlock(out_chan//k, n_heads=n_heads, kernel=kernel_attn, dropout=dropout_attn, pad=pad_attn, bias=bias)
  
  def forward(self, x):  # [batch_size, in_chan, seq_len]
    return self.attn_conv(self.conv(x))


class FeedForward(nn.Module):
  def __init__(self, input_size, output_size, d_ff=2048, dropout=0., **kwargs):
    super().__init__()
    self.ff = nn.Sequential(nn.Linear(input_size, d_ff),
                            nn.ReLU(inplace=True),
                            nn.Dropout(dropout),
                            nn.Linear(d_ff, output_size),
                            nn.Dropout(dropout),
                            nn.LayerNorm(output_size))
  
  def forward(self, x):  # [batch_size, *, input_size]
    return self.ff(x)   # [batch_size, *, output_size]


class Encoder(nn.Module):
  def __init__(self, config=None, residual=True, output_size=None, one_pred=False, input_proj=None):
    '''
    config = list of layer_config
      layer_config = list of block_config
        block_config = (block_type_name:str, block_configuration:dict)
    one_pred : Boolean, extract last output before feeding to ouput_proj layer
    '''
    super().__init__()
    self.config = config
    self.residual = residual
    self.output_proj = None
    self.one_pred = one_pred
    self.input_proj = None
    self.available_blocks = {'conv_block': ConvBlock, 'separable_conv_block': SeparableConvBlock,
                             'attention_conv_block': AttentionConvBlock, 'feed_forward': FeedForward,
                             'conv_attention_conv_block': ConvAttentionConvBlock, 'lstm': nn.LSTM}
    
    if input_proj is not None:
      self.input_proj = get_input_proj_layer(config=input_proj)
    
    if config is None:
      self.config = get_encoder_config(config='base')
    
    layers = []
    for layer in self.config:
      blocks = []
      for block in layer:
        sub_blocks = []
        for parallel_sub_block_type, block_config in block:
          sub_blocks.append(self.available_blocks[parallel_sub_block_type](**block_config))
        blocks.append(nn.ModuleList(sub_blocks))
      layers.append(nn.ModuleList(blocks))
    self.network = nn.ModuleList(layers)

    if output_size is not None:
      key = [k for k in ['output_size', 'out_chan', 'in_chan', 'input_size'] if k in self.config[-1][-1][-1][1]][0]
      self.output_proj = nn.Linear(self.config[-1][-1][-1][1][key], output_size)
  
  def forward(self, x):
    if self.input_proj is not None:
      x = self.input_proj(x)
      
    for i, layer in enumerate(self.network):
      out = x
      for j, block in enumerate(layer):
        outs = []
        for k, sub_block in enumerate(block):
          if 'conv' in self.config[i][j][k][0]:
            outs.append(sub_block(out.permute(0, 2, 1)).permute(0, 2, 1))
          elif 'lstm' in self.config[i][j][k][0]:
            sub_block.flatten_parameters()
            outs.append(sub_block(out)[0])
          else:
            outs.append(sub_block(out))
        out = torch.cat(outs, dim=-1)
      x = x + out if self.residual and out.shape == x.shape else out

    if self.output_proj is not None:
      if self.one_pred:
        x = x[:, -1]
      x = self.output_proj(x)
    return x


class Decoder(nn.Module):
  def __init__(self, config=None, residual=True, embed_x=False, **kwargs):
    super().__init__()
    self.config = config
    self.embed_x = embed_x

    if isinstance(config, list):
      pass
    elif config == 'css_decoder':
      output_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, pad_idx, device, max_seq_len,\
      score_fn, scaling_energy, multi_head, d_keys_values = get_decoder_config(config=config, **kwargs)
      embedder = TextEmbedder(output_dim, emb_dim=emb_dim, max_seq_len=max_seq_len)
      self.network = CSSDecoder(output_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, pad_idx, device, embedder=embedder,
                                max_seq_len=max_seq_len, score_fn=score_fn, scaling_energy=scaling_energy, multi_head=multi_head,
                                d_keys_values=d_keys_values)
    else:
      n_blocks, d_model, d_keys, d_values, n_heads, d_ff, dropout, max_seq_len, output_dim = get_decoder_config(config='transformer',
                                                                                                                **kwargs)
      self.embedder = TextEmbedder(output_dim, emb_dim=d_model, max_seq_len=max_seq_len)
      self.network = TransformerDecoder(n_blocks, d_model, d_keys, d_values, n_heads, d_ff, dropout=dropout)
      self.output_proj = nn.Linear(d_model, output_dim)
      
  def forward(self, x, y):  # x = [batch_size, seq_len, n_feats] | y = [batch_size, seq_len]
    if isinstance(self.config, list):
      pass
    elif self.config == 'css_decoder':
      if self.embed_x:
        x = self.network.embedder(x)
      out, _ = self.network(y, x, x)
    else:  # transformer decoder
      if self.embed_x:
        x = self.embedder(x)
      y = self.embedder(y)
      out = self.network(y, x)
      out = self.output_proj(out)
    return out


class EncoderMultiHeadObjective(nn.Module):
  '''
  Will instanciates an Encoder with the output layer that will be used for CTC-Loss
  and a small decoder (Embedder + MultiHeadAttention + Linear) for Cross-Entropy-Loss
  '''
  def __init__(self, encoder_config=None, residual=True, output_size=None, decoder_config=None, **kwargs):
    super().__init__()
    self.encoder = Encoder(config=encoder_config, residual=residual)
    self.encoder_proj = nn.Linear(decoder_config['d_model'], output_size)

    if decoder_config is None:
      decoder_config = get_decoder_config(config='multihead_objective_decoder', **kwargs)

    self.decoder_embedder = TextEmbedder(decoder_config['n_embeddings'], decoder_config['emb_dim'], decoder_config['max_seq_len'],
                                         dropout=decoder_config['embedder_dropout'])
    self.decoder_attn = MultiHeadAttention(decoder_config['d_model'], decoder_config['d_keys'], decoder_config['d_values'],
                                           decoder_config['n_heads'], dropout=decoder_config['mha_dropout'])
    self.decoder_proj = nn.Linear(decoder_config['d_model'], decoder_config['n_embeddings'])
  
  def forward(self, x, y):  # x = [batch_size, seq_len, n_feats] | y = [batch_size, seq_len]
    enc_out = self.encoder(x)
    dec_out = self.decoder_embedder(y)
    attn_out = self.decoder_attn(dec_out, enc_out, enc_out)
    return self.encoder_proj(enc_out), self.decoder_proj(attn_out)


if __name__ == "__main__":
  ## ENCODER
  print('ENCODER')
  encoder = Encoder(config=get_encoder_config('attention'))  # separable | attention | attention_glu | base
  print(f'Number of parameters = {sum(p.numel() for p in encoder.parameters() if p.requires_grad):,}')
  in_ = torch.randn(2, 1500, 512)
  out = encoder(in_)
  print(f'in_ = {in_.shape} | out = {out.shape}')

  encoder = Encoder(config=get_encoder_config('attention'), output_size=31)
  out = encoder(in_)
  print(f'in_ = {in_.shape} | out = {out.shape}')

  ## DECODER
  print('DECODER')
  decoder = Decoder(config='transformer')
  print(f'Number of parameters = {sum(p.numel() for p in decoder.parameters() if p.requires_grad):,}')
  enc_out = torch.randn(2, 375, 512)
  dec_in = torch.randint(0, 31, (2, 200))
  out = decoder(enc_out, dec_in)
  print(f'dec_in = {dec_in.shape} | out = {out.shape}')