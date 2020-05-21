import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(__file__).replace('divers_models.py', 'transformer/'))
from attention import MultiHeadAttention


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


class ConvDepthPointWise(nn.Module):
  def __init__(self, embedder, in_size=512, out_size=512, hid_size=768, kernels=[33, 39, 51, 63, 75, 87], repeat=5):
    super().__init__()
    config = [(in_size if i<3 else hid_size, in_size if i<2 else hid_size, k, repeat) for i, k in enumerate(kernels[:-1])]
    self.embedder = embedder
    self.net = nn.ModuleList([ConvDepthPointWiseBlock(ic, oc, k, r) for ic, oc, k, r in config])
    self.point_wise_joints = nn.ModuleList([nn.Conv1d(ic, oc, 1) for ic, oc, _, _ in config])
    self.relu = nn.ReLU(inplace=True)
    self.out_conv = nn.Sequential(nn.Conv1d(hid_size, hid_size, kernels[-1], padding=kernels[-1]//2, dilation=2), nn.ReLU(inplace=True),
                                  nn.Conv1d(hid_size, out_size, 1), nn.ReLU(inplace=True))
  
  def forward(self, x):
    x = self.embedder(x).permute(0, 2, 1)
    for sub_net, pw in zip(self.net, self.point_wise_joints):
      x = self.relu(sub_net(x) + pw(x))
    return self.out_conv(x).permute(0, 2, 1)


class ConvDepthPointWiseBlock(nn.Module):
  def __init__(self, in_chan, out_chan, kernel, repeat):
    super().__init__()
    net = [nn.Conv1d(in_chan, in_chan, kernel, groups=in_chan, padding=kernel//2),
           nn.Conv1d(in_chan, out_chan, 1)]
    for _ in range(repeat - 1):
      net.insert(0, nn.ReLU(inplace=True))
      net.insert(0, nn.Conv1d(in_chan, in_chan, 1))  # point-wise conv
      net.insert(0, nn.Conv1d(in_chan, in_chan, kernel, groups=in_chan, padding=kernel//2))  # depth-wise conv
    self.net = nn.Sequential(*net)

  def forward(self, x):
    return self.net(x)


class ConvLayer(nn.Module):
  def __init__(self, n_input_feats, d_model, n_heads, d_ff, kernel_size, n_blocks, embedder, dropout=0., only_see_past=True,
               full_att=False, block_type='self_attn', n_blocks_strided=None, residual=True):
    super().__init__()
    blocks_choice = {'self_attn': ConvSelfAttnBlock, 'dilated': ConvMultipleDilationBlock, 'dilated_bnd': ConvMultipleDilationBlockBNDrop}
    n_blocks_strided = n_blocks // 2 if n_blocks_strided is None else n_blocks_strided
    strides = [2 if i < n_blocks_strided else 1 for i in range(n_blocks)]
    self.residual = residual
    self.embedder = embedder
    self.input_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_input_feats, d_model), nn.ReLU(inplace=True), nn.LayerNorm(d_model))
    self.blocks = nn.ModuleList([blocks_choice[block_type](d_model, n_heads, kernel_size, d_ff, dropout=dropout, only_see_past=only_see_past,
                                                           self_attn=True if i % 2 == 0 or full_att else False, stride=s)
                                    for i, s in enumerate(strides)])
  
  def forward(self, x, y=None):
    x = self.embedder(x)
    x = self.input_proj(x)
    for block in self.blocks:
      out = block(x, y=y)
      x = x + out if self.residual and out.shape == x.shape else out
    return x


class DecodingConvLayer(nn.Module):
  def __init__(self, n_input_feats, d_model, n_heads, d_ff, kernel_size, n_blocks, embedder, dropout=0., only_see_past=True,
               full_att=False, n_blocks_strided=None, residual=True, **kwargs):
    super().__init__()
    n_blocks_strided = n_blocks // 2 if n_blocks_strided is None else n_blocks_strided
    strides = [2 if i < n_blocks_strided else 1 for i in range(n_blocks)]
    self.residual = residual
    self.embedder = embedder
    self.input_proj = nn.Sequential(nn.Linear(n_input_feats, d_model), nn.ReLU(inplace=True), nn.LayerNorm(d_model))
    self.blocks = nn.ModuleList([ConvMultipleDilationBlock(d_model, n_heads, kernel_size, d_ff, dropout=dropout, only_see_past=only_see_past,
                                                           self_attn=True if i % 2 == 0 or full_att else False, stride=s)
                                    for i, s in enumerate(strides)])
    self.attn = MultiHeadAttention(d_model, d_model // n_heads, d_model // n_heads, n_heads, dropout=dropout)
  
  def forward(self, x, y):
    x = self.embedder(x)
    x = self.input_proj(x)
    attn = self.attn(x, y, y)
    for i, block in enumerate(self.blocks):
      out = block(x)
      out = out if i < len(self.blocks) // 3 else out + attn
      x = x + out if self.residual and out.shape == x.shape else out
    return x


class ConvSelfAttnBlock(nn.Module):
  def __init__(self, d_model, n_heads, kernel_size, d_ff, dropout=0., only_see_past=True, self_attn=True, **kwargs):
    super().__init__()
    self.kernel_size = kernel_size
    self.only_see_past = only_see_past
    self.self_attn = self_attn

    padding = 0 if only_see_past else (kernel_size - 1) // 2
    self.conv = nn.Conv1d(in_channels=d_model, out_channels=2 * d_model, kernel_size=kernel_size, padding=padding)

    if self.self_attn:
      d_keys_vals = d_model // n_heads
      self.attn = MultiHeadAttention(d_model, d_keys_vals, d_keys_vals, n_heads, dropout=dropout)
      self.attn_norm = nn.LayerNorm(d_model)

      self.feed_forward = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(d_model, d_ff),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(dropout),
                                        nn.Linear(d_ff, d_model),
                                        nn.ReLU(inplace=True),
                                        nn.LayerNorm(d_model))
  
  def forward(self, x, futur_mask=None, y=None):  # [batch_size, seq_len, d_model]
    x_pad = x
    if self.only_see_past:
      x_pad = F.pad(x, (0, 0, self.kernel_size - 1, 0, 0, 0))
      futur_mask = (torch.triu(torch.ones(x.size(1), x.size(1) if y is None else y.size(1)), diagonal=1) == 0).to(x.device)

    x_pad = x_pad.permute(0, 2, 1)
    conved = self.conv(x_pad)  # [batch_size, 2 * hid_dim, seq_len]
    conved = F.glu(conved, dim=1)  # [batch_size, hid_dim, seq_len]
    conved = conved.permute(0, 2, 1)
    conved = conved + x  # residual connection

    if self.self_attn:
      if y is not None:
        self_attn = self.attn_norm(self.attn(conved, y, y))
      else:
        self_attn = self.attn_norm(self.attn(conved, conved, conved, mask=futur_mask))

      return self.feed_forward(conved + self_attn)
    return conved


class ConvMultipleDilationBlock(nn.Module):
  def __init__(self, d_model, n_heads, kernel_size, d_ff, stride=2, dropout=0., only_see_past=True, **kwargs):
    super().__init__()
    self.only_see_past = only_see_past
    dilations = list(range(1, 5))
    pad = [0 if only_see_past else (kernel_size - 1) // 2 + i for i in range(4)]
    if self.only_see_past:
      stride = 1
    self.futur_pad = [(kernel_size - 1) * dilation for dilation in dilations]

    self.conv1 = nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size, stride=stride, dilation=dilations[0], padding=pad[0]),
                                         nn.ReLU(inplace=True))
    self.conv2 = nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size, stride=stride, dilation=dilations[1], padding=pad[1]),
                                         nn.ReLU(inplace=True))
    self.conv3 = nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size, stride=stride, dilation=dilations[2], padding=pad[2]),
                                         nn.ReLU(inplace=True))
    self.conv4 = nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size, stride=stride, dilation=dilations[3], padding=pad[3]),
                                         nn.ReLU(inplace=True))
    self.feed_forward = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(d_model if self.only_see_past else 4*d_model, d_ff),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(dropout),
                                      nn.Linear(d_ff, d_model),
                                      nn.LayerNorm(d_model))
  
  def forward(self, x, y=None):
    x = x.permute(0, 2, 1)
    out1 = self.conv1(F.pad(x, (self.futur_pad[0], 0)) if self.only_see_past else x).permute(0, 2, 1)
    out2 = self.conv2(F.pad(x, (self.futur_pad[1], 0)) if self.only_see_past else x).permute(0, 2, 1)
    out3 = self.conv3(F.pad(x, (self.futur_pad[2], 0)) if self.only_see_past else x).permute(0, 2, 1)
    out4 = self.conv4(F.pad(x, (self.futur_pad[3], 0)) if self.only_see_past else x).permute(0, 2, 1)
    out = out1 + out2 + out3 + out4 if self.only_see_past else torch.cat((out1, out2, out3, out4), dim=-1)
    return self.feed_forward(out)


class ConvMultipleDilationBlockBNDrop(nn.Module):
  def __init__(self, d_model, n_heads, kernel_size, d_ff, stride=2, dropout=0., only_see_past=True, **kwargs):
    super().__init__()
    self.only_see_past = only_see_past
    dilations = list(range(1, 5))
    pad = [0 if only_see_past else (kernel_size - 1) // 2 + i for i in range(4)]
    if self.only_see_past:
      stride = 1
    self.futur_pad = [(kernel_size - 1) * dilation for dilation in dilations]

    self.conv1 = nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size, stride=stride, dilation=dilations[0], padding=pad[0]),
                                         nn.BatchNorm1d(d_model), nn.ReLU(inplace=True), nn.Dropout(dropout))
    self.conv2 = nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size, stride=stride, dilation=dilations[1], padding=pad[1]),
                                         nn.BatchNorm1d(d_model), nn.ReLU(inplace=True), nn.Dropout(dropout))
    self.conv3 = nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size, stride=stride, dilation=dilations[2], padding=pad[2]),
                                         nn.BatchNorm1d(d_model), nn.ReLU(inplace=True), nn.Dropout(dropout))
    self.conv4 = nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size, stride=stride, dilation=dilations[3], padding=pad[3]),
                                         nn.BatchNorm1d(d_model), nn.ReLU(inplace=True), nn.Dropout(dropout))
    self.feed_forward = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(d_model if self.only_see_past else 4*d_model, d_ff),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(dropout),
                                      nn.Linear(d_ff, d_model),
                                      nn.LayerNorm(d_model))
  
  def forward(self, x, y=None):
    x = x.permute(0, 2, 1)
    out1 = self.conv1(F.pad(x, (self.futur_pad[0], 0)) if self.only_see_past else x).permute(0, 2, 1)
    out2 = self.conv2(F.pad(x, (self.futur_pad[1], 0)) if self.only_see_past else x).permute(0, 2, 1)
    out3 = self.conv3(F.pad(x, (self.futur_pad[2], 0)) if self.only_see_past else x).permute(0, 2, 1)
    out4 = self.conv4(F.pad(x, (self.futur_pad[3], 0)) if self.only_see_past else x).permute(0, 2, 1)
    out = out1 + out2 + out3 + out4 if self.only_see_past else torch.cat((out1, out2, out3, out4), dim=-1)
    return self.feed_forward(out)


class ConvStackedDilationBlock(nn.Module):
  def __init__(self, d_model, n_heads, kernel_size, d_ff, stride=2, dropout=0., only_see_past=True, **kwargs):
    super().__init__()
  
  def forward(self, x, y=None):
    pass


class Seq2Seq(nn.Module):
  def __init__(self, output_dim, enc_emb_dim, dec_emb_dim, d_model, n_heads, enc_d_ff, dec_d_ff, kernel_size, enc_n_blocks, dec_n_blocks,
               enc_max_seq_len, dec_max_seq_len, dropout=0., n_step_aheads=1, enc_block_type='self_attn', dec_block_type='self_attn',
               decoder_layer='conv_layer'):
    super().__init__()
    self.n_step_aheads = n_step_aheads
    pos_emb = PositionalEncoding(enc_emb_dim, dropout=dropout, max_len=enc_max_seq_len)
    embedder = TextEmbedder(output_dim, emb_dim=dec_emb_dim, max_seq_len=dec_max_seq_len)
    # self.encoder = ConvDepthPointWise(pos_emb, out_size=d_model, repeat=5, kernels=[3, 5, 9, 15, 21, 33])
    self.encoder = ConvLayer(enc_emb_dim, d_model, n_heads, enc_d_ff, kernel_size, enc_n_blocks, pos_emb, dropout=dropout,
                             only_see_past=False, block_type=enc_block_type)
    if decoder_layer == 'conv_layer':
      self.decoder = ConvLayer(dec_emb_dim, d_model, n_heads, dec_d_ff, kernel_size, dec_n_blocks, embedder, dropout=dropout,
                              only_see_past=True, block_type=dec_block_type)
    else:
      self.decoder = DecodingConvLayer(dec_emb_dim, d_model, n_heads, dec_d_ff, kernel_size, dec_n_blocks, embedder, dropout=dropout,
                                       only_see_past=True, block_type=dec_block_type)
    self.output_proj = nn.Linear(d_model, n_step_aheads*output_dim)
  
  def forward(self, x, y):
    out = self.encoder(x)
    out = self.decoder(y, y=out)
    return self.output_proj(out)
  
  def greedy_decoding(self, x, sos_idx, eos_idx, max_seq_len=100):
    enc_out = self.encoder(x)

    dec_in = torch.LongTensor(x.size(0), 1).fill_(sos_idx).to(x.device)

    finished = [False] * x.size(0)

    for _ in range(max_seq_len):
      output = self.decoder(dec_in, y=enc_out)
      output = self.output_proj(output)
      pred = output[:, -1, :].argmax(-1).unsqueeze(1)

      for idx in range(x.size(0)):
        if not finished[idx] and pred[idx].item() == eos_idx:
          finished[idx] = True

      dec_in = torch.cat((dec_in, pred), dim=1)

      if all(finished):
        break
    
    return dec_in[:, 1:]


class Seq2SeqReview(nn.Module):
  def __init__(self, output_dim, enc_emb_dim, dec_emb_dim, d_model, n_heads, d_ff, kernel_size, n_blocks, enc_max_seq_len, dec_max_seq_len,
               dropout=0., n_step_aheads=1):
    super().__init__()
    pos_emb = PositionalEncoding(enc_emb_dim, dropout=dropout, max_len=enc_max_seq_len)
    embedder = TextEmbedder(output_dim, emb_dim=dec_emb_dim, max_seq_len=dec_max_seq_len)
    # self.encoder = ConvDepthPointWise(pos_emb, out_size=d_model, repeat=5, kernels=[3, 5, 9, 15, 21, 33])
    self.encoder = ConvLayer(enc_emb_dim, d_model, n_heads, d_ff, kernel_size, n_blocks, pos_emb, dropout=dropout, only_see_past=False)
    self.decoder = ConvLayer(dec_emb_dim, d_model, n_heads, d_ff, kernel_size, n_blocks, embedder, dropout=dropout, only_see_past=True)
    self.output_proj = nn.Linear(d_model, n_step_aheads * output_dim)
    self.review_decoder = ConvLayer(output_dim, d_model, n_heads, d_ff, kernel_size, n_blocks, lambda x: x,
                                       dropout=dropout, only_see_past=False)
    self.output_review_proj = nn.Linear(d_model, n_step_aheads * output_dim)
  
  def forward(self, x, y):
    enc_out = self.encoder(x)
    out = self.decoder(y, y=enc_out)
    first_out = self.output_proj(out)
    out = self.review_decoder(F.softmax(first_out.detach(), dim=-1), y=enc_out.detach())
    return self.output_review_proj(out), first_out
  
  def greedy_decoding(self, x, sos_idx, eos_idx, max_seq_len=100):
    enc_out = self.encoder(x)

    dec_in = torch.LongTensor(x.size(0), 1).fill_(sos_idx).to(x.device)
    outputs = []

    finished = [False] * x.size(0)

    for _ in range(max_seq_len):
      output = self.decoder(dec_in, y=enc_out)
      output = self.output_proj(output)
      
      outputs.append(output[:, -1, :].unsqueeze(1))
      pred = output[:, -1, :].argmax(-1).unsqueeze(1)

      for idx in range(x.size(0)):
        if not finished[idx] and pred[idx].item() == eos_idx:
          finished[idx] = True

      dec_in = torch.cat((dec_in, pred), dim=1)

      if all(finished):
        break
    
    output = self.review_decoder(F.softmax(torch.cat(outputs, dim=1), dim=-1), y=enc_out)
    output = self.output_review_proj(output)
    
    return output.argmax(-1), dec_in[:, 1:]