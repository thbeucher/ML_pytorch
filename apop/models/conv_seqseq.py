import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils as u

from collections import OrderedDict
from transformer.attention import MultiHeadAttention

############################################################################################################
### Implementation of Convolutional Sequence to Sequence Learning (https://arxiv.org/pdf/1705.03122.pdf) ###
############################################################################################################
class EncoderEmbedder(nn.Module):
  def __init__(self, input_dim, emb_dim, hid_dim, max_seq_len, dropout, device, reduce_dim=False, kernel=3, stride=2):
    super().__init__()
    self.device = device
    self.reduce_dim = reduce_dim

    self.positional_embedding = u.create_positional_embedding(max_seq_len, emb_dim, hid_dim)

    if reduce_dim:
      self.conv = nn.Sequential(nn.Conv2d(1, 1, kernel, stride=stride),
                                nn.ReLU(),
                                nn.Conv2d(1, 1, kernel, stride=stride),
                                nn.ReLU())
      input_dim = ((input_dim - (kernel - 1) - 1) // stride + 1) // 2

    self.dropout = nn.Dropout(dropout)

    self.proj = nn.Linear(input_dim, emb_dim)
    self.proj_act = nn.ReLU()
  
  def forward(self, x):
    '''
    Params:
      * x : [batch_size, seq_len]
    '''
    if self.reduce_dim:
      x = self.conv(x.unsqueeze(1).float()).squeeze(1)

    batch_size, seq_len, _ = x.shape

    index_x = torch.LongTensor(range(seq_len)).unsqueeze(0).repeat(batch_size, 1).to(self.device)
    pos_emb = self.positional_embedding(index_x).float()

    x_proj = self.proj_act(self.proj(x.float()))

    return self.dropout(x_proj + pos_emb)


class DecoderEmbedder(nn.Module):
  def __init__(self, input_dim, emb_dim, max_seq_len, dropout, device):
    super().__init__()
    self.device = device

    self.tok_embedding = nn.Embedding(input_dim, emb_dim)
    self.pos_embedding = nn.Embedding(max_seq_len, emb_dim)

    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    '''
    Params:
      * x : [batch_size, seq_len]
    '''
    tok_embedded = self.tok_embedding(x)

    pos = torch.arange(0, x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1).to(self.device)

    pos_embedded = self.pos_embedding(pos)

    return self.dropout(tok_embedded + pos_embedded)


class LMEncoderEmbedder(nn.Module):
  def __init__(self, input_dim, emb_dim, max_seq_len, device):
    super().__init__()
    self.device = device
    self.input_embedding = nn.Linear(input_dim, emb_dim)
    self.pos_embedding = nn.Embedding(max_seq_len, emb_dim)
  
  def forward(self, x):
    pos = torch.arange(0, x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1).to(self.device)

    return self.input_embedding(x) + self.pos_embedding(pos)


class Encoder(nn.Module):
  def __init__(self, emb_dim, hid_dim, n_layers, kernel_size, dropout, device, embedder=None, max_seq_len=100, input_dim=None):
    super().__init__()
    assert kernel_size % 2 == 1, 'Kernel size must be odd!'
    self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
    self.dropout = nn.Dropout(dropout)

    self.emb2hid = nn.Linear(emb_dim, hid_dim)
    self.encoders = nn.ModuleList([EncoderBlock(hid_dim, kernel_size, dropout, device) for _ in range(n_layers)])
    self.hid2emb = nn.Linear(hid_dim, emb_dim)

    self.embedder = EncoderEmbedder(input_dim, emb_dim, hid_dim, max_seq_len, dropout, device) if embedder is None else embedder
  
  def forward(self, x):
    # x -> [batch_size, seq_len]
    embedded = self.embedder(x)  # [batch_size, seq_len, emb_dim]
    conv_in = self.emb2hid(embedded)  # [batch_size, seq_len, hid_dim]
    conv_in = conv_in.permute(0, 2, 1)  # prepare for convolutional layers

    for encoder in self.encoders:
      conv_in = encoder(conv_in)  # [batch_size, hid_dim, seq_len]

    conved = conv_in.permute(0, 2, 1)  # [batch_size, seq_len, hid_dim]
    conved = self.hid2emb(conved)  # [batch_size, seq_len, emb_dim]
    combined = (conved + embedded) * self.scale  # elementwise sum output (conved) and input (embedded) to be used for attention
    return conved, combined


class EncoderBlock(nn.Module):
  def __init__(self, hid_dim, kernel_size, dropout, device):
    super().__init__()
    self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
    self.dropout = nn.Dropout(dropout)

    # nf_out = (nf_in + 2 * padding - dilation * (kernel_size -1) - 1) / stride + 1
    self.conv = nn.Conv1d(in_channels=hid_dim, out_channels=2 * hid_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
  
  def forward(self, x):
    # x -> [batch_size, hid_dim, seq_len]
    conved = self.conv(self.dropout(x))  # [batch_size, 2 * hid_dim, seq_len]
    conved = F.glu(conved, dim=1)  # [batch_size, hid_dim, seq_len]
    conved = (conved + x) * self.scale  # residual connection
    return conved


class Decoder(nn.Module):
  def __init__(self, output_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, pad_idx, device, embedder=None, max_seq_len=100,
               score_fn=F.softmax, scaling_energy=False, multi_head=False, d_keys_values=64):
    super().__init__()
    self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
    self.dropout = nn.Dropout(dropout)

    self.embedder = DecoderEmbedder(output_dim, emb_dim, max_seq_len, dropout, device) if embedder is None else embedder

    self.emb2hid = nn.Linear(emb_dim, hid_dim)
    self.decoders = nn.ModuleList([DecoderBlock(hid_dim, emb_dim, kernel_size, pad_idx, dropout, device, score_fn=score_fn,
                                                 scaling_energy=scaling_energy, multi_head=multi_head, d_keys_values=d_keys_values)
                                                  for _ in range(n_layers)])
    self.hid2emb = nn.Linear(hid_dim, emb_dim)

    self.out = nn.Linear(emb_dim, output_dim)
  
  def forward(self, x, encoder_conved, encoder_combined):
    '''
    Params:
      * x : [batch_size, seq_len]
      * encoder_conved : [batch_size, seq_len, emb_dim]
      * encoder_combined : [batch_size, seq_len, emb_dim]
    '''
    embedded = self.embedder(x)  # [batch_size, seq_len, emb_dim]
    conv_in = self.emb2hid(embedded)  # [batch_size, seq_len, hid_dim]
    conv_in = conv_in.permute(0, 2, 1)  # prepare for convolution layers

    for decoder in self.decoders:
      attention, conv_in = decoder(embedded, conv_in, encoder_conved, encoder_combined)

    conved = conv_in.permute(0, 2, 1)  # [batch_size, seq_len, hid_dim]
    conved = self.hid2emb(conved)  # [batch_size, seq_len, emb_dim]
    output = self.out(self.dropout(conved))  # [batch_size, seq_len, output_dim]
    return output, attention


class DecoderBlock(nn.Module):
  def __init__(self, hid_dim, emb_dim, kernel_size, pad_idx, dropout, device, score_fn=F.softmax, scaling_energy=False,
               multi_head=False, d_keys_values=64):
    super().__init__()
    self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
    self.dropout = nn.Dropout(dropout)

    self.kernel_size = kernel_size
    self.pad_idx = pad_idx
    self.device = device

    self.conv = nn.Conv1d(in_channels=hid_dim, out_channels=2 * hid_dim, kernel_size=kernel_size)

    self.attention = Attention(hid_dim, emb_dim, device, score_fn=score_fn, scaling_energy=scaling_energy,
                               multi_head=multi_head, d_keys_values=d_keys_values, dropout=dropout)
  
  def forward(self, embedded, conv_in, encoder_conved, encoder_combined):
    conv_in = self.dropout(conv_in)  # [batch_size, hid_dim, seq_len]
    padding = torch.zeros(conv_in.shape[0], conv_in.shape[1], self.kernel_size - 1).fill_(self.pad_idx).to(self.device)
    padded_conv_in = torch.cat((padding, conv_in), dim=2)  # [batch_size, hid_dim, seq_len + kernel_size - 1]
    conved = self.conv(padded_conv_in)  # [batch_size, 2 * hid_dim, seq_len]
    conved = F.glu(conved, dim=1)  # [batch_size, hid_dim, seq_len]
    attention, conved = self.attention(embedded, conved, encoder_conved, encoder_combined)
    conved = (conved + conv_in) * self.scale  # residual connection
    return attention, conved


class Attention(nn.Module):
  def __init__(self, hid_dim, emb_dim, device, score_fn=F.softmax, scaling_energy=False, multi_head=False, d_keys_values=64, dropout=0.):
    super().__init__()
    self.score_fn = score_fn
    self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
    self.multi_head = multi_head

    self.scaling_energy = torch.sqrt(torch.Tensor([emb_dim])) if scaling_energy else 1

    self.attention_hid2emb = nn.Linear(hid_dim, emb_dim)
    self.attention_emb2hid = nn.Linear(emb_dim, hid_dim)

    if multi_head:
      self.multi_head_att = MultiHeadAttention(emb_dim, d_keys_values, d_keys_values, emb_dim//d_keys_values, dropout=dropout)
  
  def forward(self, embedded, conved, encoder_conved, encoder_combined):
    '''
    Params:
      * embedded : [batch_size, dec_seq_len, emb_dim]
      * conved : [batch_size, hid_dim, dec_seq_len]
      * encoder_conved : [batch_size, enc_seq_len, emb_dim]
      * encoder_combined : [batch_size, enc_seq_len, emb_dim]
    '''
    conved_emb = self.attention_hid2emb(conved.permute(0, 2, 1))  # [batch_size, dec_seq_len, emb_dim]
    combined = (embedded + conved_emb) * self.scale

    if self.multi_head:
      attented_encoding = self.multi_head_att(combined, encoder_conved, encoder_combined)
      attention = self.multi_head_att.attention.energy.sum(1).softmax(-1)  # energy = [batch_size, n_heads, dec_seq_len, enc_seq_len]
    else:
      energy = combined.matmul(encoder_conved.permute(0, 2, 1)) / self.scaling_energy  # [batch_size, dec_seq_len, enc_seq_len]
      attention = self.score_fn(energy, dim=2)
      attented_encoding = attention.matmul(encoder_combined)  # [batch_size, dec_seq_len, emb_dim]
      # attented_encoding = attention.matmul(encoder_conved + encoder_combined)  # [batch_size, dec_seq_len, emb_dim]

    attented_encoding = self.attention_emb2hid(attented_encoding)  # [batch_size, dec_seq_len, hid_dim]
    attented_combined = (conved + attented_encoding.permute(0, 2, 1)) * self.scale  # [batch_size, hid_dim, dec_seq_len]
    return attention, attented_combined


class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, device):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.device = device
  
  def forward(self, enc_in, dec_in):
    '''
    Params:
      * enc_in : [batch_size, enc_seq_len]
      * dec_in : [batch_size, dec_seq_len]
    '''
    # encoder_conved is output from final encoder convolution block | [batch_size, enc_seq_len, emb_dim]
    # encoder_combined is encoder_conved + enc_in_emb + enc_in_pos_emb | [batch_size, enc_seq_len, emb_dim]
    encoder_conved, encoder_combined = self.encoder(enc_in)

    # compute predictions of next words
    # output is a batch of predictions for each word in the target sentence (dec_in)
    # attention is a batch of attention scores across the source sentence (enc_in) for each word in the target sentence (dec_in)
    output, attention = self.decoder(dec_in, encoder_conved, encoder_combined)
    
    return output, attention  # [batch_size, dec_seq_len, output_dim], [batch_size, dec_seq_len, enc_seq_len]
  
  def greedy_decoding(self, enc_in, sos_idx, eos_idx, max_seq_len=100):
    encoder_conved, encoder_combined = self.encoder(enc_in)

    batch_size = enc_in.shape[0]

    dec_in = torch.LongTensor(batch_size, 1).fill_(sos_idx).to(self.device)

    finished = [False] * batch_size

    for _ in range(max_seq_len):
      output, attention = self.decoder(dec_in, encoder_conved, encoder_combined)
      pred = output[:, -1, :].argmax(-1).unsqueeze(1)

      for idx in range(batch_size):
        if not finished[idx] and pred[idx].item() == eos_idx:
          finished[idx] = True

      dec_in = torch.cat((dec_in, pred), dim=1)

      if all(finished):
        break
    
    return dec_in[:, 1:], attention
  
  def mark_progress(self, output, finished, eos_idx, pad_idx, sentence_lengths, output_seq):
    batch_size, beam_size = output.shape
    for j in range(batch_size):
      for k in range(beam_size):
        if finished[j][k]:
          output[j, k] = pad_idx
        elif output[j, k].item() == eos_idx:
          finished[j][k] = 1
          sentence_lengths[j][k] = len(output_seq)
  
  def beam_search_decoding(self, enc_in, sos_idx, eos_idx, pad_idx, max_seq_len=600, beam_size=5, alpha=0.7):
    '''
    Performs beam search decoding ie using the best n predictions (n=beam_size) instead of just the best one.

    Beam_search_decoding using a beam_size of 1 is equivalent of greedy_decoding but less optimized.

    Params:
      * enc_in : tensor
      * eos_idx : int
      * pad_idx : int
      * max_seq_len : int, the maximum number of decoding steps
      * beam_size : int
      * alpha : float, used during final n
    
    Returns:
      * final_sentences : tensor, shape = (batch, seq_len)
    '''
    encoder_conved, encoder_combined = self.encoder(enc_in)

    batch_size = enc_in.shape[0]
    output_seq = []
    sentence_lengths = torch.zeros(batch_size, beam_size).to(self.device)
    beam_scores = torch.zeros(batch_size * beam_size, 1).to(self.device)
    finished = torch.zeros(batch_size, beam_size, dtype=torch.uint8).to(self.device)
    
    dec_in = torch.LongTensor(batch_size, 1).fill_(sos_idx).to(self.device)
    for_rows = torch.LongTensor([i * beam_size for i in range(batch_size)]).unsqueeze(1).repeat(1, beam_size).to(self.device)
    
    for i in range(max_seq_len):
      output, attention = self.decoder(dec_in, encoder_conved, encoder_combined)
      preds = F.softmax(output[:, -1, :], dim=-1)

      topk_vals, topk_idxs = torch.topk(preds, beam_size)  # (batch, beam) then (batch * beam, beam)

      if i == 0:
        beam_scores -= torch.log(topk_vals.view(-1).unsqueeze(1))  # (batch * beam_size)

        self.mark_progress(topk_idxs, finished, eos_idx, pad_idx, sentence_lengths, output_seq)

        output_seq.append(topk_idxs.view(-1).unsqueeze(1))  # [(batch * beam_size, 1)]
        dec_in = torch.cat((dec_in.repeat(beam_size, 1), topk_idxs.view(-1).unsqueeze(1)), dim=1)
        encoder_conved = encoder_conved.repeat(beam_size, 1, 1)
        encoder_combined = encoder_combined.repeat(beam_size, 1, 1)
        continue

      scores = beam_scores.repeat(1, beam_size) - torch.log(topk_vals)  # (batch * beam, beam)
      seq_scores = scores.view(batch_size, beam_size ** 2)  # (batch, beam * beam)
      seq_topk_vals, seq_topk_idxs = torch.topk(seq_scores, beam_size, largest=False)  # (batch, beam)

      beam_scores = seq_topk_vals.view(-1).unsqueeze(1)  # (batch * beam, 1)

      row_idxs = (seq_topk_idxs // beam_size + for_rows).view(-1)  # (batch * beam)
      col_idxs = (seq_topk_idxs % beam_size).view(-1)  # (batch * beam)

      out_idxs = topk_idxs[row_idxs, col_idxs]  # (batch * beam)

      self.mark_progress(out_idxs.view(batch_size, beam_size), finished, eos_idx, pad_idx, sentence_lengths, output_seq)

      output_seq.append(out_idxs.view(-1).unsqueeze(1))  # [(batch * beam, 1), ...]

      if torch.all(finished):
        break

      dec_in = torch.cat((dec_in, out_idxs.view(-1).unsqueeze(1)), dim=1)
    
    # choose the final sentence applying length normalization
    final_scores = beam_scores.view(batch_size, beam_size) / sentence_lengths.pow(alpha)  # (batch, beam)
    final_idxs = final_scores.argmax(dim=-1)  # (batch)
    all_out_seq = torch.cat(output_seq, dim=1)  # (batch * beam, seq_len)
    final_sentences = torch.stack([all_out_seq.view(batch_size, beam_size, -1)[i,j,:] for i, j in enumerate(final_idxs)])
    return final_sentences, attention


class AudioTextLM(nn.Module):
  def __init__(self, audio_encoder, audio_decoder, lm_encoder, lm_decoder, device):
    super().__init__()
    self.audio_encoder = audio_encoder
    self.audio_decoder = audio_decoder
    self.lm_encoder = lm_encoder
    self.lm_decoder = lm_decoder
    self.device = device

    self.softmax = nn.Softmax(dim=-1)
  
  def forward(self, audio_enc_in, audio_dec_in):
    audio_enc_conved, audio_enc_combined = self.audio_encoder(audio_enc_in)
    audio_output, audio_attention = self.audio_decoder(audio_dec_in, audio_enc_conved, audio_enc_combined)

    lm_enc_in = self.softmax(audio_output)
    lm_enc_conved, lm_enc_combined = self.lm_encoder(lm_enc_in)
    lm_output, lm_attention = self.lm_decoder(audio_dec_in, lm_enc_conved, lm_enc_combined)

    return audio_output, audio_attention, lm_output, lm_attention


class EncoderRelu(nn.Module):
  def __init__(self, emb_dim, hid_dim, n_layers, kernel_size, dropout, device, embedder=None, max_seq_len=100, input_dim=None):
    super().__init__()
    assert kernel_size % 2 == 1, 'Kernel size must be odd!'
    self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
    self.dropout = nn.Dropout(dropout)

    self.emb2hid = nn.Linear(emb_dim, hid_dim)
    self.encoders = nn.ModuleList([EncoderBlock(hid_dim, kernel_size, dropout, device) for _ in range(n_layers)])
    self.hid2emb = nn.Linear(hid_dim, emb_dim)

    self.relu = nn.ReLU(inplace=True)

    self.embedder = EncoderEmbedder(input_dim, emb_dim, hid_dim, max_seq_len, dropout, device) if embedder is None else embedder
  
  def forward(self, x):
    # x -> [batch_size, seq_len]
    embedded = self.embedder(x)  # [batch_size, seq_len, emb_dim]
    conv_in = self.relu(self.emb2hid(embedded))  # [batch_size, seq_len, hid_dim]
    conv_in = conv_in.permute(0, 2, 1)  # prepare for convolutional layers

    for encoder in self.encoders:
      conv_in = encoder(conv_in)  # [batch_size, hid_dim, seq_len]

    conved = conv_in.permute(0, 2, 1)  # [batch_size, seq_len, hid_dim]
    conved = self.relu(self.hid2emb(conved))  # [batch_size, seq_len, emb_dim]
    combined = (conved + embedded) * self.scale  # elementwise sum output (conved) and input (embedded) to be used for attention
    return conved, combined


class DecoderRelu(nn.Module):
  def __init__(self, output_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, pad_idx, device, embedder=None, max_seq_len=100):
    super().__init__()
    self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
    self.dropout = nn.Dropout(dropout)

    self.embedder = DecoderEmbedder(output_dim, emb_dim, max_seq_len, dropout, device) if embedder is None else embedder

    self.emb2hid = nn.Linear(emb_dim, hid_dim)
    self.decoders = nn.ModuleList([DecoderBlockRelu(hid_dim, emb_dim, kernel_size, pad_idx, dropout, device) for _ in range(n_layers)])
    self.hid2emb = nn.Linear(hid_dim, emb_dim)

    self.relu = nn.ReLU(inplace=True)

    self.out = nn.Linear(emb_dim, output_dim)
  
  def forward(self, x, encoder_conved, encoder_combined):
    '''
    Params:
      * x : [batch_size, seq_len]
      * encoder_conved : [batch_size, seq_len, emb_dim]
      * encoder_combined : [batch_size, seq_len, emb_dim]
    '''
    embedded = self.embedder(x)  # [batch_size, seq_len, emb_dim]
    conv_in = self.relu(self.emb2hid(embedded))  # [batch_size, seq_len, hid_dim]
    conv_in = conv_in.permute(0, 2, 1)  # prepare for convolution layers

    for decoder in self.decoders:
      attention, conv_in = decoder(embedded, conv_in, encoder_conved, encoder_combined)

    conved = conv_in.permute(0, 2, 1)  # [batch_size, seq_len, hid_dim]
    conved = self.relu(self.hid2emb(conved))  # [batch_size, seq_len, emb_dim]
    output = self.out(self.dropout(conved))  # [batch_size, seq_len, output_dim]
    return output, attention


class DecoderBlockRelu(nn.Module):
  def __init__(self, hid_dim, emb_dim, kernel_size, pad_idx, dropout, device):
    super().__init__()
    self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
    self.dropout = nn.Dropout(dropout)

    self.kernel_size = kernel_size
    self.pad_idx = pad_idx
    self.device = device

    self.conv = nn.Conv1d(in_channels=hid_dim, out_channels=2 * hid_dim, kernel_size=kernel_size)

    self.attention = AttentionRelu(hid_dim, emb_dim, device)
  
  def forward(self, embedded, conv_in, encoder_conved, encoder_combined):
    conv_in = self.dropout(conv_in)  # [batch_size, hid_dim, seq_len]
    padding = torch.zeros(conv_in.shape[0], conv_in.shape[1], self.kernel_size - 1).fill_(self.pad_idx).to(self.device)
    padded_conv_in = torch.cat((padding, conv_in), dim=2)  # [batch_size, hid_dim, seq_len + kernel_size - 1]
    conved = self.conv(padded_conv_in)  # [batch_size, 2 * hid_dim, seq_len]
    conved = F.glu(conved, dim=1)  # [batch_size, hid_dim, seq_len]
    attention, conved = self.attention(embedded, conved, encoder_conved, encoder_combined)
    conved = (conved + conv_in) * self.scale  # residual connection
    return attention, conved


class AttentionRelu(nn.Module):
  def __init__(self, hid_dim, emb_dim, device):
    super().__init__()
    self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

    self.attention_hid2emb = nn.Linear(hid_dim, emb_dim)
    self.attention_emb2hid = nn.Linear(emb_dim, hid_dim)

    self.relu = nn.ReLU(inplace=True)
  
  def forward(self, embedded, conved, encoder_conved, encoder_combined):
    '''
    Params:
      * embedded : [batch_size, dec_seq_len, emb_dim]
      * conved : [batch_size, hid_dim, dec_seq_len]
      * encoder_conved : [batch_size, enc_seq_len, emb_dim]
      * encoder_combined : [batch_size, enc_seq_len, emb_dim]
    '''
    conved_emb = self.relu(self.attention_hid2emb(conved.permute(0, 2, 1)))  # [batch_size, dec_seq_len, emb_dim]
    combined = (embedded + conved_emb) * self.scale
    energy = combined.matmul(encoder_conved.permute(0, 2, 1))  # [batch_size, dec_seq_len, enc_seq_len]
    attention = F.softmax(energy, dim=2)
    attented_encoding = attention.matmul(encoder_combined)  # [batch_size, dec_seq_len, emb_dim]
    # attented_encoding = attention.matmul(encoder_conved + encoder_combined)  # [batch_size, dec_seq_len, emb_dim] OLD_ATT
    attented_encoding = self.relu(self.attention_emb2hid(attented_encoding))  # [batch_size, dec_seq_len, hid_dim]
    attented_combined = (conved + attented_encoding.permute(0, 2, 1)) * self.scale  # [batch_size, hid_dim, dec_seq_len]
    return attention, attented_combined


class EncoderMixKernelRelu(nn.Module):
  def __init__(self, emb_dim, hid_dim, n_layers, kernel_sizes, dropout, device, embedder=None, max_seq_len=100, input_dim=None):
    '''
    Params:
      n_layers : list of int
      kernel_sizes : list of odd int
    '''
    super().__init__()
    assert all([ker % 2 == 1 for ker in kernel_sizes]), 'Kernel size must be odd!'
    self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
    self.dropout = nn.Dropout(dropout)

    self.emb2hid = nn.Linear(emb_dim, hid_dim)
    self.encoders = nn.ModuleList(
                      [
                        nn.Sequential(
                          OrderedDict(
                            [(f'encoder{i}_{j}', EncoderBlock(hid_dim, ker, dropout, device)) for j in range(n_layers[i])]
                          )
                        )
                      for i, ker in enumerate(kernel_sizes)
                      ]
                    )
    self.hid2emb = nn.Linear(hid_dim * len(kernel_sizes), emb_dim)

    self.relu = nn.ReLU(inplace=True)

    self.embedder = EncoderEmbedder(input_dim, emb_dim, hid_dim, max_seq_len, dropout, device) if embedder is None else embedder
  
  def forward(self, x):
    # x -> [batch_size, seq_len]
    embedded = self.embedder(x)  # [batch_size, seq_len, emb_dim]
    conv_in = self.relu(self.emb2hid(embedded))  # [batch_size, seq_len, hid_dim]
    conv_in = conv_in.permute(0, 2, 1)  # prepare for convolutional layers

    conv_in = torch.cat([encoder(conv_in) for encoder in self.encoders], dim=1)  # [batch_size, hid_dim * len(kernel_sizes), seq_len]

    conved = conv_in.permute(0, 2, 1)  # [batch_size, seq_len, hid_dim * len(kernel_sizes)]
    conved = self.relu(self.hid2emb(conved))  # [batch_size, seq_len, emb_dim]
    combined = (conved + embedded) * self.scale  # elementwise sum output (conved) and input (embedded) to be used for attention
    return conved, combined


if __name__ == '__main__':
  ## GLU(a, b) = a * sigmoid(b)
  ## ReGLU(a, b) = a * ReLU(b)
  pass
