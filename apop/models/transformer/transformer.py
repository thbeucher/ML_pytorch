import os
import sys
import torch
import utils as u
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from encoder import TransformerEncoder
from decoder import TransformerDecoder
from embedder import PositionalEmbedder

sys.path.append(os.path.abspath(__file__).replace('models/transformer/transformer.py', ''))
import utils as u


class Transformer(nn.Module):
  def __init__(self,
               n_encoder_blocks,
               n_decoder_blocks,
               d_model,
               d_keys,
               d_values,
               n_heads,
               d_ff,
               output_size,
               encoder_embedder=None,
               decoder_embedder=None,
               encoder_embedding_dim=80,
               decoder_embedding_dim=100,
               enc_max_seq_len=900,
               dec_max_seq_len=600,
               encoder_reduce_dim=False,
               decoder_reduce_dim=False,
               apply_softmax=False,
               scaling=True,
               pad_idx=2,
               dropout=0.1,
               device=None):
    '''
    Params:
      * n_encoder_blocks : int
      * n_decoder_blocks : int
      * d_model : int
      * d_keys : int
      * d_values : int
      * n_heads : int
      * d_ff : int
      * output_size : int
      * encoder_embedder (optional) : nn.Embedding
      * decoder_embedder (optional) : nn.Embedding
      * encoder_embedding_dim (optional) : int, default to 80
      * decoder_embedding_dim (optional) : int, default to 100
      * enc_max_seq_len (optional) : int, default to 900
      * dec_max_seq_len (optional) : int, default to 600
      * encoder_reduce_dim (optional) : Boolean, default to False
      * decoder_reduce_dim (optional) : Boolean, default to False
      * apply_softmax (optional) : Boolean, default to False
      * scaling (optional) : Boolean, default to True
      * pad_idx (optional) : int, default to 2
      * dropout (optional) : float, default to 0.1
      * device (optional) : torch.device
    '''
    super().__init__()
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.apply_softmax = apply_softmax

    self.encoder_reduce_dim = encoder_reduce_dim
    self.pad_idx = pad_idx

    self.encoder_embedder = encoder_embedder
    self.decoder_embedder = decoder_embedder

    if encoder_embedder is None:
      self.encoder_embedder = PositionalEmbedder(enc_max_seq_len, encoder_embedding_dim, d_model, scaling=scaling,
                                                 reduce_dim=encoder_reduce_dim, dropout=dropout, device=device)

    if decoder_embedder is None:
      self.decoder_embedder = PositionalEmbedder(dec_max_seq_len, decoder_embedding_dim, d_model, scaling=scaling,
                                                 reduce_dim=decoder_reduce_dim, dropout=dropout, device=device,
                                                 output_size=output_size)

    self.encoder = TransformerEncoder(n_encoder_blocks, d_model, d_keys, d_values, n_heads, d_ff, dropout=dropout)
    self.decoder = TransformerDecoder(n_decoder_blocks, d_model, d_keys, d_values, n_heads, d_ff, dropout=dropout, device=self.device)

    self.encoder_input_projection = nn.Linear(self.encoder_embedder.embedding_size, d_model)
    self.decoder_input_projection = nn.Linear(self.decoder_embedder.embedding_size, d_model)

    self.output_projection = nn.Linear(d_model, output_size)

    if apply_softmax:
      self.softmax = nn.Softmax(dim=-1)
  
  def forward(self, enc_in, dec_in, padding_mask=None):
    enc_in, dec_in = self.embed_input(enc_in, dec_in)

    if self.encoder_reduce_dim:
      padding_mask = u.create_padding_mask(enc_in, self.pad_idx)

    enc_out = self.encoder(enc_in, padding_mask=padding_mask)

    dec_out = self.decoder(dec_in, enc_out, padding_mask=padding_mask)

    output = self.output_projection(dec_out)

    return self.softmax(output) if self.apply_softmax else output
  
  def embed_input(self, enc_in=None, dec_in=None):
    if enc_in is not None:
      enc_in = self.encoder_embedder(enc_in)
      enc_in = self.encoder_input_projection(enc_in)
    
    if dec_in is not None:
      dec_in = self.decoder_embedder(dec_in)
      dec_in = self.decoder_input_projection(dec_in)

    return enc_in, dec_in
  
  def greedy_decoding(self, enc_in, eos_idx, pad_idx, max_seq_len=600):
    '''
    Params:
      * enc_in : torch tensor, encoder input, shape = (batch, seq_len) or (batch, seq_len, num_feat)
      * eos_idx : int
      * pad_idx : int
      * max_seq_len : int
    
    Returns:
      * tensor, shape = (batch, seq_len)
    '''
    self.decoder.reset_memory()

    batch_size = enc_in.shape[0]
    output_seq = []
    finished = [False] * batch_size

    enc_in, _ = self.embed_input(enc_in)
    enc_out = self.encoder(enc_in)

    for_pred = torch.LongTensor(batch_size, 1).zero_().to(self.device)

    for _ in range(max_seq_len):
      current_dec_in = torch.cat([for_pred] + output_seq, dim=1)
      _, current_dec_in = self.embed_input(dec_in=current_dec_in)

      dec_out = self.decoder(current_dec_in, enc_out, save=True, aggregate=True)[:, -1, :]
      preds = self.output_projection(dec_out)

      output = preds.argmax(dim=-1)

      for idx in range(batch_size):
        if finished[idx]:
          output[idx] = pad_idx
        elif output[idx].item() == eos_idx:
          finished[idx] = True

      output_seq.append(output.unsqueeze(1))

      if all(finished):
        break

    return torch.cat(output_seq, dim=1)
  
  def mark_progress(self, output, finished, eos_idx, pad_idx, sentence_lengths, output_seq):
    batch_size, beam_size = output.shape
    for j in range(batch_size):
      for k in range(beam_size):
        if finished[j][k]:
          output[j, k] = pad_idx
        elif output[j, k].item() == eos_idx:
          finished[j][k] = 1
          sentence_lengths[j][k] = len(output_seq)
  
  def beam_search_decoding(self, enc_in, eos_idx, pad_idx, max_seq_len=600, beam_size=3, alpha=0.7):
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
    self.decoder.reset_memory()

    enc_in, _ = self.embed_input(enc_in)
    enc_out = self.encoder(enc_in)

    batch_size = enc_in.shape[0]
    output_seq = []
    sentence_lengths = torch.zeros(batch_size, beam_size)
    beam_scores = torch.zeros(batch_size * beam_size, 1)
    finished = torch.zeros(batch_size, beam_size, dtype=torch.uint8)
    
    current_dec_in = torch.cat(output_seq + [torch.LongTensor(batch_size, 1).zero_().to(self.device)], dim=1)
    for_pred = [torch.LongTensor(batch_size * beam_size, 1).zero_().to(self.device)]
    for_rows = torch.LongTensor([i * beam_size for i in range(batch_size)]).unsqueeze(1).repeat(1, beam_size)
    
    for i in range(max_seq_len):
      _, current_dec_in = self.embed_input(dec_in=current_dec_in)

      dec_out = self.decoder(current_dec_in, enc_out, save=True, aggregate=True)[:, -1, :]
      preds = self.output_projection(dec_out)  # (batch, vocab_size) then (batch * beam, vocab_size)

      topk_vals, topk_idxs = torch.topk(preds, beam_size)  # (batch, beam) then (batch * beam, beam)

      if i == 0:
        beam_scores += torch.log(topk_vals).view(-1).unsqueeze(1)  # (batch * beam_size)

        self.mark_progress(topk_idxs, finished, eos_idx, pad_idx, sentence_lengths, output_seq)

        output_seq.append(topk_idxs.view(-1).unsqueeze(1))  # [(batch * beam_size, 1)]
        current_dec_in = torch.cat(output_seq + for_pred, dim=1)  # (batch * beam_size, 2)
        enc_out = enc_out.repeat(beam_size, 1, 1)
        continue
      
      scores = beam_scores.repeat(1, beam_size) + torch.log(topk_vals)  # (batch * beam, beam)
      seq_scores = scores.view(batch_size, beam_size ** 2)  # (batch, beam * beam)
      seq_topk_vals, seq_topk_idxs = torch.topk(seq_scores, beam_size)  # (batch, beam)

      beam_scores += torch.log(seq_topk_vals).view(-1).unsqueeze(1)  # (batch * beam, 1)

      row_idxs = (seq_topk_idxs // beam_size + for_rows).view(-1)  # (batch * beam)
      col_idxs = (seq_topk_idxs % beam_size).view(-1)  # (batch * beam)

      out_idxs = topk_idxs[row_idxs, col_idxs]  # (batch * beam)

      self.mark_progress(out_idxs.view(batch_size, beam_size), finished, eos_idx, pad_idx, sentence_lengths, output_seq)

      output_seq.append(out_idxs.view(-1).unsqueeze(1))  # [(batch * beam, 1), ...]

      if torch.all(finished):
        break

      current_dec_in = torch.cat(output_seq + for_pred, dim=1)  # (batch * beam, n)
    
    # choose the final sentence applying length normalization
    final_scores = beam_scores.view(batch_size, beam_size) / sentence_lengths.pow(alpha)  # (batch, beam)
    final_idxs = final_scores.argmax(dim=-1)  # (batch)
    all_out_seq = torch.cat(output_seq, dim=1)  # (batch * beam, seq_len)
    final_sentences = torch.stack([all_out_seq.view(batch_size, beam_size, -1)[i,j,:] for i, j in enumerate(final_idxs)])
    return final_sentences
