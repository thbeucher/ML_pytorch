import os
import sys
import json
import math
import torch
import numpy as np
import pickle as pk

from tqdm import tqdm

sys.path.append(os.path.abspath(__file__).replace('ASR/beam_search.py', ''))
import utils as u

from data import Data
from convnet_experiments import Experiment29
from ngrams_experiments import ngrams_encoding
from multitasks_experiments import STTReviewTrainer2, Seq2SeqReview


@torch.no_grad()
def beam_search_experiment(beam_size=5):
  exp = Experiment29(logfile='_DUMPS_logs.txt')

  u.load_model(exp.model, exp.save_name_model, restore_only_similars=True)
  exp.model.eval()

  targets, predictions = [], []
  for enc_in, dec_in in tqdm(exp.test_data_loader):
    encoder_conved, encoder_combined = exp.model.encoder(enc_in.to(exp.device))

    for i in range(encoder_conved.shape[0]):      
      encC1, encC2 = encoder_conved[i:i+1].repeat(beam_size, 1, 1), encoder_combined[i:i+1].repeat(beam_size, 1, 1)
      sequences = [[[exp.sos_idx], 0]] * beam_size

      for j in range(dec_in.shape[1]):
        current_dec_in = torch.LongTensor([s for s, _ in sequences]).to(exp.device)
        output, _ = exp.model.decoder(current_dec_in, encC1, encC2)
        pred_vals, pred_idxs = output[:, -1, :].softmax(-1).topk(beam_size)

        if j == 0:
          sequences = [[[exp.sos_idx, pi.item()], math.log(pv)] for pv, pi in zip(pred_vals[0], pred_idxs[0])]
          continue

        cur_seqs = []
        for k in range(beam_size):
          for l in range(beam_size):
            cur_seqs.append([sequences[k][0] + [pred_idxs[k, l].item()], sequences[k][1] + math.log(pred_vals[k, l])])
        
        sequences = sorted(cur_seqs, key=lambda x: x[1], reverse=True)[:beam_size]
      
      targets.append(dec_in[i, 1:].tolist())
      predictions.append(sequences[0][0][1:])
  
  targets_sentences = Data.reconstruct_sources(targets, exp.data.idx_to_tokens, exp.eos_idx, joiner='')
  preds_sentences = Data.reconstruct_sources(predictions, exp.data.idx_to_tokens, exp.eos_idx, joiner='')

  with open('beam_search_results.json', 'w') as f:
    json.dump([{'target': t, 'prediction': p} for t, p in zip(targets_sentences, preds_sentences)], f)


@torch.no_grad()
def beam_search_experiment2(beam_size=10):
  exp = STTReviewTrainer2(logfile='_DUMPS_logs.txt')

  # u.load_model(exp.model, exp.save_name_model, restore_only_similars=True)
  exp.model.eval()

  targets, predictions = [], []
  for enc_in, dec_in in tqdm(exp.test_data_loader):
    enc_out = exp.model.encoder(enc_in.to(exp.device))

    for i in range(enc_out.shape[0]):
      enc_out1 = enc_out[i:i+1].repeat(beam_size, 1, 1)
      sequences = [[[exp.sos_idx], 0, []]] * beam_size

      for j in range(dec_in.shape[1]):
        current_dec_in = torch.LongTensor([s for s, _, _ in sequences]).to(exp.device)
        output = exp.model.decoder(current_dec_in, y=enc_out1)
        output = exp.model.output_proj(output[:, -1]).softmax(-1)

        pred_vals, pred_idxs = output.topk(beam_size)

        if j == 0:
          sequences = [[[exp.sos_idx, pi.item()], math.log(pv), [out]] for pv, pi, out in zip(pred_vals[0], pred_idxs[0], output)]
          continue

        cur_seqs = []
        for k in range(beam_size):
          for l in range(beam_size):
            cur_seqs.append([sequences[k][0] + [pred_idxs[k, l].item()], sequences[k][1] + math.log(pred_vals[k, l]),
                             sequences[k][2] + [output[k]]])
        
        sequences = sorted(cur_seqs, key=lambda x: x[1], reverse=True)[:beam_size]
      
      review_in = torch.stack([torch.stack(s[2]) for s in sequences])
      output = exp.model.review_decoder(review_in, y=enc_out1)
      output = exp.model.output_review_proj(output).softmax(-1)

      scores = torch.max(output, -1)[0].sum(-1)
      best_sentence = output[scores.argmax()].argmax(-1).tolist()

      targets.append(dec_in[i, 1:].tolist())
      predictions.append(best_sentence)
  
  targets_sentences = Data.reconstruct_sources(targets, exp.data.idx_to_tokens, exp.eos_idx, joiner='')
  preds_sentences = Data.reconstruct_sources(predictions, exp.data.idx_to_tokens, exp.eos_idx, joiner='')

  with open('beam_search2_results.json', 'w') as f:
    json.dump([{'target': t, 'prediction': p} for t, p in zip(targets_sentences, preds_sentences)], f)


def test_utility_bs():
  with open('_Data_metadata_multigrams_wav2vec.pk', 'rb') as f:
    metadata = pk.load(f)
  
  with open('_multitaks_preds.json', 'r') as f:
    res = json.load(f)
  
  tokens_to_idx = metadata['tokens_to_idx']
  letters_to_idx = {k: v for k, v in tokens_to_idx.items() if v < 31}
  words_to_idx = {k: v for k, v in tokens_to_idx.items() if v >= len(tokens_to_idx) - 3000}
  trigrams_to_idx = {k: v for k, v in tokens_to_idx.items() if v >= len(tokens_to_idx) - 4500 and v < len(tokens_to_idx) - 3000}
  bigrams_to_idx = {k: v for k, v in tokens_to_idx.items() if v >= 31 and v < len(tokens_to_idx) - 4500}

  model = Seq2SeqReview(len(metadata['idx_to_tokens']), metadata['n_signal_feats'], 50, 256, 4, 320, 3, 5,
                        metadata['max_signal_len'], metadata['max_source_len'], dropout=0.)
  u.load_model(model, 'convnet/sttReview_trainer2.pt', restore_only_similars=True)

  rep = []
  with tqdm(total=len(res)) as t:
    for target, first_greedy_pred in tqdm([(el['target'], el['first_greedy_prediction']) for el in res]):
      te = [tokens_to_idx['<sos>']] + ngrams_encoding(target, letters_to_idx, bigrams_to_idx, trigrams_to_idx, words_to_idx)
      fgpe = [tokens_to_idx['<sos>']] + ngrams_encoding(first_greedy_pred, letters_to_idx, bigrams_to_idx, trigrams_to_idx, words_to_idx)
      # te = [tokens_to_idx['<sos>']] + [tokens_to_idx[el] for el in target]
      # fgpe = [tokens_to_idx['<sos>']] + [tokens_to_idx[el] for el in first_greedy_pred]

      id_ = [k for k, v in metadata['ids_to_transcript_test'].items() if v.lower() == target]

      if len(id_) == 0:
        continue
      
      id_ = id_[0]

      enc_in = torch.tensor(np.load(metadata['ids_to_audiofile_test'][id_].replace('.flac', '.features.npy'))).unsqueeze(0)

      dec_in_t = torch.LongTensor(te).unsqueeze(0)
      dec_in_fgp = torch.LongTensor(fgpe).unsqueeze(0)

      out_t, first_out_t = model(enc_in, dec_in_t)
      out_fgp, first_out_fgp = model(enc_in, dec_in_fgp)

      score_target = torch.max(out_t.softmax(-1), -1)[0].sum()
      score_fgp = torch.max(out_fgp.softmax(-1), -1)[0].sum()

      rep.append(score_target > score_fgp)

      t.set_description(f'{sum(rep)}/{len(rep)}')
      t.update(1)
  
  print(f'ok -> {sum(rep)} / {len(rep)}')


if __name__ == "__main__":
  # beam_search_experiment()
  # beam_search_experiment2()

  test_utility_bs()