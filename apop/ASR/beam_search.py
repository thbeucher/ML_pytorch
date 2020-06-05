import os
import sys
import json
import math
import torch
import numpy as np
import pickle as pk
import editdistance as ed

from tqdm import tqdm
from itertools import groupby

sys.path.append(os.path.abspath(__file__).replace('ASR/beam_search.py', ''))
import utils as u

from data import Data
from convnet_experiments import Experiment29
from ngrams_experiments import ngrams_encoding
# from multitasks_experiments import STTReviewTrainer2, Seq2SeqReview


@torch.no_grad()
def beam_search_experiment(beam_size=10):
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


def bs_test(model_name='bert-large-uncased-whole-word-masking'):  # distilbert-base-uncased
  sys.path.append('/Users/i350230/GITHUB/CTCDecoder/src/')
  import editdistance as ed
  from BKTree import BKTree
  from collections import defaultdict
  # from transformers import AutoModelWithLMHead, AutoTokenizer

  # tokenizer = AutoTokenizer.from_pretrained(model_name)
  # model = AutoModelWithLMHead.from_pretrained(model_name)

  with open('_Data_metadata_letters_wav2vec.pk', 'rb') as f:
    data = pk.load(f)
  
  with open('_ctc_exp3_predictions.pk', 'rb') as f:
    res = pk.load(f)
  
  idx_to_tokens = ['<blank>'] + data['idx_to_tokens'][3:]
  tokens_to_idx = {t: i for i, t in enumerate(idx_to_tokens)}
  
  greedy_preds = [np.array(p).argmax(-1).tolist() for p in res['predictions']]
  target_sentences = [''.join([idx_to_tokens[i] for i in t[:t.index(0) if 0 in t else None]]) for t in res['targets']]
  greedy_preds_sentences = [[i for i, _ in groupby(p)] for p in greedy_preds]
  greedy_preds_sentences = [''.join([idx_to_tokens[i] for i in p if i != 0]) for p in greedy_preds_sentences]
  print(Data.compute_scores(targets=target_sentences, predictions=greedy_preds_sentences, rec=False))

  vocabs = list(set([w for s in data['ids_to_transcript_train'].values() for w in s.lower().split(' ')]))
  vocabs += list(set([w for s in data['ids_to_transcript_test'].values() for w in s.lower().split(' ')]))
  # bk_tree = BKTree(vocabs)
  vocabs_set = set(vocabs)
  print(f'Vocab size = {len(vocabs_set)}')

  # for t, p in zip(target_sentences, greedy_preds_sentences):
  #   if t != p:
  #     for tw, pw in zip(t.split(' '), p.split(' ')):
  #       if tw != pw and pw not in vocabs_set:

  #         candidats = defaultdict(list)
  #         best_d = 100
  #         for w in vocabs_set:
  #           d = ed.eval(w, pw)
  #           if d == best_d:
  #             candidats[d].append(w)
  #           elif d < best_d:
  #             candidats = defaultdict(list)
  #             candidats[d].append(w)
  #             best_d = d
  #           else:
  #             continue
  #         print(f'target = {tw} | pred = {pw}')
  #         input(candidats)

          # resp = bk_tree.query(pw, 2)
          # input(f'{tw} | {pw}\n{resp}')
  
  lm_preds = []
  for t, p in tqdm(zip(target_sentences, greedy_preds_sentences), total=len(target_sentences)):
    new_source = p
    pw = p.split(' ')
    if any([w not in vocabs_set for w in pw]):
      source = ' '.join([tokenizer.mask_token if w not in vocabs_set else w for w in pw])
      enc_source = tokenizer.encode(source, return_tensors='pt')
      mask_token_index = torch.where(enc_source == tokenizer.mask_token_id)[1]

      token_logits = model(enc_source)[0]
      mask_token_logits = token_logits[0, mask_token_index, :]

      top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

      for token in top_5_tokens:
        if tokenizer.decode([token]) in vocabs_set:
          new_source = source.replace(tokenizer.mask_token, tokenizer.decode([token]))
          break
      # new_source = source.replace(tokenizer.mask_token, tokenizer.decode([top_5_tokens[0]]))
    lm_preds.append(new_source)
  
  print(Data.compute_scores(targets=target_sentences, predictions=lm_preds, rec=False))



if __name__ == "__main__":
  # beam_search_experiment()
  # beam_search_experiment2()

  # test_utility_bs()
  bs_test()