import torch
import numpy as np

from itertools import groupby, product


def transition_loss_test():
  # vocab = '$abc'  # correct transitions = ab, ac | all others are incorrect
  not_ok_combis = [(1, 1), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
  #          1  1  0  0  0  0  0  0  0
  # [0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]
  # candidat1 : aabac -> abac | 2ok,1not
  candidat1 = [[0.1, 0.7, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]]
  # candidat2 : ab$ac -> abac | 2ok,1not
  candidat2 = [[0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.7, 0.1, 0.1, 0.1], [0.1, 0.8, 0.05, 0.05], [0.1, 0.1, 0.1, 0.7]]
  # candidat3 : aa$aab -> aab | 1ok,1not
  candidat3 = [[0.1, 0.7, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1],
               [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]]
  # candidat4 : bcc$aa -> bca | 2not
  candidat4 = [[0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7], [0.1, 0.1, 0.1, 0.7], [0.7, 0.1, 0.1, 0.1],
               [0.1, 0.7, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]]
  
  def get_transitions(candidat, not_ok_combis, to_torch=True):
    cand_idxs = candidat.argmax(-1).tolist()
    run_idxs = []
    for i in range(0, len(cand_idxs)):
      if cand_idxs[i] != 0:
        if len(run_idxs) == 0:
          run_idxs.append((i, cand_idxs[i]))
        else:
          if cand_idxs[i] != cand_idxs[i-1]:
            run_idxs.append((i, cand_idxs[i]))
    transition_idxs = []
    for i in range(0, len(run_idxs) - 1):
      comb = (run_idxs[i][1], run_idxs[i+1][1])
      if comb in not_ok_combis:
        err = [1 if j == c else 0 for c in comb for j in range(candidat.shape[-1])]
        err = torch.Tensor(err) if to_torch else np.array(err)
        transition_idxs.append((run_idxs[i][0], run_idxs[i+1][0], err))
    return transition_idxs, len(run_idxs) - 1
  
  def compute_loss(candidat, not_ok_combis):
    candidat = np.array(candidat)
    transitions, n_trans = get_transitions(candidat, not_ok_combis, to_torch=False)
    loss = 0
    for i, j, err in transitions:
      current = np.concatenate([candidat[i], candidat[j]])
      loss += (current * err).sum()
    return loss / n_trans
  
  for cand in [candidat1, candidat2, candidat3, candidat4]:
    loss = compute_loss(cand, not_ok_combis)
    print(f'loss = {loss}')
  
  def compute_loss_torch(candidat, not_ok_combis):
    transitions, n_trans = get_transitions(candidat, not_ok_combis)
    loss = 0
    for i, j, err in transitions:
      current = torch.cat([candidat[i], candidat[j]], dim=0)
      loss += (current * err).sum()
    return loss / n_trans
  
  print('TORCH VERSION:')
  candidats = [torch.Tensor(candidat1), torch.Tensor(candidat2), torch.Tensor(candidat3), torch.Tensor(candidat4)]
  weights = [torch.nn.Parameter(torch.ones(5, 4)), torch.nn.Parameter(torch.ones(5, 4)), torch.nn.Parameter(torch.ones(6, 4)),
             torch.nn.Parameter(torch.ones(6, 4))]
  for cand, w in zip(candidats, weights):
    cand = cand * w
    loss = compute_loss_torch(cand, not_ok_combis)
    print(f'loss = {loss}')
    loss.backward()
    print(f'grad = {w.grad}')


def get_groupby_idxs(candidat, blank_token=0):
  cand_idxs = candidat.argmax(-1).tolist()
  run_idxs = []
  for i in range(0, len(cand_idxs)):
    if cand_idxs[i] != blank_token:  # ignore blank token
      if len(run_idxs) == 0:
        run_idxs.append((i, cand_idxs[i]))
      else:
        if cand_idxs[i] != cand_idxs[i-1]:
          run_idxs.append((i, cand_idxs[i]))
  return run_idxs


def get_transitions(candidat, not_ok_transitions, blank_token=0):
  '''
  Params:
    * candidat
    * not_ok_transitions
  
  Returns:
    * transition_idxs : list of tuple
    * n_transitions : int
  '''
  run_idxs = get_groupby_idxs(candidat, blank_token=blank_token)
  transition_idxs = []
  for i in range(0, len(run_idxs) - 1):
    comb = (run_idxs[i][1], run_idxs[i+1][1])
    if comb in not_ok_transitions:
      err = torch.Tensor([1 if j == c else 0 for c in comb for j in range(candidat.shape[-1])]).to(candidat.device)
      transition_idxs.append((run_idxs[i][0], run_idxs[i+1][0], err))
  return transition_idxs, max(len(run_idxs) - 1, 1)


def get_not_ok_words(candidat, ok_words, blank_token=0, space_token=1):
  run_idxs = get_groupby_idxs(candidat, blank_token=blank_token)
  words, tmp_word, tmp_idx = [], [], []
  for pos, idx in run_idxs:
    if len(tmp_word) == 0:
      tmp_word.append(idx)
      tmp_idx.append(pos)
    elif idx == space_token:  # end of current word
      words += [(tuple(tmp_word), tmp_idx)]
      tmp_word = []
      tmp_idx = []
    else:
      tmp_word.append(idx)
      tmp_idx.append(pos)
  not_ok_words = []
  for word, idxs in words:
    if word not in ok_words:
      err = torch.Tensor([1 if j == i else 0 for i in word for j in range(candidat.shape[-1])]).to(candidat.device)
      not_ok_words.append((idxs, err))
  return not_ok_words, max(len(words), 1)


def test_get_transitions():
  candidat = np.array([[0.1, 0.7, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]])
  not_ok_combis = [(1, 1), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
  print(get_transitions(candidat, not_ok_combis))


def compute_transition_loss(predictions, not_ok_transitions):
  # predictions = [batch_size, seq_len, output_size]
  # transition_mat = [2 * output_size, n_total_transition]
  loss = 0
  for batch_pred in predictions:
    transitions, n_trans = get_transitions(batch_pred, not_ok_transitions)
    batch_pred_loss = 0
    for i, j, err in transitions:
      current = torch.cat([batch_pred[i], batch_pred[j]], dim=0)
      batch_pred_loss += (current * err).sum()
    loss += batch_pred_loss / n_trans
  return loss / predictions.shape[0]


def compute_transition_loss2(predictions, transition_mat):
  # predictions = [batch_size, seq_len, output_size]
  # transition_mat = [2 * output_size, n_total_transition]
  loss = 0
  for batch_pred in predictions:
    preds_idx = batch_pred.argmax(-1)
    preds = batch_pred[preds_idx != 0]  # ignore blanks
    preds_idx_wo_blanks = preds.argmax(-1)
    batch_pred_loss = 0
    for i in range(0, len(preds) - 1):
      if preds_idx_wo_blanks[i] != preds_idx_wo_blanks[i+1]:
        current = preds[i:i+2].reshape(transition_mat.size(0), 1)
        loss += (current * transition_mat).sum(0).sum()
    n_trans = max(1, len(list(groupby(preds_idx))) - len(batch_pred[preds_idx == 0]))
    loss += batch_pred_loss / n_trans
  return -loss / predictions.shape[0]  # batch mean loss


def compute_wrong_words_loss(predictions, ok_words, blank_token=0, space_token=1):
  loss = 0
  for batch_pred in predictions:
    not_ok_words, n_words = get_not_ok_words(batch_pred, ok_words, blank_token=blank_token, space_token=space_token)
    batch_pred_loss = 0
    for idxs, err in not_ok_words:
      current = torch.cat([batch_pred[i] for i in idxs], dim=0)
      batch_pred_loss += (current * err).sum()
    loss += batch_pred_loss / n_words
  return loss / predictions.shape[0]


def create_transition_mat(sources, idx_to_tokens):  # idx_to_tokens = [<blanck>, ...]
  combis = [''.join(comb) for comb in product(''.join(idx_to_tokens[1:]), repeat=2)]
  ok_combis = []
  for comb in combis:
    for s in sources:
      if comb in s:
        ok_combis.append(comb)
        break
  trans_mat = []
  for comb in combis:
    val = len(combis) if comb in ok_combis else -1
    trans_mat.append([val if i == idx_to_tokens.index(c) else 0 for c in comb for i in range(len(idx_to_tokens))])
  return np.array(trans_mat).T


def get_not_ok_transitions(sources, idx_to_tokens):
  '''
  Params:
    * sources : list of str
    * idx_to_tokens : list of str
  
  Returns:
    * not_ok_combis : set of tuple of int
    * readable_noc : list of str
  '''
  combis = [''.join(comb) for comb in product(''.join(idx_to_tokens[1:]), repeat=2)]
  ok_combis = []
  for comb in combis:
    for s in sources:
      if comb in s:
        ok_combis.append(comb)
        break
  not_ok_combis, readable_noc = [], []
  for comb in combis:
    if comb not in ok_combis:
      not_ok_combis.append((idx_to_tokens.index(comb[0]), idx_to_tokens.index(comb[1])))
      readable_noc.append(comb)
  return set(not_ok_combis), readable_noc


def get_ok_words(sources, tokens_to_idx):
  '''words here are defined as every sequence of characters between two space token'''
  words = list(set([w for s in sources for w in s.split(' ')]))
  words_idx_seq = [tuple([tokens_to_idx[c] for c in w]) for w in words]
  return set(words_idx_seq), words


def test_trans_mat():
  print(create_transition_mat(['ab', 'ac'], '$abc'))


def test_not_ok_combis():
  not_ok_trans, readable_not = get_not_ok_transitions(['ab', 'ac'], '$abc')
  print(not_ok_trans, readable_not)


def test_not_ok_words():
  tokens_to_idx = {k:i for i, k in enumerate([' ', 'i', 's', 'e', 'y', 'o', 'u', 'd', 'm'])}
  get_ok_words(['i see you', 'do you see me'], tokens_to_idx)


def test_trans_mat_trans_loss():
  import pickle as pk
  from ctc_experiments import CTCTrainer

  with open('_Data_metadata_letters_wav2vec.pk', 'rb') as f:
    data = pk.load(f)
  
  idx_to_tokens = ['<blank>'] + data['idx_to_tokens'][3:]
  tokens_to_idx = {t: i for i, t in enumerate(idx_to_tokens)}
  
  sources = [s.lower() for s in data['ids_to_transcript_train'].values()] + [s.lower() for s in data['ids_to_transcript_test'].values()]
  not_ok_transitions, readable_not = get_not_ok_transitions(sources, ['$'] + data['idx_to_tokens'][3:])
  ok_words, readable_words = get_ok_words(sources, tokens_to_idx)

  with open('_ctc_exp3_predictions.pk', 'rb') as f:
    res = pk.load(f)
  
  def compute_acc(targets, preds):
    # mask = np.ones(targets.shape)
    # mask[targets == 0] = 0
    targets = torch.cat([targets, torch.zeros(targets.size(0), preds.size(1) - targets.size(1))], dim=1)
    return ((targets == preds.argmax(-1)).sum(1).float() / targets.shape[1]).mean()

  for i in range(0, 256, 32):
    t_sent, p_sent = CTCTrainer.reconstruct_sentences(res['targets'][i:i+32],
                                                      [np.array(p).argmax(-1).tolist() for p in res['predictions'][i:i+32]],
                                                      idx_to_tokens, tokens_to_idx)
    
    if not any([True for rnot in readable_not for ps in p_sent if rnot in ps]):
      continue

    scores = CTCTrainer.scorer(t_sent, p_sent, rec=False)
    print(f"batch{i} scores = {' | '.join([f'{k} = {v:.3f}' for k, v in scores.items()])}")
    loss = compute_transition_loss(torch.Tensor(res['predictions'][i:i+32]), not_ok_transitions)
    print(f"loss = {loss} | {loss*1e2}")
    # print(f"Acc batch{i+1} = {compute_acc(torch.Tensor(res['targets'][i:i+32]), torch.Tensor(res['predictions'][i:i+32]))}")

    wrong_words = [w for s in p_sent for w in s.split() if w not in readable_words]
    # print(f'wrong_words = {wrong_words}')
    loss = compute_wrong_words_loss(torch.Tensor(res['predictions'][i:i+32]), ok_words)
    print(f'n_wrong_words = {len(wrong_words)}\nloss = {loss}\n')
  

if __name__ == "__main__":
  # transition_loss_test()
  # test_trans_mat()
  # test_not_ok_combis()
  # test_get_transitions()
  # test_not_ok_words()

  test_trans_mat_trans_loss()