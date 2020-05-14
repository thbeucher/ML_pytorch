import os
import sys
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(__file__).replace('ASR/multitasks_experiments.py', ''))
import utils as u

from data import Data
from custom_exp import TextEmbedder
from models.transformer.attention import MultiHeadAttention


class ConvSelfAttn(nn.Module):
  def __init__(self, n_input_feats, d_model, n_heads, d_ff, kernel_size, n_blocks, embedder, dropout=0., only_see_past=True):
    super().__init__()
    self.embedder = embedder
    self.input_proj = nn.Sequential(nn.Linear(n_input_feats, d_model), nn.ReLU(inplace=True), nn.LayerNorm(d_model))
    self.blocks = nn.ModuleList([ConvSelfAttnBlock(d_model, n_heads, kernel_size, d_ff, dropout=dropout, only_see_past=only_see_past)
                                    for _ in range(n_blocks)])
  
  def forward(self, x):
    x = self.embedder(x)
    x = self.input_proj(x)
    for block in self.blocks:
      x = block(x)
    return x


class ConvSelfAttnBlock(nn.Module):
  def __init__(self, d_model, n_heads, kernel_size, d_ff, dropout=0., only_see_past=True):
    super().__init__()
    self.kernel_size = kernel_size
    self.only_see_past = only_see_past

    padding = 0 if only_see_past else (kernel_size - 1) // 2
    self.conv = nn.Conv1d(in_channels=d_model, out_channels=2 * d_model, kernel_size=kernel_size, padding=padding)

    d_keys_vals = d_model // n_heads
    self.attn = MultiHeadAttention(d_model, d_keys_vals, d_keys_vals, n_heads, dropout=dropout)
    self.attn_norm = nn.LayerNorm(d_model)

    self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(inplace=True),
                                      nn.Linear(d_ff, d_model), nn.ReLU(inplace=True), nn.LayerNorm(d_model))
  
  def forward(self, x, futur_mask=None):  # [batch_size, seq_len, d_model]
    x_pad = x
    if self.only_see_past:
      x_pad = F.pad(x, (0, 0, self.kernel_size - 1, 0, 0, 0))
      futur_mask = (torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1) == 0).to(x.device)

    x_pad = x_pad.permute(0, 2, 1)
    conved = self.conv(x_pad)  # [batch_size, 2 * hid_dim, seq_len]
    conved = F.glu(conved, dim=1)  # [batch_size, hid_dim, seq_len]
    conved = conved.permute(0, 2, 1)
    conved = conved + x  # residual connection

    self_attn = self.attn_norm(self.attn(conved, conved, conved, mask=futur_mask))

    return self.feed_forward(self_attn)


class TextModel(nn.Module):
  def __init__(self, output_dim, emb_dim, d_model, n_heads, d_ff, kernel_size, n_blocks, max_seq_len,
               dropout=0., only_see_past=True, n_step_aheads=1):
    super().__init__()
    self.n_step_aheads = n_step_aheads
    embedder = TextEmbedder(output_dim, emb_dim=emb_dim, max_seq_len=max_seq_len)
    self.net = ConvSelfAttn(emb_dim, d_model, n_heads, d_ff, kernel_size, n_blocks, embedder, dropout=dropout, only_see_past=only_see_past)
    self.output_proj = nn.Linear(d_model, n_step_aheads*output_dim)
  
  def forward(self, x):
    x = self.net(x)
    return self.output_proj(x)


class TextDataset(Dataset):
  def __init__(self, ids_to_encodedsources, sort_by_target_len=True):
    self.ids_to_encodedsources = ids_to_encodedsources
    self.identities = list(sorted(ids_to_encodedsources.keys()))

    if sort_by_target_len:
      self.identities = TextDataset._sort_by_targets_len(self.identities, ids_to_encodedsources)
  
  @staticmethod
  def _sort_by_targets_len(ids, ids2es):
    return list(map(lambda x: x[0], sorted([(i, len(ids2es[i])) for i in ids], key=lambda x: x[1])))
  
  def __len__(self):
    return len(self.identities)
  
  def __getitem__(self, idx):
    return torch.LongTensor(self.ids_to_encodedsources[self.identities[idx]])


class TextCollator(object):
  def __init__(self, pad_idx):
    self.pad_idx = pad_idx
  
  def __call__(self, batch):
    return pad_sequence(batch, batch_first=True, padding_value=self.pad_idx)


class TextTrainer(object):
  def __init__(self, device=None, encoding_fn=Data.letters_encoding, metadata_file='_Data_metadata_letters.pk', smoothing_eps=0.1,
               train_folder='../../../datasets/openslr/LibriSpeech/train-clean-100/', batch_size=32, lr=1e-3, n_epochs=500,
               test_folder='../../../datasets/openslr/LibriSpeech/test-clean/', logfile='_logs/_textTrainer_logs.txt',
               scorer=Data.compute_accuracy, save_name_model='convnet/text_trainer.pt'):
    logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.encoding_fn = encoding_fn
    self.metadata_file = metadata_file
    self.train_folder = train_folder
    self.test_folder = test_folder
    self.batch_size = batch_size
    self.save_name_model = save_name_model
    self.smoothing_eps = smoothing_eps
    self.n_epochs = n_epochs
    self.scorer = scorer

    self.set_data()
    self.sos_idx = self.data.tokens_to_idx['<sos>']
    self.eos_idx = self.data.tokens_to_idx['<eos>']
    self.pad_idx = self.data.tokens_to_idx['<pad>']
    self.mask_idx = self.data.tokens_to_idx['<mask>']

    self.set_data_loader()

    self.model = self.instanciate_model()
    logging.info(f'The model has {u.count_trainable_parameters(self.model):,} trainable parameters')

    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.criterion = u.CrossEntropyLoss(self.pad_idx)
  
  def set_data(self):
    self.data = Data()

    if not os.path.isfile(self.metadata_file):
      self.data.process_all_transcripts(self.train_folder, self.test_folder, encoding_fn=self.encoding_fn)
      self.data.idx_to_tokens.append('<mask>')
      self.data.tokens_to_idx['<mask>'] = self.data.idx_to_tokens.index('<mask>')
      self.data.save_metadata(save_name=self.metadata_file)
    else:
      self.data.load_metadata(save_name=self.metadata_file)
  
  def set_data_loader(self):
    train_dataset = TextDataset(self.data.ids_to_encodedsources_train)
    test_dataset = TextDataset(self.data.ids_to_encodedsources_test)
    collator = TextCollator(self.pad_idx)
    self.train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collator,
                                        pin_memory=True, drop_last=True)
    self.test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collator,
                                       pin_memory=True, drop_last=True)

  def instanciate_model(self, emb_dim=100, d_model=512, n_heads=8, d_ff=1024, kernel_size=3, n_blocks=6):
    self.output_dim = len(self.data.idx_to_tokens)
    self.n_step_aheads = 1
    return TextModel(self.output_dim, emb_dim, d_model, n_heads, d_ff, kernel_size, n_blocks, self.data.max_source_len,
                     dropout=0., only_see_past=False, n_step_aheads=self.n_step_aheads).to(self.device)
  
  def train(self):
    print('Start Training...')
    eval_accuracy_memory = 0
    for epoch in tqdm(range(self.n_epochs)):
      epoch_loss, accs = self.train_pass()
      logging.info(f"Epoch {epoch} | train_loss = {epoch_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")
      eval_loss, accs = self.evaluation()
      logging.info(f"Epoch {epoch} | test_loss = {eval_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")

      oea = accs.get('word_accuracy', None)

      if oea is not None and oea > eval_accuracy_memory:
        logging.info(f'Save model with eval_accuracy = {oea:.3f}')
        u.save_checkpoint(self.model, None, self.save_name_model)
        eval_accuracy_memory = oea
  
  @torch.no_grad()
  def evaluation(self):
    losses = 0
    targets, predictions = [], []

    self.model.eval()

    for enc_text in tqdm(self.test_data_loader):
      batch_size, seq_len = enc_text.shape
      mask = np.random.choice([True, False], p=[0.15, 0.85], size=(seq_len-2)*batch_size).reshape((batch_size, -1))
      mask[(tuple(range(batch_size)), np.random.randint(0, seq_len-2, batch_size))] = True
      enc_text[:, 1:-1][mask] = self.mask_idx
      enc_text = enc_text.to(self.device)

      target = enc_text[:, 2:].clone()
      if self.n_step_aheads > 1:
        target = torch.cat([target[:, i:i+self.n_step_aheads] for i in range(target.shape[-1])], dim=-1)
        target = F.pad(target, (0, self.n_step_aheads * (enc_text.shape[-1] - 2) - target.shape[-1]))
      target[mask==False] = self.pad_idx
      

      preds = self.model(enc_text[:, 1:-1])
      preds_sep = preds.reshape(preds.shape[0], preds.shape[1] * self.n_step_aheads, self.output_dim)

      losses += self.criterion(preds.reshape(-1, preds_sep.shape[-1]), target.reshape(-1), epsilon=self.smoothing_eps).item()

      targets += [et[m].tolist() for et, m in zip(enc_text[:, 2:], mask)]
      predictions += [p[m].tolist() for p, m in zip(preds[:, :, :self.output_dim].argmax(dim=-1), mask)]
    
    self.model.train()

    accs = self.scorer(**{'targets': targets, 'predictions': predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                          'idx_to_tokens': self.data.idx_to_tokens})

    return losses / len(self.test_data_loader), accs
  
  def train_pass(self, repeat_batch=1):
    # Task1 = predict masked words
    losses = 0
    targets, predictions = [], []

    for enc_text in tqdm(self.train_data_loader):
      batch_size, seq_len = enc_text.shape
      mask = np.random.choice([True, False], p=[0.15, 0.85], size=(seq_len-2)*batch_size).reshape((batch_size, -1))
      mask[(tuple(range(batch_size)), np.random.randint(0, seq_len-2, batch_size))] = True
      enc_text[:, 1:-1][mask] = self.mask_idx
      enc_text = enc_text.to(self.device)

      target = enc_text[:, 2:].clone()
      if self.n_step_aheads > 1:
        target = torch.cat([target[:, i:i+self.n_step_aheads] for i in range(target.shape[-1])], dim=-1)
        target = F.pad(target, (0, self.n_step_aheads * (enc_text.shape[-1] - 2) - target.shape[-1]))
      target[mask==False] = self.pad_idx

      for _ in range(repeat_batch):
        preds = self.model(enc_text[:, 1:-1])
        preds_sep = preds.reshape(preds.shape[0], preds.shape[1] * self.n_step_aheads, self.output_dim)

        self.optimizer.zero_grad()
        current_loss = self.criterion(preds_sep.reshape(-1, preds_sep.shape[-1]), target.reshape(-1), epsilon=self.smoothing_eps)
        current_loss.backward()
        self.optimizer.step()

      targets += [et[m].tolist() for et, m in zip(enc_text[:, 2:], mask)]
      predictions += [p[m].tolist() for p, m in zip(preds[:, :, :self.output_dim].argmax(dim=-1), mask)]

      losses += current_loss.item()

    accs = self.scorer(**{'targets': targets, 'predictions': predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                          'idx_to_tokens': self.data.idx_to_tokens})
    
    return losses / len(self.train_data_loader), accs


if __name__ == "__main__":
  ## SEEDING FOR REPRODUCIBILITY
  SEED = 42
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)

  tt = TextTrainer()
  tt.train()