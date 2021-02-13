import os
import sys
import torch
import logging
import numpy as np
import pickle as pk
import torch.optim as optim

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(__file__).replace('ASR/text_image_experiments.py', ''))
import utils as u
import models.conv_seqseq as css

from data import Data


def get_letters_size(font_size=10, font_file='arial.ttf'):
  font = ImageFont.truetype(font_file, font_size)
  letter_to_size = {}
  for letter in "abcdefghijklmnopqrstuvwxyz'.":
    img = Image.new('1', (10, 12), color=1)
    d = ImageDraw.Draw(img)
    d.text((0, 0), letter, fill='black', font=font)
    arr = np.array(img)
    letter_to_size[letter] = (arr.T.sum(1) != 12).sum()
  return letter_to_size


def text_to_image(text, font_size=10, font_file='arial.ttf', save=False, save_name='text.png'):
  letter_to_size = get_letters_size()
  font = ImageFont.truetype(font_file, font_size)

  text += ' ' * 6  # needed to avoid that the last letters was cutted
  # width = len(text) * (font_size // 2)
  width = sum([2 if l == ' ' else letter_to_size[l]+2 for l in text])
  height = font_size + 2
  png_size = (width, height)
  start_text_pos = (0, 0)

  img = Image.new('1', png_size, color=1)
  d = ImageDraw.Draw(img)
  d.text(start_text_pos, text, fill='black', font=font)

  if save:
    img.save(save_name)

  arr = np.array(img)
  idx = [i for i, ar in enumerate(arr.T) if sum(ar) != font_size+2][-1]
  return arr[:, :idx+2]  # img = Image.fromarray(arr)


def analyze():
  with open('divers/_Data_metadata_letters_raw0025.pk', 'rb') as f:
    data = pk.load(f)
  
  sources = [v.lower() for v in data['ids_to_transcript_train'].values()]
  sources += [v.lower() for v in data['ids_to_transcript_test'].values()]

  seq_len = []
  for s in tqdm(sources):
    seq_len.append(len(s) * 5)
  
  print(f'min = {min(seq_len)} | max = {max(seq_len)} | mean = {np.mean(seq_len):.2f} | std = {np.std(seq_len):.2f}')


def test():
  with open('divers/_Data_metadata_letters_raw0025.pk', 'rb') as f:
    data = pk.load(f)
    
  data['ids_to_transcript_train'] = {k: v + '.' for k, v in data['ids_to_transcript_train'].items()}
  data['ids_to_transcript_test'] = {k: v + '.' for k, v in data['ids_to_transcript_test'].items()}
  
  train_data_loader = ReadExperiment.get_data_loader(data['ids_to_transcript_train'], data['ids_to_encodedsources_train'])

  a = torch.nn.Conv1d(12, 12, 6, stride=3)
  b = torch.nn.Conv1d(12, 12, 4, stride=2)

  model = torch.nn.Sequential(torch.nn.Conv1d(12, 80, 6, stride=3), torch.nn.ReLU(inplace=True),
                              torch.nn.Conv1d(80, 80, 4, stride=2), torch.nn.ReLU(inplace=True),
                              torch.nn.Conv1d(80, 80, 3, stride=1), torch.nn.ReLU(inplace=True),
                              torch.nn.Conv1d(80, 80, 3, stride=1), torch.nn.ReLU(inplace=True),
                              torch.nn.Conv1d(80, 80, 3, stride=1), torch.nn.ReLU(inplace=True),
                              torch.nn.Conv1d(80, 256, 3, stride=1), torch.nn.ReLU(inplace=True))

  for enc_in, dec_in in train_data_loader:
    to_conv = enc_in.permute(0, 2, 1)
    aa = a(to_conv)
    ab = b(aa)
    am = model(to_conv)
    input(f'enc_in = {to_conv.shape} | aa = {aa.shape} | ab = {ab.shape} | am = {am.shape}')


class CustomDataset(Dataset):
  def __init__(self, ids_to_transcript, ids_to_encodedsources):
    self.ids_to_transcript = ids_to_transcript
    self.ids_to_encodedsources = ids_to_encodedsources

    self.identities = list(sorted(ids_to_transcript.keys()))

  def __len__(self):
    return len(self.identities)
  
  def __getitem__(self, idx):
    identity = self.identities[idx]

    signal = text_to_image(self.ids_to_transcript[identity].lower()).T

    encoder_input = torch.Tensor(signal)
    decoder_input = torch.LongTensor(self.ids_to_encodedsources[identity])
    return encoder_input, decoder_input


class CustomCollator(object):
  def __init__(self, enc_pad_val, dec_pad_val):
    self.enc_pad_val = enc_pad_val
    self.dec_pad_val = dec_pad_val

  def __call__(self, batch):
    encoder_inputs, decoder_inputs = zip(*batch)
    encoder_input_batch = pad_sequence(encoder_inputs, batch_first=True, padding_value=self.enc_pad_val).float()
    decoder_input_batch = pad_sequence(decoder_inputs, batch_first=True, padding_value=self.dec_pad_val)
    return encoder_input_batch, decoder_input_batch


class ConvModel(torch.nn.Module):
  def __init__(self, device=None, output_dim=31, emb_dim=256, hid_dim=512, n_layers=8, kernel_size=3, dropout=0.25, pad_idx=2,
               max_seq_len=600, scaling_energy=True, multi_head=True, d_keys_values=64):
    super().__init__()
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.decoder = css.Decoder(output_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, pad_idx, max_seq_len=max_seq_len,
                               multi_head=multi_head, d_keys_values=d_keys_values, scaling_energy=scaling_energy)
    self.encoder = torch.nn.Sequential(torch.nn.Conv1d(12, 80, 6, stride=3), torch.nn.ReLU(inplace=True),
                                       torch.nn.Conv1d(80, 80, 4, stride=2), torch.nn.ReLU(inplace=True),
                                       torch.nn.Conv1d(80, 80, 3, stride=1), torch.nn.ReLU(inplace=True),
                                       torch.nn.Conv1d(80, 80, 3, stride=1), torch.nn.ReLU(inplace=True),
                                       torch.nn.Conv1d(80, 80, 3, stride=1), torch.nn.ReLU(inplace=True),
                                       torch.nn.Conv1d(80, emb_dim, 3, stride=1), torch.nn.ReLU(inplace=True))
  
  def forward(self, enc_in, dec_in):
    enc_in = self.encoder(enc_in.permute(0, 2, 1)).permute(0, 2, 1)
    return self.decoder(dec_in, enc_in, enc_in)
  
  def greedy_decoding(self, enc_in, sos_idx, eos_idx, max_seq_len=100):
    enc_in = self.encoder(enc_in.permute(0, 2, 1)).permute(0, 2, 1)
    encoder_conved, encoder_combined = enc_in, enc_in

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


class ReadExperiment(object):
  def __init__(self, device=None, n_epochs=500, logfile='_text_image_experiments_logs.txt', eval_step=5, lr=1e-5,
               save_name_model='readexperiment.pt', scorer=Data.compute_scores, batch_size=32):
    logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.n_epochs = n_epochs
    self.eval_step = eval_step
    self.save_name_model = save_name_model
    self.scorer = scorer
    self.batch_size = batch_size

    self.prepare_data()
    self.sos_idx = self.data['tokens_to_idx']['<sos>']
    self.eos_idx = self.data['tokens_to_idx']['<eos>']
    self.pad_idx = self.data['tokens_to_idx']['<pad>']
    self.idx_to_tokens = self.data['idx_to_tokens']

    self.model = self.instanciate_model(output_dim=len(self.idx_to_tokens))
    logging.info(f'The model has {u.count_trainable_parameters(self.model):,} trainable parameters')

    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.criterion = u.CrossEntropyLoss(self.pad_idx)

    u.load_model(self.model, save_name_model, restore_only_similars=True)
  
  @staticmethod
  def get_data_loader(ids_to_transcript, ids_to_sources, pad_idx=2, batch_size=32):
    custom_dataset = CustomDataset(ids_to_transcript, ids_to_sources)
    custom_collator = CustomCollator(1, pad_idx)
    return DataLoader(custom_dataset, batch_size=batch_size, num_workers=4, collate_fn=custom_collator, shuffle=True, pin_memory=True)
  
  def prepare_data(self):
    with open('_Data_metadata_letters_raw0025.pk', 'rb') as f:
      self.data = pk.load(f)
    
    # add . at end of each transcript to correspond to <eos> token
    self.data['ids_to_transcript_train'] = {k: v + '.' for k, v in self.data['ids_to_transcript_train'].items()}
    self.data['ids_to_transcript_test'] = {k: v + '.' for k, v in self.data['ids_to_transcript_test'].items()}
    
    self.train_data_loader = ReadExperiment.get_data_loader(self.data['ids_to_transcript_train'], self.data['ids_to_encodedsources_train'])
    self.test_data_loader = ReadExperiment.get_data_loader(self.data['ids_to_transcript_test'], self.data['ids_to_encodedsources_test'])
  
  def instanciate_model(self, output_dim=31, emb_dim=256, hid_dim=512, n_layers=8, kernel_size=3, dropout=0.25, pad_idx=2,
                        max_seq_len=600, scaling_energy=True, multi_head=True, d_keys_values=64):
    return ConvModel(output_dim=output_dim, emb_dim=emb_dim, hid_dim=hid_dim, n_layers=n_layers, kernel_size=kernel_size,
                     dropout=dropout, pad_idx=pad_idx, max_seq_len=max_seq_len, scaling_energy=scaling_energy, multi_head=multi_head,
                     d_keys_values=d_keys_values).to(self.device)
  
  def train(self):
    print('Start Training...')
    eval_accuracy_memory = 0
    for epoch in tqdm(range(self.n_epochs)):
      epoch_loss, accs = self.train_pass(only_loss=False if epoch % self.eval_step == 0 else True)
      logging.info(f"Epoch {epoch} | train_loss = {epoch_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")
      eval_loss, accs = self.evaluation(only_loss=False if epoch % self.eval_step == 0 else True)
      logging.info(f"Epoch {epoch} | test_loss = {eval_loss:.3f} | {' | '.join([f'{k} = {v:.3f}' for k, v in accs.items()])}")

      oea = accs['preds_acc'] if 'preds_acc' in accs else accs.get('word_accuracy', None)

      if oea is not None and oea > eval_accuracy_memory:
        logging.info(f'Save model with eval_accuracy = {oea:.3f}')
        u.save_checkpoint(self.model, None, self.save_name_model)
        eval_accuracy_memory = oea
  
  def train_pass(self, only_loss=True):
    losses, accs = 0, {}
    targets, predictions = [], []

    for enc_in, dec_in in tqdm(self.train_data_loader):
      enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
      preds, att = self.model(enc_in, dec_in[:, :-1])

      self.optimizer.zero_grad()

      current_loss = self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1))

      current_loss.backward()

      self.optimizer.step()

      targets += dec_in[:, 1:].tolist()
      predictions += preds.argmax(dim=-1).tolist()

      losses += current_loss.item()
    
    if not only_loss:
      accs = self.scorer(**{'targets': targets, 'predictions': predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.idx_to_tokens})
    
    return losses / len(self.train_data_loader), accs
  
  @torch.no_grad()
  def evaluation(self, only_loss=True):
    losses, accs = 0, {}
    targets, predictions = [], []

    self.model.eval()

    for enc_in, dec_in in tqdm(self.test_data_loader):
      enc_in, dec_in = enc_in.to(self.device), dec_in.to(self.device)
      preds, att = self.model(enc_in, dec_in[:, :-1])

      losses += self.criterion(preds.reshape(-1, preds.shape[-1]), dec_in[:, 1:].reshape(-1)).item()
      
      if not only_loss:
        preds, _ = self.model.greedy_decoding(enc_in, self.sos_idx, self.eos_idx, max_seq_len=dec_in.shape[1])
        targets += dec_in[:, 1:].tolist()
        predictions += preds.tolist()
    
    self.model.train()

    if not only_loss:
      accs = self.scorer(**{'targets': targets, 'predictions': predictions, 'eos_idx': self.eos_idx, 'pad_idx': self.pad_idx,
                            'idx_to_tokens': self.idx_to_tokens})

    return losses / len(self.test_data_loader), accs


if __name__ == "__main__":
  re = ReadExperiment()
  re.train()

  # test()