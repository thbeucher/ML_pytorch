import os
import re
import sys
import torch
import random
import logging
import numpy as np
import torch.optim as optim

from tqdm import tqdm

sys.path.append(os.path.abspath(__file__).replace('ASR/naiveconv_experiments.py', ''))

import utils as u
import models.naive_conv as nc

from data import Data
from convnet_trainer import ConvnetTrainer


class Experiment11(object):
  '''Conv-ConvTranspose to creates compressed representation by reconstruction task'''
  def __init__(self, device=None, logfile='_logs/_logs_experiment11.txt', save_name_model='convnet/naive_convnet_experiment11.pt',
               batch_size=32, lr=1e-4, metadata_file='_Data_metadata_letters_raw0025.pk', n_epochs=500,
               train_folder='../../../datasets/openslr/LibriSpeech/train-clean-100/',
               test_folder='../../../datasets/openslr/LibriSpeech/test-clean/'):
    logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.save_name_model = save_name_model
    self.batch_size = batch_size
    self.lr = lr
    self.metadata_file = metadata_file
    self.n_epochs = n_epochs
    self.train_folder = train_folder
    self.test_folder = test_folder

    self.set_data()

    self.model = nc.NaiveConvED(n_feats=10).to(self.device)
    logging.info(f'The model has {u.count_trainable_parameters(self.model):,} trainable parameters')

    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.criterion = torch.nn.MSELoss(reduction='sum')

    self.train_data_loader = self.data.get_dataset_generator(batch_size=batch_size)
    self.test_data_loader = self.data.get_dataset_generator(train=False, batch_size=batch_size)
    
  def set_data(self):
    self.data = Data()

    if not os.path.isfile(self.metadata_file):
      self.data.set_audio_metadata(self.train_folder, self.test_folder)
      self.data.process_all_transcripts(self.train_folder, self.test_folder)
      self.data.save_metadata(save_name=self.metadata_file)
    else:
      self.data.load_metadata(save_name=self.metadata_file)
  
  def train(self):
    print('Start Training...')
    eval_loss_memory = None
    for epoch in tqdm(range(self.n_epochs)):
      epoch_loss = self.train_pass()
      logging.info(f"Epoch {epoch} | train_loss = {epoch_loss:.3f}")
      eval_loss = self.evaluation()
      logging.info(f"Epoch = {epoch} | test_loss = {eval_loss:.3f}")

      if eval_loss_memory is None or eval_loss < eval_loss_memory:
        u.save_checkpoint(self.model, None, self.save_name_model)
        eval_loss_memory = eval_loss
  
  @torch.no_grad()
  def evaluation(self):
    losses = 0

    self.model.eval()

    for enc_in, _ in tqdm(self.test_data_loader):
      enc_in = enc_in.to(self.device)

      if enc_in.shape[1] % 2 == 1:
        enc_in = enc_in[:, :-1]

      preds = self.model(enc_in)

      losses += self.criterion(preds, enc_in).item()
    
    self.model.train()

    return losses / len(self.test_data_loader)
  
  def train_pass(self):
    losses = 0

    for enc_in, _ in tqdm(self.train_data_loader):
      enc_in = enc_in.to(self.device)

      if enc_in.shape[1] % 2 == 1:
        enc_in = enc_in[:, :-1]

      preds = self.model(enc_in)

      self.optimizer.zero_grad()

      current_loss = self.criterion(preds, enc_in)

      current_loss.backward()

      self.optimizer.step()

      losses += current_loss.item()
    
    return losses / len(self.train_data_loader)


class EncoderEmbedderNaiveConv(torch.nn.Module):
  def __init__(self, enc_input_dim, emb_dim, hid_dim, enc_max_seq_len, enc_dropout, device, reduce_dim=False,
               n_feats=10, in_size=400, out_size=80):
    super().__init__()
    self.naive_encoder = nc.NaiveConvEncoder(n_feats=n_feats, in_size=in_size, out_size=out_size)
    self.projection = torch.nn.Linear(out_size, emb_dim)
    self.relu = torch.nn.ReLU(inplace=True)
  
  def forward(self, x):  # x = [batch_size, seq_len, in_size]
    out, _ = self.naive_encoder(x.unsqueeze(1))  # out = [batch_size, 1, new_seq_len, out_size]
    out = self.relu(self.projection(out.squeeze(1)))  ## out = [batch_size, new_seq_len, emb_dim]
    return out


# STATUS = STOPPED, epoch 90, flat testing accuracy curve
class Experiment15(ConvnetTrainer):  # epoch 160 of experiment11 NaiveConvEncoder
  '''Convnet letters prediction, adam, Attention-CrossEntropy loss, window-raw-sliced, EncoderEmbedder = NaiveConvEncoder'''
  def __init__(self, logfile='_logs/_logs_experiment15.txt', save_name_model='convnet/convnet_experiment15.pt', batch_size=8,
               metadata_file='_Data_metadata_letters_raw0025.pk'):
    convnet_config = {'encoder_embedder': EncoderEmbedderNaiveConv, 'enc_layers': 2}
    super().__init__(logfile=logfile, save_name_model=save_name_model, batch_size=batch_size, metadata_file=metadata_file,
                    convnet_config=convnet_config)
    nc.load_model_from_NaiveConvED(self.model.encoder.embedder.naive_encoder, 'convnet/naive_convnet_experiment11.pt')


if __name__ == "__main__":
  ## SEEDING FOR REPRODUCIBILITY
  SEED = 42
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)

  experiments = {k.replace('Experiment', ''): v for k, v in locals().items() if re.search(r'Experiment\d+', k) is not None}
  
  rep = input(f'Which Experiment do you want to start? ({",".join(experiments.keys())}): ')
  exp = experiments[rep]()
  exp.train()
