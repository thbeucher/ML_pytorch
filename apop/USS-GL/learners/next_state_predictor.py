import os
import sys
import torch
import visdom

from tqdm import tqdm
from torchvision.io import read_image
from torchvision.utils import make_grid, save_image

sys.path.append(os.path.abspath(__file__).replace('next_state_predictor.py', ''))

from visual_feature_learner import VFLTrainer, VFL
from utils import load_model, save_model, int_to_bin, plot_metric


class NSP(torch.nn.Module):  # NSP=Next State Predictor
  BASE_CONFIG = {'embed': 'embedding',  # embed = 'embedding' or 'binary'
                 'n_actions': 5}
  def __init__(self, config={}):
    super().__init__()
    self.config = {**NSP.BASE_CONFIG, **config}

    self.nsp_in_proj = torch.nn.Linear(32*10*10 + 5 + 8 + 8, 512)
    self.nsp = torch.nn.GRUCell(input_size=512, hidden_size=1024, bias=True)
    self.nsp_out_proj = torch.nn.Linear(1024, 32*10*10)

    if self.config['embed'] == 'embedding':
      self.action_embedder = torch.nn.Embedding(num_embeddings=5, embedding_dim=5)
      self.shoulder_embedder = torch.nn.Embedding(num_embeddings=91, embedding_dim=8)
      self.elbow_embedder = torch.nn.Embedding(num_embeddings=181, embedding_dim=8)
  
  def _embed_body_infos(self, actions, shoulders, elbows, device):
    if self.config['embed'] == 'binary':
      actions_emb = torch.nn.functional.one_hot(torch.LongTensor(actions), num_classes=self.config['n_actions']).float().to(device)
      shoulders_emb = torch.FloatTensor([int_to_bin(s) for s in shoulders]).to(device)
      elbows_emb = torch.FloatTensor([int_to_bin(e) for e in elbows]).to(device)
    else:
      actions_emb = self.action_embedder(torch.LongTensor(actions))  # [B] -> [B, 5]
      shoulders_emb = self.shoulder_embedder(torch.LongTensor(shoulders))  # [B] -> [B, 8]
      elbows_emb = self.elbow_embedder(torch.LongTensor(elbows))  # [B] -> [B, 8]
    return actions_emb, shoulders_emb, elbows_emb

  def forward(self, quantized, actions, shoulders, elbows, hidden=None):
    '''
    Parameters:
      * quantized : torch.FloatTensor, shape = [bs, n_feats, h, w]
      * actions : list of int
      * shoulders : list of int
      * elbows : list of int
    '''
    actions_emb, shoulders_emb, elbows_emb = self._embed_body_infos(actions, shoulders, elbows, quantized.device)
    enriched_quantized = self.nsp_in_proj(torch.cat([quantized.view(len(actions), -1),
                                                     actions_emb, shoulders_emb, elbows_emb], 1))
    next_hidden = self.nsp(enriched_quantized, hidden)  # [B, 512] -> [B, 1024]
    next_quantized = self.nsp_out_proj(next_hidden)  # [B, 1024] -> [B, 32*10*10]
    return next_quantized.view(quantized.shape), next_hidden  # [B, 32*10*10] -> [B, 32, 10, 10]


class NSPTrainer(object):
  BASE_CONFIG = {'batch_size': 30, 'use_visdom': True, 'memory_size': 7200, 'max_ep_len': 60, 'nsp_config': {},
                 'save_name': 'next_state_predictor.pt', 'models_folder': 'models/',
                 'save_vfl_name': 'visual_feature_learner.pt'}
  def __init__(self, config={}):
    self.config = {**NSPTrainer.BASE_CONFIG, **config}

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.next_state_predictor = NSP(self.config['nsp_config'])
    self.nsp_optimizer = torch.optim.AdamW(self.next_state_predictor.parameters())
    self.mse_criterion = torch.nn.MSELoss()

    if self.config['use_visdom']:
      self.vis = visdom.Visdom()
    
    self.vfl_learner = VFL()
    load_model(self.vfl_learner, models_folder=self.config['models_folder'], save_name=self.config['save_vfl_name'],
               device=self.device)

  @torch.no_grad()
  def fill_nsp_memory(self):
    self.memory = VFLTrainer.fill_memory(memory_size=self.config['memory_size'], max_ep_len=self.config['max_ep_len'])

    for i in tqdm(range(0, len(self.memory), self.config['batch_size'])):
      states, next_states, actions, shoulders, elbows, rewards, dones = zip(*self.memory[i:i+self.config['batch_size']])

      states = torch.stack(states, 0).to(self.device)
      next_states = torch.stack(next_states).to(self.device)

      _, _, quantized = self.vfl_learner(states)
      _, _, next_quantized = self.vfl_learner(next_states)

      for j, (s, ns, q, nq, a, sh, e, r, d) in enumerate(zip(states.cpu(), next_states.cpu(), quantized.cpu(),
                                                       next_quantized.cpu(), actions, shoulders, elbows,
                                                       rewards, dones)):
        self.memory[i+j] = [s, ns, q, nq, a, sh, e, r, d]
  
  def train(self, n_iterations=3000, plot_progress=True, save_model=True, fill_memory=True, **kwargs):
    if fill_memory:
      print('Fill nsp_memory...')
      self.fill_nsp_memory()

    print(f'Train next_state_predictor for {n_iterations} iterations...')
    group_batch_len = self.config['batch_size'] * self.config['max_ep_len']
    with tqdm(total=n_iterations) as t:
      for i in range(n_iterations):
        for b_start in range(0, len(self.memory), group_batch_len):
          hidden = None
          for ep_t in range(self.config['max_ep_len']):
            # as we provide previous hidden step because we use GruCell and
            # as the memory is organized as [ep1_t1, ep1_t2...ep1_tMAX, ep2_t1, ...]
            # we need to create our batch using element every max_ep_len
            _, next_states, quantized, next_quantized, actions, shoulders, elbows, _, _\
              = zip(*self.memory[b_start:b_start+group_batch_len][ep_t::self.config['max_ep_len']])

            next_states = torch.stack(next_states, 0)
            quantized = torch.stack(quantized, 0).to(self.device)
            next_quantized = torch.stack(next_quantized, 0).to(self.device)

            pred_next_quantized, hidden = self.next_state_predictor(quantized, actions, shoulders, elbows, hidden=hidden)
            hidden = hidden.detach()

            self.nsp_optimizer.zero_grad()
            loss = self.mse_criterion(pred_next_quantized, next_quantized)
            loss.backward()
            self.nsp_optimizer.step()

        if plot_progress:
          with torch.no_grad():
            next_state_rec = self.vfl_learner.decoder(next_quantized)
            pred_next_state_rec = self.vfl_learner.decoder(pred_next_quantized)

          self.vis.image(make_grid(next_states[:3], nrow=3), win='next_state', opts={'title': 'next_state'})

          save_image(make_grid(next_state_rec[:3].cpu(), nrow=3), 'test_make_grid1.png')
          self.vis.image(read_image('test_make_grid1.png'), win='next_state_rec', opts={'title': 'next_state_rec'})

          save_image(make_grid(pred_next_state_rec[:3].cpu(), nrow=3), 'test_make_grid2.png')
          self.vis.image(read_image('test_make_grid2.png'), win='pred_next_state_rec', opts={'title': 'pred_next_state_rec'})

          plot_metric(self.vis, loss.item(), i, win='nsp_loss', title='NSP loss evolution')

        t.set_description(f'Loss={loss.item():.3f}')
        t.update(1)
    
    if save_model:
      self.save_model()
  
  def save_model(self, model_name=None):
    save_model(self.next_state_predictor, models_folder=self.config['models_folder'],
               save_name=self.config['save_name'] if model_name is None else model_name)
  
  def load_model(self, model_name=None):
    load_model(self.next_state_predictor, models_folder=self.config['models_folder'],
               save_name=self.config['save_name'] if model_name is None else model_name)


if __name__ == '__main__':
  nsp_trainer = NSPTrainer({'save_name': 'test_nsp.pt', 'memory_size': 120})
  nsp_trainer.train(n_iterations=1)
  nsp_trainer.load_model()
  os.remove('models/test_nsp.pt')