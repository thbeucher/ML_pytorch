import os
import ast
import sys
import torch
import random
import logging
import argparse

from pg_exps import train_ppo, train_reinforce

sys.path.append(os.path.abspath(__file__).replace('USS-GL/pgmem_exps.py', ''))
import utils as u


class Summarizer(torch.nn.Module):
  def __init__(self, vector_dim, hidden_size, n_output_tokens, method='MLP'):
    '''
    Summarize a sequence of p tokens of dim d to a sequence of k tokens
    '''
    super().__init__()
    self.method = method
    self.vector_dim = vector_dim
    if self.method == 'MLP':
      self.mlp = torch.nn.Sequential(
        torch.nn.Linear(vector_dim, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, n_output_tokens)
      )
    else:
      self.query_vector = torch.nn.Parameter(torch.randn(1, n_output_tokens, vector_dim))  # [1, k, d]
          
  def forward(self, x):  # [batch_size, n_input_tokens, vector_dim] = [b, p, d]
    if self.method == 'MLP':
      weight = torch.softmax(self.mlp(x), dim=-2)  # [b, p, d] -> [b, p, k]
      weight = weight.transpose(-1, -2)  # [b, p, k] -> [b, k, p]
    else:
      weight = torch.matmul(self.query_vector, x.transpose(-1, -2))  # [1, k, d] * [b, d, p] = [b, k, p]
      weight = torch.softmax(weight / (self.vector_dim ** 0.5), dim=-1)
    z = torch.matmul(weight, x)  # [b, k, p] * [b, p, d] = [b, k, d]
    return z


class ToyACmem(torch.nn.Module):
  BASE_CONFIG = {'concat': True, 'memory_size': 16, 'expand_memory_emb': True, 'expanded_memory_emb_size': 12,
                 'summarizer_hidden_size': 200, 'state_size': 4, 'shared_AC_hidden_size': 200,
                 'processor_hidden_size': 100}
  def __init__(self, configuration={}):
    super().__init__()
    self.config = {**ToyACmem.BASE_CONFIG, **configuration}
    memory_emb_size = self.config['expanded_memory_emb_size'] if self.config['expand_memory_emb'] else self.config['state_size']
    fstate_size = 2*memory_emb_size if self.config['concat'] else memory_emb_size

    self.shared = torch.nn.Linear(fstate_size, self.config['shared_AC_hidden_size'])
    self.actor = torch.nn.Linear(self.config['shared_AC_hidden_size'], 5)
    self.critic = torch.nn.Linear(self.config['shared_AC_hidden_size'], 1)

    self.register_buffer('memory', torch.FloatTensor(1, self.config['memory_size'], memory_emb_size).uniform_(0, 1))
    self.positional_encoder = u.PositionalEncoding(memory_emb_size)

    self.reader = Summarizer(memory_emb_size, self.config['summarizer_hidden_size'], 1)
    self.writer = Summarizer(memory_emb_size, self.config['summarizer_hidden_size'], self.config['memory_size'])

    if self.config['expand_memory_emb']:
      self.embedder = torch.nn.Linear(4, memory_emb_size)
      self.processor = torch.nn.Sequential(torch.nn.Linear(5+1+2*memory_emb_size, self.config['processor_hidden_size']),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(self.config['processor_hidden_size'], memory_emb_size))
  
  def forward(self, state, critic=False, keep_action_logits=False):  # [state_size] or [ep_len, state_size]
    if self.config['expand_memory_emb']:
      state = self.embedder(state)
    
    if len(state.size()) == 1:
      state = state.view(1, 1, -1)  # [emb_dim=d] -> [1=bs, 1=k, emb_dim=d]
    elif len(state.size()) == 2:
      state = state.unsqueeze(1)  # [bs, emb_dim] -> [bs, 1, emb_dim]
    
    bs, *_ = state.shape

    action_probs = torch.empty(bs, 1, 5)
    state_values = torch.empty(bs, 1, 1)
    new_memory = self.memory.clone()
    for i in range(bs):
      cstate = state[i:i+1]

      # [bs, 1, d] || [1, mem_size, d] = [bs, 1+mem_size, d] -> [bs, 1, d]
      mem_state = self.reader(torch.cat([cstate, self.positional_encoder(new_memory)], dim=-2))

      # [bs, 1, d] || [bs, 1, d] = [bs, 1, 2*d] if concat
      cstate = torch.cat([cstate, mem_state], dim=-1) if self.config['concat'] else mem_state

      out = torch.nn.functional.relu(self.shared(cstate))  # [bs, 1, d or 2d] -> [bs, 1, 200]

      action_logits = self.actor(out)  # [bs, 1, 200] -> [bs, 1, 5]
      caction_probs = action_logits if keep_action_logits else torch.softmax(action_logits, dim=-1)

      cstate_values = self.critic(out)  # [bs, 1, 200] -> [bs, 1, 1]

      if self.config['expand_memory_emb']:
        # [bs, 1, d] || [bs, 1, d] || [bs, 1, 5] || [bs, 1, 1] = [bs, 1, d+d+5+1] -> [bs, 1, d]
        mem_state = self.processor(torch.cat([state[i:i+1], mem_state, caction_probs, cstate_values], dim=-1))

      # [bs, 1, d] || [bs, 1, d] || [1, mem_size, d] = [bs, 1+1+mem_size, d] -> [bs, mem_size, d]
      new_memory = self.writer(torch.cat([state[i:i+1].detach(),
                                          mem_state.detach(),
                                          self.positional_encoder(new_memory.detach())], dim=-2))
      new_memory -= new_memory.min(-1, keepdim=True)[0]
      new_memory /= new_memory.max(-1, keepdim=True)[0]

      action_probs[i] = caction_probs[0]
      state_values[i] = cstate_values[0]
    
    self.memory = new_memory.detach().clone()

    return action_probs.squeeze(), state_values.squeeze(-1) if critic else None


if __name__ == '__main__':
  argparser = argparse.ArgumentParser(prog='pgmem_exps.py', description='')
  argparser.add_argument('--log_file', default='_tmp_pgmem_exps_logs.txt', type=str)
  argparser.add_argument('--use_visdom', default=True, type=ast.literal_eval)
  argparser.add_argument('--game_view', default=False, type=ast.literal_eval)
  argparser.add_argument('--save_model', default=True, type=ast.literal_eval)
  argparser.add_argument('--load_model', default=False, type=ast.literal_eval)
  argparser.add_argument('--save_name', default='models/PGmemModel.pt', type=str)
  argparser.add_argument('--seed', default=42, type=int)
  argparser.add_argument('--pretraining', default=False, type=ast.literal_eval)
  argparser.add_argument('--concat', default=True, type=ast.literal_eval)
  argparser.add_argument('--expand_memory_emb', default=True, type=ast.literal_eval)
  argparser.add_argument('--memory_size', default=16, type=int)
  argparser.add_argument('--expanded_memory_emb_size', default=12, type=int)
  argparser.add_argument('--summarizer_hidden_size', default=100, type=int)
  argparser.add_argument('--state_size', default=4, type=int)
  argparser.add_argument('--shared_AC_hidden_size', default=200, type=int)
  argparser.add_argument('--processor_hidden_size', default=100, type=int)
  argparser.add_argument('--algo', default='ppo', type=str, choices=['reinforce', 'ppo'])
  argparser.add_argument('--force_training', default=False, type=ast.literal_eval)
  args = argparser.parse_args()

  logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')
  
  trainers = {'reinforce': train_reinforce, 'ppo': train_ppo}
  model_conf = {'concat': args.concat, 'memory_size': args.memory_size, 'expand_memory_emb': args.expand_memory_emb,
                'expanded_memory_emb_size': args.expanded_memory_emb_size, 'summarizer_hidden_size': args.summarizer_hidden_size,
                'state_size': args.state_size, 'shared_AC_hidden_size': args.shared_AC_hidden_size,
                'processor_hidden_size': args.processor_hidden_size}
  
  rep = input('Start training? (y or n): ') if not args.force_training else 'y'
  if rep == 'y' or args.force_training:
    # seeding for reproducibility
    random.seed(args.seed * args.seed)
    torch.manual_seed(args.seed)

    u.dump_argparser_parameters(args)
    trainers[args.algo](game_view=args.game_view, use_visdom=args.use_visdom, load_model=args.load_model, episode_batch=True,
                        save_model=args.save_model, save_name=args.save_name, AC=True, model=ToyACmem, pretraining=args.pretraining,
                        model_conf=model_conf)