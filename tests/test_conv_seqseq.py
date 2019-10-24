import os
import sys
import math
import time
import spacy
import torch
import random
import torch.nn as nn
import torch.optim as optim

from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset, Multi30k

sys.path.append(os.path.abspath(__file__).replace('tests/test_conv_seqseq.py', ''))
import utils as u
from models.conv_seqseq import Encoder, Decoder, DecoderEmbedder, Seq2Seq


if __name__ == "__main__":
  SEED = 1234

  random.seed(SEED)
  torch.manual_seed(SEED)
  torch.backends.cudnn.deterministic = True

  spacy_de = spacy.load('de')
  spacy_en = spacy.load('en')


  def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


  def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]
  
  SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)

  TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)

  train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

  SRC.build_vocab(train_data, min_freq=2)
  TRG.build_vocab(train_data, min_freq=2)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  BATCH_SIZE = 128

  train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), 
                                                                        batch_size=BATCH_SIZE,
                                                                        device=device)

  INPUT_DIM = len(SRC.vocab)
  OUTPUT_DIM = len(TRG.vocab)
  EMB_DIM = 256
  HID_DIM = 512
  ENC_LAYERS = 10
  DEC_LAYERS = 10
  ENC_KERNEL_SIZE = 3
  DEC_KERNEL_SIZE = 3
  ENC_DROPOUT = 0.25
  DEC_DROPOUT = 0.25
  PAD_IDX = TRG.vocab.stoi['<pad>']
  SOS_IDX = TRG.vocab.stoi['<sos>']
  EOS_IDX = TRG.vocab.stoi['<eos>']
      
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      
  enc_embedder = DecoderEmbedder(INPUT_DIM, EMB_DIM, 100, ENC_DROPOUT, device)
  enc = Encoder(EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, device, embedder=enc_embedder)
  dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, PAD_IDX, device)

  model = Seq2Seq(enc, dec, device).to(device)

  def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

  print(f'The model has {count_parameters(model):,} trainable parameters')

  optimizer = optim.Adam(model.parameters())

  criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

  SM = u.ScoresMaster(idx_2_letter=TRG.vocab.itos, pad_idx=PAD_IDX, joiner=' ')


  def train(model, iterator, optimizer, criterion, clip):
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
      src = batch.src
      trg = batch.trg
      
      optimizer.zero_grad()
      
      output, _ = model(src, trg[:,:-1])
      
      #output = [batch size, trg sent len - 1, output dim]
      #trg = [batch size, trg sent len]
      
      output = output.contiguous().view(-1, output.shape[-1])
      trg = trg[:,1:].contiguous().view(-1)
      
      #output = [batch size * trg sent len - 1, output dim]
      #trg = [batch size * trg sent len - 1]
      
      loss = criterion(output, trg)
      
      loss.backward()
      
      torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
      
      optimizer.step()
      
      epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)
  

  def evaluate(model, iterator, criterion):
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
      for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        output, _ = model(src, trg[:,:-1])
    
        #output = [batch size, trg sent len - 1, output dim]
        #trg = [batch size, trg sent len]

        SM.partial_feed(trg[:, 1:].tolist(), output.argmax(-1).tolist())

        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:,1:].contiguous().view(-1)

        #output = [batch size * trg sent len - 1, output dim]
        #trg = [batch size * trg sent len - 1]
        
        loss = criterion(output, trg)

        epoch_loss += loss.item()
    
    _, word_acc, _, awer = SM.get_scores(None, None, stop_idx=EOS_IDX, from_feed=True, strategy='other')
    SM.reset_feed()
        
    return epoch_loss / len(iterator), word_acc, awer
  
  def test(model, iterator):
    model.eval()

    with torch.no_grad():
      for batch in iterator:
        src = batch.src
        trg = batch.trg

        output, _ = model.greedy_decoding(src, SOS_IDX, EOS_IDX)

        SM.partial_feed(trg[:, 1:].tolist(), output.tolist())
    
    _, word_acc, _, awer = SM.get_scores(None, None, stop_idx=EOS_IDX, from_feed=True, strategy='other')
    SM.reset_feed()

    return word_acc, awer
  

  def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
  

  N_EPOCHS = 10
  CLIP = 1

  best_valid_loss = float('inf')

  for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss, word_acc, awer = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      # torch.save(model.state_dict(), 'tut5-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    print(f'\t  Word Acc: {word_acc:.3f} |       WER: {awer:7.3f}')
  
  # model.load_state_dict(torch.load('tut5-model.pt'))

  test_loss, word_acc, awer = evaluate(model, test_iterator, criterion)

  print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | Word Acc: {word_acc:.3f} | WER: {awer:.3f}')

  word_acc, awer = test(model, test_iterator)

  print(f'Greedy Decoding -> Word accuracy = {word_acc:.3f} | WER = {awer:.3f}')
