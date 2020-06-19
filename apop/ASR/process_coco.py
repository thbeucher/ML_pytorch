import os
import json
import torch
import numpy as np
import pickle as pk
import soundfile as sf

from tqdm import tqdm
from pydub import AudioSegment
from scipy.io.wavfile import write
from collections import defaultdict


def get_waveglow():
  waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
  waveglow = waveglow.remove_weightnorm(waveglow)
  waveglow = waveglow.to('cuda')
  waveglow.eval()
  return waveglow


def get_tacotron2():
  tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
  tacotron2 = tacotron2.to('cuda')
  tacotron2.eval()
  return tacotron2


def preprocess_text(text, tacotron2):
  if isinstance(text, str):
    sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
  else:
    sequence = np.array([tacotron2.text_to_sequence(t, ['english_cleaners']) for t in text])

  sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)
  return sequence


def get_soundwaves(sequence, waveglow, tacotron2):
  with torch.no_grad():
    _, mel, _, _ = tacotron2.infer(sequence)
    audio = waveglow.infer(mel)

  audio_numpy = audio[0].data.cpu().numpy() if audio.shape[0] == 1 else audio.data.cpu().numpy()
  return audio_numpy


def save_audio(signal, save_name, rate=22050):
  write(save_name, rate, signal)


def downsampling_signal(filename, initial_rate=22050, new_rate=16000, saving_format='wav'):
  _, sample_rate = sf.read(filename)
  if sample_rate != new_rate:
    sound = AudioSegment.from_file(filename, format=saving_format, frame_rate=initial_rate)
    sound = sound.set_frame_rate(new_rate)  # Downsample to 16Khz
    sound.export(filename, format=saving_format)


def downsample_signal_n_save(signal, save_name, initial_rate=22050, new_rate=16000, saving_format='wav'):
  write(save_name, initial_rate, signal)
  downsampling_signal(save_name, initial_rate=initial_rate, new_rate=new_rate, saving_format=saving_format)


def process_captions_file(captions_file, save_folder, waveglow, tacotron2, by_batch=True):
  if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

  with open(captions_file, 'r') as f:
    data = json.load(f)
  
  print('Processing annotations...')
  tmp_seq_file = '_tmp_sequences.pk'
  if os.path.isfile(tmp_seq_file):
    sequences = pk.load(open(tmp_seq_file, 'rb'))
  else:
    sequences = [(annot['image_id'], annot['id'], preprocess_text(annot['caption'], tacotron2)) for annot in tqdm(data['annotations'])]
    sequences = [(os.path.join(save_folder, f"{s[0]}-{s[1]}.wav"), s[2]) for s in sequences]
    sequences = sorted(sequences, key=lambda x: x[1].shape[-1])
    pk.dump(sequences, open(tmp_seq_file, 'wb'))
  
  if not by_batch:
    print('Text-to-Speech and downsampling processing...')
    for save_name, sequence in tqdm(sequences):
      if not os.path.isfile(save_name):
        signal = get_soundwaves(sequence, waveglow, tacotron2)
        downsample_signal_n_save(signal, save_name)
  else:
    batchs = group_by_seq_len_then_batch(sequences)
    print('Text-to-Speech processing...')
    for batch in tqdm(batchs):
      save_names, seqs = zip(*batch)
      if not all([os.path.isfile(sn) for sn in save_names]):
        signals = get_soundwaves(torch.cat(seqs), waveglow, tacotron2)
        for signal, save_name in zip(signals, save_names):
          save_audio(signal, save_name)
    print('Downsampling from 22khz to 16khz...')
    for f in tqdm(os.listdir(save_folder)):
      downsampling_signal(f)
    
  os.remove(tmp_seq_file)


def group_by_seq_len_then_batch(sequences, batch_size=16):
  groups = defaultdict(list)
  for seq in sequences:
    groups[seq[-1].shape[-1]].append(seq)
  batchs = sorted([(k, g[i:i+batch_size]) for k, g in groups.items() for i in range(0, len(g), batch_size)], key=lambda x: x[0])
  return [b[1] for b in batchs]


def create_sound_file_from_captions():
  # Gather models to perform Text-to-Speech
  waveglow = get_waveglow()
  tacotron2 = get_tacotron2()

  # Process train captions
  # train_captions_file = '../../../datasets/coco/annotations/captions_train2014.json'
  # save_folder = '../../../datasets/coco/train2014_wav/'
  # process_captions_file(train_captions_file, save_folder, waveglow, tacotron2)
  
  # Process test captions
  test_captions_file = '../../../datasets/coco/annotations/captions_val2014.json'
  save_folder = '../../../datasets/coco/val2014_wav/'
  process_captions_file(test_captions_file, save_folder, waveglow, tacotron2)


if __name__ == "__main__":
  create_sound_file_from_captions()