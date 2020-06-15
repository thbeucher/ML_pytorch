import os
import json
import torch
import numpy as np

from tqdm import tqdm
from pydub import AudioSegment
from scipy.io.wavfile import write


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
  sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
  sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)
  return sequence


def get_soundwaves(sequence, waveglow, tacotron2):
  with torch.no_grad():
    _, mel, _, _ = tacotron2.infer(sequence)
    audio = waveglow.infer(mel)

  audio_numpy = audio[0].data.cpu().numpy()
  return audio_numpy


def downsample_signal_n_save(signal, save_name, initial_rate=22050, new_rate=16000, saving_format='wav'):
  write(save_name, initial_rate, signal)
  sound = AudioSegment.from_file(save_name, format=saving_format, frame_rate=initial_rate)
  sound = sound.set_frame_rate(new_rate)  # Downsample to 16Khz
  sound.export(save_name, format=saving_format)


def process_captions_file(captions_file, save_folder, waveglow, tacotron2):
  if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

  with open(captions_file, 'r') as f:
    data = json.load(f)
  
  print('Processing annotations...')
  for annotation in tqdm(data['annotations']):
    save_name = os.path.join(save_folder, f"{annotation['image_id']}-{annotation['id']}.wav")

    if not os.path.isfile(save_name):
      sequence = preprocess_text(annotation['caption'], tacotron2)
      signal = get_soundwaves(sequence, waveglow, tacotron2)
      downsample_signal_n_save(signal, save_name)


def create_sound_file_from_captions():
  # Gather models to perform Text-to-Speech
  waveglow = get_waveglow()
  tacotron2 = get_tacotron2()

  # Process train captions
  train_captions_file = '../../../datasets/coco/annotations/captions_train2014.json'
  save_folder = '../../../datasets/coco/train2014_wav/'
  process_captions_file(train_captions_file, save_folder, waveglow, tacotron2)
  
  # Process test captions
  test_captions_file = '../../../datasets/coco/annotations/captions_val2014.json'
  save_folder = '../../../datasets/coco/val2014_wav/'
  process_captions_file(test_captions_file, save_folder, waveglow, tacotron2)


if __name__ == "__main__":
  create_sound_file_from_captions()