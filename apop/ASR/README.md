# ASR (Automatic Speech Recognition)

## Speech-to-Text (STT)

Experiments conduct in this repository are currently using openslr dataset that you can download on http://www.openslr.org/12/

### data.py

This file contains every functions to process data for ASR experiments

**Usage**:

```python
metadata_file = 'filename_where_metadata_are_saved.pk'
train_folder = 'path/to/openslr/training/folder'
test_folder = 'path/to/openslr/testing/folder'

data = Data()

if not os.path.isfile(metadata_file):
  data.set_audio_metadata(train_folder, test_folder)
  data.process_all_transcripts(train_folder, test_folder)
  data.save_metadata(save_name=metadata_file)
else:
  data.load_metadata(save_name=metadata_file)
```

### ctc_experiments.py
 
Default configuration (available experiment) use [wav2vec](https://arxiv.org/abs/1904.05862) model so you have to download it first!

Usage:
```python
python3 ctc_experiments.py
```

Experiment1 ends up with following performance score:
 ```character_accuracy: 0.72, 'word_accuracy': 0.86, sentence_accuracy: 0.5, 'wer': 0.057```
