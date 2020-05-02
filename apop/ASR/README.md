# ASR (Automatic Speech Recognition)

Experiments conduct in this repository are currently using openslr dataset that you can download on http://www.openslr.org/12/

## data.py

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

## convnet_trainer.py

Contains class to train/eval/dump_predictions for convnet_seqseq experiments

## convnet_experiments.py

Contains experiments that uses convnet_seqseq architecture

Usage:
```python
python3 convnet_experiments.py
```
