# ASR (Automatic Speech Recognition)

Experiments conduct in this repository are currently using openslr dataset that you can download on http://www.openslr.org/12/

## data.py

This file contains every functions to process data for ASR experiments

**Usage**:

```python3 data.py --dct_type 2 --extract_type mfcc --train_folder path_to_train_folder --test_folder path_to_test_folder```

## transformer_pretraining.py

Script to pretrain a transformer network to repeat the sentence (identity training)
