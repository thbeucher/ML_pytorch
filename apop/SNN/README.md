# SNN (Spiking Neural Network)

## Image Classification using CSNN and STDP

Experiments on image classification (e.g. MNIST dataset) using SNN and unsupervised learning algorithm like STDP

### sdcnn_or_rl.py

This file is a script to train & test on MNIST dataset a 3-convolution-layer-network trained with STDP and R-STDP

**Usage**:

```python
python sdcnn_or_rl.py
```

### sdcnn_or.py

This file is a script to train & test on MNIST dataset a 2-convolution-layer-network trained with STDP and LinearSVC as classifier

**Usage**:

```python
python sdcnn_or.py
```

**Script information**:
* Running time (on GPU) : 8mn
* GPU memory usage : 1Go
* F1-score : 0.98
