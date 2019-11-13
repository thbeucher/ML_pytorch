# ML_pytorch

This repository contains experiments and models implemented using pytorch

Installation : 
```python3 setup.py install```

## Profiling
To visualize potential bottleneck, pytorch provide some tools:
```python -m torch.utils.bottleneck script.py```

But I prefer to use cProfile and snakevize (```pip3 install snakeviz```):
1. Launch profiling
```python -m cProfile -o cpu_usage.prof script_to_profile.py```
2. Visualize
```snakeviz cpu_usage.prof```
