# TinyML and Efficient Deep Learning Computing

This repository is a collection of resources and labs for the course "TinyML and Efficient Deep Learning Computing" from [MIT HAN LAB](https://hanlab.mit.edu/courses/2023-fall-65940).


## Setting up the environment

```bash
# check gpu
nvidia-smi

# inmport torch
pip3 install torch torchvision torchaudio

# save the environment
pip freeze > requirements.txt
``` 

If you want to check the installation, you can run the following code:

```python
import torch
torch.cuda.is_available()
```

