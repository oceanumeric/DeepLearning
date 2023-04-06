# DeepLearning


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

