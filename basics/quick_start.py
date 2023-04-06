# %%
import os
import math
import torch
import time
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt


# check torch version
print("Using torch", torch.__version__)

## understand tensors and operations in pytorch
# function y = 1/|x| * \sum_{i=1}^n (x_i+2)^2 + 3
# %%
x = torch.arange(3, dtype=torch.float32, requires_grad=True)
a = x + 2
b = a ** 2
c = b + 3
y = c.mean()
print("y = ", y)

# since we set requires_grad=True
# we can compute the gradient of y with respect to x
# call backward() on y
y.backward()
print(x.grad)  # dy/dx


### ---------- work with GPU ----------- ###
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)

# create a tensor on GPU
x = torch.arange(3, dtype=torch.float32, requires_grad=True, device=device)
print(x)

# or you can put it 
x = torch.zeros(3, 2)
x = x.to(device)
print(x)


x = torch.randn(5000, 5000)

## CPU version
start_time = time.time()
_ = torch.matmul(x, x)
end_time = time.time()
print(f"CPU time: {(end_time - start_time):6.5f}s")

## GPU version
x = x.to(device)
_ = torch.matmul(x, x)  # First operation to 'burn in' GPU
# CUDA is asynchronous, so we need to use different timing functions
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
_ = torch.matmul(x, x)
end.record()
torch.cuda.synchronize()  # Waits for everything to finish running on the GPU
print(f"GPU time: {0.001 * start.elapsed_time(end):6.5f}s")  # Milliseconds to seconds


# GPU operations have a separate seed we also want to set
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
