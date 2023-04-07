# %% 
# import packages that are not related to torch
import os
import math
import time
import numpy as np
import seaborn as sns 
from tqdm import tqdm
import matplotlib.pyplot as plt


# torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tu_data
from torchvision.datasets import FashionMNIST



