# Full Stack PyTorch

Back in 2017, I have to use `numpy` to train a neural network. It was a pain. One has to write a lot of code 
to train a simple neural network. The chance of having a bug is high. Later, I went to study math and statistics. I did not follow the deep learning trend. I thought it would be better to learn the math and statistics first, which should be the foundation of deep learning, such as optimization, probability, and statistics.

I gained a lot of knowledge from the math and statistics courses. This year, I started to train neural networks again. I found that the deep learning frameworks have become much more mature. I am very impressed by the `pytorch` framework. 

In this post, I will show you how to train a neural network with `pytorch`. I will use the `pytorch` framework to train a neural network to classify the fashion-MNIST dataset. The fashion-MNIST dataset is a dataset of Zalando's article images. It is a drop-in replacement for the MNIST dataset. It has 10 classes, such as T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot. The dataset has 60,000 training examples and 10,000 testing examples. 


## The complexity of training a neural network

The training of a neural network is a complex process. It involves many steps. It also involves many hyperparameters turning. The hyperparameters include the learning rate, the number of epochs, the batch size, the optimizer, the loss function, the activation function, the initialization method, and so on. 

All those hyperparameters are important. They can affect the performance of the neural network. Therefore, training a neural network is a time-consuming process. It is becoming more or less an art, which is highly of practical skills.


Since modules and functions of  `pytorch` itself have grown a lot, it
is better to import packages separately. The following code imports
the packages that we will use in this post.

```python
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
``` 





