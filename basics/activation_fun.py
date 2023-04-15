# %%
# import packages that are not related to torch
import os
import math
import time
import inspect
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt


# torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tu_data
from torchvision import transforms
from torchvision.datasets import FashionMNIST


### --------- environment setup --------- ###
# set up the data path
DATA_PATH = "../data"


# function for setting seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# set up seed globally and deterministically
set_seed(76)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### --------- data preprocessing --------- ###


def _get_data():
    """
    download the dataset from FashionMNIST and transfom it to tensor
    """
    # set up the transformation
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # download the dataset
    train_set = FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    return train_set, test_set


def _visualize_data(train_dataset, test_dataset):
    """
    visualize the dataset by randomly sampling
    9 images from the dataset
    """
    # set up the figure
    fig = plt.figure(figsize=(8, 8))
    col, row = 3, 3

    for i in range(1, col * row + 1):
        sample_idx = np.random.randint(0, len(train_dataset))
        img, label = train_dataset[sample_idx]
        fig.add_subplot(row, col, i)
        plt.title(train_dataset.classes[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")


## build up a class for different activation functions
class AcFun(nn.Module):
    def __init__(self):
        super(AcFun, self).__init__()
        self.name = self.__class__.__name__
        self.config = {"name": self.name}


# inherit from AcFun
class Sigmoid(AcFun):
    # it is important to know that pytorch class
    # has __call__ function, which calls the forward automatically
    # therefore when you intialize the class, you are calling
    # the forward function directly
    def forward(self, x):
        return torch.sigmoid(x)


class Tanh(AcFun):
    def forward(self, x):
        return torch.tanh(x)


class ReLU(AcFun):
    def forward(self, x):
        return torch.relu(x)


class LeakyReLU(AcFun):
    def __init__(self, negative_slope=0.1):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.config["negative_slope"] = negative_slope

    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope)


class ELU(AcFun):
    def forward(self, x):
        return F.elu(x)


class Swish(AcFun):
    def forward(self, x):
        return x * torch.sigmoid(x)


# function to get gradient of the activation function
def _get_grad(ac_fun, x):
    """
    get the gradient of the activation function at x
    """
    # copy the input tensor and set requires_grad to True
    x = x.clone().detach().requires_grad_(True)
    y = ac_fun(x)
    # slower version y.backward(torch.ones_like(y))
    y.sum().backward()  # faster version
    return x.grad


def _vis_grad(ac_fun, x, ax):
    """
    visualize the gradient of the activation function
    Input:
        ac_fun: activation function
        x: input tensor
        ax: matplotlib axis
    """
    # calculate the output
    y = ac_fun(x)
    # get the gradient
    grad = _get_grad(ac_fun, x)

    # pass the data to cpu and convert to numpy
    x, y, grad = x.cpu().numpy(), y.cpu().numpy(), grad.cpu().numpy()
    # plot the gradient
    ax.plot(x, y, "k-", label="AcFun")
    ax.plot(x, grad, "k--", label="Gradient")
    ax.set_title(ac_fun.name)
    ax.legend()
    ax.set_ylim(-1.5, x.max())


def vis_ac_fun(ac_fun_dict):
    """
    visualize the activation function and its gradient
    """

    # get number of activation functions
    num = len(ac_fun_dict)
    columns = math.ceil(num / 2)

    # initialize the input tensor
    x = torch.linspace(-5, 5, 1000)

    # set up the figure
    fig, axes = plt.subplots(2, columns, figsize=(4 * columns, 8))
    axes = axes.flatten()
    for i, ac_fun in enumerate(ac_fun_dict.values()):
        _vis_grad(ac_fun, x, axes[i])
    fig.subplots_adjust(hspace=0.3)


if __name__ == "__main__":
    print(os.getcwd())
    print("Using torch", torch.__version__)
    print("Using device", device)
    # set up finger config
    # it is only used for interactive mode
    %config InlineBackend.figure_format = 'retina'

    # download the dataset
    # train_set, test_set = _get_data()
    # _visualize_data(train_set, test_set)

    # create a dictionary of activation functions
    # 6 activation functions
    ac_fun_dict = {
        "Sigmoid": Sigmoid(),
        "Tanh": Tanh(),
        "ReLU": ReLU(),
        "LeakyReLU": LeakyReLU(),
        "ELU": ELU(),
        "Swish": Swish(),
    }
    # set seaborn style
    sns.set_style("ticks")
    vis_ac_fun(ac_fun_dict)


# %%
