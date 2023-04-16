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


# set up the data path
DATA_PATH = "../data"
SAVE_PATH = "../pretrained/ac_fun"


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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# region --------- data prepocessing --------- ###
def _get_data():
    """
    download the dataset from FashionMNIST and transfom it to tensor
    """
    # set up the transformation
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
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

# endregion



if __name__ == "__main__":
    print(os.getcwd())
    print("Using torch", torch.__version__)
    print("Using device", DEVICE)
    # set up finger config
    # it is only used for interactive mode
    # %config InlineBackend.figure_format = "retina"
    
    # download the dataset
    train_set, test_set = _get_data()
    print("Train set size:", len(train_set))
    print("Test set size:", len(test_set))
    print("The shape of the image:", train_set[0][0].shape)
    print("The size of vectorized image:", train_set[0][0].view(-1).shape)
    # calculate the mean of the image
    print("check features", train_set.data.shape)
    print("check labels", train_set.targets.shape)
    # calculate the mean of the image by dividing 255
    # 255 is the maximum value of the image (0 - 255 pixel value)
    print("The mean of the image:", train_set.data.float().mean()/255)
    print("The standard deviation of the image:", train_set.data.float().std()/255)
    
    # load the dataset
    train_loader = tu_data.DataLoader(train_set, batch_size=1024, shuffle=True, drop_last=False)
    imgs, _ = next(iter(train_loader))
    print(f"Mean: {imgs.mean().item():5.3f}")
    print(f"Standard deviation: {imgs.std().item():5.3f}")
    print(f"Maximum: {imgs.max().item():5.3f}")
    print(f"Minimum: {imgs.min().item():5.3f}")


# %%
