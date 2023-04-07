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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # download the dataset
    train_dataset = FashionMNIST(root='./data', train=True,
                                    download=True, transform=transform)
    test_dataset = FashionMNIST(root='./data', train=False,
                                    download=True, transform=transform)
    
    return train_dataset, test_dataset


def _visualize_data(train_dataset, test_dataset):
    """
    visualize the dataset by randomly sampling
    9 images from the dataset
    """
    # set up the figure
    fig = plt.figure(figsize=(8, 8))
    col, row = 3, 3
    
    for i in range(1, col*row+1):
        sample_idx = np.random.randint(0, len(train_dataset))
        img, label = train_dataset[sample_idx]
        fig.add_subplot(row, col, i)
        plt.title(train_dataset.classes[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
        

# build the model
class FashionClassifier(nn.Module):

    def __init__(self):
        # initialize the parent class
        # super() allows us to access methods from a parent class
        super(FashionClassifier, self).__init__()
        # define the neural network
        # based on architecture of LeNet-5
        self.net = nn.Sequential(
            # convolutional layers
            # input channel 1 (grayscale), output channel 6
            # kernel size 5 and stride is default 1
            # kernel is just a fancy name for filter
            # here we are using 5 x 5 filter
            # we are using padding to keep the output size the same
            # the output size is (28 - 5 + 2 * 2) / 1 + 1 = 28
            nn.Conv2d(in_channels=1, out_channels=6,
                                     kernel_size=5,
                                     padding=2),
            # activation function sigmoid
            nn.Sigmoid(),
            # average pooling layer
            # the output size is (28 - 2) / 2 + 1 = 14
            nn.AvgPool2d(kernel_size=2, stride=2),
            # the output size is (14 - 5) / 1 + 1 = 10
            nn.Conv2d(in_channels=6, out_channels=16,
                                     kernel_size=5),
            nn.Sigmoid(),
            # the output size is (10 - 2) / 2 + 1 = 5
            nn.AvgPool2d(kernel_size=2, stride=2),
            # dense layers or fully connected layers
            nn.Flatten()
        )
        
        
    def _layer_summary(self, input_size):
        """
        print the summary of the model
        Input:
            input_size: the size of the input tensor
                        in the form of (batch_size, channel, height, width)
        note: using * to unpack the tuple
        """
        # generate a random input tensor
        # 
        X = torch.rand(*input_size)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, "output shape:\t", X.shape)
        
    

# function for training the model
def train_the_model(training_dataset, val_dataset):
    
    # create the loader
    training_loader = tu_data.DataLoader(training_dataset,
                                            batch_size=64,
                                            shuffle=True)
    validation_loader = tu_data.DataLoader(val_dataset,
                                            batch_size=64,
                                            shuffle=True)
    
def main():
    # download the dataset
    train_dataset, test_dataset = _get_data()
    _visualize_data(train_dataset, test_dataset)
    
    # split the dataset into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = tu_data.random_split(train_dataset,
                                                      [train_size, val_size])

    

if __name__ == "__main__":
    print(os.getcwd())
    print("Using torch", torch.__version__)
    print("Using device", device)

    # check layer summary
    foo_model = FashionClassifier()
    foo_model._layer_summary((1, 1, 28, 28))
    
    
    



# %%
