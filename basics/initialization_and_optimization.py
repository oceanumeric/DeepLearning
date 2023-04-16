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


# region --------- neural network model --------- ###
class BaseNet(nn.Module):
    """
    A simple neural network to show the effect of activation functions
    """

    def __init__(
        self, ac_fun, input_size=784, num_class=10, hidden_sizes=[512, 256, 256, 128]
    ):
        """
        Inputs:
            ac_fun: activation function
            input_size: size of the input = 28*28
            num_class: number of classes = 10
            hidden_sizes: list of hidden layer sizes that specify the layer sizes
        """
        super().__init__()

        # create a list of layers
        layers = []
        layers_sizes = [input_size] + hidden_sizes
        for idx in range(1, len(layers_sizes)):
            layers.append(nn.Linear(layers_sizes[idx - 1], layers_sizes[idx]))
            layers.append(ac_fun)
        # add the last layer
        layers.append(nn.Linear(layers_sizes[-1], num_class))
        # create a sequential neural network
        self.net = nn.Sequential(*layers)  # * is used to unpack the list
        self.num_nets = len(layers_sizes)

        # set up the config dictionary
        self.config = {
            "ac_fun": ac_fun.__class__.__name__,
            "input_size": input_size,
            "num_class": num_class,
            "hidden_sizes": hidden_sizes,
            "num_of_nets": self.num_nets,
        }

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # flatten the input as 1 by 784 vector
        return self.net(x)

    def _layer_summary(self, input_size):
        """
        print the summary of the model
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


class Identity(nn.Module):
    """
    A class for identity activation function
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# a dictionary of activation functions
AC_FUN_DICT = {"Tanh": nn.Tanh, "ReLU": nn.ReLU, "Identity": Identity}

# endregion


# region --------- visualize weights, gradients and activations --------- ###
def _plot_distribution(val_dict, color="C0", xlabel=None, stat="count", use_kde=True):
    """
    A helper function for plotting the distribution of the values.
    Input:
        val_dict: a dictionary of values
        color: the color of the plot
        xlabel: the label of the x-axis
        stat: the type of the statistic
        use_kde: whether to use kernel density estimation
    """
    columns = len(val_dict)  # number of columns
    fig, axes = plt.subplots(1, columns, figsize=(columns * 3.5, 3))
    axes = axes.flatten()
    for idx, (name, val) in enumerate(val_dict.items()):
        # only use kde if the range of the values is larger than 1e-7
        sns.histplot(
            val,
            color=color,
            ax=axes[idx],
            bins=50,
            stat=stat,
            kde=use_kde and (val.max() - val.min() > 1e-7),
        )
        axes[idx].set_title(name)
        axes[idx].set_xlabel(xlabel)
        axes[idx].set_ylabel(stat)
    fig.subplots_adjust(wspace=0.4)

    return fig


def visualize_weights(model, color="C0"):
    weights = {}
    # weights are stored in the model as a dictionary called named_parameters
    for name, param in model.named_parameters():
        if "weight" in name:
            weights_key = f"Layer {name.split('.')[1]}"
            weights[weights_key] = param.detach().cpu().numpy().flatten()
    # plot the distribution of the weights
    fig = _plot_distribution(weights, color=color, xlabel="Weight value")
    fig.suptitle("Distribution of the weights", fontsize=14, y=1.05)


def visualize_gradients(model, train_dataset, color="C0", print_variance=False):
    # get the gradients by passing a batch of data through the model
    model.eval()
    data_loader = tu_data.DataLoader(train_dataset, batch_size=1024, shuffle=False)
    imgs, labels = next(iter(data_loader))
    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

    # calculate the gradients
    model.zero_grad()
    preds = model(imgs)
    loss = F.cross_entropy(preds, labels)
    loss.backward()
    gradients = {
        name: param.grad.detach().cpu().numpy().flatten()
        for name, param in model.named_parameters()
        if "weight" in name
    }
    model.zero_grad()

    # plot the distribution of the gradients
    fig = _plot_distribution(gradients, color=color, xlabel="Gradient value")
    fig.suptitle("Distribution of the gradients", fontsize=14, y=1.05)

    if print_variance:
        for name, grad in gradients.items():
            print(f"{name} gradient variance: {np.var(grad):.2e}")


def visualize_activations(model, train_dataset, color="C0", print_variance=False):
    model.eval()
    data_loader = tu_data.DataLoader(train_dataset, batch_size=1024, shuffle=False)
    imgs, labels = next(iter(data_loader))
    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

    # pass the data through the model
    img_features = imgs.view(imgs.shape[0], -1)
    activations = {}

    with torch.no_grad():
        for idx, layer in enumerate(model.net):
            img_features = layer(img_features)
            if isinstance(layer, nn.Linear):
                activations[f"Layer {idx}"] = (
                    img_features.detach().cpu().numpy().flatten()
                )
    # plot the distribution of the activations
    fig = _plot_distribution(
        activations, color=color, stat="density", xlabel="Activation value"
    )

    fig.suptitle("Distribution of the activations", fontsize=14, y=1.05)

    if print_variance:
        for name, act in activations.items():
            print(f"{name} activation variance: {np.var(act):.3f}")


# endregion


# constant initialization
def const_init(model, val=0.0):
    for name, param in model.named_parameters():
        param.data.fill_(val)


def const_variance_init(model, std=0.01):
    for name, param in model.named_parameters():
        param.data.normal_(0, std)
        
        
def xavier_init(model):
    for name, param in model.named_parameters():
        if "weight" in name:
            param.data.normal_(0, 1 / np.sqrt(param.shape[1]))
        elif "bias" in name:
            param.data.fill_(0.0)
            

def kaiming_init(model):
    for name, param in model.named_parameters():
        if "bias" in name:
            param.data.fill_(0.0)
        elif name.startswith("layers.0"):
            # the first layer does not have ReLU
            param.data.normal_(0, 1 / np.sqrt(param.shape[1]))
        else:
            param.data.normal_(0, np.sqrt(2) / np.sqrt(param.shape[1]))


if __name__ == "__main__":
    print(os.getcwd())
    print("Using torch", torch.__version__)
    print("Using device", DEVICE)
    set_seed(42)
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
    print("The mean of the image:", train_set.data.float().mean() / 255)
    print("The standard deviation of the image:", train_set.data.float().std() / 255)

    # load the dataset
    train_loader = tu_data.DataLoader(
        train_set, batch_size=1024, shuffle=True, drop_last=False
    )
    imgs, _ = next(iter(train_loader))
    print(f"Mean: {imgs.mean().item():5.3f}")
    print(f"Standard deviation: {imgs.std().item():5.3f}")
    print(f"Maximum: {imgs.max().item():5.3f}")
    print(f"Minimum: {imgs.min().item():5.3f}")
    print(f"Shape of tensor: {imgs.shape}")

    our_model = BaseNet(ac_fun=Identity()).to(DEVICE)

    #----  constant initialization
    const_init(our_model, val=0.005)
    visualize_gradients(our_model, train_set)
    visualize_activations(our_model, train_set, print_variance=True)
    print("-" * 80)

    #---- constant variance initialization
    const_variance_init(our_model, std=0.01)
    visualize_activations(our_model, train_set, print_variance=True)
    print("-" * 80)

    const_variance_init(our_model, std=0.1)
    visualize_activations(our_model, train_set, print_variance=True)
    
    #---- xavier initialization
    xavier_init(our_model)
    visualize_weights(our_model)
    visualize_activations(our_model, train_set, print_variance=True)
    
    #---- kaiming initialization
    our_model = BaseNet(ac_fun=nn.ReLU()).to(DEVICE)
    kaiming_init(our_model)
    visualize_gradients(our_model, train_set, print_variance=True)
    visualize_activations(our_model, train_set, print_variance=True)


# %%
