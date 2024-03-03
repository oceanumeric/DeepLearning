import copy
import math
import random
import time
from collections import OrderedDict, defaultdict
from typing import Union, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm

from torchprofile import profile_macs

assert torch.cuda.is_available(), (
    "The current runtime does not have CUDA support."
    "Please go to menu bar (Runtime - Change runtime type) and select GPU"
)


### --------- environment setup --------- ###
# set up the data path
DATA_PATH = "./data"


# function for setting seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# set up seed globally and deterministically
set_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


assert torch.cuda.is_available(), (
    "The current runtime does not have CUDA support."
    "Please make sure you can have access to a GPU."
)


def download_url(url, model_dir=".", overwrite=False):
    import os, sys, ssl
    from urllib.request import urlretrieve

    ssl._create_default_https_context = ssl._create_unverified_context
    target_dir = url.split("/")[-1]
    model_dir = os.path.expanduser(model_dir)
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = os.path.join(model_dir, target_dir)
        cached_file = model_dir
        if not os.path.exists(cached_file) or overwrite:
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)
        return cached_file
    except Exception as e:
        # remove lock file so download can be executed next time.
        os.remove(os.path.join(model_dir, "download.lock"))
        sys.stderr.write("Failed to download from url %s" % url + "\n" + str(e) + "\n")
        return None


### --------- neural network model --------- ###
class VGG(nn.Module):
    ARCH = [64, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    def __init__(self) -> None:
        super().__init__()

        layers = []
        layer_counts = defaultdict(int)

        def add(name: str, layer: nn.Module) -> None:
            layers.append((f"{name}_{layer_counts[name]}", layer))
            layer_counts[name] += 1

        input_channels = 3  # RGB for input image
        for x in self.ARCH:
            if x != "M":
                # conv-batchnorm-ReLU
                add(
                    "conv",
                    nn.Conv2d(input_channels, x, kernel_size=3, padding=1, bias=False),
                )
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(inplace=True))
                input_channels = x  # update the input_channels for the next layer
            else:
                add("pool", nn.MaxPool2d(kernel_size=2))

        self.network = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(512, 10)  # last layer input is 512 and output is 10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input x [batch_size - N, 3, 32, 32] -> [N, 512, 2, 2]
        x = self.network(x)
        # do average pooling -> [N, 512]
        x = x.mean(dim=[2, 3])
        # do classification -> [N, 10]
        x = self.classifier(x)
        return x


def train_the_model(
    network_model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optimizer,
    scheduler: LambdaLR,
    callbacks=None,
) -> None:
    network_model.train()

    for inputs, target in tqdm(dataloader, desc="Training", leave=False):
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = network_model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        # callbacks is a list of objects that have on_batch_end method
        if callbacks is not None:
            for callback in callbacks:
                callback.on_batch_end(loss.item())



@torch.inference_mode()
def evaluate_the_model(
    network_model: nn.Module, dataloader: DataLoader, verbose: bool = True
) -> float:
    network_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, target in tqdm(dataloader, desc="Evaluating", leave=False, disable=not verbose):
            inputs, target = inputs.to(device), target.to(device)
            outputs = network_model(inputs)
            predicted = outputs.argmax(dim=1)

            # calculate accuracy
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total * 100
    if verbose:
        print(f"Accuracy: {accuracy:.2f} %")

    return accuracy


if __name__ == "__main__":
    print("Running lab1.py as main program.")
