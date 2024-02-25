"""
Lab0: Introduction to PyTorch and Training a Simple Network
author: oceanumeric
date: 2024-02-25
"""
# import packages that are not related to torch
import os
import math
import time
import random
import numpy as np
import seaborn as sns
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict


# import torch packages
import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *


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


### --------- data loading --------- ###
def _download_data():
    # set up the data transformation
    transform = {
        "train": Compose(
            [
                RandomCrop(32, padding=4),  # random crop with padding
                RandomHorizontalFlip(),
                ToTensor(),
            ]
        ),
        "test": ToTensor(),
    }

    dataset = {}
    # load the CIFAR10 dataset
    for data_group in ["train", "test"]:
        dataset[data_group] = CIFAR10(
            DATA_PATH,
            train=(data_group == "train"),
            transform=transform[data_group],
            download=True,
        )

    # show the dataset size
    logger.info(f"Training dataset size: {len(dataset['train'])}")
    logger.info(f"Testing dataset size: {len(dataset['test'])}")
    logger.info(f"The dimension of the training data: {dataset['train'].data.shape}")
    logger.info(f"The dimension of the testing data: {dataset['test'].data.shape}")
    logger.info(f"The number of classes (or labels): {len(dataset['train'].classes)}")
    logger.info(f"The classes (or labels): {dataset['train'].classes}")
    logger.info(
        f"The classes with their corresponding index: {dataset['train'].class_to_idx}"
    )
    logger.info(f"The dimension of the image: {dataset['train'][0][0].shape}")

    return dataset


def _visualize_data(dataset):
    # visualize the data by selecting 40 images from the training set
    plt.figure(figsize=(20, 10))
    for i in range(40):
        plt.subplot(4, 10, i + 1)
        plt.imshow(dataset["train"].data[i])
        plt.title(dataset["train"].classes[dataset["train"].targets[i]])
        plt.axis("off")
    plt.show()


def _split_data(dataset):
    # split the training dataset into batches and
    # we set batch size to 512
    data_loader = {}
    for data_group in ["train", "test"]:
        data_loader[data_group] = DataLoader(
            dataset[data_group],
            batch_size=512,
            shuffle=(data_group == "train"),
            num_workers=0,
            pin_memory=True,
        )

    # now have a look at the first batch of the training data and its shape
    for data_group in ["train", "test"]:
        logger.critical(f"Data group: {data_group}")
        for input_data, target in data_loader[data_group]:
            logger.info(f"The shape of the input data: {input_data.shape}")
            logger.info(f"The shape of the target: {target.shape}")
            break

    return data_loader


# set up the network
class VGG(nn.Module):
    # define architecture
    # 'M' means max pooling
    ARCH = [64, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    def __init__(self) -> None:
        super().__init__()

        layers = []
        layers_counts = defaultdict(int)  # it could sort the layers by the index

        def __add(name: str, layer: nn.Module) -> None:
            layers.append((f"{name}_{layers_counts[name]}", layer))
            layers_counts[name] += 1  # this helps to keep track of the layer count

        in_channels = 3  # RGB for input image
        for layer in self.ARCH:
            if layer == "M":
                __add("pool", nn.MaxPool2d(kernel_size=2))
            else:
                # conv layer - batch norm - ReLU
                __add(
                    "conv",
                    nn.Conv2d(in_channels, layer, kernel_size=3, padding=1, bias=False),
                )
                __add("bn", nn.BatchNorm2d(layer))
                __add("relu", nn.ReLU(inplace=True))
                in_channels = layer  # update the in_channels for the next layer

        # fully connected layers
        self.network = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(512, 10)  # last layer input is 512 and output is 10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input x [batch_size, 3, 32, 32]
        x = self.network(
            x
        )  # [batch_size, 512, 2, 2], it is the last layer is max pooling with kernel size 2

        # do average pooling
        x = x.mean(dim=[2, 3])  # [batch_size, 512]

        # do classification
        x = self.classifier(x)  # [batch_size, 10]

        return x

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
        X = torch.rand(*input_size).to(device)
        for layer in self.network:
            X = layer(X)
            print(layer.__class__.__name__, "output shape:\t", X.shape)

        X = X.mean(dim=[2, 3])
        print("Flatten output shape:\t", X.shape)
        X = self.classifier(X)
        print("Output shape:\t", X.shape)


def _profile_model(model: nn.Module, input_size) -> None:
    # profile the model
    num_params = 0
    for param in model.parameters():
        if param.requires_grad:
            num_params += param.numel()
    # check how many million parameters with 2 decimal places
    logger.info(f"Number of parameters: {num_params / 1e6:.2f}M")
    num_macs = profile_macs(model, torch.zeros(input_size).to(device))
    logger.info(f"Number of MACs: {num_macs / 1e6:.2f}M")


def _schedule_training(
    dataloader: DataLoader, num_epochs: int = 20, optimizer: Optimizer = None
):
    """
    # based on  this https://myrtle.ai/learn/how-to-train-your-resnet/
    """
    steps_per_epoch = len(dataloader["train"])

    # Define the piecewise linear scheduler
    lr_lambda = lambda step: np.interp(
        [step / steps_per_epoch], [0, num_epochs * 0.3, num_epochs], [0, 1, 0]
    )[0]

    # Visualize the learning rate schedule
    steps = np.arange(num_epochs * steps_per_epoch)
    plt.plot(steps, [lr_lambda(step) * 0.4 for step in steps])
    plt.xlabel("Number of Steps")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.show()

    scheduler = LambdaLR(optimizer, lr_lambda)

    return scheduler


def train_the_model(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: _schedule_training,
) -> None:
    
    model.train()

    for inputs, targets in tqdm(dataloader["train"], desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()



@torch.inference_mode()
def model_evaluation(
    model: nn.Module, dataloader: DataLoader) -> float:

    model.eval()

    num_samples = 0
    num_correct = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader["test"], desc="Evaluation", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)

            num_samples += inputs.size(0)
            num_correct += (predictions == targets).sum()

    accuracy = num_correct / num_samples * 100

    return accuracy.item()  # convert to scalar


def main():
    # download the dataset
    dataset = _download_data()
    # _visualize_data(dataset)
    dataloader = _split_data(dataset)

    # initialize the network
    model = VGG().to(device)
    
    # print(model.network)
    # print(model.classifier)

    # check the layer summary with a random input
    # random_input = (1, 3, 32, 32) # batch size is 1
    # model._layer_summary((1, 3, 32, 32))

    # profile the model
    # _profile_model(model, (1, 3, 32, 32))

    # initialize the cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    # initialize the optimizer
    optimizer = SGD(model.parameters(), lr=0.4, momentum=0.9, weight_decay=5e-4)

    # number of epochs
    num_epochs = 20

    # schedule the learning rate
    scheduler = _schedule_training(
        dataloader, num_epochs=num_epochs, optimizer=optimizer
    )

    for epoch_idx in tqdm(range(1, num_epochs + 1), desc="Epoch"):
        train_the_model(model, dataloader, criterion, optimizer, scheduler)
        accuracy = model_evaluation(model, dataloader)
        logger.info(f"Epoch {epoch_idx}/{num_epochs}, Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    print("Running lab0.py...")
    main()
