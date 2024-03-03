#%%
import copy
import math
import random
import time
from loguru import logger
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
            # the format has to be {name}.{layer_counts[name]}
            # as the default format in pytorch is {name}.{index}
            layers.append((f"{name}{layer_counts[name]}", layer))
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

        # we have to use backbone to keep the same name as the pretrained model
        # because the pretrained model is saved with the name backbone
        # if you use different name, you have to change the name when loading the model
        self.backbone = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(512, 10)  # last layer input is 512 and output is 10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input x [batch_size - N, 3, 32, 32] -> [N, 512, 2, 2]
        x = self.backbone(x)
        # do average pooling -> [N, 512]
        x = x.mean(dim=[2, 3])
        # do classification -> [N, 10]
        x = self.classifier(x)
        return x


def train_the_model(
    network_model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
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


def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)


def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements

def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width


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



def plot_weight_distribution(model, bins=256, count_nonzero_only=False):
    # we have 8 layers in the model and one final layer for classification
    # we will plot the histogram of the weights for each layer
    fig, axes = plt.subplots(3,3, figsize=(10, 6))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        # check if the parameter is weight
        # the weight is the parameter that has 2 dimensions
        # the bias is the parameter that has 1 dimension
        if param.dim() > 1:
            ax = axes[plot_index]
            if count_nonzero_only:
                param_cpu = param.detach().view(-1).cpu()
                param_cpu = param_cpu[param_cpu != 0].view(-1)
                ax.hist(param_cpu, bins=bins, density=True,
                        color = 'blue', alpha = 0.5)
            else:
                ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True,
                        color = 'blue', alpha = 0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()



# constants for calculating the model size
Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB



def fine_grained_prune(tensor: torch.Tensor, sparsity : float) -> torch.Tensor:
    """
    magnitude-based pruning for single tensor
    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
        sparsity = #zeros / #elements = (1 - #nonzeros) / #elements
    :return:
        torch.(cuda.)Tensor, mask for zeros
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()

    ##################### YOUR CODE STARTS HERE #####################
    # Step 1: calculate the #zeros (please use round())
    # tensor.numel() returns the number of elements in the tensor
    num_zeros = torch.round(num_elements - tensor.count_nonzero())
    # Step 2: calculate the importance of weight with absolute value
    importance = tensor.abs()
    # Step 3: calculate the pruning threshold based on sparsity
    # 3.1 we need to calculate the k-th (smallest) value in the tensor
    k = num_elements * sparsity + 1
    k = int(k)
    threshold = torch.kthvalue(importance.view(-1), k).values
    print(f"threshold: {threshold}")
    # Step 4: get binary mask (1 for nonzeros, 0 for zeros)
    mask = importance > threshold
    ##################### YOUR CODE ENDS HERE #######################

    # Step 5: apply mask to prune the tensor
    tensor.mul_(mask)

    return mask


def test_fine_grained_prune(
    test_tensor=torch.tensor([[-0.46, -0.40, 0.39, 0.19, 0.37],
                              [0.00, 0.40, 0.17, -0.15, 0.16],
                              [-0.20, -0.23, 0.36, 0.25, 0.03],
                              [0.24, 0.41, 0.07, 0.13, -0.15],
                              [0.48, -0.09, -0.36, 0.12, 0.45]]),
    test_mask=torch.tensor([[True, True, False, False, False],
                            [False, True, False, False, False],
                            [False, False, False, False, False],
                            [False, True, False, False, False],
                            [True, False, False, False, True]]),
    target_sparsity=0.75, target_nonzeros=None):
    def plot_matrix(tensor, ax, title):
        ax.imshow(tensor.cpu().numpy() == 0, vmin=0, vmax=1, cmap='tab20c')
        ax.set_title(title)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        for i in range(tensor.shape[1]):
            for j in range(tensor.shape[0]):
                text = ax.text(j, i, f'{tensor[i, j].item():.2f}',
                                ha="center", va="center", color="k")

    test_tensor = test_tensor.clone()
    fig, axes = plt.subplots(1,2, figsize=(6, 10))
    ax_left, ax_right = axes.ravel()
    plot_matrix(test_tensor, ax_left, 'dense tensor')

    sparsity_before_pruning = get_sparsity(test_tensor)
    mask = fine_grained_prune(test_tensor, target_sparsity)
    sparsity_after_pruning = get_sparsity(test_tensor)
    sparsity_of_mask = get_sparsity(mask)

    plot_matrix(test_tensor, ax_right, 'sparse tensor')
    fig.tight_layout()
    plt.show()

    print('* Test fine_grained_prune()')
    print(f'    target sparsity: {target_sparsity:.2f}')
    print(f'        sparsity before pruning: {sparsity_before_pruning:.2f}')
    print(f'        sparsity after pruning: {sparsity_after_pruning:.2f}')
    print(f'        sparsity of pruning mask: {sparsity_of_mask:.2f}')

    if target_nonzeros is None:
        if test_mask.equal(mask):
            print('* Test passed.')
        else:
            print('* Test failed.')
    else:
        if mask.count_nonzero() == target_nonzeros:
            print('* Test passed.')
        else:
            print('* Test failed.')


if __name__ == "__main__":
    print("Running lab1.py as main program.")
    # download the dataset
    # dataset = _download_data()
    # # load the dataset
    # dataloader = _split_data(dataset)

    # # download the pretrained model
    # checkpoint_url = "https://hanlab18.mit.edu/files/course/labs/vgg.cifar.pretrained.pth"
    # checkpoint = torch.load(download_url(checkpoint_url), map_location="cpu")
    # model = VGG().cuda()
    # print(f"=> loading checkpoint '{checkpoint_url}'")
    # model.load_state_dict(checkpoint['state_dict'])
    # recover_model = lambda: model.load_state_dict(checkpoint['state_dict'])

    # # evaluate the model
    # dense_model_accuracy = evaluate_the_model(model, dataloader["test"])
    # dense_model_size = get_model_size(model)
    # logger.info(f"Accuracy of the dense model: {dense_model_accuracy:.2f} %")
    # logger.info(f"Size of the dense model: {dense_model_size / MiB:.2f} MiB")
    # plot_weight_distribution(model)

    # test fine_grained_prune
    test_fine_grained_prune()
    target_sparsity = 0.57
    test_fine_grained_prune(target_sparsity=target_sparsity, target_nonzeros=10)
# %%
