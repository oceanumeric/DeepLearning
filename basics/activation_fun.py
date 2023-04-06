# %%
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import urllib.request
from urllib.error import HTTPError

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim


# set seaborn style
sns.set()
# set up path
DATASET_PATH = "../data"
# checkpoint path
CHECKPOINT_PATH = "../checkpoints"

# function for setting seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
set_seed(76)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActivationFunction(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.config = {"name": self.name}
        

# set up activation functions
class Sigmoid(ActivationFunction):
    
    def forward(self, x):
        return 1/(1+torch.exp(-x))
    

class Tanh(ActivationFunction):
    
    def forward(self, x):
        exp_x, exp_neg_x = torch.exp(x), torch.exp(-x)
        return (exp_x - exp_neg_x)/(exp_x + exp_neg_x)
    

class ReLU(ActivationFunction):
    
    def forward(self, x):
        return x * (x > 0).float()
    

class LeakyReLU(ActivationFunction):
        
        def __init__(self, alpha=0.1):
            super().__init__()
            self.config["alpha"] = alpha
            
        def forward(self, x):
            return torch.where(x > 0, x, self.config["alpha"]*x)
        

class ELU(ActivationFunction):
    
    def forward(self, x):
        return torch.where(x > 0, x, torch.exp(x)-1)
    

class Swish(ActivationFunction):
    
    def forward(self, x):
        return x * torch.sigmoid(x)


# create a dictionary of activation functions
activation_fun_dict = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "ReLU": ReLU,
    "LeakyReLU": LeakyReLU,
    "elu": ELU,
    "swish": Swish
}


def get_grads(activation_fun, x):
    """
    Compute gradients of activation function based on input x
    
    Inputs:
        activation_fun: activation function
        x: 1D tensor
    Output:
        1D tensor of gradients
    """
    # intialize input vector with greadients
    x = x.clone().requires_grad_(True)
    out = activation_fun(x)
    # calculate gradients
    out.sum().backward()
    
    return x.grad


def visualize_activation_funs(act_fun, x, ax):
    y = act_fun(x)
    y_grads = get_grads(act_fun, x)
    # push to cpu
    x, y, y_grads = x.cpu().numpy(), y.cpu().numpy(), y_grads.cpu().numpy()   
    # plot activation function
    ax.plot(x, y, label = act_fun.config["name"], linewidth=2)
    ax.plot(x, y_grads, label = act_fun.config["name"] + " grads", linewidth=2)
    ax.set_title(act_fun.name)
    ax.legend()
    ax.set_ylim(-1.5, x.max())
    

def download_pretrained_models():
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial3/"
    pretrained_files = ["FashionMNIST_elu.config", "FashionMNIST_elu.tar",
                    "FashionMNIST_leakyrelu.config", "FashionMNIST_leakyrelu.tar",
                    "FashionMNIST_relu.config", "FashionMNIST_relu.tar",
                    "FashionMNIST_sigmoid.config", "FashionMNIST_sigmoid.tar",
                    "FashionMNIST_swish.config", "FashionMNIST_swish.tar",
                    "FashionMNIST_tanh.config", "FashionMNIST_tanh.tar"]
    # Create checkpoint path if it doesn't exist yet
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # For each file, check whether it already exists. If not, try downloading it.
    for file_name in pretrained_files:
        file_path = os.path.join(CHECKPOINT_PATH, file_name)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f"Downloading {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print("Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n", e)
    


if __name__ == "__main__":
    print(os.getcwd())
    print("Cuda version: ", torch.version.cuda)
    print("Using device: ", device)
    
    # download pretrained models
    download_pretrained_models()
    
    # visualize activation functions
    # act_fns = [act_fn() for act_fn in activation_fun_dict.values()]
    # x = torch.linspace(-5, 5, 1000)
    # fig_rows = math.ceil(len(act_fns)/2)
    # fig, ax = plt.subplots(fig_rows, 2, figsize=(8, fig_rows*4))
    # for i, act_fun in enumerate(act_fns):
    #     visualize_activation_funs(act_fun, x, ax[divmod(i, 2)])
    # fig.subplots_adjust(hspace=0.3)
# %%
