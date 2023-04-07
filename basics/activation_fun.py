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
import torch.utils.data as tu_data
import torch.optim as optim
import torchvision
from torchvision.datasets import FashionMNIST
from torchvision import transforms


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
    

# set up a network
class BaseNet(nn.Module):
    
    def __init__(self, activation_fun,
                 input_dim = 784,
                 output_dim = 10,
                 hidden_dims = [512, 256, 256, 128]):
        """
        Inputs:
            activation_fun: activation function
            input_dim: input dimension of images as flattened vector
            output_dim: output dimension of network (number of classes)
            hidden_dims: list of hidden dimensions (4 hidden layers)
        
        we have defined default values for input_dim, output_dim
        and hidden_dims, but you can change them if you want
        """
        super().__init__()
        # build up network in the loop
        # we can do this because each layer has the same structure
        layers = []
        layer_sizes = [input_dim] + hidden_dims
        
        for i in range(1, len(layer_sizes)):
            # all layers are linear except the last one
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            # liear layers are followed by activation function
            layers.append(activation_fun)
        # add last linear layer
        layers.append(nn.Linear(layer_sizes[-1], output_dim))
        
        self.layers = nn.Sequential(*layers)
        
        # save config
        self.config = {
            "activation_fun": activation_fun.config,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": hidden_dims
        }
    
    def forward(self, x):
        # flatten images
        x = x.view(x.shape[0], -1)
        return self.layers(x)


def _get_config_file(model_path, model_name):
    # Name of the file for storing hyperparameter details
    return os.path.join(model_path, model_name + ".config")


def _get_model_file(model_path, model_name):
    # Name of the file for storing network parameters
    return os.path.join(model_path, model_name + ".tar")


def load_model(model_path, model_name, net=None):
    """
    Load a model from a file.
    
    Inputs:
        model_path: path to the folder containing the model
        model_name: name of the model (without extension)
    Output:
        model: the loaded model
    """
    # Load the config file
    config_file = _get_config_file(model_path, model_name)
    assert os.path.isfile(config_file), f"Config file {config_file} does not exist"
    with open(config_file, "r") as f:
        config = json.load(f)
    
    # Load the model
    model_file = _get_model_file(model_path, model_name)
    assert os.path.isfile(model_file), f"Model file {model_file} does not exist"
    
    if net is None:
        act_fn_name = config["act_fn"].pop("name").lower()
        act_fn = activation_fun_dict[act_fn_name](**config.pop("act_fn"))
        net = BaseNet(act_fn=act_fn, **config)
    net.load_state_dict(torch.load(model_file, map_location=device))
    return net


def save_model(model, model_path, model_name):
    """
    Given a model, we save the state_dict and hyperparameters.

    Inputs:
        model - Network object to save parameters from
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
    """
    config_dict = model.config
    os.makedirs(model_path, exist_ok=True)
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    with open(config_file, "w") as f:
        json.dump(config_dict, f)
    torch.save(model.state_dict(), model_file)
    
    
def visualize_gradients(net, color="c0"):
    """
    Inputs:
        net: network, class object
        color: color of the plot
    remark: this function is designed to be called in a loop
    """
    # we are loading a pretrained model, so we don't need to train it
    # we only need to evaluate it
    net.eval()
    small_loader = tu_data.DataLoader(train_set, batch_size=256,
                                      shuffle=False)
    imgs, labels = next(iter(small_loader))
    imgs, labels = imgs.to(device), labels.to(device)
    
    # pass images through the network
    net.zero_grad()  # reset gradients
    preds = net(imgs)  # forward pass and compute predictions
    loss = F.cross_entropy(preds, labels)  # compute loss
    loss.backward()  # compute gradients
    
    # extract graidients for weights but not for biases
    grads = {name: param.grad.data.view(-1).cpu().clone().numpy() for name,
                param in net.named_parameters()
                if "weight"  in name}
    net.zero_grad()  # reset gradients
    
    ## Plotting
    columns = len(grads)
    fig, ax = plt.subplots(1, columns, figsize=(columns*3.5, 2.5))
    fig_index = 0
    for key in grads:
        key_ax = ax[fig_index%columns]
        sns.histplot(data=grads[key], bins=30, ax=key_ax, color=color, kde=True)
        key_ax.set_title(str(key))
        key_ax.set_xlabel("Grad magnitude")
        fig_index += 1
    fig.suptitle(f"Gradient magnitude distribution for activation function {net.config['activation_fun']['name']}", fontsize=14, y=1.05)
    fig.subplots_adjust(wspace=0.45)
    plt.show()
    plt.close()
    

def train_model(net, model_name, max_epochs=50, patience=7,
                    batch_size=256, overwrite=False):
    """
    Train a model on the training set of FashionMNIST

    Inputs:
        net - Object of BaseNetwork
        model_name - (str) Name of the model, used for creating the checkpoint names
        max_epochs - Number of epochs we want to (maximally) train for
        patience - If the performance on the validation set has not
                    improved for #patience epochs, we stop training early
        batch_size - Size of batches used in training
        overwrite - Determines how to handle the case when there already
                    exists a checkpoint. If True, it will be overwritten.
                    Otherwise, we skip training.
    """
    file_exists = os.path.isfile(_get_model_file(CHECKPOINT_PATH, model_name))
    if file_exists and not overwrite:
        print("Model file already exists. Skipping training...")
    else:
        if file_exists:
            print("Model file exists, but will be overwritten...")
    
    
        
    

if __name__ == "__main__":
    print(os.getcwd())
    print("Cuda version: ", torch.version.cuda)
    print("Using device: ", device)
    
    # download pretrained models
    # download_pretrained_models()
    
    # visualize activation functions
    # act_fns = [act_fn() for act_fn in activation_fun_dict.values()]
    # x = torch.linspace(-5, 5, 1000)
    # fig_rows = math.ceil(len(act_fns)/2)
    # fig, ax = plt.subplots(fig_rows, 2, figsize=(8, fig_rows*4))
    # for i, act_fun in enumerate(act_fns):
    #     visualize_activation_funs(act_fun, x, ax[divmod(i, 2)])
    # fig.subplots_adjust(hspace=0.3)
    
    # download data
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
    train_dataset = FashionMNIST(root=DATASET_PATH, train=True,
                                 download=True, transform=transform)
    train_set, val_set = tu_data.random_split(train_dataset,
                                                       [50000, 10000])

    # Loading the test set
    test_set = FashionMNIST(root=DATASET_PATH, train=False,
                            transform=transform,
                            download=True)
    train_loader = tu_data.DataLoader(train_set,
                                   batch_size=1024,
                                   shuffle=True,
                                   drop_last=False)
    val_loader = tu_data.DataLoader(val_set,
                                 batch_size=1024,
                                 shuffle=False,
                                 drop_last=False)
    test_loader = tu_data.DataLoader(test_set,
                                  batch_size=1024,
                                  shuffle=False,
                                  drop_last=False)
    
    # check images
    # exmp_imgs = [train_set[i][0] for i in range(16)]
    # # Organize the images into a grid for nicer visualization
    # img_grid = torchvision.utils.make_grid(torch.stack(exmp_imgs, dim=0),
    #                                        nrow=4,
    #                                        normalize=True,
    #                                        pad_value=0.5)
    # img_grid = img_grid.permute(1, 2, 0)

    # plt.figure(figsize=(8,8))
    # plt.title("FashionMNIST examples")
    # plt.imshow(img_grid)
    # plt.axis('off')
    # plt.show()
    # plt.close()
    
    # plot gradient distribution
    # for i, act_fn_name in enumerate(activation_fun_dict):
    #     set_seed(716) # Setting the seed ensures that we have the same weight initialization for each activation function
    #     act_fn = activation_fun_dict[act_fn_name]()
    #     net_actfn = BaseNet(activation_fun=act_fn).to(device)
    #     visualize_gradients(net_actfn, color=f"C{i}")
    
# %%
