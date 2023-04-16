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


# region --------- environment setup --------- ###
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

# endregion


# region --------- data prepocessing --------- ###
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


# endregion


# region --------- activation functions --------- ###
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
    ax.set_title(ac_fun.name, fontweight="bold")
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


# endregion


# region --------- build up a neural network --------- ###
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
            "ac_fun": ac_fun.config,
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


# endregion


# region --------- visualize the distribution of gradient --------- ###
def _vis_grad_dist(neural_net, training_dataset, ac_fun_dict):
    """
    Input:
        nueral_net: neural network model with activation function
                    such as foo_net = BaseNet(ReLU())
        training_dataset: training dataset
        ac_fun_dict: dictionary of activation functions
    """

    fig_rows = len(ac_fun_dict)
    fig_cols = 5  # number of nets

    # load the data from the training dataset
    train_loader = tu_data.DataLoader(training_dataset, batch_size=256, shuffle=False)

    # get the first batch of data
    imgs, labels = next(iter(train_loader))
    # push the data to the device
    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

    fig, axes = plt.subplots(
        fig_rows, fig_cols, figsize=(3.7 * fig_cols, 3 * fig_rows)
    )

    for row_idx, ac_key in enumerate(ac_fun_dict):
        set_seed(42)
        # push model to device
        ac_fun_net = neural_net(ac_fun_dict[ac_key]).to(DEVICE)
        # change the model to evaluation mode
        ac_fun_net.eval()
        # pass the data through the network and get gradient
        ac_fun_net.zero_grad()  # set the gradient to zero
        preds = ac_fun_net(imgs)
        loss = F.cross_entropy(preds, labels)
        loss.backward()
        # extract the gradient of the first layer
        gradients = {
            name: param.grad.view(-1).cpu().detach().numpy()
            for name, param in ac_fun_net.named_parameters()
            if "weight" in name
        }
        ac_fun_net.zero_grad()  # set the gradient to zero

        for col_idx, key in enumerate(gradients):
            ax = axes[row_idx, col_idx]
            sns.histplot(gradients[key], bins=30, ax=ax, kde=True, color=f"C{row_idx}")
            ax.set_title(f"{ac_key}:{key}")

    fig.subplots_adjust(hspace=0.4, wspace=0.4)


# endregion



#region ######  -------- function to train the model  -------- #######
def train_the_model(network_model, training_dataset, val_dataset,
                                    num_epochs=13,
                                    patience=7):
    """
    Train the neural network model, we stop if the validation loss
    does not improve for a certain number of epochs (patience=7)
    Inputs:
        network_model: the neural network model
        training_dataset: the training dataset
        val_dataset: the validation dataset
        num_epochs: the number of epochs
        patience: the number of epochs to wait before early stopping
    Output:
        the trained model
    """
    
    # initialize the model
    model = network_model
    
    ac_fun_name = model.config["ac_fun"]["name"]
    # push the model to the device
    model.to(DEVICE)
    
    # hyperparameters setting
    learning_rate = 0.001
    batch_size = 64
    
    # create the loader
    training_loader = tu_data.DataLoader(training_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
    validation_loader = tu_data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
    
    # define the loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # define the optimizer
    # we are using stochastic gradient descent
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=0.9)
    
    # print out the model summary
    print(model)
    
    # loss tracker
    loss_scores = []
    
    # validation score tracker
    val_scores = []
    best_val_score = -1
    epoch_count = 0 # count the number of epochs
    time_start = time.time()
    
    # begin training
    for epoch in tqdm(range(num_epochs)):
        # set the model to training mode
        model.train()
        correct_preds, total_preds = 0, 0
        for imgs, labels in training_loader:
            # push the data to the device
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # forward pass
            preds = model(imgs)
            loss = loss_fn(preds, labels)
            
            # backward pass
            # zero the gradient
            optimizer.zero_grad()
            # calculate the gradient
            loss.backward()
            # update the weights
            optimizer.step()
            
            # calculate the accuracy
            correct_preds += preds.argmax(dim=1).eq(labels).sum().item()
            total_preds += len(labels)
            
        epoch_count += 1
        
        # append the loss score
        loss_scores.append(loss.item())
        
        # calculate the training accuracy
        train_acc = correct_preds / total_preds
        # calculate the validation accuracy
        val_acc = test_the_model(model, validation_loader)
        val_scores.append(val_acc)
        
        # print out the training and validation accuracy
        print(f"### ----- Epoch {epoch+1:2d} Training accuracy: {train_acc*100.0:03.2f}")
        print(f"                    Validation accuracy: {val_acc*100.0:03.2f}")
        
        if val_acc > val_scores[best_val_score] or best_val_score == -1:
            best_val_score = epoch
        else:
            # one could save the model here
            torch.save(model.state_dict(), SAVE_PATH + f"/{ac_fun_name}.pt")
            print(f"We have not improved for {epoch_count} epochs, stopping...")
            time_end = time.time()
            print(f"Took {time_end - time_start:.2f} seconds to train the model")
            break
    # save the model
    torch.save(model.state_dict(), SAVE_PATH + f"/{ac_fun_name}.pt")
    time_end = time.time()
    print(f"Took {time_end - time_start:.2f} seconds to train the model")
    
    
        
    # plot the loss scores and validation scores
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].plot([i for i in range(1, len(loss_scores)+1)], loss_scores)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].plot([i for i in range(1, len(val_scores)+1)], val_scores)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Accuracy")
    fig.suptitle("Loss and Validation Accuracy of LeNet-5 for Fashion MNIST")
    fig.subplots_adjust(wspace=0.45)
    fig.show()


def test_the_model(model, val_data_loader):
    """
    Test the model on the validation dataset
    Input:
        model: the trained model
        val_data_loader: the validation data loader
    """
    # set the model to evaluation mode
    model.eval()
    
    correct_preds, total_preds = 0, 0
    for imgs, labels in val_data_loader:
        # push the data to the device
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # no need to calculate the gradient
        with torch.no_grad():
            preds = model(imgs)
            # get the index of the max log-probability
            # output is [batch_size, 10]
            preds = preds.argmax(dim=1, keepdim=True)
            # item() is used to get the value of a tensor
            # move the tensor to the cpu
            correct_preds += preds.eq(labels.view_as(preds)).sum().item()
            total_preds += len(imgs)
    
    test_acc = correct_preds / total_preds
    
    return test_acc
    
 #endregion



#region --- visualize the output for each layer --------- #####
def _visualize_output(train_set, ac_fun_dict):
    """
    visualize the output for each layer based on pretrained model
    """
    
    # will only do this for three activation functions
    models_list = ["Sigmoid", "Tanh", "ReLU"]

    # initialize a dictionary to store the output of each layer
    output_dict = {} 
        
    for ac_fun_name in models_list:
        # load the data
        data_loader = tu_data.DataLoader(train_set, batch_size=1024)
        imgs, labels = next(iter(data_loader))
        # load the model
        ac_fun = ac_fun_dict[ac_fun_name]
        nn_model = BaseNet(ac_fun).to(DEVICE)
        saved_model = torch.load(SAVE_PATH + f"/{ac_fun_name}.pt", map_location=DEVICE)
        nn_model.load_state_dict(saved_model)
        
        # evaluate the model
        nn_model.eval()
        with torch.no_grad():
            imgs = imgs.to(DEVICE)
            imgs = imgs.view(imgs.shape[0], -1)
            for layer_idx, layer in enumerate(nn_model.net[:-1]):
                imgs = layer(imgs)
                layer_name = layer.__class__.__name__
                output_dict_key = ac_fun_name + "_" + str(layer_idx) + "_" + layer_name
                output_dict[output_dict_key] = imgs.view(-1).cpu().numpy()
    

    fig_rows = 2 * len(models_list)
    fig_cols = 4
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols*3.7, fig_rows*3))
    axes = axes.flatten()
    
    color_map = {
        "Sigmoid": "C0",
        "Tanh": "C1",
        "ReLU": "C2"
    }
    
    for idx, output_key in enumerate(output_dict):
        # get the output
        output = output_dict[output_key]
        # get the activation function name
        ac_fun_name = output_key.split("_")[0]
        # get the layer index
        layer_idx = int(output_key.split("_")[1])
        layer_name = output_key.split("_")[2]
        # get the axis
        ax = axes[idx]
        sns.histplot(output, ax=ax, kde=True, bins=50, color=color_map[ac_fun_name])
        ax.set_title(f"{ac_fun_name} - Layer {layer_idx}: {layer_name}")
    
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    
        
        

#endregion



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
    # _visualize_data(train_set, test_set)

    # create a dictionary of activation functions 6 activation functions
    ac_fun_dict = {
        "Sigmoid": Sigmoid(),
        "Tanh": Tanh(),
        "ReLU": ReLU(),
        "LeakyReLU": LeakyReLU(),
        "ELU": ELU(),
        "Swish": Swish(),
    }
    # set seaborn style
    # sns.set_style("ticks")
    # vis_ac_fun(ac_fun_dict)
    
    # set seed
    set_seed(42)
    # check layer summary
    foo = BaseNet(ac_fun_dict["Tanh"])
    # print(foo.config)
    # good habit to check the dimension dynamically in the network
    foo._layer_summary((1, 28 * 28))
    # foo._layer_summary((28*28, 1)) will raise an error
    # for param_name, param_val in foo.named_parameters():
    #     print(param_name, param_val.shape, param_val.grad)

    # split the dataset into train and validation
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_dataset, val_dataset = tu_data.random_split(train_set, [train_size, val_size])
    # train_the_model(foo, train_dataset, val_dataset)
    # _vis_grad_dist(BaseNet, train_dataset, ac_fun_dict)
    _visualize_output(train_set, ac_fun_dict)




# %%
