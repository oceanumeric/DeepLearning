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


#region ###### ---------  environment setup --------- ######

# set up the data path
DATA_PATH = "../data"
SAVE_PATH = "../pretrained"

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
#endregion


#region ######  -------- data preprocessing -------- #######
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
 #endregion       


#region ######  -------- build up neural network -------- #######
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
            # Flatten is recommended to use instead of view
            nn.Flatten(),
            # lienar layer
            # intput size is 16 * 5 * 5 = 400
            # why 400? because the output size of the last layer is 5 x 5
            # with 16 channels
            # using _layer_summary() to check the output size
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.Linear(in_features=120, out_features=84),
            # output layer, 10 classes
            nn.Linear(in_features=84, out_features=10)
        )


    def forward(self, x):
        return self.net(x)
    
        
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
 #endregion     
           
    
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
    model = network_model()
    # push the model to the device
    model.to(device)
    
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
    
    # begin training
    for epoch in tqdm(range(num_epochs)):
        # set the model to training mode
        model.train()
        correct_preds, total_preds = 0, 0
        for imgs, labels in training_loader:
            # push the data to the device
            imgs = imgs.to(device)
            labels = labels.to(device)
            
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
            torch.save(model.state_dict(), SAVE_PATH + "/best_model.pt")
            print(f"We have not improved for {patience} epochs, stopping...")
            break 
        
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
        imgs = imgs.to(device)
        labels = labels.to(device)
        
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
 
 
    
def main():
    # download the dataset
    train_dataset, test_dataset = _get_data()
    # _visualize_data(train_dataset, test_dataset)
    
    # split the dataset into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = tu_data.random_split(train_dataset,
                                                      [train_size, val_size])
    train_the_model(FashionClassifier, train_dataset, val_dataset)
    

if __name__ == "__main__":
    print(os.getcwd())
    print("Using torch", torch.__version__)
    print("Using device", device)

    main()


# %%
