import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


print("Using torch", torch.__version__)

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device", device)

# set seeds for reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed(719)
    torch.cuda.manual_seed_all(719)


class DigitNet(nn.Module):

    # define a nerual network with parameters for the layers
    # those parameters are initialized in the constructor 
    # and are used in the forward method
    # and will be learned (updated) during training
    def __init__(self):
        super(DigitNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 6 input image channel, 16 output channels, 5x5 square convolution
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        # another affine operation: y = Wx + b
        self.fc2 = nn.Linear(120, 84)
        # final affine operation: y = Wx + b
        self.fc3 = nn.Linear(84, 10) 
  

    def forward(self, x):
        # max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
def fun_add_two_columns(df, col1, col2, new_col_name):
    df[new_col_name] = df[col1] + df[col2]
    return df
    

if __name__ == "__main__":
    # create a network
    net = DigitNet()
    # move the network to the device
    net.to(device)
    # print the network
    print(net)

    # check parameters
    for name, param in net.named_parameters():
        print(name, param.shape)

    # create a random input
    input = torch.randn(1, 1, 32, 32)
    # move the input to the device
    input = input.to(device)
    out = net(input)
    print("Output", out.shape, out)
    # clear the gradients
    net.zero_grad()
    out.backward(torch.randn(1, 10).to(device))