# %%
import os
import math
import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import seaborn as sns 
import matplotlib.pyplot as plt


print("Using torch", torch.__version__)


class XorClassify(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        # initialize the model
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        # perform the forward pass
        # the dimension of x is different from 
        # the dimension of the input layer
        # we update x by calling three functions
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        
        return x
    

class XorDataset(data.Dataset):
    
    def __init__(self, num_samples, std = 0.1):
        """
        Inputs:
            num_samples: number of samples
            std: standard deviation of the noise
        """
        super().__init__()
        self.size = num_samples
        self.std = std
        self.generate_continuous_xor()
        
    def generate_continuous_xor(self):
        # Each data point in the XOR dataset has two variables, x and y, that can be either 0 or 1
        # The label is their XOR combination, i.e. 1 if only x or only y is 1 while the other is 0.
        # If x=y, the label is 0.
        data = torch.randint(0, 2, (self.size, 2), dtype=torch.float32)
        label = (data.sum(dim=1) == 1).to(torch.long)
        data += self.std * torch.randn_like(data)
        
        self.data = data
        self.label = label
        
    def __len__(self):
        return self.size
    
    
    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.label[idx]
        
        return data_point, data_label
    

def visualize_sample(features, label):
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    if isinstance(label, torch.Tensor):
        label = label.numpy()
        
    data_0 = features[label == 0]
    data_1 = features[label == 1]
    
    plt.figure(figsize=(4,4))
    plt.scatter(data_0[:,0], data_0[:,1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:,0], data_1[:,1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()
        
        
if __name__ == "__main__":
    model = XorClassify(2, 4, 1)
    print(model)
    
    for name, param in model.named_parameters():
        print(name, param.shape)
        
    # create a dataset
    # set seeds for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
    xor_dataset = XorDataset(200)
    print("Size of the dataset", len(xor_dataset))
    print("First data point", xor_dataset[0])
    
    visualize_sample(xor_dataset.data, xor_dataset.label)
    
    # create a data loader
    data_loader = data.DataLoader(xor_dataset, batch_size=8, shuffle=True)
    data_inputs, data_labels = next(iter(data_loader))
    
    print("Data inputs", data_inputs.shape, "\n", data_inputs)
    print("Data labels", data_labels.shape, "\n", data_labels)
# %%
