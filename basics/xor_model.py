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
from tqdm import tqdm
from matplotlib.colors import to_rgba
# from tqdm.notebook import tqdm


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
    
    
def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):
    
    # set model to training mode
    model.train()
    
    # train the model
    for epoch in tqdm(range(num_epochs)):
        for data_inputs, data_labels in data_loader:
            ## Step 1: Move input data to device
            # (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            
            # Step 2 run the model by passing the inputs
            # aka forward pass
            preds = model(data_inputs)
            # squeeze the output from [batch_size, 1] to [batch_size]
            preds = preds.squeeze(dim=1)
            
            # Step 3 calculate the loss
            loss = loss_module(preds, data_labels.float())
            
            ## Step 4: Perform backpropagation
            # clear the gradients from the previous iteration
            optimizer.zero_grad()
            # perform backprogation
            loss.backward()
            # step 5 update the parameters
            optimizer.step()
            
def eval_model(model, test_data_loader):
    # set model to evaluation mode
    model.eval()
    true_preds, num_preds = 0., 0.
    
    # deactivate autograd engine to reduce memory usage
    # and speed up computations
    with torch.no_grad():
        for data_inputs, data_labels in test_data_loader:
            # move data to device
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds)
            pred_labels = (preds >= 0.5).long()
            
            true_preds += (pred_labels == data_labels).sum()
            num_preds += len(data_labels)
            
    acc = true_preds / num_preds
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")
    
    
# decorator to deactivate autograd engine
@torch.no_grad() 
def visualize_classification(model, data, label):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    fig = plt.figure(figsize=(4,4), dpi=500)
    plt.scatter(data_0[:,0], data_0[:,1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:,0], data_1[:,1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()

    # Let's make use of a lot of operations we have learned above
    model.to(device)
    c0 = torch.Tensor(to_rgba("C0")).to(device)
    c1 = torch.Tensor(to_rgba("C1")).to(device)
    x1 = torch.arange(-0.5, 1.5, step=0.01, device=device)
    x2 = torch.arange(-0.5, 1.5, step=0.01, device=device)
    xx1, xx2 = torch.meshgrid(x1, x2, indexing='ij')  # Meshgrid function as in numpy
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    preds = model(model_inputs)
    preds = torch.sigmoid(preds)
    output_image = (1 - preds) * c0[None,None] + preds * c1[None,None]  # Specifying "None" in a dimension creates a new one
    output_image = output_image.cpu().numpy()  # Convert to numpy array. This only works for tensors on CPU, hence first push to CPU
    plt.imshow(output_image, origin='lower', extent=(-0.5, 1.5, -0.5, 1.5))
    plt.grid(False)
    return fig

    
    
if __name__ == "__main__":
    # set up device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device", device)
    
    # set up the model
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
    data_loader = data.DataLoader(xor_dataset,
                                    batch_size=8,
                                        shuffle=True)
    data_inputs, data_labels = next(iter(data_loader))
    
    print("Data inputs", data_inputs.shape, "\n", data_inputs)
    print("Data labels", data_labels.shape, "\n", data_labels)
    
    # set up loss
    loss_module = nn.BCEWithLogitsLoss()
    # set up optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # now train the model with a larger dataset
    train_dataset = XorDataset(2500)
    train_dataset_loader = data.DataLoader(train_dataset,
                                            batch_size=128,
                                                shuffle=True)
    # push the model to the device
    model.to(device)
    
    # now train the model
    train_model(model, optimizer, train_dataset_loader, loss_module)
    
    state_dict = model.state_dict()
    print(state_dict)
    
    # save our model
    # torch.save(state_dict, "xor_model.tar")
    
    # eveluate the model
    test_data = XorDataset(500)
    test_data_loader = data.DataLoader(test_data,
                                        batch_size=128,
                                        shuffle=False,
                                        drop_last=False)
    eval_model(model, test_data_loader)
    
    visualize_classification(model, xor_dataset.data, xor_dataset.label)
# %%
