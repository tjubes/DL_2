# %%
!conda install scipy -y

# %%
import numpy as np
import os
from utilities import *
import h5py
from scipy.stats import zscore

# Use os.path.join() to create the correct file path
filepath = get_filepath()
print(f"Base filepath: {filepath}")

# %%
def load_and_normalize_files(directory_path, max_files=None, downsample_factor=None):
    """
    Load and normalize all h5 files in the specified directory
    
    Parameters:
    directory_path (str): Path to the directory containing h5 files
    max_files (int, optional): Maximum number of files to load
    downsample_factor (int, optional): Factor by which to downsample the data
    
    Returns:
    tuple: (data, labels) where data is a list of normalized matrices and labels are the corresponding task types
    """
    data_list = []
    labels = []
    
    # Get all h5 files in the directory
    h5_files = [f for f in os.listdir(directory_path) if f.endswith('.h5')]
    
    # Limit the number of files if specified
    if max_files is not None:
        h5_files = h5_files[:max_files]
    
    for file in h5_files:
       # Extract the task type from the filename
        if file.startswith("task_"):
            parts = file.split('_')
            task_type = '_'.join(parts[:2])  # e.g., task_motor or task_working
        else:
            task_type = file.split('_')[0]  # e.g., rest

        # Load the data
        file_path = os.path.join(directory_path, file)
        matrix = read_h5py_file(file_path)
        
        # Downsample if specified
        if downsample_factor is not None:
            matrix = matrix[:, ::downsample_factor]
        
        # Normalize the data using scipy's zscore
        normalized_matrix = zscore(matrix, axis=1, nan_policy='propagate')
        normalized_matrix = np.nan_to_num(normalized_matrix, nan=0.0)
        
        # Add to lists
        data_list.append(normalized_matrix)
        labels.append(task_type)
    
    return data_list, labels

# Example usage for Intra-subject classification
intra_train_path = os.path.join(filepath, "Intra", "train")
intra_test_path = os.path.join(filepath, "Intra", "test")

# Load a small subset of files to test the function
# Downsample factor is set to 16 to speed up the process, CHANGE LATER!
train_data, train_labels = load_and_normalize_files(intra_train_path, downsample_factor=16)
test_data, test_labels = load_and_normalize_files(intra_test_path, downsample_factor=16)
unique_labels = set(train_labels + test_labels)
print(f"Unique labels: {unique_labels}")

# Print summary
print(f"\nLoaded {len(train_data)} training files and {len(test_data)} test files")
print(f"Training labels: {train_labels}")
print(f"Test labels: {test_labels}")
print(f"Shape of first training sample after downsampling: {train_data[0].shape}")



# %%
# Defining a simple CNN model
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# %%
# Defining a loss function and optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Alternative: You might also consider Adam optimizer (optim.Adam()) which adapts learning rates 
# automatically and often works well for neural networks without needing to tune momentum manually.

# %%
# Training the network
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# %%
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)



# %%



