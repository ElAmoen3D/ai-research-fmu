# CNN FIRST TRIAL - Gabriel Gonzalez
# This is a simple CNN model for image classification using PyTorch.
# This model should import the MNIST dataset and train a CNN to classify handwritten digits.

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Convolutional layer 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer 1
        self.fc1 = nn.Linear(64 * 7 * 7, 128)

        # Fully connected layer 2
        self.fc2 = nn.Linear(128, 10)

        # Activation function
        self.relu = nn.ReLU()
        # Scratching regularization for now
        # self.dropout = nn.Dropout(p=0.5)

    # Forward pass
    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# initialize model to avoid re-initialization when importing this file in other scripts
def make_model():
    model = Model()
    return model

# data loading method - import MNIST dataset and return train and test loaders
def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # normalize with mean and std of MNIST dataset
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

if __name__ == "__main__":
    # Create the model
    model = make_model()
    print(model)

    print("TRAINING MODEL - MNIST DATASET \n ************************************\n")

    # Load data
    train_loader, test_loader = load_data(batch_size=64)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=5)

    

