"""Single-layer CNN trial with saved training plots and feature maps."""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

RESULTS_DIR = "../results/trial2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3

os.makedirs(RESULTS_DIR, exist_ok=True)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Convolutional layer 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer 1
        self.fc1 = nn.Linear(32 * 14 * 14, 10)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.fc1(x)
        return x
    
def make_model():
    model = Model()
    return model

def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, epochs=EPOCHS, lr=LR):
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(model.parameters(), lr=lr) # test with SGD now, try adam later

    train_losses = [] 
    test_losses = []
    test_accuracies = []

    # training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

    # testing loop
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

    return train_losses, test_losses, test_accuracies

# more magic code for plotting
def plot_results(train_losses, test_losses, test_accuracies, model=None, test_loader=None):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(test_losses, label='Test Loss', color='orange')
    plt.title('Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(test_accuracies, label='Test Accuracy', color='green')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/training_results.png")
    plt.show()

    if model is not None and test_loader is not None:
        plot_feature_maps(model, test_loader)
        plot_feature_maps(model, test_loader)
        plot_feature_maps(model, test_loader)

# magic ai code for plotting
def plot_feature_maps(model, test_loader, num_maps=16):
    model.eval()

    images, _ = next(iter(test_loader))
    image = images[0:1].to(DEVICE)

    with torch.no_grad():
        feature_maps = model.relu(model.conv1(image)).squeeze(0).cpu()

    num_maps = min(num_maps, feature_maps.size(0))
    rows = 4
    cols = (num_maps + rows - 1) // rows

    plt.figure(figsize=(2 * (cols + 1), 2 * rows))

    plt.subplot(rows, cols + 1, 1)
    plt.imshow(image.squeeze(0).squeeze(0).cpu(), cmap="gray")
    plt.title("Input")
    plt.axis("off")

    for idx in range(num_maps):
        plt.subplot(rows, cols + 1, idx + 2)
        plt.imshow(feature_maps[idx], cmap="viridis")
        plt.title(f"Map {idx + 1}", fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/feature_maps.png", dpi=150)
    plt.show()

# main method
if __name__ == "__main__":
    model = make_model().to(DEVICE)
    train_loader, test_loader = load_data(batch_size=BATCH_SIZE)
    train_losses, test_losses, test_accuracies = train_model(model, train_loader, test_loader, epochs=EPOCHS, lr=LR)
    plot_results(train_losses, test_losses, test_accuracies, model=model, test_loader=test_loader)