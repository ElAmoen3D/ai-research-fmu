import os
 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
 
import torchvision
import torchvision.transforms as transforms
 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
 
 
RESULTS_DIR = "../results/trial1"
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
 
    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 
 
# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
 
train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
 
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
 
# ---------------------------------------------------------------------------
# Model, loss, optimizer
# ---------------------------------------------------------------------------
model = Model().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
 
# ---------------------------------------------------------------------------
# Training loop (tracks loss/accuracy each epoch for plotting)
# ---------------------------------------------------------------------------
train_losses, train_accs = [], []
test_losses, test_accs = [], []
 
for epoch in range(1, EPOCHS + 1):
    # --- train ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
 
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
 
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
 
    train_loss = running_loss / total
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
 
    # --- evaluate ---
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
 
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
 
    test_loss = running_loss / total
    test_acc = correct / total
    test_losses.append(test_loss)
    test_accs.append(test_acc)
 
    print(f"Epoch {epoch:2d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
 
print("Training finished.")
 
# ---------------------------------------------------------------------------
# Results & plots
# ---------------------------------------------------------------------------
 
# Loss / accuracy curves
epochs_range = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
axes[0].plot(epochs_range, train_losses, label="Train Loss", marker="o")
axes[0].plot(epochs_range, test_losses, label="Test Loss", marker="o")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss vs Epoch")
axes[0].legend()
axes[0].grid(alpha=0.3)
 
axes[1].plot(epochs_range, train_accs, label="Train Acc", marker="o")
axes[1].plot(epochs_range, test_accs, label="Test Acc", marker="o")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy vs Epoch")
axes[1].legend()
axes[1].grid(alpha=0.3)
 
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150)
plt.close()
 
# Confusion matrix
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
 
cm = confusion_matrix(all_labels, all_preds)
 
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Confusion Matrix")
ax.set_xticks(range(10))
ax.set_yticks(range(10))
fig.colorbar(im, ax=ax)
 
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=8)
 
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
 
# Sample predictions grid
idxs = np.random.choice(len(test_set), 16, replace=False)
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for ax, idx in zip(axes.flatten(), idxs):
    image, label = test_set[idx]
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(DEVICE))
        pred = output.argmax(dim=1).item()
 
    img_np = image.squeeze().numpy() * 0.3081 + 0.1307
    ax.imshow(img_np, cmap="gray")
    color = "green" if pred == label else "red"
    ax.set_title(f"T:{label} P:{pred}", color=color, fontsize=10)
    ax.axis("off")
 
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "sample_predictions.png"), dpi=150)
plt.close()
 
print(f"Plots saved to {os.path.abspath(RESULTS_DIR)}")