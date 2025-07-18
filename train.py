import random
import numpy as np
import time
from tqdm import tqdm
import argparse
import medmnist
from medmnist import INFO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# ------------------ Define Arguments Directly ------------------
# Due to issues with argparse in Colab, we will define the arguments directly.
epochs = 5
batch_size = 64
lr = 0.001
data_flag = 'pathmnist'

# ------------------ Reproducibility ------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ------------------ Config & Info ------------------
# Use the directly defined variables from the previous cell
# data_flag, epochs, batch_size, lr are defined in cell 0luarqKB0yje

download = True
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])
transform = transforms.Compose([transforms.ToTensor()])

# Get the number of classes from the info dictionary's label key
n_classes = len(info['label'])


print(f"Using dataset: {data_flag}, Num classes: {n_classes}")
print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

# ------------------ Device Info ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
if device.type == "cuda":
    print(f"CUDA available: {torch.cuda.is_available()}, GPU name: {torch.cuda.get_device_name(0)}")

# ------------------ Data ------------------
train_dataset = DataClass(split='train', transform=transform, download=download)
test_dataset = DataClass(split='test', transform=transform, download=download)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# ------------------ Model ------------------
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=n_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*13*13, num_classes)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32*13*13)
        return self.fc1(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# ------------------ Training Loop ------------------
for epoch in range(epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0

    # Training progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.squeeze().long().to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(train_loader)

    # ------------------ Validation ------------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total

    # ------------------ Logging ------------------
    epoch_time = time.time() - start_time
    gpu_mem = torch.cuda.memory_allocated(device)/1024**2 if device.type == "cuda" else 0
    print(f"\n[Epoch {epoch+1}/{epochs}] "
          f"Loss: {avg_loss:.4f} | Test Acc: {accuracy:.2f}% | "
          f"Time: {epoch_time:.2f}s | GPU Mem: {gpu_mem:.2f} MB\n")

print("✅ Training complete!")