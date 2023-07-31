import ssl
ssl._create_default_https_context = ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader

# Load the pre-trained ResNet18 model
base_model = models.resnet18(pretrained=True)

# Replace the classifier with a new one for pneumonia detection
num_classes = 2  # Two classes: pneumonia and healthy
in_features = base_model.fc.in_features
base_model.fc = nn.Linear(in_features, num_classes)

# Freeze the pre-trained layers (optional)
for param in base_model.parameters():
    param.requires_grad = False

# Define data transformations (resize, normalize, augment, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the dataset and create data loaders
train_dataset = datasets.ImageFolder("C:\\Users\\leej179\\git\\pdudl\\chest_xray\\train", transform=transform)
test_dataset = datasets.ImageFolder("C:\\Users\\leej179\\git\\pdudl\\chest_xray\\test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Move the model to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = base_model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(base_model.fc.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    base_model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = base_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Fine-tuning (optional)
for param in base_model.parameters():
    param.requires_grad = True

fine_tune_optimizer = optim.SGD(base_model.parameters(), lr=0.0001, momentum=0.9)

for epoch in range(num_epochs):
    base_model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        fine_tune_optimizer.zero_grad()
        outputs = base_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        fine_tune_optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Evaluation on the test set
base_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = base_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
