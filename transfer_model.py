import ssl
ssl._create_default_https_context = ssl._create_default_https_context = ssl._create_unverified_context


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader

class PneumoniaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaCNN, self).__init__()
        # Load the pre-trained ResNet18 model using torch.hub
        self.model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def train_model(self, train_loader, num_epochs_frozen=5, num_epochs_fine_tuning=5, lr=0.001):
        # Move the model to the appropriate device (CPU or GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Training loop (frozen training)
        for epoch in range(num_epochs_frozen):
            self.model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs_frozen}, Loss: {running_loss/len(train_loader)}")

        # Unfreeze the pre-trained layers for fine-tuning
        for param in self.model.parameters():
            param.requires_grad = True

        # Fine-tuning loop
        for epoch in range(num_epochs_fine_tuning):
            self.model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs_fine_tuning}, Loss: {running_loss/len(train_loader)}")

    def evaluate_model(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    # Define data transformations (resize, normalize, augment, etc.)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the dataset and create data loaders
    train_dataset = datasets.ImageFolder("C:\\Users\\leej179\\git\\pdudl\\chest_xray\\train", transform=transform)
    test_dataset = datasets.ImageFolder("C:\\Users\\leej179\\git\\pdudl\\chest_xray\\test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Create an instance of PneumoniaCNN and train the model
    pneumonia_model = PneumoniaCNN(num_classes=2)
    pneumonia_model.train_model(train_loader, num_epochs_frozen=5, num_epochs_fine_tuning=5)

    # Evaluate the model on the test set
    pneumonia_model.evaluate_model(test_loader)

    # Save the model to a file
    model_file = "pneumonia_model.pth"
    pneumonia_model.save_model(model_file)

    # Load the model from the file
    loaded_model = PneumoniaCNN(num_classes=2)
    loaded_model.load_model(model_file)

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
# from torchvision import datasets, models
# from torch.utils.data import DataLoader

# # Load the pre-trained ResNet18 model
# base_model = models.resnet18(pretrained=True)

# # Replace the classifier with a new one for pneumonia detection
# num_classes = 2  # Two classes: pneumonia and healthy
# in_features = base_model.fc.in_features
# base_model.fc = nn.Linear(in_features, num_classes)

# # Define data transformations (resize, normalize, augment, etc.)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# # Load the dataset and create data loaders
# train_dataset = datasets.ImageFolder("C:\\Users\\leej179\\git\\pdudl\\chest_xray\\train", transform=transform)
# test_dataset = datasets.ImageFolder("C:\\Users\\leej179\\git\\pdudl\\chest_xray\\test", transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Move the model to the appropriate device (CPU or GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# base_model = base_model.to(device)

# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(base_model.fc.parameters(), lr=0.001)

# # Training loop (frozen training)
# num_epochs_frozen = 5  # Change this as needed

# for epoch in range(num_epochs_frozen):
#     base_model.train()
#     running_loss = 0.0

#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = base_model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

#     print(f"Epoch {epoch+1}/{num_epochs_frozen}, Loss: {running_loss/len(train_loader)}")

# # Unfreeze the pre-trained layers for fine-tuning
# for param in base_model.parameters():
#     param.requires_grad = True

# # Fine-tuning loop
# num_epochs_fine_tuning = 5  # Change this as needed

# fine_tune_optimizer = optim.SGD(base_model.parameters(), lr=0.0001, momentum=0.9)

# for epoch in range(num_epochs_fine_tuning):
#     base_model.train()
#     running_loss = 0.0

#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         fine_tune_optimizer.zero_grad()
#         outputs = base_model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         fine_tune_optimizer.step()
#         running_loss += loss.item()

#     print(f"Epoch {epoch+1}/{num_epochs_fine_tuning}, Loss: {running_loss/len(train_loader)}")

# # Evaluation on the test set
# base_model.eval()
# correct = 0
# total = 0

# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = base_model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# accuracy = 100 * correct / total
# print(f"Test Accuracy: {accuracy:.2f}%")
