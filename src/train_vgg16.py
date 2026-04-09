import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_vgg16_classifier(num_classes):
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model


def train_model(model, train_loader, device=None, epochs=5, lr=0.001, momentum=0.9):
    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / max(len(train_loader), 1)
        epoch_acc = correct / max(total, 1)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

    return model
