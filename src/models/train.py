import torch
import torch.nn as nn
import torch.optim as optim
from src.data.dataset_loader import get_dataloaders
from src.models.model import build_model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


if __name__ == "__main__":
    device = torch.device("cpu")  # votre PC : pas de GPU
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(batch_size=16)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)  # on entraîne seulement la dernière couche

    epochs = 2  # juste pour tester que ça tourne, on augmentera plus tard
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} - val_accuracy: {val_acc:.4f}")