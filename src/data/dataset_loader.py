from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_dataloaders(batch_size=16):
    train_data = datasets.ImageFolder("data/processed/train", transform=transform)
    val_data = datasets.ImageFolder("data/processed/val", transform=transform)
    test_data = datasets.ImageFolder("data/processed/test", transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_data.class_to_idx

if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders()
    print("Mapping des classes :", class_to_idx)

    # Test rapide : récupérer un seul batch pour vérifier que tout fonctionne
    images, labels = next(iter(train_loader))
    print("Forme d'un batch d'images :", images.shape)  # attendu : [16, 3, 224, 224]
    print("Labels du batch :", labels)