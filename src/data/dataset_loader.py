from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

if __name__ == "__main__":
    train_data = datasets.ImageFolder("data/processed/train", transform=transform)
    val_data = datasets.ImageFolder("data/processed/val", transform=transform)
    test_data = datasets.ImageFolder("data/processed/test", transform=transform)

    print("Mapping des classes :", train_data.class_to_idx)
    print("Nombre d'images train :", len(train_data))
    print("Nombre d'images val   :", len(val_data))
    print("Nombre d'images test  :", len(test_data))