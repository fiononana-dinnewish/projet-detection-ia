from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if __name__ == "__main__":
    train_data = datasets.ImageFolder("data/processed/train", transform=transform)
    val_data = datasets.ImageFolder("data/processed/val", transform=transform)
    test_data = datasets.ImageFolder("data/processed/test", transform=transform)

    print("Mapping des classes :", train_data.class_to_idx)
    print("Nombre d'images train :", len(train_data))
    print("Nombre d'images val   :", len(val_data))
    print("Nombre d'images test  :", len(test_data))