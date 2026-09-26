import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def build_model(num_classes=2, freeze_backbone=True):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Remplace la dernière couche (toujours dynamique, jamais en dur)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def unfreeze_last_block(model):
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

if __name__ == "__main__":
    model = build_model()
    print(model.fc)  # doit afficher : Linear(in_features=512, out_features=2, bias=True)