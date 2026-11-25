
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(
    num_classes: int = 2,
    freeze_backbone: bool = True, 
    pretrained: bool = True
):
    """
    Construit un modèle ResNet18 pour la classification binaire.

    Args:
        num_classes: Nombre de classes.
        freeze_backbone: Gèle les couches du backbone.
        pretrained: Utilise les poids pré-entraînés.

    Returns:
        Un modèle ResNet18 adapté à la classification.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model
