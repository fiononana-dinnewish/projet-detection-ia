from pathlib import Path
from PIL import Image
import torch
import numpy as np

from src.models.model import build_model
from src.common.preprocessing import preprocess_image as shared_preprocess

# Convention officielle des labels :
# 0 = fake (image générée/retouchée par IA)
# 1 = real (image réelle)


def load_model(checkpoint_path, device="cpu"):
    """Charge le modèle entraîné et son mapping de classes depuis un checkpoint .pth"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(num_classes=2, freeze_backbone=True).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["class_to_idx"]


def predict(image: Image.Image, model, class_to_idx, device="cpu"):
    """
    Retourne (label: str, confidence: float) pour une image PIL donnée.
    Utilise le prétraitement partagé (src.common.preprocessing) pour garantir
    la cohérence entre entraînement et inférence (pas de train/serve skew).
    """
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # 1. Prétraitement partagé -> numpy array (H, W, C), déjà normalisé
    arr = shared_preprocess(image)

    # 2. Conversion en tenseur PyTorch : (H, W, C) -> (C, H, W), puis ajout de la dimension batch
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    label_idx = probs.argmax().item()
    label = idx_to_class[label_idx]
    confidence = float(probs[label_idx])

    return label, confidence


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    checkpoint_path = project_root / "models" / "best_model.pth"

    device = torch.device("cpu")
    model, class_to_idx = load_model(checkpoint_path, device)
    print(f"[INFO] Modèle chargé, mapping : {class_to_idx}")

    test_dir = project_root / "data" / "processed" / "test"
    correct, total = 0, 0

    for class_name in ["real", "fake"]:
        class_dir = test_dir / class_name
        for img_path in list(class_dir.glob("*.*"))[:20]:
            image = Image.open(img_path)
            label, confidence = predict(image, model, class_to_idx, device)
            is_correct = (label == class_name)
            correct += is_correct
            total += 1

    print(f"\n[VÉRIFICATION] Accuracy sur échantillon (40 images) : {correct}/{total} = {correct/total:.2%}")