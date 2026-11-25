import os
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image

from src.common.preprocessing import preprocess_image
from src.models.architecture import build_model  # TODO: vérifier le nom exact avec Membre 2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "best_model.pth"))


@lru_cache(maxsize=1)
def get_model():
    """Charge le modèle UNE seule fois, puis le réutilise à chaque requête."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    model = build_model(num_classes=2)  # TODO: mêmes arguments que lors de l'entraînement
    model.load_state_dict(checkpoint["model_state_dict"])  # TODO: clé exacte du checkpoint
    model.eval()

    class_names = checkpoint.get("class_names")  # TODO: ordre exact
    return model, class_names


def predict_image(image: Image.Image) -> tuple[str, float]:
    model, class_names = get_model()

    arr = preprocess_image(image)  # le MÊME prétraitement que l'entraînement
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    idx = int(probs.argmax())
    return class_names[idx], float(probs[idx])