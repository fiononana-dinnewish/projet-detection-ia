import os
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image

from src.common.preprocessing import preprocess_image
from src.models.architecture import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "best_model.pth"))


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    model = build_model(num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_names = checkpoint.get("class_names")
    if not class_names:
        class_to_idx = checkpoint.get("class_to_idx")
        if class_to_idx:
            class_names = [n for n, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
        else:
            class_names = ["fake", "real"]
    return model, class_names


def predict_image(image: Image.Image) -> tuple[str, float]:
    model, class_names = get_model()

    arr = preprocess_image(image)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    idx = int(probs.argmax())
    return class_names[idx], float(probs[idx])