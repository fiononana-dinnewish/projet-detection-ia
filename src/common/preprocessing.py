from PIL import Image
import numpy as np
from pathlib import Path

IMG_SIZE = (224, 224)

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

"""Garantit que le modèle voit exactement le même prétraitement, peu importe
    qu'il soit appelé pendant l'entraînement ou pendant une prédiction en production."""
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")

    image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)

    arr = np.asarray(image).astype("float32") / 255.0

    arr = (arr - MEAN) / STD

    return arr

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    for split in ["train", "val", "test"]:
        data_dir = PROJECT_ROOT / "data" / "processed" / split
        image_paths = [p for p in data_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]

        n_success, n_failed = 0, 0
        shapes_seen, formats_seen = set(), set()

        for path in image_paths:
            try:
                img = Image.open(path)
                formats_seen.add(path.suffix.lower())
                result = preprocess_image(img)
                shapes_seen.add(result.shape)
                n_success += 1
            except Exception as e:
                n_failed += 1
                print(f"[ÉCHEC] {path} : {e}")

        print(f"\n--- {split} ---")
        print(f"Images trouvées : {len(image_paths)} | Réussies : {n_success} | Échouées : {n_failed}")
        print(f"Formats : {formats_seen} | Shapes : {shapes_seen}")
        if n_success > 0:
            print(f"Exemple min/max : {result.min():.2f} / {result.max():.2f}")