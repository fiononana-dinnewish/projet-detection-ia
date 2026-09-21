from evidently import Report
from evidently.presets import DataDriftPreset
from PIL import Image
import numpy as np
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
import random
random.seed(42)

def extract_image_features(image_path: Path) -> dict:
    try:
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            arr = np.asarray(img_rgb)
            width, height = img.size
            file_size_kb = os.path.getsize(image_path) / 1024
            
            brightness = arr.mean()
            
            grayscale = np.asarray(img.convert("L"), dtype=np.float32)
            sharpness = np.var(np.gradient(grayscale))
            return {
                "width": width,
                "height": height,
                "aspect_ratio": width / height,
                "file_size_kb": file_size_kb,
                "brightness": brightness,
                "sharpness": sharpness,
            }
    except Exception as e:
        print(f"[WARNING] Impossible de traiter {image_path}: {e}")
        return None
    
"""Construit un DataFrame de features, une ligne par image traitée avec succès."""
def build_features_dataframe(image_paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in image_paths:
        features = extract_image_features(path)
        if features is not None:
            rows.append(features)
    return pd.DataFrame(rows)

def sample_balanced(root_dir: Path, n_per_class: int = 100) -> list[Path]:
    paths = []
    for cls in ["real", "fake"]:
        cls_paths = [p for p in (root_dir / cls).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        random.shuffle(cls_paths)
        paths.extend(cls_paths[:n_per_class])
    return paths

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]

reference_paths = sample_balanced(PROJECT_ROOT / "data/processed/train", n_per_class=100)

current_paths = sample_balanced(PROJECT_ROOT / "data/processed/val", n_per_class=200)

reference_data = build_features_dataframe(reference_paths)
current_data = build_features_dataframe(current_paths)

report = Report(metrics=[DataDriftPreset()])

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
my_eval = report.run(reference_data=reference_data, current_data=current_data)
os.makedirs("reports", exist_ok=True)
my_eval.save_html(f"reports/drift_{timestamp}.html")

print(f"Rapport généré : monitoring/reports/drift_{timestamp}.html")