from pathlib import Path
from PIL import Image
import torch

from src.models.inference import load_model, predict


def run_robustness_tests(edge_cases_dir, model, class_to_idx, device="cpu"):
    edge_cases_dir = Path(edge_cases_dir)
    image_paths = [p for p in edge_cases_dir.glob("*.*")
                   if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]

    if not image_paths:
        print(f"[ATTENTION] Aucune image trouvée dans {edge_cases_dir}")
        print("Ajoutez quelques images hors contexte (memes, captures d'écran, etc.)")
        return

    print(f"\n{'Image':<30} {'Prédiction':<10} {'Confiance':<10}")
    print("-" * 50)

    for path in image_paths:
        try:
            image = Image.open(path)
            label, confidence = predict(image, model, class_to_idx, device)
            flag = "  ⚠️ Très confiant sur image hors contexte" if confidence > 0.95 else ""
            print(f"{path.name:<30} {label:<10} {confidence:.2%}{flag}")
        except Exception as e:
            print(f"{path.name:<30} ERREUR : {e}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    checkpoint_path = project_root / "models" / "best_model.pth"
    edge_cases_dir = project_root / "tests" / "edge_cases"

    device = torch.device("cpu")
    model, class_to_idx = load_model(checkpoint_path, device)

    run_robustness_tests(edge_cases_dir, model, class_to_idx, device)