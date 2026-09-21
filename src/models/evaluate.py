from pathlib import Path
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.data.dataset_loader import get_dataloaders
from src.models.model import build_model


def evaluate_full(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # proba classe "real" (index 1)
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=["fake", "real"]))
    print("Matrice de confusion :\n", confusion_matrix(all_labels, all_preds))
    print("AUC :", roc_auc_score(all_labels, all_probs))


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    checkpoint_path = project_root / "models" / "best_model.pth"

    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(num_classes=2, freeze_backbone=True).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"[INFO] Modèle chargé (val_accuracy au moment de la sauvegarde : {checkpoint['val_accuracy']:.4f})")
    print(f"[INFO] Mapping des classes : {checkpoint['class_to_idx']}")

    _, _, test_loader, _ = get_dataloaders(batch_size=16)
    evaluate_full(model, test_loader, device)