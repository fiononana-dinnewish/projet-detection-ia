import argparse
from pathlib import Path
 
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
 
from src.data.dataset_loader import get_dataloaders
from src.models.model import build_model,unfreeze_last_block 
 
 
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
 
 
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0
 
 
def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement du modèle real/fake")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Taille des batchs")
    parser.add_argument("--epochs", type=int, default=3, help="Nombre d'epochs")
    parser.add_argument("--run-name", type=str, default="sanity_check", help="Nom du run MLflow")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        default=True,
        help="Geler le backbone ResNet18 (n'entraîner que la dernière couche)",
    )
    parser.add_argument(
        "--unfreeze-last-block",
        action="store_true",
        help="Dégeler layer4 + fc pour fine-tuning (au lieu de juste fc)",
    )
    return parser.parse_args()
 
 
def main():
    args = parse_args()
 
    # --- Config fixe et explicite pour MLflow (évite les surprises de config externe) ---
    project_root = Path(__file__).resolve().parents[2]
    tracking_uri = f"sqlite:///{project_root / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("detection-real-fake")  # créée automatiquement si elle n'existe pas
 
    print(f"[INFO] Tracking URI  : {mlflow.get_tracking_uri()}")
    print(f"[INFO] Expérience    : detection-real-fake")
 
    device = torch.device("cpu")
    print(f"[INFO] Device utilisé : {device}")
 
    # --- Données ---
    try:
        train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
            batch_size=args.batch_size
        )
    except Exception as e:
        print(f"[ERREUR] Impossible de charger les données : {e}")
        print("Vérifiez que data/processed/train, val, test existent (voir preprocessing.py).")
        return
 
    print(f"[INFO] Mapping des classes : {class_to_idx}")
    print(f"[INFO] Batchs train/val/test : {len(train_loader)}/{len(val_loader)}/{len(test_loader)}")
 
    # --- Modèle ---
    model = build_model(num_classes=2, freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Si l'option --unfreeze-last-block est utilisée, on dégèle les derniers blocs
    model = build_model(num_classes=2, freeze_backbone=args.freeze_backbone).to(device)
    if args.unfreeze_last_block:
        model = unfreeze_last_block(model)
        print("[INFO] Fine-tuning activé : layer4 + fc dégelés")
 
    # On n'entraîne que les paramètres qui ont requires_grad=True
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    
    # --- Entraînement + tracking MLflow ---
    best_val_acc = 0.0
    checkpoints_dir = project_root / "models"
    checkpoints_dir.mkdir(exist_ok=True)
 
    try:
        with mlflow.start_run(run_name=args.run_name) as run:
            mlflow.log_params(
                {
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "freeze_backbone": args.freeze_backbone,
                    "architecture": "resnet18",
                }
            )
 
            for epoch in range(args.epochs):
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_acc = evaluate(model, val_loader, device)
 
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
 
                print(
                    f"Epoch {epoch + 1}/{args.epochs} - "
                    f"train_loss: {train_loss:.4f} - val_accuracy: {val_acc:.4f}"
                )
 
                # Sauvegarde du meilleur modèle uniquement
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    checkpoint_path = checkpoints_dir / "best_model.pth"
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "class_to_idx": class_to_idx,
                            "val_accuracy": val_acc,
                            "epoch": epoch,
                        },
                        checkpoint_path,
                    )
                    print(f"  -> Nouveau meilleur modèle sauvegardé ({val_acc:.4f})")
 
            mlflow.log_metric("best_val_accuracy", best_val_acc)
 
            print(f"\n[INFO] Run ID        : {run.info.run_id}")
            print(f"[INFO] Experiment ID  : {run.info.experiment_id}")
            print(f"[INFO] Meilleure val_accuracy : {best_val_acc:.4f}")
            print(f"[INFO] Checkpoint sauvegardé  : {checkpoints_dir / 'best_model.pth'}")
 
    except Exception as e:
        print(f"[ERREUR] L'entraînement a échoué : {e}")
        raise
 
 
if __name__ == "__main__":
    main()