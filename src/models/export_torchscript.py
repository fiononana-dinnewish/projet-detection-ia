from pathlib import Path
import torch

from src.models.model import build_model


def export_to_torchscript(checkpoint_path, output_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(num_classes=2, freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scripted_model = torch.jit.script(model)
    scripted_model.save(str(output_path))

    print(f"[INFO] Modèle TorchScript sauvegardé : {output_path}")
    return scripted_model


def verify_scripted_model(scripted_model, original_checkpoint_path, device="cpu"):
    """Vérifie que le modèle scripté donne les mêmes résultats que l'original."""
    checkpoint = torch.load(original_checkpoint_path, map_location=device)
    original_model = build_model(num_classes=2, freeze_backbone=True)
    original_model.load_state_dict(checkpoint["model_state_dict"])
    original_model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        original_output = original_model(dummy_input)
        scripted_output = scripted_model(dummy_input)

    diff = (original_output - scripted_output).abs().max().item()
    print(f"[VÉRIFICATION] Écart max entre modèle original et scripté : {diff:.8f}")

    if diff < 1e-5:
        print("[VÉRIFICATION] OK — les deux modèles produisent des résultats identiques.")
    else:
        print("[ATTENTION] Écart non négligeable détecté, à investiguer.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    checkpoint_path = project_root / "models" / "best_model.pth"
    output_path = project_root / "models" / "best_model_scripted.pt"

    scripted_model = export_to_torchscript(checkpoint_path, output_path)
    verify_scripted_model(scripted_model, checkpoint_path)