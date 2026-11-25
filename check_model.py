from pathlib import Path
import torch

MODEL_PATH = Path("models/best_model.pth")

print("Fichier trouvé :", MODEL_PATH.exists())
print("Chemin absolu :", MODEL_PATH.resolve())

try:
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    print("\nType :", type(checkpoint))

    if isinstance(checkpoint, dict):
        print("Clés :", checkpoint.keys())
        print("Classes :", checkpoint.get("class_names"))

except Exception as e:
    print("\nErreur :", repr(e))