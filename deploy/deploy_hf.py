import os
from pathlib import Path
from huggingface_hub import HfApi

HF_USER = "dinnewish"          # à remplacer
BACKEND = f"{HF_USER}/detection-ia-backend"
FRONTEND = f"{HF_USER}/detection-ia-frontend"
ROOT = Path(__file__).resolve().parents[1]

api = HfApi(token=os.environ["HF_TOKEN"])

# ---------- Backend (Docker) ----------
api.create_repo(BACKEND, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(
    folder_path=ROOT, repo_id=BACKEND, repo_type="space",
    allow_patterns=["api/**", "src/**", "models/best_model.pth"],
    ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
)
for name in ["Dockerfile", "README.md"]:
    api.upload_file(
        path_or_fileobj=str(ROOT / "deploy" / "backend" / name),
        path_in_repo=name, repo_id=BACKEND, repo_type="space",
    )

# ---------- Frontend (Gradio) ----------
api.create_repo(FRONTEND, repo_type="space", space_sdk="gradio", exist_ok=True)
api.add_space_variable(
    FRONTEND, "API_URL",
    f"https://{HF_USER.lower()}-detection-ia-backend.hf.space",
)
api.upload_folder(
    folder_path=ROOT / "frontend", repo_id=FRONTEND, repo_type="space",
    ignore_patterns=["**/__pycache__/**"],
)

def app_url(repo_id: str) -> str:
    user, name = repo_id.split("/")
    return f"https://{user.lower()}-{name.lower()}.hf.space"

print("\n=== Déploiement lancé ===")
print(f"Backend  (page du Space) : https://huggingface.co/spaces/{BACKEND}")
print(f"Backend  (API directe)   : {app_url(BACKEND)}/docs")
print(f"Frontend (page du Space) : https://huggingface.co/spaces/{FRONTEND}")
print(f"Frontend (application)   : {app_url(FRONTEND)}")
print("Le build prend quelques minutes : suivez-le dans l'onglet Logs de chaque Space.")