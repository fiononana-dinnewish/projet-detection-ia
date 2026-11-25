import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo = "dinnewish/test-zerogpu"

api.create_repo(repo, repo_type="space", space_sdk="gradio",
                space_hardware="zero-a10g", exist_ok=True)
print("OK : votre compte peut créer un Space Gradio gratuit")
api.delete_repo(repo, repo_type="space")   # on supprime le Space de test