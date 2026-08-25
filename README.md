Dataset utiliser au premier 
745 real (75%) vs 250 fake (25%) 

## Backend API - Installation

### Prérequis

- Python 3.11 recommandé
- pip
- Git

### Installation des dépendances

```bash
python -m pip install --upgrade pip
pip install -r api/requirements-api.txt
```

Les principales dépendances du backend sont :

```bash
pip install fastapi
pip install "uvicorn[standard]"
pip install python-multipart
pip install pillow
pip install httpx
pip install pytest
pip install torch
pip install torchvision
```

### Lancer le backend

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

```text
http://localhost:8000
```

Depuis la racine du projet :

```bash
pytest -v
```

### Vérification rapide

L'endpoint de contrôle est :

```text
GET /
```

L'endpoint de prédiction est :

```text
POST /predict
```

Les images prises en charge par le backend sont JPEG/JPG et PNG.
