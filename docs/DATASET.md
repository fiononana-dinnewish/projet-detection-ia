# Fiche descriptive du dataset

**Nom :** Real vs Fake Image Dataset (Kaggle) 
**Licence :** Usage académique uniquement  
**Version DVC :** data-v1

## Volumétrie
- Total images : 995
- Répartition : 75% real / 25% fake

## Prétraitement
- Redimensionnement : 224x224
- Format : JPEG
- Split : 70% train / 15% val / 15% test
- Stratification : par classe et catégorie

## Biais connus
- Images majoritairement de haute résolution
- Possibles différences de compression selon la source

## Reproductibilité
```bash
dvc pull
python src/data/preprocessing.py