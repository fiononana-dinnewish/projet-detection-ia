import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

def preprocess_dataset(dataset_path, processed_path, img_size=(224, 224), train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    
    data = []  # Liste de tuples (chemin_image, classe, categorie)
    for class_name in ['real', 'fake']:
        class_dir = Path(dataset_path) / class_name
        for category in os.listdir(class_dir):
            category_dir = class_dir / category
            if not category_dir.is_dir():
                continue
            for img_file in category_dir.glob('*.*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    data.append((str(img_file), class_name, category))

    print(f"Total images trouvées : {len(data)}")
    if not data:
        print("Aucune image trouvée. Vérifiez le chemin.")
        return

    # 2. Split stratifié par (classe, catégorie)
    # Créer une clé de stratification
    stratify_keys = [f"{cls}_{cat}" for _, cls, cat in data]
    
    # Premier split: train vs (val+test)
    train_idx, temp_idx = train_test_split(
        range(len(data)), 
        test_size=(1- train_ratio), 
        random_state=42, 
        stratify=stratify_keys
    )
    
    # Second split: val vs test
    temp_data = [data[i] for i in temp_idx]
    temp_keys = [stratify_keys[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        range(len(temp_data)), 
        test_size=test_ratio/(val_ratio+test_ratio), 
        random_state=42, 
        stratify=temp_keys
    )

    # Construire les ensembles finaux
    train_set = [data[i] for i in train_idx]
    val_set = [temp_data[i] for i in val_idx]
    test_set = [temp_data[i] for i in test_idx]

    # Traiter et sauvegarder les images
    for split_name, split_data in [('train', train_set), ('val', val_set), ('test', test_set)]:
        for img_path, class_name, category in split_data:
            
            # Définir le chemin de sortie
            out_dir = Path(processed_path) / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Générer un nom de fichier unique pour éviter les collisions
            img_filename = Path(img_path).name
            # Optionnel : ajouter la catégorie au nom pour traçabilité
            new_filename = f"{category}_{img_filename}"
            out_path = out_dir / new_filename
            
            try:
                # Ouvrir, redimensionner et sauvegarder
                with Image.open(img_path) as img:
                    # Convertir en RGB si nécessaire (ex: pour les PNG en niveaux de gris)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Redimensionner
                    img_resized = img.resize(img_size, Image.Resampling.LANCZOS)
                    # Sauvegarder
                    img_resized.save(out_path, 'JPEG', quality=90)
            except Exception as e:
                print(f"Erreur avec l'image {img_path}: {e}")

    print(f"Prétraitement terminé. Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

if __name__ == "__main__":
    dataset_path = Path(os.getcwd()).parent.parent / "data" / "raw" / "real_and_fake_image_dataset"
    processed_path = Path(os.getcwd()).parent.parent / "data" / "processed"
    preprocess_dataset(dataset_path, processed_path)