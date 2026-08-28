from PIL import Image
import os
from pathlib import Path

def explore_dataset(dataset_path):
    stats = {}
    for class_name in ["real", "fake"]:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            print(f"Dossier {class_name} non trouvé")
            continue
        print(f"Classe :{class_name}")
        
        valid, invalid = 0, 0
        total_images = 0
        
        for category in os.listdir(class_path):
            category_path = os.path.join(class_path, category)
            if not os.path.isdir(category_path):
                print(f"Dossier {category} non trouvé")
                continue
            
            print(f"Catégorie : {category}")
            
            images = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
            
            for f in images:
                img_path = os.path.join(category_path, f)
                
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                    valid += 1
                except Exception as e:
                    invalid += 1
                    print(f"[INVALID] {img_path} -> {e}")
            total_images += len(images)
        stats[class_name] = {"valid": valid, "invalid": invalid}
        print(f"Total des images: {total_images}")
    return stats

if __name__ == "__main__":
    dataset_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "real_and_fake_image_dataset"
    print(f"Chemin du dataset : {dataset_path}")
    print(explore_dataset(dataset_path))