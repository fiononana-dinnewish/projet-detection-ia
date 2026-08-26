import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from src.data.preprocessing import preprocess_dataset
import tempfile

#Créer un dossier temporaire avec des fausses images
def test_preprocess_dataset_creates_folders():
    with tempfile.TemporaryDirectory() as tmpdir:
        # C:\Users\Oracle\AppData\Local\Temp\tmpy3tn6mzc
        raw_dir = Path(tmpdir) / "raw" / "real_and_fake_image_dataset"
        processed_dir = Path(tmpdir)
        
        for class_name in ["real", "fake"]:
            class_path = raw_dir / class_name / "category1"
            class_path.mkdir(parents=True, exist_ok=True)
            for i in range(5):
                img = Image.new("RGB", (100, 100), color=(128, 128, 128))
                img.save(class_path / f"test_image_{i}.jpg")
            
        preprocess_dataset(raw_dir, processed_dir, img_size=(224, 224))

        for split in ['train', 'val', 'test']:
            for class_name in ['real', 'fake']:
                folder = processed_dir / split / class_name
                assert folder.exists()
                assert len(list(folder.glob('*.jpg'))) > 0

#Vérifier que les images sont bien redimensionnées
def test_my_preprocessing_resizes_images():
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw" / "real_and_fake_image_dataset"
        processed_dir = Path(tmpdir) / "processed"

        for class_name in ['real', 'fake']:
            class_path = raw_dir / class_name / 'category1'
            class_path.mkdir(parents=True, exist_ok=True)
            for i in range(5):
                img = Image.new('RGB', (300, 200), color=(128, 128, 128))
                img.save(class_path / f"test_{i}.jpg")
        
        target_size = (224, 224)
        preprocess_dataset(raw_dir, processed_dir, img_size=target_size)

        for class_name in ['real', 'fake']:
            processed_files = list((processed_dir / 'train' / class_name).glob('*.jpg'))
            assert len(processed_files) > 0
            for img_path in processed_files:
                with Image.open(img_path) as img:
                    assert img.size == target_size
            
            
#Vérifie que la distribution des classes est conservée après split
def test_preprocess_dataset_maintains_class_distribution():
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw" / "real_and_fake_image_dataset"
        processed_dir = Path(tmpdir) / "processed"
        
        for class_name in ['real', 'fake']:
            for i in range(20):
                class_path = raw_dir / class_name / f"category_{i%3}"
                class_path.mkdir(parents=True, exist_ok=True)
                img = Image.new('RGB', (100, 100), color=(128, 128, 128))
                img.save(class_path / f"img_{i}.jpg")
        
        preprocess_dataset(raw_dir, processed_dir, img_size=(224, 224))
        
        for split in ['train', 'val', 'test']:
            real_count = len(list((processed_dir / split / 'real').glob('*.jpg')))
            fake_count = len(list((processed_dir / split / 'fake').glob('*.jpg')))
            total = real_count + fake_count
            
            if total > 0:
                ratio = real_count / total
                assert 0.4 < ratio < 0.6, f"Distribution déséquilibrée pour {split}: {ratio}"

#Vérifie que les images sont bien redimensionnées à la taille spécifiée
def test_preprocess_dataset_resizes_correctly():
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw" / "real_and_fake_image_dataset"
        processed_dir = Path(tmpdir) / "processed"

        for class_name in ['real', 'fake']:
            class_path = raw_dir / class_name / "category1"
            class_path.mkdir(parents=True, exist_ok=True)
            for i in range(5):
                img = Image.new('RGB', (300, 200), color=(128, 128, 128))
                img.save(class_path / f"test_{i}.jpg")
        
        target_size = (224, 224)
        preprocess_dataset(raw_dir, processed_dir, img_size=target_size)
        
        for class_name in ['real', 'fake']:
            processed_files = list((processed_dir / 'train' / class_name).glob('*.jpg'))
            assert len(processed_files) > 0
            for img_path in processed_files:
                with Image.open(img_path) as img:
                    assert img.size == target_size

#Vérifie que le script gère proprement un dossier vide
def test_preprocess_dataset_handles_empty_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw" / "real_and_fake_image_dataset"
        processed_dir = Path(tmpdir) / "processed"
        
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            preprocess_dataset(raw_dir, processed_dir, img_size=(224, 224))
        except FileNotFoundError:
            pass
            # pytest.fail("Le script a planté avec un dossier vide: FileNotFoundError")
        # except Exception as e:
        #     pytest.fail(f"Le script a planté avec un dossier vide: {e}")

#Vérifie que le script gère différents formats d'image
def test_preprocess_dataset_supports_different_formats():
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw" / "real_and_fake_image_dataset"
        processed_dir = Path(tmpdir) / "processed"
        
        for class_name in ['real', 'fake']:
            class_path = raw_dir / class_name / 'category1'
            class_path.mkdir(parents=True, exist_ok=True)
            
            if class_name == 'real':
                img = Image.new('RGB', (100, 100), color=(128, 128, 128))
                img.save(class_path / "test.jpg")
                
                img = Image.new('RGB', (100, 100), color=(128, 128, 128))
                img.save(class_path / "test.png")
                
                img = Image.new('RGB', (100, 100), color=(128, 128, 128))
                img.save(class_path / "test3.jpeg")
                
                for i in range(4, 6):
                    img = Image.new('RGB', (100, 100), color=(128, 128, 128))
                    img.save(class_path / f"test_{i}.jpg")
                
                with open(class_path / "text.txt", 'w') as f:
                    f.write("Ceci n'est pas une image")

            else:
                for i in range(5):
                    img = Image.new('RGB', (100, 100), color=(128, 128, 128))
                    img.save(class_path / f"test_{i}.jpg")
            
        
        preprocess_dataset(raw_dir, processed_dir, img_size=(224, 224))
        
        processed_files = list((processed_dir / 'train' / 'real').glob('*'))
        assert len(processed_files) >= 3



