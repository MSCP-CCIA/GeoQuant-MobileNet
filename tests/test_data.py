import torch
import os
import shutil
from PIL import Image
import numpy as np
from src.data.dataloaders import create_dataloader

def setup_dummy_data():
    """Crea una estructura de carpetas temporal con imagenes falsas para testear."""
    test_dir = "./data/test_dummy"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Crear 2 clases con 2 imagenes cada una
    for i in range(2):
        class_dir = os.path.join(test_dir, f"person_{i}")
        os.makedirs(class_dir)
        for j in range(2):
            img_data = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
            img = Image.fromarray(img_data)
            img.save(os.path.join(class_dir, f"img_{j}.jpg"))
    return test_dir

def test_dataloader_loading():
    print("--- Iniciando Test de DataLoader ---")
    dummy_path = setup_dummy_data()
    
    # Mock de config temporal para el test
    config_content = f"""
system:
  num_workers: 0 # 0 para evitar problemas de multiprocessing en tests rapidos
data:
  data_root: "{dummy_path}"
  train_batch_size: 2
  val_batch_size: 2
  input_size: [112, 112]
"""
    with open("configs/test_config.yaml", "w") as f:
        f.write(config_content)
    
    try:
        loader = create_dataloader("configs/test_config.yaml", split='train')
        
        # Verificar longitud
        print(f"Total imagenes encontradas: {len(loader.dataset)}")
        assert len(loader.dataset) == 4
        
        # Probar iteracion
        for images, labels in loader:
            print(f"Batch Shape: {images.shape}") # Deberia ser [2, 3, 112, 112]
            print(f"Labels: {labels}")
            assert images.shape == (2, 3, 112, 112)
            break
            
        print("¡Test de DataLoader exitoso!")
        
    finally:
        # Limpieza
        if os.path.exists(dummy_path):
            shutil.rmtree(dummy_path)
        if os.path.exists("configs/test_config.yaml"):
            os.remove("configs/test_config.yaml")

if __name__ == "__main__":
    test_dataloader_loading()
