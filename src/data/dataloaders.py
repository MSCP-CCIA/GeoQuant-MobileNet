import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data.dataset import FaceDataset
import yaml
import os

def get_transforms(config: dict, is_training: bool = True):
    """
    Define el pipeline de transformaciones estandar para ArcFace/InsightFace.
    """
    input_size = tuple(config['data']['input_size'])
    
    if is_training:
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

def create_dataloader(config_path: str, split: str = 'train'):
    """
    Factory function para crear el DataLoader basado en la configuracion YAML.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    is_training = (split == 'train')
    
    # Determinar ruta segun el split (asumiendo carpetas train/val en processed o raw)
    data_root = config['data']['data_root']
    target_dir = os.path.join(data_root, 'processed' if is_training else 'raw', split)
    
    # Fallback si no existe la estructura detallada aun para pruebas iniciales
    if not os.path.exists(target_dir):
        target_dir = data_root 

    transform = get_transforms(config, is_training=is_training)
    
    dataset = FaceDataset(
        root_dir=target_dir,
        transform=transform
    )
    
    batch_size = config['data']['train_batch_size'] if is_training else config['data']['val_batch_size']
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=config['system']['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=is_training # Evita batches incompletos en entrenamiento
    )
    
    return loader
