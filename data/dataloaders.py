from torchvision import datasets
from torch.utils.data import DataLoader
from data.transforms import get_transforms
import yaml
import os

def get_dataloader(dataset_name, batch_size=32, is_training=False):
    """
    Cargador universal optimizado para el proyecto (112x112, 16GB RAM).
    
    Args:
        dataset_name (str): "imagenet-1k-subset" o "cifar100"
        batch_size (int): Tamaño del lote para control de VRAM/RAM
        is_training (bool): Si es True, activa el modo entrenamiento para QAT
    """
    with open("configs/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Buscar configuración específica del dataset
    ds_cfg = next(ds for ds in config['datasets'] if ds['name'] == dataset_name)
    
    # Obtener el pipeline de 112x112 y normalización correspondiente
    transform = get_transforms(
        dataset_name=("cifar100" if "cifar" in dataset_name else "imagenet"),
        is_training=is_training
    )

    # 1. Cargar el Dataset correspondiente
    if "cifar10" in dataset_name:
        dataset = datasets.CIFAR10(
            root=ds_cfg['path'], 
            train=is_training, 
            download=False, 
            transform=transform
        )
    else: # ImageNet (asumido estructura ImageFolder)
        dataset = datasets.ImageFolder(
            root=os.path.join(ds_cfg['path'], "train" if is_training else "val"),
            transform=transform
        )

    # 2. Configurar el DataLoader con parámetros de eficiencia
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=4,
        pin_memory=True
    )
