"""
CUBDataset: wrapper sobre CUB-200-2011 para análisis geométrico de embeddings.
Usa ImageFolder con splits train/test pre-organizados en data/raw/cub200/.
"""

from pathlib import Path
from typing import Optional

import torchvision
from torch.utils.data import DataLoader, Dataset

from geoquant.data.transforms import get_transforms
from geoquant.utils.logging import get_logger

logger = get_logger(__name__)


class CUBDataset:
    """
    Factory para el dataset CUB-200-2011.
    Devuelve Dataset de PyTorch listos para usar en DataLoader.
    """

    def __init__(self, root: str, image_size: int = 224):
        self.root = Path(root)
        self.image_size = image_size

    def get_split(self, split: str = "train") -> Dataset:
        """
        Args:
            split: 'train' o 'test'
        Returns:
            Dataset de PyTorch con transformaciones aplicadas.
        """
        split_path = self.root / split
        if not split_path.exists():
            raise FileNotFoundError(
                f"Split '{split}' no encontrado en {split_path}. "
                "Ejecuta 'python data/download_cub200.py' para preparar el dataset."
            )

        is_training = split == "train"
        transform = get_transforms({"image_size": self.image_size}, is_training=is_training)

        logger.info(f"Cargando CUB-200 [{split}] desde {split_path}")
        return torchvision.datasets.ImageFolder(root=str(split_path), transform=transform)

    def get_loaders(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """Devuelve (train_loader, val_loader)."""
        train_ds = self.get_split("train")
        val_ds = self.get_split("test")

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader


def get_dataloaders(config: dict):
    """
    Punto de entrada funcional. Recibe el dict de configuración completo.
    """
    data_cfg = config["data"]
    hw_cfg = config.get("environment", {}).get("hardware", {})

    dataset = CUBDataset(
        root=data_cfg["raw_dir"],
        image_size=data_cfg.get("image_size", 224),
    )
    return dataset.get_loaders(
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", hw_cfg.get("num_workers", 4)),
        pin_memory=data_cfg.get("pin_memory", True),
    )