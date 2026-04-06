"""
Data loading orchestration for the GeoQuant-MobileNet repository.
Implements a factory pattern to decouple dataset instantiation from training logic.
Exclusively focused on the CUB-200-2011 dataset for fine-grained geometric analysis.
"""

from pathlib import Path
from typing import Tuple, Dict, Any

import torchvision
from torch.utils.data import DataLoader, Dataset
from src.utils.logger import logger
from data.transforms import get_transforms


class DataLoaderFactory:
    """
    Handles the instantiation of Datasets and DataLoaders using centralized config.
    """

    @staticmethod
    def get_dataset(
            dataset_name: str,
            data_path: str,
            is_training: bool,
            image_size: int
    ) -> Dataset:
        """
        Maps configuration names to PyTorch Dataset objects.
        Uses ImageFolder assuming data is pre-organized into train/test directories.
        """
        transform = get_transforms({"image_size": image_size}, is_training=is_training)
        normalized_name = dataset_name.lower()

        if "cub200" in normalized_name:
            # Apunta a la subcarpeta correcta dependiendo de la fase
            split_folder = 'train' if is_training else 'test'

            # Resolución absoluta de rutas para evitar problemas con el IDE/Terminal
            project_root = Path(__file__).resolve().parents[1]

            # Limpiamos el "./" o ".\" del inicio si existe en el YAML (ej. "./data/raw/...")
            clean_data_path = data_path.lstrip("./\\")

            split_path = project_root / clean_data_path / split_folder

            # Validación crítica: Si no existe, lanza error detallado con la ruta absoluta
            if not split_path.exists():
                logger.error(f"Dataset path not found: {split_path}")
                raise FileNotFoundError(
                    f"CUB-200 split not found at {split_path}. "
                    "Please run 'python data/download_cub200.py' first to format the dataset."
                )

            return torchvision.datasets.ImageFolder(
                root=str(split_path),  # Convertimos el objeto Path a string para PyTorch
                transform=transform
            )

        raise ValueError(f"Dataset '{dataset_name}' is not supported in the GeoQuant pipeline.")

    @classmethod
    def create_loader(
            cls,
            loader_config: Dict[str, Any],
            hardware_config: Dict[str, Any],
            is_training: bool
    ) -> DataLoader:
        """
        Creates a high-performance DataLoader with pinned memory and multi-process loading.
        """
        dataset = cls.get_dataset(
            dataset_name=loader_config['name'],
            data_path=loader_config['path'],
            is_training=is_training,
            image_size=loader_config.get('image_size', 224)  # Ajustado a 224 para CUB-200
        )

        return DataLoader(
            dataset,
            batch_size=loader_config.get('batch_size', 32),  # Ajustado a 32 para evitar OOM
            shuffle=is_training,
            num_workers=hardware_config.get('num_workers', 4),
            pin_memory=hardware_config.get('pin_memory', True)
        )


def get_project_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Orchestrates the creation of train and validation loaders from global config.

    Args:
        config: Full configuration dictionary loaded from base_config.yaml.
    """
    env_cfg = config.get("environment", {}).get("hardware", {})
    data_cfg = config.get("data", {}).get("loader_params", {})

    if not data_cfg:
        logger.error("Configuration error: 'data.loader_params' section is missing.")
        raise KeyError("Check your base_config.yaml structure.")

    logger.info(f"Initializing GeoQuant loaders for: {data_cfg.get('name')}")

    train_loader = DataLoaderFactory.create_loader(data_cfg, env_cfg, is_training=True)
    val_loader = DataLoaderFactory.create_loader(data_cfg, env_cfg, is_training=False)

    return train_loader, val_loader