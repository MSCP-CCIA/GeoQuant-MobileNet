"""
Data loading orchestration for the GeoQuant-MobileNet repository.
Implements a factory pattern to decouple dataset instantiation from training logic.
"""

import os
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
        """
        transform = get_transforms({"image_size": image_size}, is_training=is_training)

        normalized_name = dataset_name.lower()

        if "imagenette" in normalized_name:
            split = 'train' if is_training else 'val'
            return torchvision.datasets.Imagenette(
                root=data_path,
                split=split,
                download=True,
                transform=transform
            )

        # Extensibility point for future biometric datasets (LFW, etc.)
        raise ValueError(f"Dataset '{dataset_name}' is not currently supported in the GeoQuant pipeline.")

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
            image_size=loader_config.get('image_size', 112)
        )

        return DataLoader(
            dataset,
            batch_size=loader_config.get('batch_size', 64),
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
    # Navigating the professional YAML hierarchy
    env_cfg = config.get("environment", {}).get("hardware", {})
    data_cfg = config.get("data", {}).get("loader_params", {})

    if not data_cfg:
        logger.error("Configuration error: 'data.loader_params' section is missing.")
        raise KeyError("Check your base_config.yaml structure.")

    logger.info(f"Initializing GeoQuant loaders for: {data_cfg.get('name')}")

    train_loader = DataLoaderFactory.create_loader(data_cfg, env_cfg, is_training=True)
    val_loader = DataLoaderFactory.create_loader(data_cfg, env_cfg, is_training=False)

    return train_loader, val_loader