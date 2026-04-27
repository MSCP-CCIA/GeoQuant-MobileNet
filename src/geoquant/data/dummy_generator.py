"""
Genera imágenes dummy para stress benchmark: originales del CUB-200 + ruido gaussiano.
Las imágenes resultantes se guardan en disco con estructura ImageFolder para reutilización.
"""

import random
from pathlib import Path
from typing import Optional, Union

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from geoquant.utils.logging import get_logger

logger = get_logger(__name__)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _eval_transform_no_norm(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.143), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])


def generate_dummy_dataset(
    config: dict,
    output_dir: Union[str, Path] = "data/dummy",
    split: str = "test",
    sigma: float = 0.05,
    n_images: int = 500,
    force: bool = False,
) -> Path:
    """
    Genera imágenes dummy sumando ruido gaussiano a imágenes reales del CUB-200.

    El ruido se aplica en espacio [0, 1] antes de normalizar y el resultado
    se clipa para mantener valores válidos de píxel. Las imágenes se guardan
    en estructura ImageFolder: output_dir/{class_name}/{idx}_noisy.png

    Args:
        config: Configuración del proyecto (data.raw_dir, data.image_size).
        output_dir: Directorio de salida para las imágenes dummy.
        split: Split del dataset fuente ('train' o 'test').
        sigma: Desviación estándar del ruido gaussiano (espacio [0, 1]).
        n_images: Número de imágenes dummy a generar.
        force: Si True, regenera aunque ya existan imágenes previas.

    Returns:
        Path al directorio con las imágenes dummy.
    """
    out_path = Path(output_dir)

    if out_path.exists() and not force and any(out_path.rglob("*.png")):
        logger.info(f"Imágenes dummy ya existen en {out_path}. Usa --regenerate para forzar.")
        return out_path

    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    image_size = data_cfg.get("image_size", 224)

    split_path = raw_dir / split
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split '{split}' no encontrado en {split_path}. "
            "Ejecuta 'python data/download_cub200.py' para preparar el dataset."
        )

    transform = _eval_transform_no_norm(image_size)
    source_dataset = torchvision.datasets.ImageFolder(root=str(split_path), transform=transform)

    n_images = min(n_images, len(source_dataset))
    indices = random.sample(range(len(source_dataset)), n_images)
    class_names = source_dataset.classes

    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generando {n_images} imágenes dummy (sigma={sigma}) desde {split_path}...")

    for i, idx in enumerate(indices):
        tensor, label = source_dataset[idx]

        noise = torch.randn_like(tensor) * sigma
        noisy = (tensor + noise).clamp(0.0, 1.0)

        class_dir = out_path / class_names[label]
        class_dir.mkdir(exist_ok=True)

        torchvision.utils.save_image(noisy, class_dir / f"{idx:06d}_noisy.png")

        if (i + 1) % 100 == 0:
            logger.info(f"  {i + 1}/{n_images} imágenes generadas")

    logger.info(f"Generación completada: {n_images} imágenes en {out_path}")
    return out_path


def get_dummy_loader(
    dummy_dir: Union[str, Path],
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    """
    DataLoader sobre las imágenes dummy con pipeline de eval completo (con normalización).

    Args:
        dummy_dir: Directorio con estructura ImageFolder de las dummy.
        image_size: Resolución de la imagen (debe coincidir con la generación).
        batch_size: Batch size para inferencia.
        num_workers: Workers del DataLoader (0 para CPU-only sin subprocesos).

    Returns:
        DataLoader listo para inferencia.
    """
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.143), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    dataset = torchvision.datasets.ImageFolder(root=str(dummy_dir), transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
