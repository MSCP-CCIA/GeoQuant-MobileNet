"""
Pipelines de transformación para experimentos GeoQuant.
Optimizados para análisis de embeddings de grano fino (CUB-200) y estabilidad INT8.
"""

from typing import Any, Dict, List

from torchvision import transforms
from torchvision.transforms import InterpolationMode


class TransformFactory:
    """Pipelines de transformación estandarizados para MobileNetV3."""

    IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
    IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

    @classmethod
    def get_pipeline(
        cls,
        image_size: int = 224,
        is_training: bool = False,
    ) -> transforms.Compose:
        """
        Construye el pipeline según la fase.

        Args:
            image_size: Resolución objetiva (224 para CUB-200).
            is_training: Si True, aplica augmentation para robustez QAT.
        """
        normalization = transforms.Normalize(
            mean=cls.IMAGENET_MEAN,
            std=cls.IMAGENET_STD,
        )

        if is_training:
            return transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.5, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalization,
            ])

        # Eval: escala estándar 1.143 antes del CenterCrop
        return transforms.Compose([
            transforms.Resize(
                int(image_size * 1.143),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalization,
        ])


def get_transforms(config: Dict[str, Any], is_training: bool = False) -> transforms.Compose:
    """Punto de entrada funcional."""
    return TransformFactory.get_pipeline(
        image_size=config.get("image_size", 224),
        is_training=is_training,
    )