"""
Transformation pipelines for GeoQuant experiments.
Optimized for embedding analysis and INT8 quantization stability.
"""

from typing import Dict, Any, List
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class TransformFactory:
    """
    Service to generate standardized image transformation pipelines.
    Designed for MobileNetV3 architectures and geometric evaluation.
    """

    # ImageNet normalization constants
    IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
    IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

    @classmethod
    def get_pipeline(
        cls,
        image_size: int = 112,
        is_training: bool = False
    ) -> transforms.Compose:
        """
        Builds a transformation pipeline based on geometric requirements.

        Args:
            image_size: Target resolution (standard 112 for biometrics).
            is_training: If True, applies data augmentation for QAT robustness.
        """
        normalization = transforms.Normalize(
            mean=cls.IMAGENET_MEAN,
            std=cls.IMAGENET_STD
        )

        if is_training:
            return transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size,
                    interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalization,
            ])

        # Validation/Geometric Inference pipeline
        return transforms.Compose([
            transforms.Resize(
                int(image_size * 1.14), # Maintaining aspect ratio before center crop
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalization,
        ])


def get_transforms(config: Dict[str, Any], is_training: bool = False) -> transforms.Compose:
    """
    Functional entry point for transformation generation.
    Extracts configuration parameters from the YAML-based dictionary.
    """
    image_size = config.get("image_size", 112)
    return TransformFactory.get_pipeline(
        image_size=image_size,
        is_training=is_training
    )