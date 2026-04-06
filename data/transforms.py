"""
Transformation pipelines for GeoQuant experiments.
Optimized for fine-grained embedding analysis (CUB-200) and INT8 quantization stability.
"""

from typing import Dict, Any, List
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class TransformFactory:
    """
    Service to generate standardized image transformation pipelines.
    Designed for MobileNetV3 architectures and geometric evaluation of fine-grained features.
    """

    # ImageNet normalization constants (Required for pretrained MobileNetV3)
    IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
    IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

    @classmethod
    def get_pipeline(
            cls,
            image_size: int = 224,
            is_training: bool = False
    ) -> transforms.Compose:
        """
        Builds a transformation pipeline based on geometric requirements.

        Args:
            image_size: Target resolution (Standard 224 for fine-grained datasets like CUB-200).
            is_training: If True, applies data augmentation for QAT robustness.
        """
        normalization = transforms.Normalize(
            mean=cls.IMAGENET_MEAN,
            std=cls.IMAGENET_STD
        )

        if is_training:
            # Training pipeline optimized to preserve fine-grained details
            return transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.5, 1.0),  # Evita recortes muy pequeños donde el ave desaparezca
                    interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalization,
            ])

        # Validation / Geometric Inference pipeline
        # The 1.143 ratio mathematically scales 224 up to ~256 before the CenterCrop.
        # This is the industry standard for evaluating ImageNet-pretrained models.
        return transforms.Compose([
            transforms.Resize(
                int(image_size * 1.143),
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
    # Safe default explicitly set to 224 for CUB-200
    image_size = config.get("image_size", 224)

    return TransformFactory.get_pipeline(
        image_size=image_size,
        is_training=is_training
    )