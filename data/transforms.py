from torchvision import transforms
from torchvision.transforms import InterpolationMode

def get_transforms(dataset_name="imagenet", is_training=False):
    """
    Pipeline de preprocesamiento optimizado para 112x112 y arquitecturas ligeras.
    
    Args:
        dataset_name (str): "imagenet" o "cifar100".
        is_training (bool): Si es True, aplica aumentos (solo para QAT).
    """
    
    # 1. Definir parámetros según el dataset
    if "cifar10" in dataset_name:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.247, 0.243, 0.261]
    else: # imagenet y otros
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    # 2. Pipeline base (Validation/Inference/Topology)
    # Target: 112x112 con Interpolación Bicúbica
    base_transforms = [
        transforms.Resize(128, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    # 3. Aumentos para QAT (Opcional, solo si se entrena)
    if is_training:
        training_transforms = [
            transforms.RandomResizedCrop(112, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
        ] + base_transforms[2:] # Mantener ToTensor y Normalize
        return transforms.Compose(training_transforms)

    return transforms.Compose(base_transforms)
