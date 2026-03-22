import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os


def create_imagenette_dataloaders(batch_size=64, data_dir='./data'):
    os.makedirs(data_dir, exist_ok=True)

    # Normalización estándar de ImageNet (Obligatoria para Transfer Learning)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Pipeline de entrenamiento: Recorte aleatorio y volteo para evitar sobreajuste
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Pipeline de validación: Recorte central preciso a 224x224
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    print("Verificando/Descargando el dataset Imagenette (10 clases, alta resolución)...")
    try:
        train_dataset = torchvision.datasets.Imagenette(root=data_dir, split='train', download=True,
                                                        transform=train_transform)
        val_dataset = torchvision.datasets.Imagenette(root=data_dir, split='val', download=True,
                                                      transform=val_transform)
    except AttributeError:
        # Fallback de seguridad por si tu versión de torchvision es anterior a 0.16
        raise RuntimeError(
            "Tu versión de torchvision no incluye Imagenette nativo. Actualiza con 'pip install -U torchvision' o descarga la carpeta manualmente y usa torchvision.datasets.ImageFolder.")

    # Al ser imágenes de 224x224, un batch_size de 64 es un buen balance de memoria VRAM
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader