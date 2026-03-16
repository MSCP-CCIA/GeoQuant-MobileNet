import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar10_dataloaders(batch_size=64, resize=224, num_workers=4):
    """
    Retorna DataLoaders para CIFAR-10 preparados para Transfer Learning.
    Las redes como MobileNet y ShuffleNet (entrenadas en ImageNet) esperan
    una entrada más grande que 32x32 y una normalización específica.
    """
    print(f"Cargando CIFAR-10 y aplicando transformaciones de Transfer Learning (resize={resize})...")

    # ImageNet normalization parameters
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Transformaciones de Entrenamiento: Data Augmentation + Resize + Normalize
    train_transform = transforms.Compose([
        transforms.Resize(resize), # Escalar imagen (e.g. 224x224)
        transforms.RandomHorizontalFlip(), # Aumento: volteo horizontal
        transforms.RandomRotation(15),     # Aumento: pequeña rotación
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Aumento: color
        transforms.ToTensor(),             # Tensor entre 0 y 1
        transforms.Normalize(mean, std)    # Normalización ImageNet
    ])

    # Transformaciones de Validación: Solo Resize + Normalize (Sin Augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Descargar y cargar datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    
    valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=val_transform)

    # Crear DataLoaders
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers, pin_memory=True)
    
    valloader = DataLoader(valset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    return trainloader, valloader
