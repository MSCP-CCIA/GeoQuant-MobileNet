import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import os

from src.data.cifar10_dataloaders import get_cifar10_dataloaders
from src.models.factory import get_model

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, architecture):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train - {architecture}]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100. * correct / total})
        
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device, epoch, architecture):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val - {architecture}]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100. * correct / total})
            
    return running_loss / len(loader), 100. * correct / total

def main():
    # --- Configuración ---
    ARCHITECTURES = ['mobilenet_v2', 'shufflenet_v2']
    NUM_EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Usando dispositivo: {device}")
    
    # --- Cargar Datos ---
    train_loader, val_loader = get_cifar10_dataloaders(batch_size=BATCH_SIZE)
    
    # --- Bucle de Entrenamiento para cada Arquitectura ---
    for arch in ARCHITECTURES:
        logger.info(f"--- Iniciando entrenamiento para: {arch} ---")
        
        # Modelo
        model = get_model(arch).to(device)

        # Loss y Optimizador
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        best_val_acc = 0.0
        checkpoint_dir = f'./checkpoints/{arch}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, arch)
            val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, arch)
            
            logger.info(f"Epoch {epoch}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # Guardar mejor modelo
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Mejor modelo para {arch} guardado en {checkpoint_path} (Val Acc: {best_val_acc:.2f}%)")
                
if __name__ == '__main__':
    main()
