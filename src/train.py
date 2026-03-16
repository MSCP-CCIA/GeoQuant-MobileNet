import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from src.utils.logger import logger
from src.models.factory import get_model
from src.data.dataloaders import create_imagenette_dataloaders
from src.topology.hooks import EmbeddingExtractor
from src.topology.metrics import calculate_s_index


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100. * correct / total})

    return running_loss / len(loader), 100. * correct / total


def validate_epoch(model, loader, criterion, device, epoch, hook_extractor):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_targets = []

    hook_extractor.clear()

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_targets.append(targets.cpu())
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100. * correct / total})

    embeddings = hook_extractor.get_features()
    targets_tensor = torch.cat(all_targets, dim=0)

    # Cálculo del S-Index (Inverso del DBI) según la sección 1.4.3 de tu tesis
    s_index, intra_sim, inter_sim = calculate_s_index(embeddings, targets_tensor)

    return running_loss / len(loader), 100. * correct / total, s_index, intra_sim, inter_sim


def run_baseline(arch_name, device, train_loader, val_loader, epochs=15):
    logger.info(f"=== Iniciando entrenamiento de Baseline (Transfer Learning): {arch_name} ===")

    # Obtiene modelo con pesos de ImageNet filtrados (Cabeza aleatoria)
    model, feature_layer = get_model(arch_name, num_classes=10)
    model = model.to(device)

    extractor = EmbeddingExtractor(model, feature_layer)

    # Label Smoothing para mejorar la geometría del espacio latente
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizador SGD con momentum para mejor generalización que Adam en visión
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_s_index = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, s_index, intra_sim, inter_sim = validate_epoch(
            model, val_loader, criterion, device, epoch, extractor
        )

        scheduler.step()

        logger.info(f"[{arch_name}] Epoch {epoch}/{epochs}")
        logger.info(f"  Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        logger.info(f"  Val   -> Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        logger.info(f"  Topo  -> S-Index: {s_index:.4f} (Intra: {intra_sim:.4f}, Inter: {inter_sim:.4f})")

        if val_acc > best_acc:
            best_acc = val_acc
            best_s_index = s_index
            os.makedirs(f'checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/{arch_name}_best_fp32.pth')
            logger.info(f"  [!] Nuevo mejor modelo guardado.")

    extractor.cleanup()
    return best_acc, best_s_index


def main():
    logger.info("Iniciando pipeline de entrenamiento para Imagenette (Head-only Training)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Los dataloaders ahora descargan y procesan Imagenette a 224x224
    train_loader, val_loader = create_imagenette_dataloaders(batch_size=64)

    metrics = {}
    architectures = ['mobilenet_v3_small', 'shufflenet_v2']

    for arch in architectures:
        best_acc, best_s_index = run_baseline(arch, device, train_loader, val_loader, epochs=15)
        metrics[arch] = {'Accuracy': best_acc, 'S-Index': best_s_index}

    logger.info("=== REPORTE FINAL DE MÉTRICAS (BASELINES FP32) ===")
    for arch, m in metrics.items():
        logger.info(f"Arquitectura: {arch:<18} | Accuracy: {m['Accuracy']:>6.2f}% | S-Index: {m['S-Index']:>6.4f}")


if __name__ == '__main__':
    main()