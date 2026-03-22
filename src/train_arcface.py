import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from src.utils.logger import logger
from src.models.factory import get_arcface_model
from data.dataloaders import create_imagenette_dataloaders
from src.topology.metrics import calculate_s_index


def train_epoch(model, arcface_head, loader, criterion, optimizer, device, epoch):
    model.train()
    arcface_head.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train ArcFace]")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # 1. Extraer embeddings normalizados
        embeddings = model(inputs)

        # 2. Calcular logits angulares
        outputs = arcface_head(embeddings, targets)

        # 3. Penalización estándar sobre logits angulares
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

    return running_loss / len(loader)


def validate_epoch(model, loader, device, epoch):
    model.eval()
    all_targets = []
    all_embeddings = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val Topología]")
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            # Extraemos los embeddings directamente (ya vienen de BatchNorm)
            embeddings = model(inputs)

            all_embeddings.append(embeddings.cpu())
            all_targets.append(targets.cpu())

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)

    # Evaluar la topología cruda: S-Index
    s_index, intra_sim, inter_sim = calculate_s_index(embeddings_tensor, targets_tensor)
    return s_index, intra_sim, inter_sim


def run_arcface_experiment(arch_name, device, train_loader, val_loader, epochs=20):
    logger.info(f"=== Iniciando Entrenamiento Topológico (ArcFace): {arch_name} ===")

    model, arcface_head = get_arcface_model(arch_name, num_classes=10)
    model = model.to(device)
    arcface_head = arcface_head.to(device)

    criterion = nn.CrossEntropyLoss()

    # Entrenamos ambos módulos juntos
    params = list(model.parameters()) + list(arcface_head.parameters())
    optimizer = optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_s_index = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, arcface_head, train_loader, criterion, optimizer, device, epoch)
        s_index, intra_sim, inter_sim = validate_epoch(model, val_loader, device, epoch)

        scheduler.step()

        logger.info(f"[{arch_name}] Epoch {epoch}/{epochs} | Loss: {train_loss:.4f}")
        logger.info(
            f"  [+] Métrica Topológica -> S-Index: {s_index:.4f} (Intra: {intra_sim:.4f}, Inter: {inter_sim:.4f})")

        # LA CLAVE DE TU TESIS: Guardamos el modelo con la geometría más robusta, ignorando el Accuracy
        if s_index > best_s_index:
            best_s_index = s_index
            os.makedirs(f'checkpoints', exist_ok=True)
            # Solo guardamos el extractor (la cabeza ArcFace se descarta en inferencia/cuantización)
            torch.save(model.state_dict(), f'checkpoints/{arch_name}_best_arcface_fp32.pth')
            logger.info(f"  [!] Nuevo S-Index máximo alcanzado. Modelo guardado.")

    return best_s_index


def main():
    logger.info("Iniciando pipeline ArcFace (S-Index Optimization)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_imagenette_dataloaders(batch_size=64)

    architectures = ['mobilenet_v3_small', 'shufflenet_v2']

    for arch in architectures:
        run_arcface_experiment(arch, device, train_loader, val_loader, epochs=20)


if __name__ == '__main__':
    main()