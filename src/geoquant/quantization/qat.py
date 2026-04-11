"""
QAT (Quantization-Aware Training) via torch.ao FX Graph Mode.
Fine-tunes con ruido INT8 simulado (FakeQuantize) manteniendo la cabeza ArcFace
para preservar la geometría del espacio latente.
"""

import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from geoquant.utils.logging import get_logger

logger = get_logger(__name__)


def apply_qat(
    model: torch.nn.Module,
    arcface_head: torch.nn.Module,
    train_loader,
    output_path: str,
    epochs: int = 10,
    lr: float = 5e-5,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    """
    Fine-tuning QAT con FX Graph Mode.

    Args:
        model: Backbone FP32 ya entrenado.
        arcface_head: Cabeza ArcFace (guía topológica durante QAT).
        train_loader: DataLoader de entrenamiento.
        output_path: Ruta para guardar el modelo INT8 resultante.
        epochs: Épocas de fine-tuning.
        lr: Learning rate (bajo para no destruir la geometría aprendida).
        device: Dispositivo de entrenamiento.

    Returns:
        Modelo INT8 convertido (en CPU).
    """
    from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
    from torch.ao.quantization import get_default_qat_qconfig_mapping

    logger.info(f"Iniciando QAT FX Graph Mode ({epochs} épocas, lr={lr})...")

    model_to_qat = copy.deepcopy(model)
    model_to_qat.train().to(device)

    qconfig_mapping = get_default_qat_qconfig_mapping("fbgemm")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    logger.info("Trazando grafo FX e insertando FakeQuantize...")
    qat_model = prepare_qat_fx(model_to_qat, qconfig_mapping, example_inputs=(dummy_input,))

    arcface_head = copy.deepcopy(arcface_head).to(device).train()
    criterion = nn.CrossEntropyLoss()
    params = list(qat_model.parameters()) + list(arcface_head.parameters())
    optimizer = optim.SGD(params, lr=lr, momentum=0.9)

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch}/{epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            embeddings = qat_model(inputs)
            logits = arcface_head(embeddings, targets)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})

        logger.info(f"  QAT Epoch {epoch} | loss={running_loss / len(train_loader):.4f}")

    logger.info("Convirtiendo grafo QAT a INT8 definitivo...")
    qat_model.eval().to("cpu")
    model_int8 = convert_fx(qat_model)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_int8.state_dict(), out)
    logger.info(f"Modelo QAT INT8 guardado → {out}")

    return model_int8