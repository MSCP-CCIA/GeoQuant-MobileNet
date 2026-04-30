"""
Módulo de Cuantización Consciente del Entrenamiento (QAT) - Destilación Pura Ganadora.

Este script implementa la Destilación de Espacio Latente exacta que logró el Cosine Drift de 0.22.
Utiliza MSELoss, AdamW, y congela el BatchNorm en la época 5.
Integra nombrado dinámico (512d/1024d), evaluación k-NN (1-NN y 5-NN) y registro CSV.
"""

import copy
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from geoquant.utils.logging import get_logger

logger = get_logger(__name__)


def compute_knn_accuracy(gallery_embs: torch.Tensor, gallery_labels: torch.Tensor,
                         query_embs: torch.Tensor, query_labels: torch.Tensor, k: int = 1) -> float:
    dist_matrix = torch.cdist(query_embs, gallery_embs)

    if gallery_embs is query_embs:
        dist_matrix.fill_diagonal_(float('inf'))

    knn_indices = dist_matrix.topk(k, largest=False, dim=1).indices
    knn_labels = gallery_labels[knn_indices]

    if k == 1:
        preds = knn_labels.squeeze()
    else:
        preds = torch.mode(knn_labels, dim=1).values

    acc = (preds == query_labels).float().mean().item()
    return acc


def apply_qat_distillation(
        model_fp32: torch.nn.Module,
        arcface_head: torch.nn.Module,
        train_loader,
        val_loader,
        output_path: str,
        epochs: int = 10,
        lr: float = 1e-4,  # AdamW rate
        device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:

    from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
    from torch.ao.quantization import get_default_qat_qconfig_mapping

    # --- EXTRACCIÓN DINÁMICA DE DIMENSIÓN ---
    model_fp32.eval().to(device)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        emb_size = model_fp32(dummy_input).shape[1]

    logger.info(f"Iniciando QAT con Destilación Pura (Dimensión: {emb_size}d)...")

    # --- CORRECCIÓN DINÁMICA DEL MOTOR ---
    current_engine = torch.backends.quantized.engine
    qconfig_mapping = get_default_qat_qconfig_mapping(current_engine)

    # Preparar Maestro (FP32 - Congelado)
    teacher = copy.deepcopy(model_fp32).eval().to(device)
    for param in teacher.parameters():
        param.requires_grad = False

    # Preparar ArcFace (Congelado, no participa en la loss)
    arcface_head = copy.deepcopy(arcface_head).to(device).eval()
    for param in arcface_head.parameters():
        param.requires_grad = False

    # Preparar Estudiante (FakeQuantize - Descongelado)
    student = copy.deepcopy(model_fp32).train().to(device)
    qat_model = prepare_qat_fx(student, qconfig_mapping, example_inputs=(dummy_input,))

    criterion_kd = nn.MSELoss()
    optimizer = optim.AdamW(qat_model.parameters(), lr=lr, weight_decay=1e-4)

    experiment_history = []
    best_metric = 0.0
    best_qat_state = None

    for epoch in range(1, epochs + 1):
        qat_model.train()

        # Estabilizar ruido de cuantización a mitad del entrenamiento (La clave de la destilación)
        if epoch >= 5:
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch}/{epochs}")

        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_emb = teacher(inputs)

            qat_emb = qat_model(inputs)

            # Destilación Pura: Solo castigamos la distancia al vector maestro
            loss_kd = criterion_kd(qat_emb, teacher_emb)

            loss_kd.backward()
            optimizer.step()
            running_loss += loss_kd.item()
            pbar.set_postfix({"MSE_Loss": running_loss / (pbar.n + 1)})

        train_loss_epoch = running_loss / len(train_loader)

        # --- BUCLE DE VALIDACIÓN (k-NN) ---
        qat_model.eval()
        all_student_emb, all_val_labels = [], []

        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs = val_inputs.to(device)
                s_emb = qat_model(val_inputs)

                all_student_emb.append(s_emb.cpu())
                all_val_labels.append(val_targets.cpu())

        s_tensor = torch.cat(all_student_emb)
        val_labels = torch.cat(all_val_labels)

        # Calculamos tus dos métricas topológicas clave
        val_1nn_acc = compute_knn_accuracy(s_tensor, val_labels, s_tensor, val_labels, k=1)
        val_5nn_acc = compute_knn_accuracy(s_tensor, val_labels, s_tensor, val_labels, k=5)

        logger.info(f"  Epoch {epoch} | MSE_Loss={train_loss_epoch:.4f} | 1-NN={val_1nn_acc:.4f} | 5-NN={val_5nn_acc:.4f}")

        experiment_history.append({
            "epoch": epoch,
            "train_loss": train_loss_epoch,
            "val_1nn_acc": val_1nn_acc,
            "val_5nn_acc": val_5nn_acc
        })

        # Seleccionamos el mejor modelo basados en la rigurosidad biométrica (1-NN)
        if val_1nn_acc > best_metric:
            best_metric = val_1nn_acc
            best_qat_state = copy.deepcopy(qat_model.state_dict())
            logger.info("  ^-- Topología mejorada, guardando estado!")

    # --- NOMBRADO DINÁMICO DE ARCHIVOS ---
    base_out = Path(output_path)
    base_out.parent.mkdir(parents=True, exist_ok=True)

    final_model_name = f"{base_out.stem}_{emb_size}d{base_out.suffix}"
    final_out_path = base_out.parent / final_model_name

    history_name = f"training_history_qat_{emb_size}d.csv"
    history_path = base_out.parent / history_name

    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_1nn_acc", "val_5nn_acc"])
        writer.writeheader()
        writer.writerows(experiment_history)
    logger.info(f"Historial CSV guardado -> {history_path}")

    # --- RESTAURACIÓN Y CONVERSIÓN ---
    if best_qat_state is not None:
        qat_model.load_state_dict(best_qat_state)

    logger.info("Convirtiendo modelo QAT a INT8 definitivo...")
    qat_model.eval().to("cpu")
    model_int8 = convert_fx(qat_model)

    torch.save(model_int8.state_dict(), final_out_path)
    logger.info(f"Modelo QAT (Destilado) guardado -> {final_out_path}")

    return model_int8