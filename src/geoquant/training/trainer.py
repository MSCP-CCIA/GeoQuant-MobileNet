"""
Módulo de Entrenamiento (Trainer) para el Baseline FP32.

Este script gestiona el ciclo de entrenamiento y validación del modelo maestro.
Incluye extracción dinámica de la dimensión del embedding, evaluación topológica
con k-NN (1-NN para rigurosidad, 5-NN para cohesión) y registro en CSV.
"""

import os
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
    """
    Calcula el Accuracy Topológico.
    Si k=1 usa emparejamiento exacto (Rank-1).
    Si k>1 usa votación mayoritaria.
    """
    dist_matrix = torch.cdist(query_embs, gallery_embs)

    if gallery_embs is query_embs:
        dist_matrix.fill_diagonal_(float('inf'))

    knn_indices = dist_matrix.topk(k, largest=False, dim=1).indices
    knn_labels = gallery_labels[knn_indices]

    if k == 1:
        preds = knn_labels.squeeze()
    else:
        # Votación mayoritaria
        preds = torch.mode(knn_labels, dim=1).values

    acc = (preds == query_labels).float().mean().item()
    return acc


class Trainer:
    def __init__(self, backbone, arcface_head, train_loader, val_loader, config, device):
        self.backbone = backbone.to(device)
        self.arcface_head = arcface_head.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        train_cfg = config.get("training", {})
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=train_cfg.get("label_smoothing", 0.1)
        )
        self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "outputs/checkpoints"))
        self.optimizer = None
        self.scheduler = None

    def _setup_phase(self, freeze_backbone: bool, lr: float, epochs: int):
        if freeze_backbone:
            for param in self.backbone.model.features.parameters():
                param.requires_grad = False
            for param in self.backbone.model.classifier.parameters():
                param.requires_grad = True
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

        params = list(filter(lambda p: p.requires_grad, self.backbone.parameters())) + \
                 list(self.arcface_head.parameters())

        train_cfg = self.config.get("training", {})
        self.optimizer = optim.SGD(
            params,
            lr=lr,
            momentum=train_cfg.get("momentum", 0.9),
            weight_decay=train_cfg.get("weight_decay", 5e-4),
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

    def _train_epoch(self, epoch: int) -> float:
        self.backbone.train()
        self.arcface_head.train()
        running_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            embeddings = self.backbone(inputs)
            logits = self.arcface_head(embeddings, targets)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})

        return running_loss / len(self.train_loader)

    def _val_epoch(self, epoch: int):
        self.backbone.eval()
        all_emb, all_targets = [], []

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]"):
                inputs = inputs.to(self.device)
                emb = self.backbone(inputs)
                all_emb.append(emb.cpu())
                all_targets.append(targets.cpu())

        emb_tensor = torch.cat(all_emb)
        tgt_tensor = torch.cat(all_targets)
        return emb_tensor, tgt_tensor

    def fit_phase(self, epochs: int, lr: float, freeze_backbone: bool, phase_name: str) -> dict:
        self.epochs = epochs
        self._setup_phase(freeze_backbone, lr, epochs)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # --- EXTRACCIÓN DINÁMICA DE DIMENSIÓN ---
        self.backbone.eval()
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            emb_size = self.backbone(dummy_input).shape[1]

        logger.info(f"--- Iniciando Fase: {phase_name} ({epochs} epochs, lr={lr}, Dim: {emb_size}d) ---")

        best_1nn_acc = 0.0
        best_metrics = {}
        experiment_history = []

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(epoch)
            emb, targets = self._val_epoch(epoch)

            # Evaluar separabilidad topológica (Rigurosidad y Cohesión)
            val_1nn_acc = compute_knn_accuracy(emb, targets, emb, targets, k=1)
            val_5nn_acc = compute_knn_accuracy(emb, targets, emb, targets, k=5)

            self.scheduler.step()

            logger.info(
                f"[{phase_name}] Epoch {epoch} | Loss={train_loss:.4f} | 1-NN={val_1nn_acc:.4f} | 5-NN={val_5nn_acc:.4f}")

            experiment_history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_1nn_acc": val_1nn_acc,
                "val_5nn_acc": val_5nn_acc
            })

            # Selección del mejor modelo (Basado siempre en 1-NN para el Maestro FP32)
            if val_1nn_acc > best_1nn_acc:
                best_1nn_acc = val_1nn_acc
                best_metrics = {
                    "epoch": epoch,
                    "1nn_acc": val_1nn_acc,
                    "5nn_acc": val_5nn_acc,
                    "train_loss": train_loss
                }

                # Nombrado dinámico del checkpoint
                ckpt_path = self.checkpoint_dir / f"best_fp32_{phase_name}_{emb_size}d.pth"
                torch.save(self.backbone.state_dict(), ckpt_path)
                logger.info(f"  ^-- Topología FP32 Superior! Guardado -> {ckpt_path}")

        # --- GUARDAR HISTORIAL CSV CON NOMBRADO DINÁMICO ---
        history_path = self.checkpoint_dir.parent / f"training_history_{phase_name}_{emb_size}d.csv"
        with open(history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_1nn_acc", "val_5nn_acc"])
            writer.writeheader()
            writer.writerows(experiment_history)
        logger.info(f"Historial CSV guardado -> {history_path}")

        return best_metrics