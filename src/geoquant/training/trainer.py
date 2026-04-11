"""
Trainer unificado: cubre entrenamiento FP32 baseline y QAT fine-tuning.
Optimiza por S-Index (separabilidad geométrica) además de accuracy.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from geoquant.utils.logging import get_logger
from geoquant.evaluation.block_e import knn_accuracy

logger = get_logger(__name__)


class Trainer:
    """
    Trainer unificado para FP32 y QAT.

    Args:
        backbone: MobileNetV3Backbone.
        arcface_head: ArcFaceHead.
        train_loader: DataLoader de entrenamiento.
        val_loader: DataLoader de validación.
        config: dict de configuración completo.
        device: torch.device objetivo.
    """

    def __init__(self, backbone, arcface_head, train_loader, val_loader, config, device):
        self.backbone = backbone.to(device)
        self.arcface_head = arcface_head.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        train_cfg = config.get("training", {})
        params = list(self.backbone.parameters()) + list(self.arcface_head.parameters())

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=train_cfg.get("label_smoothing", 0.1)
        )
        self.optimizer = optim.SGD(
            params,
            lr=train_cfg.get("lr", 0.01),
            momentum=train_cfg.get("momentum", 0.9),
            weight_decay=train_cfg.get("weight_decay", 5e-4),
        )
        self.epochs = train_cfg.get("epochs", 30)
        self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "outputs/checkpoints"))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )

    # ------------------------------------------------------------------
    # Bucle de entrenamiento
    # ------------------------------------------------------------------

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
        """Extrae embeddings del split val y calcula kNN accuracy."""
        self.backbone.eval()
        all_emb, all_targets = [], []

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]"):
                inputs = inputs.to(self.device)
                emb = self.backbone(inputs)
                all_emb.append(emb.cpu())
                all_targets.append(targets)

        emb_tensor = torch.cat(all_emb)
        tgt_tensor = torch.cat(all_targets)
        acc = knn_accuracy(emb_tensor, tgt_tensor, k=5)
        return acc, emb_tensor, tgt_tensor

    # ------------------------------------------------------------------
    # Entrenamiento completo
    # ------------------------------------------------------------------

    def fit(self) -> dict:
        """Entrena y retorna métricas del mejor checkpoint."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_acc = 0.0
        best_metrics = {}

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_acc, emb, targets = self._val_epoch(epoch)
            self.scheduler.step()

            logger.info(
                f"Epoch {epoch}/{self.epochs} | loss={train_loss:.4f} | kNN@5={val_acc:.4f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                best_metrics = {"epoch": epoch, "knn_acc": val_acc, "train_loss": train_loss}
                ckpt_path = self.checkpoint_dir / "best_fp32.pth"
                torch.save(self.backbone.state_dict(), ckpt_path)
                logger.info(f"  Nuevo mejor modelo guardado → {ckpt_path}")

        return best_metrics