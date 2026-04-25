import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from geoquant.utils.logging import get_logger
from geoquant.evaluation.block_b import alignment

logger = get_logger(__name__)

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
                all_targets.append(targets)

        emb_tensor = torch.cat(all_emb)
        tgt_tensor = torch.cat(all_targets)
        return emb_tensor, tgt_tensor

    def fit_phase(self, epochs: int, lr: float, freeze_backbone: bool, phase_name: str) -> dict:
        self.epochs = epochs
        self._setup_phase(freeze_backbone, lr, epochs)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # --- LÓGICA DE MINIMIZACIÓN DE ALIGNMENT ---
        best_align = float('inf')  # Ahora buscamos el valor más bajo
        best_metrics = {}

        logger.info(f"--- Iniciando Fase: {phase_name} ({epochs} epochs, lr={lr}) ---")

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(epoch)
            emb, targets = self._val_epoch(epoch)

            # Evaluar compacidad topológica con Bloque B
            val_align = alignment(emb, targets)
            self.scheduler.step()

            logger.info(f"[{phase_name}] Epoch {epoch}/{epochs} | loss={train_loss:.4f} | Alignment={val_align:.4f}")

            # Guardar si las clases están más compactas
            if val_align < best_align:
                best_align = val_align
                best_metrics = {"epoch": epoch, "alignment": val_align, "train_loss": train_loss}
                ckpt_path = self.checkpoint_dir / f"best_fp32_{phase_name}.pth"
                torch.save(self.backbone.state_dict(), ckpt_path)
                logger.info(f"  Nuevo mejor modelo (Clústeres más densos) guardado → {ckpt_path}")

        return best_metrics