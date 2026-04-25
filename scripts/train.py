"""
scripts/train.py — Entrenamiento FP32 baseline con ArcFace.
Uso: python scripts/train.py [--config configs/config.yaml] [--experiment configs/experiment/baseline_fp32.yaml]
"""

import argparse
from pathlib import Path
import yaml
import torch

from geoquant.utils.reproducibility import seed_everything
from geoquant.utils.logging import get_logger
from geoquant.data.dataset import get_dataloaders
from geoquant.models.backbone import build_backbone
from geoquant.models.arcface import build_arcface
from geoquant.training.trainer import Trainer
from geoquant.evaluation.embeddings import extract_and_save
logger = get_logger(__name__)


def load_config(base: str, experiment: str = None) -> dict:
    with open(base) as f:
        config = yaml.safe_load(f)
    if experiment:
        with open(experiment) as f:
            override = yaml.safe_load(f)
        # Merge superficial de secciones
        for section, values in override.items():
            if isinstance(values, dict):
                config.setdefault(section, {}).update(values)
            else:
                config[section] = values
    return config

def main():
    parser = argparse.ArgumentParser(description="GeoQuant — Entrenamiento FP32")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--experiment", default="configs/experiment/baseline_fp32.yaml")
    args = parser.parse_args()

    config = load_config(args.config, args.experiment)
    seed_everything(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")

    train_loader, val_loader = get_dataloaders(config)

    # 1. Construir modelos leyendo configuración
    backbone = build_backbone(config)
    arcface = build_arcface(config, in_features=backbone.in_features)

    trainer = Trainer(backbone, arcface, train_loader, val_loader, config, device)

    # 2. Configuración de hiperparámetros de las fases
    train_cfg = config.get("training", {})
    base_lr = train_cfg.get("lr", 0.01)

    # Puedes añadir "warmup_epochs: 5" a tu config.yaml o hardcodearlo aquí
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    ft_epochs = train_cfg.get("epochs", 30)

    # --- FASE 1: Warm-up del Head (Backbone congelado) ---
    logger.info("Iniciando Fase 1: Estabilización de la capa Lineal y ArcFace")
    trainer.fit_phase(
        epochs=warmup_epochs,
        lr=base_lr,
        freeze_backbone=True,
        phase_name="warmup"
    )

    # --- FASE 2: Fine-tuning Biométrico (No Freezing) ---
    logger.info("Iniciando Fase 2: Fine-Tuning Total del Backbone")
    # Reducimos el learning rate a una décima parte para no destruir los pesos
    ft_lr = base_lr * 0.1
    metrics = trainer.fit_phase(
        epochs=ft_epochs,
        lr=ft_lr,
        freeze_backbone=False,
        phase_name="finetuning"
    )

    logger.info(f"Entrenamiento completado: {metrics}")

    # Guardar embeddings FP32 del mejor checkpoint de la fase de finetuning
    eval_cfg = config.get("eval", {})
    emb_dir = Path(eval_cfg.get("embeddings_dir", "outputs/embeddings"))
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "outputs/checkpoints"))

    extract_and_save(
        ckpt_path=ckpt_dir / "best_fp32_finetuning.pth",
        output_path=emb_dir / "emb_fp32.pt",
        config=config,
        dataloader=val_loader,
        device=torch.device("cpu"),
    )
    logger.info("Embeddings FP32 guardados para evaluación.")


if __name__ == "__main__":
    main()