"""
scripts/quantize.py — Aplica PTQ o QAT sobre el backbone FP32 entrenado.
Uso: python scripts/quantize.py --approach ptq|qat [--config ...] [--experiment ...]
"""

import argparse
from pathlib import Path

import yaml
import torch

from geoquant.utils.reproducibility import seed_everything
from geoquant.utils.logging import get_logger
from geoquant.data.dataset import get_dataloaders
from geoquant.models.backbone import build_backbone, MobileNetV3Backbone
from geoquant.models.arcface import build_arcface
from geoquant.quantization.ptq import apply_ptq
from geoquant.quantization.qat import apply_qat
from geoquant.quantization.export import export_torchscript

logger = get_logger(__name__)


def load_config(base: str, experiment: str = None) -> dict:
    with open(base) as f:
        config = yaml.safe_load(f)
    if experiment:
        with open(experiment) as f:
            override = yaml.safe_load(f)
        for section, values in override.items():
            if isinstance(values, dict):
                config.setdefault(section, {}).update(values)
            else:
                config[section] = values
    return config


def main():
    parser = argparse.ArgumentParser(description="GeoQuant — Cuantización")
    parser.add_argument("--approach", choices=["ptq", "qat"], required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_fp32.pth")
    parser.add_argument("--export", action="store_true", help="Exportar a TorchScript INT8")
    args = parser.parse_args()

    config = load_config(args.config, args.experiment)
    seed_everything(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar backbone FP32
    backbone = build_backbone(config)
    ckpt = torch.load(args.checkpoint, map_location=device)
    backbone.load_state_dict(ckpt)
    backbone.eval()
    logger.info(f"Backbone cargado desde {args.checkpoint}")

    output_dir = Path(config.get("quantization", {}).get("output_dir", "outputs/checkpoints"))

    if args.approach == "ptq":
        output_path = output_dir / "mobilenet_v3_small_ptq_int8.pth"
        model_int8 = apply_ptq(backbone, str(output_path))

    elif args.approach == "qat":
        train_loader, _ = get_dataloaders(config)
        arcface = build_arcface(config, in_features=backbone.in_features)

        output_path = output_dir / "mobilenet_v3_small_qat_int8.pth"
        qat_cfg = config.get("training", {})
        model_int8 = apply_qat(
            model=backbone,
            arcface_head=arcface,
            train_loader=train_loader,
            output_path=str(output_path),
            epochs=qat_cfg.get("epochs", 10),
            lr=qat_cfg.get("lr", 5e-5),
            device=device,
        )

    if args.export:
        ts_path = output_dir / f"mobilenet_v3_small_{args.approach}_int8.pt"
        export_torchscript(model_int8, str(ts_path))

    logger.info("Cuantización completada.")


if __name__ == "__main__":
    main()