"""
scripts/evaluate.py — Evaluación geométrica completa (5 bloques).
Compara embeddings FP32 vs INT8 y genera reportes JSON/CSV/LaTeX.

Uso: python scripts/evaluate.py --fp32 <ckpt_fp32> --int8 <ckpt_int8> [--approach ptq|qat]
"""

import argparse
from pathlib import Path

import yaml
import torch

from geoquant.utils.reproducibility import seed_everything
from geoquant.utils.logging import get_logger
from geoquant.data.dataset import get_dataloaders
from geoquant.models.backbone import build_backbone
from geoquant.evaluation.suite import EvaluationSuite
from geoquant.evaluation.reporter import Reporter

logger = get_logger(__name__)


def extract_embeddings(model, loader, device):
    """Extrae todos los embeddings de un DataLoader."""
    model.eval().to(device)
    all_emb, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            emb = model(inputs)
            all_emb.append(emb.cpu())
            all_labels.append(labels)
    return torch.cat(all_emb), torch.cat(all_labels)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="GeoQuant — Evaluación Geométrica")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--fp32", default="outputs/checkpoints/best_fp32.pth")
    parser.add_argument("--int8", required=True, help="Checkpoint INT8 (PTQ o QAT)")
    parser.add_argument("--approach", default="ptq", choices=["ptq", "qat"])
    parser.add_argument("--output-dir", default="outputs/results")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.get("seed", 42))
    device = torch.device("cpu")  # Modelos INT8 solo en CPU

    _, val_loader = get_dataloaders(config)

    # Backbone FP32
    fp32_model = build_backbone(config)
    fp32_model.load_state_dict(torch.load(args.fp32, map_location=device))
    logger.info("Extrayendo embeddings FP32...")
    emb_fp32, labels = extract_embeddings(fp32_model, val_loader, device)

    # Backbone INT8
    int8_model = build_backbone(config)
    int8_model.load_state_dict(torch.load(args.int8, map_location=device))
    logger.info("Extrayendo embeddings INT8...")
    emb_int8, _ = extract_embeddings(int8_model, val_loader, device)

    # Evaluación completa
    suite = EvaluationSuite(k_neighbors=10)
    results = suite.run(emb_fp32, emb_int8, labels)

    # Reporte
    reporter = Reporter(output_dir=args.output_dir)
    paths = reporter.save_all(results, stem=f"eval_{args.approach}")
    logger.info(f"Reportes guardados: {paths}")


if __name__ == "__main__":
    main()