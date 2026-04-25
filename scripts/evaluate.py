"""
scripts/evaluate.py — Evaluación geométrica completa (5 bloques).
Compara embeddings FP32 vs INT8 y genera reportes JSON/CSV/LaTeX.

Los embeddings se cachean en outputs/embeddings/ para evitar re-inferencia.
Usa --force-extract para ignorar el caché y regenerarlos.

Uso:
  python scripts/evaluate.py --int8 <ckpt_int8> [--approach ptq|qat]
  python scripts/evaluate.py --int8 <ckpt_int8> --force-extract
"""

import argparse
from pathlib import Path

import yaml
import torch

from geoquant.utils.reproducibility import seed_everything
from geoquant.utils.logging import get_logger
from geoquant.data.dataset import get_dataloaders
from geoquant.evaluation.suite import EvaluationSuite
from geoquant.evaluation.reporter import Reporter
from geoquant.evaluation.embeddings import extract_and_save, load_embeddings

logger = get_logger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_embeddings(key, ckpt_path, emb_dir, config, val_loader, device, force):
    """Carga embeddings desde disco o los extrae y guarda si no existen."""
    emb_path = emb_dir / f"emb_{key}.pt"
    if emb_path.exists() and not force:
        logger.info(f"[{key}] Cargando embeddings desde disco...")
        return load_embeddings(emb_path)
    logger.info(f"[{key}] Extrayendo embeddings...")
    return extract_and_save(ckpt_path, emb_path, config, val_loader, device)


def main():
    parser = argparse.ArgumentParser(description="GeoQuant — Evaluación Geométrica")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--fp32", default="outputs/checkpoints/baseline_fp32/best_fp32_finetuning.pth")
    parser.add_argument("--int8", required=True, help="Checkpoint INT8 (PTQ o QAT)")
    parser.add_argument("--approach", default="ptq", choices=["ptq", "qat"])
    parser.add_argument("--output-dir", default=None,
                        help="Directorio de reportes (default: eval.output_dir del config)")
    parser.add_argument("--emb-dir", default=None,
                        help="Directorio de embeddings cacheados (default: eval.embeddings_dir del config)")
    parser.add_argument("--force-extract", action="store_true",
                        help="Ignorar caché y regenerar embeddings desde los checkpoints")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.get("seed", 42))
    device = torch.device("cpu")  # Modelos INT8 solo en CPU

    eval_cfg = config.get("eval", {})
    emb_dir = Path(args.emb_dir or eval_cfg.get("embeddings_dir", "outputs/embeddings"))
    output_dir = args.output_dir or eval_cfg.get("output_dir", "outputs/results")
    _, val_loader = get_dataloaders(config)

    emb_fp32, labels = get_embeddings(
        "fp32", args.fp32, emb_dir, config, val_loader, device, args.force_extract
    )
    emb_int8, _ = get_embeddings(
        args.approach, args.int8, emb_dir, config, val_loader, device, args.force_extract
    )

    # Evaluación completa
    suite = EvaluationSuite(k_neighbors=10)
    results = suite.run(emb_fp32, emb_int8, labels)

    # Reporte
    reporter = Reporter(output_dir=output_dir)
    paths = reporter.save_all(results, stem=f"eval_{args.approach}")
    logger.info(f"Reportes guardados: {paths}")


if __name__ == "__main__":
    main()