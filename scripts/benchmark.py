"""
scripts/benchmark.py — Benchmark de latencia e impacto en disco.
Compara FP32, PTQ y QAT en CPU restringido (1 núcleo).

Uso: python scripts/benchmark.py [--fp32 <ckpt>] [--ptq <ckpt>] [--qat <ckpt>]
"""

import argparse
import copy

import torch
import yaml

from geoquant.utils.reproducibility import seed_everything
from geoquant.utils.logging import get_logger
from geoquant.models.backbone import build_backbone
from geoquant.evaluation.latency import benchmark_models

logger = get_logger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="GeoQuant — Benchmark de Latencia")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--fp32", default="outputs/checkpoints/best_fp32.pth")
    parser.add_argument("--ptq", default=None)
    parser.add_argument("--qat", default=None)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.get("seed", 42))

    models_to_bench = {}

    # FP32
    m_fp32 = build_backbone(config)
    m_fp32.load_state_dict(torch.load(args.fp32, map_location="cpu"))
    models_to_bench["FP32"] = m_fp32

    # PTQ
    if args.ptq:
        m_ptq = build_backbone(config)
        m_ptq.load_state_dict(torch.load(args.ptq, map_location="cpu"))
        models_to_bench["PTQ"] = m_ptq

    # QAT
    if args.qat:
        m_qat = build_backbone(config)
        m_qat.load_state_dict(torch.load(args.qat, map_location="cpu"))
        models_to_bench["QAT"] = m_qat

    results = benchmark_models(models_to_bench, iterations=args.iterations)

    print("\n" + "=" * 65)
    print(f"{'MÉTODO':<10} | {'LATENCIA (ms)':<14} | {'STD (ms)':<10} | {'DISCO (MB)'}")
    print("=" * 65)
    for name, m in results.items():
        print(
            f"{name:<10} | {m['latency_ms']:>10.2f}     | "
            f"{m['latency_std_ms']:>8.2f}   | {m['disk_mb']:>8.2f}"
        )
    print("=" * 65)


if __name__ == "__main__":
    main()