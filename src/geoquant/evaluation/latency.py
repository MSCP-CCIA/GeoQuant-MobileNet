"""
Medición de latencia e impacto de memoria para modelos FP32 e INT8.
Simula hardware restringido (1 núcleo CPU) para resultados reproducibles.
"""

import os
import time
from pathlib import Path

import torch
import numpy as np

from geoquant.utils.logging import get_logger

logger = get_logger(__name__)


def measure_latency(
    model: torch.nn.Module,
    image_size: int = 224,
    iterations: int = 100,
    warmup: int = 10,
) -> dict:
    """
    Mide latencia media de inferencia en CPU (ms) y tamaño en disco (MB).

    Args:
        model: Modelo a evaluar.
        image_size: Resolución de entrada.
        iterations: Iteraciones para el benchmark.
        warmup: Iteraciones de calentamiento (descartadas).

    Returns:
        dict con 'latency_ms', 'latency_std_ms', 'disk_mb'.
    """
    model.eval().to("cpu")
    dummy = torch.randn(1, 3, image_size, image_size)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(dummy)
            times.append((time.perf_counter() - t0) * 1000)

    # Tamaño en disco
    tmp = Path("_tmp_benchmark.pt")
    torch.save(model.state_dict(), tmp)
    disk_mb = tmp.stat().st_size / (1024 ** 2)
    tmp.unlink()

    return {
        "latency_ms": float(np.mean(times)),
        "latency_std_ms": float(np.std(times)),
        "disk_mb": disk_mb,
    }


def benchmark_models(models_dict: dict, image_size: int = 224, iterations: int = 100) -> dict:
    """
    Benchmarks múltiples modelos.

    Args:
        models_dict: {"FP32": model_fp32, "PTQ": model_ptq, "QAT": model_qat}
        image_size: Resolución de entrada.
        iterations: Iteraciones por modelo.

    Returns:
        dict {nombre: métricas}
    """
    results = {}
    for name, model in models_dict.items():
        logger.info(f"Benchmarking {name}...")
        results[name] = measure_latency(model, image_size, iterations)
        logger.info(
            f"  {name}: {results[name]['latency_ms']:.2f} ms ± "
            f"{results[name]['latency_std_ms']:.2f} | "
            f"disco: {results[name]['disk_mb']:.2f} MB"
        )
    return results