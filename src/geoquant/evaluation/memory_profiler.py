"""
Medición de peak de memoria RAM durante inferencia en CPU usando tracemalloc.
No usa CUDA — diseñado exclusivamente para benchmarks en CPU restringido.
"""

import tracemalloc
from typing import Optional

import torch
from torch.utils.data import DataLoader

from geoquant.utils.logging import get_logger

logger = get_logger(__name__)


def measure_memory(
    model: torch.nn.Module,
    dataloader: DataLoader,
    n_batches: Optional[int] = None,
) -> dict:
    """
    Mide el pico de memoria RAM (MB) durante un forward pass en CPU.

    Usa tracemalloc para capturar la asignación de memoria de Python/PyTorch
    sin incluir overhead de inicialización del proceso.

    Args:
        model: Modelo a perfilar (se mueve a CPU).
        dataloader: DataLoader con imágenes de prueba (imágenes dummy o reales).
        n_batches: Número de batches a procesar. None procesa el dataloader completo.

    Returns:
        dict con:
            - 'peak_ram_mb': pico máximo de RAM durante la inferencia.
            - 'current_ram_mb': RAM en uso al terminar el último batch.
    """
    model.eval().to("cpu")

    tracemalloc.start()

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            _ = model(images.to("cpu"))
            if n_batches is not None and i + 1 >= n_batches:
                break

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "peak_ram_mb": peak / (1024 ** 2),
        "current_ram_mb": current / (1024 ** 2),
    }
