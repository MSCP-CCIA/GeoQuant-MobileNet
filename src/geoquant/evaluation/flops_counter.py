"""
Estimación de FLOPs y parámetros del modelo usando thop.
Mide operaciones del forward pass con un input de batch_size=1 en CPU.
"""

import torch
from thop import clever_format, profile

from geoquant.utils.logging import get_logger

logger = get_logger(__name__)


def count_flops(model: torch.nn.Module, image_size: int = 224) -> dict:
    """
    Estima FLOPs y número de parámetros para un input estándar de inferencia.

    Args:
        model: Modelo a analizar (se mueve a CPU).
        image_size: Resolución de la imagen de entrada (cuadrada).

    Returns:
        dict con:
            - 'flops': FLOPs crudos (float).
            - 'params': Número de parámetros (float).
            - 'flops_str': FLOPs formateados (ej. "72.868M").
            - 'params_str': Parámetros formateados (ej. "1.521M").
    """
    model.eval().to("cpu")
    dummy = torch.randn(1, 3, image_size, image_size)

    flops, params = profile(model, inputs=(dummy,), verbose=False)
    flops_str, params_str = clever_format([flops, params], "%.3f")

    logger.info(f"FLOPs: {flops_str} | Parámetros: {params_str}")

    return {
        "flops": flops,
        "flops_str": flops_str,
    }
