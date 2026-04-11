"""
PTQ (Post-Training Quantization) via torchao.
Aplica Int8DynamicActivationInt8Weight sin necesidad de reentrenamiento.
"""

import copy
from pathlib import Path

import torch

from geoquant.utils.logging import get_logger

logger = get_logger(__name__)


def apply_ptq(model: torch.nn.Module, output_path: str) -> torch.nn.Module:
    """
    Cuantiza el modelo con PTQ dinámico INT8 (torchao).

    Args:
        model: Backbone FP32 ya entrenado (en eval mode).
        output_path: Ruta donde guardar el state_dict cuantizado.

    Returns:
        Modelo cuantizado en CPU.
    """
    try:
        from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
    except ImportError as e:
        raise ImportError("Instala torchao: pip install torchao") from e

    logger.info("Aplicando PTQ (torchao Int8DynamicActivationInt8Weight)...")

    ptq_model = copy.deepcopy(model)
    ptq_model.eval()
    quantize_(ptq_model, Int8DynamicActivationInt8WeightConfig())

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ptq_model.state_dict(), out)
    logger.info(f"Modelo PTQ guardado → {out}")

    return ptq_model