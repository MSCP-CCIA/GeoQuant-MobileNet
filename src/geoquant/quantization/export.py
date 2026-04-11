"""
Exportación a TorchScript INT8 para despliegue en producción.
Convierte el modelo cuantizado a un formato portable y optimizado.
"""

from pathlib import Path

import torch

from geoquant.utils.logging import get_logger

logger = get_logger(__name__)


def export_torchscript(
    model: torch.nn.Module,
    output_path: str,
    image_size: int = 224,
) -> None:
    """
    Exporta un modelo cuantizado a TorchScript (.pt).

    Args:
        model: Modelo INT8 ya convertido (en CPU, eval mode).
        output_path: Ruta de salida para el archivo .pt.
        image_size: Tamaño de imagen de entrada (default 224).
    """
    model.eval().to("cpu")
    torch.backends.quantized.engine = "fbgemm"

    dummy_input = torch.randn(1, 3, image_size, image_size)

    logger.info("Trazando modelo a TorchScript...")
    try:
        scripted = torch.jit.trace(model, dummy_input)
    except Exception:
        logger.warning("torch.jit.trace falló, intentando torch.jit.script...")
        scripted = torch.jit.script(model)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out))
    logger.info(f"TorchScript INT8 exportado → {out}")