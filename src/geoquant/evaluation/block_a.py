"""
Bloque A: Degradación Geométrica Directa.
  - CosineDrift: desplazamiento medio del coseno entre embeddings FP32 e INT8.
  - RRE (Relative Representation Error): error relativo de representación.
"""

import torch
import torch.nn.functional as F


def cosine_drift(emb_fp32: torch.Tensor, emb_int8: torch.Tensor) -> float:
    """
    Mide cuánto se desplazan los embeddings tras la cuantización.
    Calcula 1 - cos_sim para cada par y devuelve la media.

    Args:
        emb_fp32: Embeddings originales (N, D), FP32.
        emb_int8: Embeddings cuantizados (N, D), INT8→FP32.

    Returns:
        Drift medio en [0, 2]. 0 = sin degradación.
    """
    emb_fp32 = F.normalize(emb_fp32.float(), dim=1)
    emb_int8 = F.normalize(emb_int8.float(), dim=1)
    cos_sim = (emb_fp32 * emb_int8).sum(dim=1)  # (N,)
    drift = (1.0 - cos_sim).mean().item()
    return drift


def rre(emb_fp32: torch.Tensor, emb_int8: torch.Tensor) -> float:
    """
    Relative Representation Error (RRE): norma del error relativa a FP32.

    RRE = mean(||e_fp32 - e_int8|| / ||e_fp32||)

    Args:
        emb_fp32: Embeddings originales (N, D).
        emb_int8: Embeddings cuantizados (N, D).

    Returns:
        RRE media. 0 = sin error.
    """
    emb_fp32 = emb_fp32.float()
    emb_int8 = emb_int8.float()
    error = torch.norm(emb_fp32 - emb_int8, dim=1)
    norm_fp32 = torch.norm(emb_fp32, dim=1).clamp(min=1e-8)
    return (error / norm_fp32).mean().item()


def run(emb_fp32: torch.Tensor, emb_int8: torch.Tensor) -> dict:
    """Ejecuta todas las métricas del Bloque A."""
    return {
        "cosine_drift": cosine_drift(emb_fp32, emb_int8),
        "rre": rre(emb_fp32, emb_int8),
    }