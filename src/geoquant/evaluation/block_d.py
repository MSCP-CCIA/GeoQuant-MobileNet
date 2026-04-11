"""
Bloque D: Geometría Intrínseca del Espacio Latente.
  - IsoMean: error de isometría promedio (preservación de distancias relativas).
  - EDim (Effective Dimensionality): dimensionalidad efectiva via entropía de valores singulares.
"""

import torch
import torch.nn.functional as F
import numpy as np


def iso_mean(emb_fp32: torch.Tensor, emb_int8: torch.Tensor, sample: int = 1000) -> float:
    """
    IsoMean: desviación media de las distancias relativas entre FP32 e INT8.
    Mide si la cuantización preserva la isometría del espacio latente.

    IsoMean = mean(|d_fp32(i,j) - d_int8(i,j)| / d_fp32(i,j))

    Args:
        emb_fp32: Embeddings FP32 (N, D).
        emb_int8: Embeddings INT8→FP32 (N, D).
        sample: Número de muestras para subsamplear pares.

    Returns:
        IsoMean ∈ [0, ∞). 0 = isometría perfecta.
    """
    n = emb_fp32.shape[0]
    idx = torch.randperm(n)[:sample]
    a = emb_fp32[idx].float()
    b = emb_int8[idx].float()

    dist_a = torch.cdist(a, a)   # (sample, sample)
    dist_b = torch.cdist(b, b)

    # Pares superiores (excluir diagonal)
    mask = torch.triu(torch.ones(sample, sample, dtype=torch.bool), diagonal=1)
    da = dist_a[mask].clamp(min=1e-8)
    db = dist_b[mask]

    return ((da - db).abs() / da).mean().item()


def effective_dim(emb: torch.Tensor) -> float:
    """
    EDim (Effective Dimensionality): entropía de la distribución de valores singulares.
    Mide cuántas dimensiones del espacio latente están efectivamente en uso.

    EDim = exp(-sum(p_i * log(p_i)))  donde p_i = σ_i² / sum(σ²)

    Args:
        emb: Embeddings (N, D).

    Returns:
        EDim ∈ [1, D]. Mayor = espacio más isótropo y rico.
    """
    emb = emb.float()
    # SVD truncada para eficiencia
    _, S, _ = torch.linalg.svd(emb - emb.mean(dim=0), full_matrices=False)
    S2 = S.pow(2)
    p = S2 / S2.sum().clamp(min=1e-8)
    # Entropía de Shannon sobre los valores propios
    entropy = -(p * (p + 1e-10).log()).sum()
    return entropy.exp().item()


def run(emb_fp32: torch.Tensor, emb_int8: torch.Tensor) -> dict:
    """Ejecuta todas las métricas del Bloque D."""
    return {
        "iso_mean": iso_mean(emb_fp32, emb_int8),
        "edim_fp32": effective_dim(emb_fp32),
        "edim_int8": effective_dim(emb_int8),
    }