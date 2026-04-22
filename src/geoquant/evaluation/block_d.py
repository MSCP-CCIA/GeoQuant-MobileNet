"""
Bloque D: Geometría Intrínseca del Espacio Latente.
  - EDim (Effective Dimensionality): dimensionalidad efectiva vía entropía de valores singulares.
"""

import torch

def effective_dim(emb: torch.Tensor) -> float:
    """
    EDim (Effective Dimensionality): entropía de la distribución de valores singulares.
    Mide cuántas dimensiones del espacio latente están efectivamente en uso.

    EDim = exp(-sum(p_i * log(p_i)))  donde p_i = σ_i / sum(σ)

    Args:
        emb: Embeddings (N, D).

    Returns:
        EDim >= 1. Mayor = espacio más rico espectralmente y menos colapsado.
    """
    emb = emb.float()

    emb_centered = emb - emb.mean(dim=0)
    _, S, _ = torch.linalg.svd(emb_centered, full_matrices=False)

    p = S / S.sum().clamp(min=1e-8)
    entropy = -(p * (p + 1e-10).log()).sum()

    return entropy.exp().item()


def run(emb_fp32: torch.Tensor, emb_quant: torch.Tensor) -> dict:
    """Ejecuta las métricas del Bloque D."""
    edim_fp32 = effective_dim(emb_fp32)
    edim_quant = effective_dim(emb_quant)

    return {
        "edim_fp32": edim_fp32,
        "edim_quant": edim_quant,
        "delta_edim": edim_quant - edim_fp32,
    }