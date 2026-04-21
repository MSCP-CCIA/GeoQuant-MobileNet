"""
Bloque B: Similitud de Representaciones.
  - CKA (Centered Kernel Alignment): similitud estructural global entre espacios de representación.
  - Alignment: cercanía media entre pares positivos (misma clase).
  - Uniformity: dispersión global de la distribución en la hiperesfera.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CKA (Linear)
# ---------------------------------------------------------------------------

def _center(K: torch.Tensor) -> torch.Tensor:
    """Centra una matriz kernel/Gram."""
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype, device=K.device) - \
        (1.0 / n) * torch.ones((n, n), dtype=K.dtype, device=K.device)
    return H @ K @ H


def cka_linear(emb_fp32: torch.Tensor, emb_quant: torch.Tensor) -> float:
    """
    Calcula CKA lineal entre dos matrices de embeddings.

    Args:
        emb_fp32: Embeddings originales (N, D), FP32.
        emb_quant: Embeddings cuantizados (N, D), en dominio real.

    Returns:
        Escalar CKA. Valores altos indican mayor similitud estructural global.
    """
    emb_fp32 = emb_fp32.float()
    emb_quant = emb_quant.float()

    K = emb_fp32 @ emb_fp32.T
    L = emb_quant @ emb_quant.T

    Kc = _center(K)
    Lc = _center(L)

    hsic_xy = (Kc * Lc).sum()
    hsic_xx = (Kc * Kc).sum().sqrt()
    hsic_yy = (Lc * Lc).sum().sqrt()

    return (hsic_xy / (hsic_xx * hsic_yy).clamp(min=1e-8)).item()


# ---------------------------------------------------------------------------
# Alignment y Uniformity
# ---------------------------------------------------------------------------

def alignment(emb: torch.Tensor, labels: torch.Tensor, alpha: float = 2.0) -> float:
    """
    Calcula alignment sobre una matriz de embeddings.

    Args:
        emb: Embeddings (N, D).
        labels: Etiquetas de clase (N,).
        alpha: Exponente de la distancia, típicamente 2.

    Returns:
        Escalar con la distancia media entre todos los pares positivos.
        Menor = mejor compactación intra-clase.
    """
    emb = F.normalize(emb.float(), dim=1)

    total = 0.0
    num_pairs = 0

    for c in labels.unique():
        e = emb[labels == c]
        n = e.shape[0]

        if n < 2:
            continue

        diff = e.unsqueeze(0) - e.unsqueeze(1)
        dist = diff.norm(dim=2).pow(alpha)

        upper = dist.triu(diagonal=1)
        total += upper.sum()
        num_pairs += n * (n - 1) / 2

    return (total / max(num_pairs, 1)).item()


def uniformity(emb: torch.Tensor, t: float = 2.0) -> float:
    """
    Calcula uniformity sobre una matriz de embeddings.

    Args:
        emb: Embeddings (N, D).
        t: Parámetro de escala.

    Returns:
        Escalar de uniformidad global.
        Más negativo = mejor dispersión global en la hiperesfera.
    """
    emb = F.normalize(emb.float(), dim=1)

    sq_dist = torch.pdist(emb, p=2).pow(2)
    return sq_dist.mul(-t).exp().mean().log().item()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(emb_fp32: torch.Tensor, emb_quant: torch.Tensor, labels: torch.Tensor) -> dict:
    """Ejecuta las métricas del Bloque B."""
    alignment_fp32 = alignment(emb_fp32, labels)
    alignment_quant = alignment(emb_quant, labels)

    uniformity_fp32 = uniformity(emb_fp32)
    uniformity_quant = uniformity(emb_quant)

    return {
        "cka": cka_linear(emb_fp32, emb_quant),
        "alignment_fp32": alignment_fp32,
        "alignment_quant": alignment_quant,
        "delta_alignment": alignment_quant - alignment_fp32,
        "uniformity_fp32": uniformity_fp32,
        "uniformity_quant": uniformity_quant,
        "delta_uniformity": uniformity_quant - uniformity_fp32,
    }