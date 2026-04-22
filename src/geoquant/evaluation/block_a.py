"""
Bloque A: Degradación Geométrica Directa.
  - CosineDrift: desplazamiento medio del coseno entre embeddings FP32 y cuantizados.
  - CosineDrift por muestra.
  - Separación angular por muestra (radianes y grados).
"""

import math
import torch
import torch.nn.functional as F


def cosine_similarity_per_sample(emb_fp32: torch.Tensor, emb_quant: torch.Tensor) -> torch.Tensor:
    """
    Calcula la similitud coseno por muestra.

    Args:
        emb_fp32: Embeddings originales (N, D), FP32.
        emb_quant: Embeddings cuantizados (N, D), en dominio real.

    Returns:
        Tensor (N,) con similitud coseno por muestra.
    """
    emb_fp32 = F.normalize(emb_fp32.float(), dim=1)
    emb_quant = F.normalize(emb_quant.float(), dim=1)
    cos_sim = (emb_fp32 * emb_quant).sum(dim=1)
    return torch.clamp(cos_sim, -1.0, 1.0)


def cosine_drift_per_sample(emb_fp32: torch.Tensor, emb_quant: torch.Tensor) -> torch.Tensor:
    """
    Calcula 1 - cos_sim para cada par de embeddings.

    Args:
        emb_fp32: Embeddings originales (N, D), FP32.
        emb_quant: Embeddings cuantizados (N, D), en dominio real.

    Returns:
        Tensor (N,) con drift individual.
    """
    cos_sim = cosine_similarity_per_sample(emb_fp32, emb_quant)
    return 1.0 - cos_sim


def cosine_drift(emb_fp32: torch.Tensor, emb_quant: torch.Tensor) -> float:
    """
    Calcula el cosine drift medio.

    Args:
        emb_fp32: Embeddings originales (N, D), FP32.
        emb_quant: Embeddings cuantizados (N, D), en dominio real.

    Returns:
        Drift medio en [0, 2]. 0 = sin degradación.
    """
    drift = cosine_drift_per_sample(emb_fp32, emb_quant)
    return drift.mean().item()


def angular_separation_per_sample_rad(emb_fp32: torch.Tensor, emb_quant: torch.Tensor) -> torch.Tensor:
    """
    Calcula la separación angular por muestra en radianes.

    Args:
        emb_fp32: Embeddings originales (N, D), FP32.
        emb_quant: Embeddings cuantizados (N, D), en dominio real.

    Returns:
        Tensor (N,) con ángulos en radianes.
    """
    cos_sim = cosine_similarity_per_sample(emb_fp32, emb_quant)
    return torch.acos(cos_sim)


def angular_separation_per_sample_deg(emb_fp32: torch.Tensor, emb_quant: torch.Tensor) -> torch.Tensor:
    """
    Calcula la separación angular por muestra en grados.

    Args:
        emb_fp32: Embeddings originales (N, D), FP32.
        emb_quant: Embeddings cuantizados (N, D), en dominio real.

    Returns:
        Tensor (N,) con ángulos en grados.
    """
    angles_rad = angular_separation_per_sample_rad(emb_fp32, emb_quant)
    return angles_rad * (180.0 / math.pi)


def run(emb_fp32: torch.Tensor, emb_quant: torch.Tensor) -> dict:
    """Ejecuta las métricas núcleo del Bloque A."""
    return {
        "cosine_drift": cosine_drift(emb_fp32, emb_quant),
        "cosine_similarity_per_sample": cosine_similarity_per_sample(emb_fp32, emb_quant),
        "cosine_drift_per_sample": cosine_drift_per_sample(emb_fp32, emb_quant),
        "angular_separation_per_sample_rad": angular_separation_per_sample_rad(emb_fp32, emb_quant),
        "angular_separation_per_sample_deg": angular_separation_per_sample_deg(emb_fp32, emb_quant),
    }