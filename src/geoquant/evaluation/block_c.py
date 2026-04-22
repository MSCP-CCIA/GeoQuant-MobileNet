"""
Bloque C: Preservación de Estructura de Vecindad.
  - Overlap@k: fracción de vecinos k-NN compartidos entre FP32 y cuantización.
"""

import torch
import torch.nn.functional as F

def _knn_indices(emb: torch.Tensor, k: int) -> torch.Tensor:
    """
    Retorna los índices de los k vecinos más cercanos (excluye self).

    Args:
        emb: Embeddings (N, D).
        k: Tamaño del vecindario.

    Returns:
        Tensor (N, k) con índices de vecinos por muestra.
    """
    emb = F.normalize(emb.float(), dim=1)
    sim = emb @ emb.T
    sim.fill_diagonal_(-float("inf"))
    return sim.topk(k=k, dim=1, largest=True).indices


def overlap_at_k_per_sample(
    emb_fp32: torch.Tensor,
    emb_quant: torch.Tensor,
    k: int = 10
) -> torch.Tensor:
    """
    Calcula Overlap@k por muestra.

    Args:
        emb_fp32: Embeddings originales (N, D), FP32.
        emb_quant: Embeddings cuantizados (N, D), en dominio real.
        k: Tamaño del vecindario.

    Returns:
        Tensor (N,) con la fracción de vecinos preservados por muestra.
    """
    nn_fp32 = _knn_indices(emb_fp32, k)    # (N, k)
    nn_quant = _knn_indices(emb_quant, k)  # (N, k)

    overlaps = []
    for i in range(emb_fp32.shape[0]):
        s1 = set(nn_fp32[i].tolist())
        s2 = set(nn_quant[i].tolist())
        overlaps.append(len(s1 & s2) / k)

    return torch.tensor(overlaps, dtype=torch.float32)


def overlap_at_k(
    emb_fp32: torch.Tensor,
    emb_quant: torch.Tensor,
    k: int = 10
) -> float:
    """
    Overlap@k: fracción media de vecinos k-NN preservados tras la cuantización.

    Args:
        emb_fp32: Embeddings originales (N, D), FP32.
        emb_quant: Embeddings cuantizados (N, D), en dominio real.
        k: Tamaño del vecindario.

    Returns:
        Overlap en [0, 1]. 1 = vecindario idéntico.
    """
    overlaps = overlap_at_k_per_sample(emb_fp32, emb_quant, k)
    return overlaps.mean().item()


def run(emb_fp32: torch.Tensor, emb_quant: torch.Tensor, ks=(1, 5, 10, 20)) -> dict:
    """Ejecuta las métricas del Bloque C para múltiples valores de k."""
    results = {}
    for k in ks:
        results[f"overlap_at_{k}"] = overlap_at_k(emb_fp32, emb_quant, k)
    return results