"""
Bloque C: Preservación de Estructura de Vecindad.
  - Overlap@k: fracción de vecinos k-NN compartidos entre FP32 e INT8.
  - Trustworthiness: grado en que los nuevos vecinos (INT8) son confiables.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import trustworthiness as _sklearn_trustworthiness


def _knn_indices(emb: torch.Tensor, k: int) -> torch.Tensor:
    """Retorna los índices de los k vecinos más cercanos (excluye self)."""
    emb = F.normalize(emb.float(), dim=1)
    # Similitud coseno → distancia; usamos argsort descendente
    sim = emb @ emb.T
    sim.fill_diagonal_(-float("inf"))
    return sim.argsort(dim=1, descending=True)[:, :k]


def overlap_at_k(emb_fp32: torch.Tensor, emb_int8: torch.Tensor, k: int = 10) -> float:
    """
    Overlap@k: fracción media de vecinos k-NN preservados tras la cuantización.

    Args:
        emb_fp32: Embeddings FP32 (N, D).
        emb_int8: Embeddings INT8→FP32 (N, D).
        k: Tamaño del vecindario.

    Returns:
        Overlap ∈ [0, 1]. 1 = vecindario idéntico.
    """
    nn_fp32 = _knn_indices(emb_fp32, k)   # (N, k)
    nn_int8 = _knn_indices(emb_int8, k)   # (N, k)

    overlaps = []
    for i in range(emb_fp32.shape[0]):
        s1 = set(nn_fp32[i].tolist())
        s2 = set(nn_int8[i].tolist())
        overlaps.append(len(s1 & s2) / k)

    return float(np.mean(overlaps))


def trustworthiness_score(
    emb_fp32: torch.Tensor,
    emb_int8: torch.Tensor,
    k: int = 10,
) -> float:
    """
    Trustworthiness (Venna & Kaski, 2006): mide si los vecinos en el espacio
    INT8 son también cercanos en FP32 (penaliza falsas cercanías).

    Args:
        emb_fp32: Espacio de referencia (N, D).
        emb_int8: Espacio proyectado/cuantizado (N, D).
        k: Tamaño del vecindario.

    Returns:
        Trustworthiness ∈ [0, 1]. 1 = sin vecinos espurios.
    """
    X = emb_fp32.float().numpy()
    X_emb = emb_int8.float().numpy()
    return float(_sklearn_trustworthiness(X, X_emb, n_neighbors=k, metric="cosine"))


def run(emb_fp32: torch.Tensor, emb_int8: torch.Tensor, k: int = 10) -> dict:
    """Ejecuta todas las métricas del Bloque C."""
    return {
        f"overlap_at_{k}": overlap_at_k(emb_fp32, emb_int8, k),
        "trustworthiness": trustworthiness_score(emb_fp32, emb_int8, k),
    }