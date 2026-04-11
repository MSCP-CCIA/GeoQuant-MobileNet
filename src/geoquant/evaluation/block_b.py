"""
Bloque B: Similitud de Representaciones.
  - CKA (Centered Kernel Alignment): similitud entre espacios de representación.
  - rho_sim (ρ_sim): correlación de Spearman entre matrices de distancias.
  - Alignment: alineación media de embeddings del mismo par.
  - Uniformity: dispersión de la distribución en la hiperesfera.
"""

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# CKA (Linear)
# ---------------------------------------------------------------------------

def _center(K: torch.Tensor) -> torch.Tensor:
    """Centra la matriz kernel (HSIC)."""
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype, device=K.device) - 1.0 / n
    return H @ K @ H


def cka_linear(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    CKA lineal entre matrices de activación X e Y.

    Args:
        X: (N, D1) embeddings modelo A.
        Y: (N, D2) embeddings modelo B.

    Returns:
        CKA ∈ [0, 1]. 1 = representaciones idénticas.
    """
    X = X.float()
    Y = Y.float()
    K = X @ X.T
    L = Y @ Y.T
    Kc = _center(K)
    Lc = _center(L)
    hsic_xy = (Kc * Lc).sum()
    hsic_xx = (Kc * Kc).sum().sqrt()
    hsic_yy = (Lc * Lc).sum().sqrt()
    denom = hsic_xx * hsic_yy
    return (hsic_xy / denom.clamp(min=1e-8)).item()


# ---------------------------------------------------------------------------
# ρ_sim: Correlación de Spearman entre distancias
# ---------------------------------------------------------------------------

def rho_sim(emb_a: torch.Tensor, emb_b: torch.Tensor, sample: int = 500) -> float:
    """
    Correlación de Spearman entre las distancias par-a-par de dos espacios.

    Args:
        emb_a, emb_b: (N, D) embeddings de dos modelos.
        sample: Número de pares para subsamplear (evita O(N²) con N grande).

    Returns:
        ρ_sim ∈ [-1, 1]. 1 = orden de distancias perfectamente preservado.
    """
    n = emb_a.shape[0]
    idx = torch.randperm(n)[:sample]
    a = emb_a[idx].float()
    b = emb_b[idx].float()

    dist_a = torch.cdist(a, a).triu(diagonal=1).flatten()
    dist_b = torch.cdist(b, b).triu(diagonal=1).flatten()

    mask = dist_a > 0
    rho, _ = spearmanr(dist_a[mask].numpy(), dist_b[mask].numpy())
    return float(rho)


# ---------------------------------------------------------------------------
# Alignment y Uniformity (Wang & Isola, 2020)
# ---------------------------------------------------------------------------

def alignment(emb: torch.Tensor, labels: torch.Tensor, alpha: float = 2.0) -> float:
    """
    Alineación: distancia media entre embeddings de la misma clase.
    Menor = mejor (los positivos están cerca en la hiperesfera).
    """
    emb = F.normalize(emb.float(), dim=1)
    loss = 0.0
    count = 0
    unique_classes = labels.unique()
    for c in unique_classes:
        mask = labels == c
        e = emb[mask]
        if e.shape[0] < 2:
            continue
        # Distancia L2 entre todos los pares de la clase
        diff = e.unsqueeze(0) - e.unsqueeze(1)  # (n, n, D)
        dist = diff.norm(dim=2).pow(alpha)
        n = e.shape[0]
        loss += dist.triu(diagonal=1).sum() / (n * (n - 1) / 2)
        count += 1
    return (loss / max(count, 1)).item()


def uniformity(emb: torch.Tensor, t: float = 2.0) -> float:
    """
    Uniformidad: dispersión en la hiperesfera.
    Mayor (menos negativo) = distribución más uniforme.
    """
    emb = F.normalize(emb.float(), dim=1)
    sq_dist = torch.cdist(emb, emb).pow(2)
    return sq_dist.mul(-t).exp().mean().log().item()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(emb_fp32: torch.Tensor, emb_int8: torch.Tensor, labels: torch.Tensor) -> dict:
    """Ejecuta todas las métricas del Bloque B."""
    return {
        "cka": cka_linear(emb_fp32, emb_int8),
        "rho_sim": rho_sim(emb_fp32, emb_int8),
        "alignment_fp32": alignment(emb_fp32, labels),
        "alignment_int8": alignment(emb_int8, labels),
        "uniformity_fp32": uniformity(emb_fp32),
        "uniformity_int8": uniformity(emb_int8),
    }