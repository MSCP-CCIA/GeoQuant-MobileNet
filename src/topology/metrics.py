import numpy as np
from sklearn.metrics import davies_bouldin_score
import torch
def calculate_s_index(embeddings, targets):
    """
    Implementación formal del Índice de Separabilidad Corregido (S_sep).
    Basado en el inverso del Índice de Davies-Bouldin (Ecuación 38 de la tesis).
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    if len(np.unique(targets)) < 2:
        return 0.0, 0.0, 0.0

    # 1. Calcular el Índice de Davies-Bouldin (DBI)
    # DBI mide (dispersión_intra / distancia_inter). Menor es mejor.
    dbi = davies_bouldin_score(embeddings, targets)

    # 2. Calcular el S-Index Corregido (S_sep = 1 / DBI) según sección 1.4.3
    # Esto garantiza la interpretación "Mayor es Mejor".
    s_sep = 1.0 / dbi if dbi > 0 else 0.0

    # Métricas auxiliares para tus logs (opcional)
    # Aquí podrías calcular distancias medias si lo deseas para el log
    intra_dist = dbi # Representante de la relación de dispersión
    inter_dist = 1/dbi

    return float(s_sep), float(intra_dist), float(inter_dist)