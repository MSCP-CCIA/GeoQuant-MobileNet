"""
Bloque E: Calidad de Representación para Downstream Tasks.
  - kNN accuracy: accuracy de clasificación k-NN sobre los embeddings.
  - Linear probe: accuracy de una regresión logística lineal sobre los embeddings.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize


def knn_accuracy(
    emb: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5,
) -> float:
    """
    kNN accuracy: clasificación k-NN en el espacio de embeddings.
    Usa distancia coseno. Leave-one-out cuando N es pequeño.

    Args:
        emb: Embeddings (N, D).
        labels: Etiquetas enteras (N,).
        k: Número de vecinos.

    Returns:
        Accuracy ∈ [0, 1].
    """
    X = F.normalize(emb.float(), dim=1).numpy()
    y = labels.numpy()

    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute")
    clf.fit(X, y)
    preds = clf.predict(X)
    return float((preds == y).mean())


def linear_probe(
    emb_train: torch.Tensor,
    labels_train: torch.Tensor,
    emb_val: torch.Tensor,
    labels_val: torch.Tensor,
    max_iter: int = 500,
) -> float:
    """
    Linear probe: regresión logística sobre los embeddings congelados.
    Mide la calidad de representación lineal del espacio latente.

    Args:
        emb_train: Embeddings de entrenamiento (N_train, D).
        labels_train: Etiquetas de entrenamiento (N_train,).
        emb_val: Embeddings de validación (N_val, D).
        labels_val: Etiquetas de validación (N_val,).
        max_iter: Iteraciones máximas del solver.

    Returns:
        Accuracy en validación ∈ [0, 1].
    """
    X_train = normalize(emb_train.float().numpy())
    X_val = normalize(emb_val.float().numpy())
    y_train = labels_train.numpy()
    y_val = labels_val.numpy()

    clf = LogisticRegression(max_iter=max_iter, solver="lbfgs", multi_class="multinomial")
    clf.fit(X_train, y_train)
    return float(clf.score(X_val, y_val))


def run(
    emb: torch.Tensor,
    labels: torch.Tensor,
    emb_val: torch.Tensor = None,
    labels_val: torch.Tensor = None,
    k: int = 5,
) -> dict:
    """
    Ejecuta todas las métricas del Bloque E.
    Si emb_val es None, usa emb/labels para ambos splits (útil en evaluación rápida).
    """
    result = {"knn_accuracy": knn_accuracy(emb, labels, k=k)}

    if emb_val is not None and labels_val is not None:
        result["linear_probe"] = linear_probe(emb, labels, emb_val, labels_val)

    return result