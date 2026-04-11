"""
BalancedClassSampler: muestreo balanceado por clase para CUB-200.
Garantiza que cada mini-batch contenga muestras de todas las clases,
lo cual es crítico para la geometría del espacio latente con ArcFace.
"""

import torch
from torch.utils.data import Sampler
from collections import defaultdict
from typing import Iterator, List


class BalancedClassSampler(Sampler):
    """
    Sampler que garantiza exactamente `n_samples_per_class` imágenes por clase
    en cada época. Útil para estabilizar el entrenamiento con pérdidas métricas.

    Args:
        labels: Lista de etiquetas enteras del dataset.
        n_samples_per_class: Número de muestras por clase por época.
    """

    def __init__(self, labels: List[int], n_samples_per_class: int = 4):
        self.n_samples_per_class = n_samples_per_class
        self.class_indices: dict = defaultdict(list)

        for idx, label in enumerate(labels):
            self.class_indices[int(label)].append(idx)

        self.classes = list(self.class_indices.keys())
        self.n_classes = len(self.classes)

    def __iter__(self) -> Iterator[int]:
        indices = []
        for cls in self.classes:
            cls_idx = self.class_indices[cls]
            if len(cls_idx) >= self.n_samples_per_class:
                chosen = torch.randperm(len(cls_idx))[:self.n_samples_per_class].tolist()
            else:
                chosen = torch.randint(len(cls_idx), (self.n_samples_per_class,)).tolist()
            indices.extend([cls_idx[i] for i in chosen])

        # Shuffle entre clases manteniendo el balance dentro de cada clase
        perm = torch.randperm(len(indices)).tolist()
        return iter([indices[i] for i in perm])

    def __len__(self) -> int:
        return self.n_classes * self.n_samples_per_class