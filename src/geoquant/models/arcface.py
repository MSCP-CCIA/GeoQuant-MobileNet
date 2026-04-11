"""
ArcFaceHead: cabeza de pérdida angular para entrenamiento métrico.
Implementa ArcFace (Deng et al., 2019) para maximizar la separabilidad
geométrica en la hiperesfera unitaria.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """
    Cabeza ArcFace: proyecta embeddings a logits angulares con margen aditivo.

    Args:
        in_features: Dimensión del embedding de entrada.
        num_classes: Número de clases (centroides en la hiperesfera).
        scale: Factor de escala s (norma del vector). Estándar: 30.0.
        margin: Margen angular m en radianes. Estándar: 0.5.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = scale
        self.m = margin

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Proyección a cosenos (ambos L2-normalizados)
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))

        # cos(θ + m) = cos θ · cos m − sin θ · sin m
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Relajación para θ + m > π (evita inestabilidad numérica)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Aplicar margen solo a la clase ground-truth
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s


def build_arcface(config: dict, in_features: int) -> ArcFaceHead:
    """Instancia ArcFaceHead desde configuración."""
    arc_cfg = config.get("model", {}).get("arcface", {})
    num_classes = config["data"]["num_classes"]
    return ArcFaceHead(
        in_features=in_features,
        num_classes=num_classes,
        scale=arc_cfg.get("scale", 30.0),
        margin=arc_cfg.get("margin", 0.5),
    )