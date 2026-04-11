
"""
MobileNetV3 wrapper para extracción de embeddings geométricos.
Reemplaza la cabeza clasificadora por BatchNorm1d para producir
embeddings L2-normalizables en la hiperesfera unitaria.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetV3Backbone(nn.Module):
    """
    MobileNetV3-Small pre-entrenado con ImageNet.
    La capa classifier[-1] se sustituye por BatchNorm1d para obtener
    embeddings estables antes de la pérdida ArcFace.

    Args:
        pretrained: Cargar pesos ImageNet1K.
        embedding_size: Dimensión de salida (determinada por la arquitectura).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = models.mobilenet_v3_small(weights=None)

        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            state_dict = weights.get_state_dict(progress=True)
            # Filtrar cabeza clasificadora — la reemplazamos
            filtered = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
            base.load_state_dict(filtered, strict=False)

        self.in_features: int = base.classifier[3].in_features

        # Sustituir Linear final por BatchNorm1d (sin bias aditivo)
        base.classifier[3] = nn.BatchNorm1d(self.in_features)

        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_backbone(config: dict) -> MobileNetV3Backbone:
    """Instancia el backbone a partir del dict de configuración."""
    model_cfg = config.get("model", {})
    arch = model_cfg.get("backbone", "mobilenet_v3_small")
    pretrained = model_cfg.get("pretrained", True)

    if arch != "mobilenet_v3_small":
        raise ValueError(f"Backbone '{arch}' no soportado. Usa 'mobilenet_v3_small'.")

    return MobileNetV3Backbone(pretrained=pretrained)