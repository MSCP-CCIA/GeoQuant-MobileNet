import torchvision.models as models
import torch.nn as nn
from src.models.arcface import ArcFaceLayer


def get_arcface_model(arch_name, num_classes=10):
    if arch_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=None)
        state_dict = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1.get_state_dict(progress=True)

        filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
        model.load_state_dict(filtered_dict, strict=False)

        in_features = model.classifier[3].in_features

        # EL SECRETO: El modelo termina en BatchNorm1d. No hay capa Linear.
        model.classifier[3] = nn.BatchNorm1d(in_features)

    elif arch_name == 'shufflenet_v2':
        model = models.shufflenet_v2_x1_0(weights=None)
        state_dict = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1.get_state_dict(progress=True)

        filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith('conv5') and not k.startswith('fc')}
        model.load_state_dict(filtered_dict, strict=False)

        in_features = model.fc.in_features

        # Termina en BatchNorm1d
        model.fc = nn.BatchNorm1d(in_features)

    else:
        raise ValueError(f"Arquitectura {arch_name} no soportada.")

    # Instanciamos la cabeza ArcFace de forma independiente
    arcface_head = ArcFaceLayer(in_features=in_features, num_classes=num_classes)

    return model, arcface_head