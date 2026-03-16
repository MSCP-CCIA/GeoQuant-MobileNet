import torchvision.models as models
import torch.nn as nn


def get_model(arch_name, num_classes=10):
    if arch_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=None)
        state_dict = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1.get_state_dict(progress=True)

        # Filtramos la cabeza para entrenarla desde cero
        filtered_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith('classifier')
        }
        model.load_state_dict(filtered_dict, strict=False)

        # Extraemos la dimensión de entrada de la última capa
        in_features = model.classifier[3].in_features

        # Reemplazamos el clasificador final inyectando BatchNorm1d
        model.classifier[3] = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, num_classes)
        )

        # El hook ahora apunta al BatchNorm (índice 0 dentro del Sequential del classifier[3])
        feature_layer = 'classifier.3.0'

    elif arch_name == 'shufflenet_v2':
        model = models.shufflenet_v2_x1_0(weights=None)
        state_dict = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1.get_state_dict(progress=True)

        # Filtramos la cabeza
        filtered_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith('conv5') and not k.startswith('fc')
        }
        model.load_state_dict(filtered_dict, strict=False)

        in_features = model.fc.in_features

        # Reemplazamos la capa FC inyectando BatchNorm1d
        model.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, num_classes)
        )

        # El hook ahora apunta al BatchNorm (índice 0 dentro del nuevo bloque fc)
        feature_layer = 'fc.0'

    else:
        raise ValueError(f"Arquitectura {arch_name} no soportada.")

    return model, feature_layer