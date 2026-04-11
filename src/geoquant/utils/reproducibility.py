"""
seed_everything: fija todas las fuentes de aleatoriedad del experimento.
Garantiza reproducibilidad completa entre ejecuciones.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Fija el seed en Python, NumPy, PyTorch (CPU y CUDA) y variables de entorno.

    Args:
        seed: Semilla global del experimento (default 42, desde config.yaml).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Garantiza determinismo en convoluciones CUDA (ligera pérdida de velocidad)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False