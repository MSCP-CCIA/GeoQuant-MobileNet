import argparse
from pathlib import Path
import yaml
import torch

from geoquant.utils.reproducibility import seed_everything
from geoquant.utils.logging import get_logger
from geoquant.data.dataset import get_dataloaders
from geoquant.models.backbone import build_backbone
from geoquant.models.arcface import build_arcface

# Importamos nuestros nuevos métodos robustos
from geoquant.quantization.ptq import apply_ptq_static
from geoquant.quantization.qat import apply_qat_distillation

logger = get_logger(__name__)

def load_config(base: str, experiment: str = None) -> dict:
    with open(base) as f:
        config = yaml.safe_load(f)
    if experiment:
        with open(experiment) as f:
            override = yaml.safe_load(f)
        for section, values in override.items():
            if isinstance(values, dict):
                config.setdefault(section, {}).update(values)
            else:
                config[section] = values
    return config

def main():
    parser = argparse.ArgumentParser(description="GeoQuant — Cuantización de Espacio Latente")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--experiment", default="configs/experiment/ptq_static.yaml")
    args = parser.parse_args()

    config = load_config(args.config, args.experiment)
    seed_everything(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")

    # 1. Cargar datos (necesarios para calibración PTQ y validación QAT)
    train_loader, val_loader = get_dataloaders(config)

    # 2. Construir modelo maestro
    backbone = build_backbone(config)
    arcface = build_arcface(config, in_features=backbone.in_features)

    ## 3. Cargar los pesos perfectos en FP32 de nuestro finetuning
    # Fijamos la ruta explícitamente a la carpeta donde el baseline guardó el modelo
    fp32_path = Path("outputs/checkpoints/baseline_fp32/best_fp32_finetuning.pth")

    # Fallback por si en el futuro cambiaste los nombres
    if not fp32_path.exists():
        fp32_path = Path("outputs/checkpoints/baseline_fp32/best_fp32.pth")

    if not fp32_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo maestro en {fp32_path}. ¡Asegúrate de entrenarlo primero!")

    logger.info(f"Cargando pesos maestros FP32 desde: {fp32_path}")
    backbone.load_state_dict(torch.load(fp32_path, map_location=device, weights_only=True))

    # 4. Leer configuración de cuantización
    quant_cfg = config.get("quantization", {})
    method = quant_cfg.get("approach", "ptq")  # Esto vendrá sobreescrito por el yaml de experiment
    output_path = quant_cfg.get("output_path", f"outputs/quantized/model_{method}.pth")

    # 5. Ejecutar la ruta seleccionada
    if method == "ptq":
        logger.info("Ruta seleccionada: PTQ Estático")
        apply_ptq_static(
            model=backbone,
            calib_loader=train_loader, # Usamos los datos de entrenamiento para calibrar los observadores
            output_path=output_path,
            device=device
        )
    elif method == "qat":
        logger.info("Ruta seleccionada: QAT con Destilación Geométrica")
        epochs = quant_cfg.get("epochs", 10)
        lr = quant_cfg.get("lr", 5e-5)

        apply_qat_distillation(
            model_fp32=backbone,
            arcface_head=arcface,
            train_loader=train_loader,
            val_loader=val_loader,
            output_path=output_path,
            epochs=epochs,
            lr=lr,
            device=device
        )
    else:
        logger.error(f"Método de cuantización desconocido en config: {method}")

if __name__ == "__main__":
    main()