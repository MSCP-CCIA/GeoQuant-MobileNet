import os
from torchvision import datasets
import yaml
from src.utils.logger import logger

def main():
    with open("configs/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    cifar_cfg = next(ds for ds in config['datasets'] if ds['name'] == 'cifar10')
    target_path = cifar_cfg['path']
    
    logger.info(f"Iniciando descarga de CIFAR-10 en: {target_path}")
    
    datasets.CIFAR10(root=target_path, train=True, download=True)
    datasets.CIFAR10(root=target_path, train=False, download=True)

    logger.info("Descarga de CIFAR-10 completada exitosamente.")
    logger.warning(f"No olvides ejecutar: uv run dvc add {target_path}")

if __name__ == "__main__":
    main()
