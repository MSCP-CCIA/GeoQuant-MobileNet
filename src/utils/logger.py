import logging
import sys
import os
from datetime import datetime

def setup_logger(name="FaceQuant", log_level=logging.INFO):
    """
    Configura un logger profesional con salida a consola y archivo.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Evitar duplicados si se llama varias veces
    if logger.hasHandlers():
        return logger

    # Formato de los mensajes
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. Handler para Consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. Handler para Archivo (Persistencia de experimentos)
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = f"{log_dir}/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# Instancia global para uso rápido
logger = setup_logger()
