"""
Wrapper de logging con soporte opcional para MLflow.
Provee get_logger() como punto de entrada único en todo el paquete.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str = "geoquant", level: int = logging.INFO) -> logging.Logger:
    """
    Retorna un logger configurado con handlers de consola y archivo.
    Idempotente: llamadas repetidas devuelven el mismo logger.

    Args:
        name: Nombre del logger (tipicamente __name__ del módulo).
        level: Nivel de logging.

    Returns:
        logging.Logger listo para usar.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Consola
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Archivo
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# MLflow helper (opcional — no requiere mlflow instalado)
# ---------------------------------------------------------------------------

class MLflowLogger:
    """
    Thin wrapper sobre mlflow para loguear experimentos.
    Se degrada graciosamente si mlflow no está instalado.
    """

    def __init__(self, experiment_name: str = "geoquant"):
        try:
            import mlflow
            self._mlflow = mlflow
            mlflow.set_experiment(experiment_name)
            self._active = True
        except ImportError:
            self._active = False
            get_logger().warning("mlflow no instalado — logging solo local.")

    def log_params(self, params: dict) -> None:
        if self._active:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int = None) -> None:
        if self._active:
            self._mlflow.log_metrics(metrics, step=step)

    def start_run(self, run_name: str = None):
        if self._active:
            return self._mlflow.start_run(run_name=run_name)

    def end_run(self) -> None:
        if self._active:
            self._mlflow.end_run()