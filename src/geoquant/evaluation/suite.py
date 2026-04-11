"""
EvaluationSuite: orquesta los 5 bloques de evaluación geométrica.
Recibe embeddings FP32 e INT8 y devuelve un dict unificado de métricas.
"""

import torch
from geoquant.evaluation import block_a, block_b, block_c, block_d, block_e
from geoquant.utils.logging import get_logger

logger = get_logger(__name__)


class EvaluationSuite:
    """
    Orquestador de evaluación geométrica end-to-end.

    Uso típico:
        suite = EvaluationSuite()
        results = suite.run(emb_fp32, emb_int8, labels)
    """

    def __init__(self, k_neighbors: int = 10):
        self.k = k_neighbors

    def run(
        self,
        emb_fp32: torch.Tensor,
        emb_int8: torch.Tensor,
        labels: torch.Tensor,
        emb_fp32_train: torch.Tensor = None,
        labels_train: torch.Tensor = None,
    ) -> dict:
        """
        Ejecuta los 5 bloques de evaluación.

        Args:
            emb_fp32: Embeddings FP32 del split de evaluación (N, D).
            emb_int8: Embeddings INT8 del split de evaluación (N, D).
            labels: Etiquetas del split de evaluación (N,).
            emb_fp32_train: Embeddings FP32 del split de entrenamiento (para linear probe).
            labels_train: Etiquetas del split de entrenamiento.

        Returns:
            dict con todas las métricas de los 5 bloques.
        """
        results = {}

        logger.info("Bloque A: Degradación Geométrica Directa...")
        results["block_a"] = block_a.run(emb_fp32, emb_int8)

        logger.info("Bloque B: Similitud de Representaciones...")
        results["block_b"] = block_b.run(emb_fp32, emb_int8, labels)

        logger.info("Bloque C: Preservación de Vecindad...")
        results["block_c"] = block_c.run(emb_fp32, emb_int8, k=self.k)

        logger.info("Bloque D: Geometría Intrínseca...")
        results["block_d"] = block_d.run(emb_fp32, emb_int8)

        logger.info("Bloque E: Calidad Downstream...")
        results["block_e"] = block_e.run(
            emb=emb_fp32,
            labels=labels,
            emb_val=emb_int8,
            labels_val=labels,
            k=5,
        )
        if emb_fp32_train is not None and labels_train is not None:
            results["block_e"]["linear_probe"] = block_e.linear_probe(
                emb_fp32_train, labels_train, emb_fp32, labels
            )

        return results

    def run_single(
        self,
        emb: torch.Tensor,
        labels: torch.Tensor,
        k: int = 5,
    ) -> dict:
        """Evaluación de un solo modelo (sin comparación FP32 vs INT8)."""
        return {
            "knn_accuracy": block_e.knn_accuracy(emb, labels, k=k),
            "uniformity": block_b.uniformity(emb),
            "alignment": block_b.alignment(emb, labels),
            "edim": block_d.effective_dim(emb),
        }