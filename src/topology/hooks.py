import torch

class EmbeddingExtractor:
    """Extracción segura de embeddings usando Forward Hooks de PyTorch (limpieza automática)."""
    def __init__(self, model, layer_name):
        self.handle = None
        self.features = []
        pass

    def cleanup(self):
        """Eliminación del hook para evitar memory leaks."""
        if self.handle:
            self.handle.remove()
