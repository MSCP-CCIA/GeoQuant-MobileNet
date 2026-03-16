import torch

class EmbeddingExtractor:
    """Extracción segura de embeddings usando Forward Hooks de PyTorch (limpieza automática)."""
    def __init__(self, model, layer_name):
        self.handle = None
        self.features = []
        
        # Buscar la capa en el modelo
        layer = dict([*model.named_modules()]).get(layer_name)
        if layer is None:
            raise ValueError(f"Capa '{layer_name}' no encontrada en el modelo. Capas disponibles: {list(dict([*model.named_modules()]).keys())}")
            
        def hook(module, input, output):
            # Guardar el tensor de salida, aplanándolo si es un mapa de características 2D
            out = output.detach().cpu()
            if out.dim() > 2:
                # Global Average Pooling si es un feature map (B, C, H, W) -> (B, C)
                out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
                out = out.view(out.size(0), -1)
            self.features.append(out)
            
        self.handle = layer.register_forward_hook(hook)

    def get_features(self):
        """Devuelve todos los embeddings acumulados como un solo tensor."""
        if not self.features:
            return torch.tensor([])
        return torch.cat(self.features, dim=0)
        
    def clear(self):
        """Limpia los features acumulados para la siguiente época."""
        self.features = []

    def cleanup(self):
        """Eliminación del hook para evitar memory leaks."""
        if self.handle:
            self.handle.remove()
