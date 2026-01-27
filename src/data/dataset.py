import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image
import logging

# Configurar logging para capturar errores de carga de datos
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDataset(Dataset):
    """
    Dataset genérico para reconocimiento facial compatible con la estructura de carpetas:
    root/class_x/xxx.jpg
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"El directorio {root_dir} no existe.")

        # Usamos ImageFolder para indexar eficientemente las clases
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Carga una imagen y su etiqueta con manejo de errores.
        """
        try:
            # dataset[idx] devuelve (PIL Image, int label)
            img, label = self.dataset[idx]
            
            if self.transform:
                img = self.transform(img)
                
            return img, label
            
        except Exception as e:
            logger.error(f"Error cargando imagen en el indice {idx}: {e}")
            # En caso de error, intentamos cargar la siguiente imagen (recursión simple)
            # Nota: En un entorno de produccion/investigacion real, podriamos preferir 
            # devolver un tensor de ceros o manejarlo segun la politica de datos.
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)
