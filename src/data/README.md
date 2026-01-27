# Data Module Documentation

Este módulo maneja la carga y preprocesamiento de datos para el proyecto FaceQuant-Biometric-Robustness. Contiene utilidades para definir datasets personalizados y construir dataloaders compatibles con PyTorch.

## Contenido

El directorio `src/data` contiene los siguientes scripts:

### 1. `dataset.py`

Define la clase `FaceDataset`, que hereda de `torch.utils.data.Dataset`.

#### Clase `FaceDataset`
Diseñada para cargar imágenes organizadas en carpetas por clase (formato estándar de `ImageFolder`), útil para tareas de reconocimiento facial.

- **Ubicación:** `src/data/dataset.py`
- **Uso:**
  ```python
  from src.data.dataset import FaceDataset
  
  dataset = FaceDataset(root_dir="ruta/a/datos", transform=mis_transformaciones)
  ```
- **Características Principales:**
  - **Estructura de Directorios:** Espera una estructura `root/clase_A/imagen1.jpg`.
  - **Manejo de Errores:** Implementa un mecanismo de robustez en `__getitem__`. Si una imagen falla al cargarse (archivo corrupto, etc.), captura la excepción, registra un error y recursivamente intenta cargar la siguiente imagen (`idx + 1`). Esto evita que el entrenamiento se detenga por una sola imagen corrupta.
  - **Atributos:**
    - `classes`: Lista de nombres de clases.
    - `class_to_idx`: Diccionario mapeando nombres de clases a índices enteros.

### 2. `dataloaders.py`

Provee funciones "factory" para configurar y crear instancias de `DataLoader` listas para el entrenamiento o validación, integradas con la configuración YAML del proyecto.

#### Función `get_transforms(config: dict, is_training: bool)`
Define el pipeline de preprocesamiento estándar para modelos de reconocimiento facial (como ArcFace/InsightFace).

- **Transformaciones (Entrenamiento):**
  1. `Resize`: Ajusta la imagen al tamaño definido en `config['data']['input_size']`.
  2. `RandomHorizontalFlip`: Aumentación de datos simple.
  3. `ToTensor`: Convierte a tensor PyTorch.
  4. `Normalize`: Normaliza a media 0.5 y desviación estándar 0.5 (rango [-1, 1]).
- **Transformaciones (Validación/Test):**
  - Igual al entrenamiento pero sin `RandomHorizontalFlip`.

#### Función `create_dataloader(config_path: str, split: str = 'train')`
Función principal para generar dataloaders.

- **Argumentos:**
  - `config_path`: Ruta al archivo `.yaml` de configuración.
  - `split`: 'train' para datos de entrenamiento, otro valor para validación/raw.
- **Lógica:**
  1. Lee la configuración YAML.
  2. Determina la ruta de los datos:
     - Si `split='train'`, busca en `data_root/processed/train`.
     - De lo contrario, busca en `data_root/raw/split`.
  3. Aplica las transformaciones correspondientes.
  4. Configura el `DataLoader` con parámetros como `batch_size`, `num_workers`, `pin_memory`, etc., extraídos del config.

## Configuración Esperada (YAML)

Para que `dataloaders.py` funcione correctamente, el archivo de configuración debe tener una estructura similar a esta:

```yaml
data:
  input_size: [112, 112]  # Ejemplo para ArcFace
  data_root: "data/"
  train_batch_size: 64
  val_batch_size: 64

system:
  num_workers: 4
```
