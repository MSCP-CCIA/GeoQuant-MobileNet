# FaceQuant-Biometric-Robustness

## Project Structure

Este proyecto sigue una arquitectura modular diseñada para investigación en visión artificial y sistemas embebidos. A continuación se detalla la función de cada componente:

```text
D:\FaceQuant-Biometric-Robustness\
├── configs/                # Configuración centralizada
│   └── config.yaml         # Hiperparámetros, rutas y ajustes de hardware (CPU/GPU)
│
├── data/                   # Gestión de Datos (Ignorado en git)
│   ├── raw/                # Datasets originales comprimidos o sin procesar (CASIA, LFW)
│   ├── processed/          # Datos listos para entrenamiento (alineados, tensores)
│   └── scripts/            # Scripts utilitarios para descarga y preparación de datos
│
├── notebooks/              # Experimentación y Exploración
│   └── ...                 # Jupyter notebooks para prototipado rápido y visualización
│
├── src/                    # Código Fuente Principal (Paquete Python)
│   ├── data/               # Lógica de carga de datos
│   │   ├── __init__.py
│   │   └── ...             # Dataloaders personalizados y transformaciones
│   │
│   ├── models/             # Arquitecturas de Modelos
│   │   ├── __init__.py
│   │   └── ...             # Definiciones de MobileNetV3, ResNet18, ArcFace
│   │
│   ├── quantization/       # Motor de Cuantización
│   │   ├── __init__.py
│   │   └── ...             # Lógica para PTQ (Post-Training Quantization) y FX Graph
│   │
│   └── utils/              # Herramientas Auxiliares
│       ├── __init__.py
│       └── ...             # Métricas (EER), logs, visualización
│
├── tests/                  # Pruebas Unitarias e Integración
│   └── ...                 # Tests para validar componentes críticos
│
├── .env                    # Variables de entorno (Rutas locales, API Keys) - NO COMITEAR
├── .gitignore              # Archivos y carpetas excluidos del control de versiones
├── pyproject.toml          # Configuración del proyecto Python
├── README.md               # Documentación general
└── requirements.txt        # Dependencias del proyecto (pip)
```

## Configuración Inicial

Este proyecto utiliza **uv** para la gestión de dependencias y entornos virtuales de forma centralizada en `pyproject.toml`.

1.  **Instalación de uv:**
    Si aún no tienes `uv` instalado:
    ```bash
    # Con pip
    pip install uv
    
    # O ver documentación oficial para otros métodos: https://docs.astral.sh/uv/
    ```

2.  **Instalación del Entorno y Dependencias:**
    Ejecuta el siguiente comando para crear el entorno virtual (`.venv`) e instalar todas las dependencias definidas en `pyproject.toml` de una sola vez:
    ```bash
    uv sync
    ```
    
    Activar el entorno:
    ```bash
    # Windows
    .venv\Scripts\activate
    
    # Linux/macOS
    source .venv/bin/activate
    ```

3.  **Variables de Entorno:**
    Revisar el archivo `.env` para ajustar las rutas de datos según tu sistema.
