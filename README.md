# Impacto de la Cuantización INT8 en la Topología de Espacios Latentes

Este repositorio contiene el framework experimental para la investigación titulada: "Impacto de la Metodología y Granularidad de Cuantización INT8 sobre la Representación Latente y Eficiencia Computacional en Redes Convolucionales Ligeras". El objetivo central es el análisis multiobjetivo mediante la construcción de un Frente de Pareto 3D que relacione la precisión predictiva, la eficiencia en hardware y la integridad topológica del modelo comprimido.

## Objetivo Científico
La investigación evalúa simultáneamente tres métricas críticas en arquitecturas convolucionales de baja latencia (MobileNetV3 y ShuffleNetV2):
1.  **Exactitud (Top-1 Accuracy):** Evaluación del rendimiento predictivo tras la reducción de precisión a 8 bits.
2.  **Eficiencia Computacional (Latencia):** Tiempo de inferencia medido en milisegundos (ms) sobre arquitectura CPU x86 (Intel/AMD).
3.  **Integridad Topológica (Índice S):** Cuantificación de la separabilidad en el espacio latente mediante distancias de coseno con normalización L2.

## Arquitectura del Sistema
El proyecto se estructura de forma modular para garantizar la reproducibilidad científica:

- **src/quantization/**: Implementación de flujos de cuantización PTQ y QAT utilizando el motor torch.ao.quantization.
- **src/topology/**: Módulos de extracción de características mediante Forward Hooks y cálculo del Índice S.
- **src/models/**: Factory de modelos para la carga de pesos FP32 y fusión de capas (Conv-BN-ReLU).
- **src/utils/**: Utilidades de benchmarking de latencia y gestión de logs experimentales.
- **data/**: Dataloaders optimizados para el cumplimiento de restricciones de memoria RAM (16GB).

## Configuración del Entorno de Investigación
La gestión de dependencias y entornos virtuales se realiza exclusivamente mediante **uv** (Astral).

### Requisitos del Sistema
- Python 3.9 o superior.
- Arquitectura CPU x86 (Target de inferencia INT8).
- GPU con 8GB VRAM (Requerida únicamente para la fase de entrenamiento QAT).

### Instalación y Ejecución
Para inicializar el entorno y sincronizar las dependencias definidas en pyproject.toml:
```bash
uv sync
```

Para ejecutar el pipeline de evaluación multiobjetivo:
```bash
uv run evaluate_pareto.py
```

## Gestión de Datos y Versionamiento
Este proyecto implementa **DVC** (Data Version Control) para el manejo de activos binarios en `data/raw` y `data/processed`, utilizando Google Drive como almacenamiento remoto.

## Licencia
Este software se distribuye bajo la licencia MIT.
