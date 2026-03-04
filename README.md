# Impacto de la Cuantización INT8 en la Topología de Espacios Latentes

Este repositorio contiene la arquitectura experimental para la tesis de grado enfocada en la evaluación multiobjetivo (Frente de Pareto 3D) de modelos de Deep Learning comprimidos. Se analiza el impacto de la metodología de cuantización (**PTQ** y **QAT**) y su granularidad (**Per-tensor** vs. **Per-channel**) sobre la integridad topológica del espacio latente y la eficiencia en hardware x86.

## 🔬 Objetivo Científico
Evaluar simultáneamente tres ejes críticos en arquitecturas de redes convolucionales ligeras (*MobileNetV3*, *ShuffleNetV2*):
1.  **Accuracy (Top-1):** Rendimiento predictivo bajo cuantización INT8.
2.  **Eficiencia Computacional:** Latencia de inferencia medida en milisegundos (ms) sobre arquitectura CPU x86.
3.  **Integridad Topológica (Índice S):** Medida de separabilidad en el espacio latente calculada mediante distancias de coseno normalizadas $L_2$.

## 🏗️ Arquitectura del Proyecto
Diseño modular para facilitar la experimentación científica:

- `src/quantization/`: Orquestación de `torch.ao.quantization` (PTQ/QAT).
- `src/topology/`: Extracción segura de tensores vía **Forward Hooks** y cálculo algebraico vectorizado del Índice S.
- `src/models/`: Factoría de modelos ligeros con soporte para fusión de capas.
- `src/utils/`: Benchmarking de latencia CPU y registro de métricas Pareto.
- `data/`: Dataloaders optimizados para no saturar 16GB de RAM durante la extracción masiva de embeddings.

## 🚀 Configuración del Entorno
Este proyecto utiliza **uv** (Astral) como gestor de dependencias y entornos.

### Requisitos Previos
- Python 3.9+
- Arquitectura CPU x86 (Target de inferencia)
- GPU (Opcional, solo para entrenamiento/QAT)

### Instalación
```bash
# Sincronizar entorno y dependencias
uv sync

# Ejecutar evaluación experimental
uv run evaluate_pareto.py
```

## 🛠️ Restricciones de Hardware
- **RAM:** 16 GB Total (Manejado mediante generadores de datos).
- **CPU:** Target de latencia INT8 (x86).
- **GPU:** 8 GB VRAM (Uso exclusivo en carga FP32 y fase QAT).

## 📄 Licencia
Este proyecto se distribuye bajo la licencia MIT.
