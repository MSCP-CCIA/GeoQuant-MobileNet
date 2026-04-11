# Reproducibility Guide — GeoQuant

Instrucciones para reproducir todos los experimentos del proyecto desde cero.

## Requisitos

- Python ≥ 3.9
- [uv](https://github.com/astral-sh/uv) para gestión de entornos
- GPU opcional (CUDA); la evaluación/cuantización ocurre en CPU

## 1. Entorno

```bash
uv sync
```

## 2. Dataset CUB-200-2011

El dataset se gestiona con DVC. Descarga los datos crudos:

```bash
dvc pull
```

Si es la primera vez, descarga manualmente y colócalo en `data/raw/cub200/` con la estructura:

```
data/raw/cub200/
├── train/
│   ├── 001.Black_footed_Albatross/
│   └── ...
└── test/
    ├── 001.Black_footed_Albatross/
    └── ...
```

## 3. Entrenamiento FP32 Baseline

```bash
make train
# o equivalente:
python scripts/train.py --config configs/config.yaml \
    --experiment configs/experiment/baseline_fp32.yaml
```

Checkpoint guardado en `outputs/checkpoints/best_fp32.pth`.

## 4. Cuantización

### PTQ (Post-Training Quantization)

```bash
make quantize-ptq
```

### QAT (Quantization-Aware Training)

```bash
make quantize-qat
```

Checkpoints guardados en `outputs/checkpoints/`.

## 5. Evaluación Geométrica

```bash
make evaluate-ptq   # Bloque A–E: FP32 vs PTQ
make evaluate-qat   # Bloque A–E: FP32 vs QAT
```

Reportes (JSON, CSV, LaTeX) en `outputs/results/`.

## 6. Benchmark de Latencia

```bash
make benchmark
```

## 7. Tests

```bash
make test
```

## Seed global

Todos los experimentos usan `seed: 42` definido en `configs/config.yaml`.
`seed_everything(42)` se llama al inicio de cada script.

## Estructura de outputs

```
outputs/
├── checkpoints/
│   ├── best_fp32.pth
│   ├── mobilenet_v3_small_ptq_int8.pth
│   └── mobilenet_v3_small_qat_int8.pth
├── results/
│   ├── eval_ptq.json / .csv / .tex
│   └── eval_qat.json / .csv / .tex
└── logs/
    └── experiment_YYYYMMDD_HHMMSS.log
```