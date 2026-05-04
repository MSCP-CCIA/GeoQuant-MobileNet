# GeoQuant-MobileNet

> **Geometric & Topological Embedding Analysis in MobileNetV3 under INT8 Quantization**
>
> Framework experimental para cuantificar el impacto de la cuantización INT8 (PTQ y QAT) sobre la **topología del espacio latente** de redes convolucionales ligeras, evaluado mediante un conjunto sistemático de métricas geométricas (CKA, Cosine Drift, Alignment/Uniformity, Overlap@k, Effective Dimensionality) y un benchmark multiobjetivo (Accuracy × Latencia × Integridad Topológica).

---

## Tabla de Contenidos

1. [Motivación científica](#1-motivación-científica)
2. [Hipótesis y preguntas de investigación](#2-hipótesis-y-preguntas-de-investigación)
3. [Visión general del pipeline](#3-visión-general-del-pipeline)
4. [Estructura del repositorio](#4-estructura-del-repositorio)
5. [Stack tecnológico y dependencias](#5-stack-tecnológico-y-dependencias)
6. [Instalación y entorno](#6-instalación-y-entorno)
7. [Dataset: CUB-200-2011](#7-dataset-cub-200-2011)
8. [Configuración (config.yaml + overrides)](#8-configuración-configyaml--overrides)
9. [Arquitectura del modelo](#9-arquitectura-del-modelo)
10. [Pipeline de entrenamiento (FP32 baseline)](#10-pipeline-de-entrenamiento-fp32-baseline)
11. [Cuantización: PTQ estática](#11-cuantización-ptq-estática)
12. [Cuantización: QAT con destilación geométrica](#12-cuantización-qat-con-destilación-geométrica)
13. [Suite de evaluación geométrica (Bloques A–D)](#13-suite-de-evaluación-geométrica-bloques-ad)
14. [Benchmark de latencia y disco](#14-benchmark-de-latencia-y-disco)
15. [Stress benchmark: dummy data + FLOPs + peak RAM](#15-stress-benchmark-dummy-data--flops--peak-ram)
16. [Reportes (JSON / CSV / LaTeX)](#16-reportes-json--csv--latex)
17. [Scripts CLI](#17-scripts-cli)
18. [Targets de Make](#18-targets-de-make)
19. [Notebooks](#19-notebooks)
20. [Tests](#20-tests)
21. [Reproducibilidad](#21-reproducibilidad)
22. [Outputs y artefactos generados](#22-outputs-y-artefactos-generados)
23. [Glosario de métricas](#23-glosario-de-métricas)
24. [Solución de problemas](#24-solución-de-problemas)
25. [Licencia y autoría](#25-licencia-y-autoría)

---

## 1. Motivación científica

La cuantización INT8 (paso de pesos y activaciones de FP32 a enteros de 8 bits) es la técnica estándar para desplegar redes convolucionales en hardware restringido (CPU embebido, móviles). La literatura clásica mide su impacto **únicamente en términos de Top-1 Accuracy**, pero esa visión es ciega ante una dimensión crítica para sistemas de **recuperación, biometría y aprendizaje métrico**: la **topología del espacio latente**.

Este proyecto plantea que dos modelos pueden tener exactamente la misma accuracy y, sin embargo, presentar **espacios de embedding completamente distintos** tras la cuantización: clústeres rotados, fronteras erosionadas, vecindarios destruidos. La investigación se sitúa en la intersección de tres ejes:

1. **Exactitud predictiva** — Top-1 / k-NN Accuracy.
2. **Eficiencia computacional** — Latencia (ms) y huella en disco (MB) sobre CPU x86.
3. **Integridad topológica** — Métricas geométricas que miden cuánto "se mueve" el espacio de representación bajo INT8.



---

## 2. Hipótesis y preguntas de investigación

**Hipótesis central.** Un backbone entrenado con una pérdida métrica angular (ArcFace) que maximiza la separabilidad geométrica producirá un espacio latente *robusto a la cuantización*: el ruido de discretización INT8 será absorbido por los márgenes angulares sin que los clústeres se solapen.

**Preguntas concretas que el framework responde:**

- ¿Cuánto se desplaza un embedding tras la cuantización? → **Cosine Drift** (Bloque A).
- ¿La estructura global del espacio se preserva? → **CKA lineal** (Bloque B).
- ¿Los pares positivos (misma clase) siguen igual de juntos? → **Alignment** (Bloque B).
- ¿La distribución sobre la hiperesfera se colapsa? → **Uniformity** (Bloque B).
- ¿Los vecindarios k-NN se conservan? → **Overlap@k** (Bloque C).
- ¿El espacio pierde rango espectral efectivo? → **EDim** (Bloque D).
- ¿PTQ o QAT preservan mejor la geometría con el menor coste? → **comparación cruzada** + **benchmark de latencia**.

---

## 3. Visión general del pipeline

```
                 ┌──────────────────────────────────────────┐
                 │  CUB-200-2011 (200 clases, 224×224)      │
                 └──────────────────┬───────────────────────┘
                                    │
                  ImageFolder + transforms (train/eval)
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │                                                       │
        ▼                                                       ▼
┌────────────────────┐                              ┌──────────────────────┐
│  FP32 Training     │                              │  Calibration loader  │
│  (2 fases)         │                              │  (mismo train_loader)│
│  ─ Phase 1: warmup │                              └──────────┬───────────┘
│  ─ Phase 2: FT     │                                         │
│  Backbone +        │                                         │
│  ArcFace head      │                                         │
└─────────┬──────────┘                                         │
          │                                                    │
          ▼                                                    ▼
   best_fp32_*.pth ────────────►  PTQ static (FX Graph) ──► model_ptq_*d.pth
          │                                                    │
          ├──────────────────►   QAT distillation (FX) ───► model_qat_*d.pth
          │                       (MSE teacher↔student)
          ▼
  ┌─────────────────────┐
  │  extract_embeddings │ (FP32, PTQ, QAT)  →  outputs/embeddings/*.pt
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────────────────────────────┐
  │  EvaluationSuite                            │
  │  Bloque A: CosineDrift, ángulos por muestra │
  │  Bloque B: CKA, Alignment, Uniformity       │
  │  Bloque C: Overlap@k                        │
  │  Bloque D: EDim                             │
  └──────────┬──────────────────────────────────┘
             │
             ▼
  Reporter →  outputs/results/eval_{ptq,qat}.{json,csv,tex}

  Paralelo: benchmark.py → latencia ms + disco MB
```

---

## 4. Estructura del repositorio

```
GeoQuant-MobileNet/
├── configs/
│   ├── config.yaml                   # Fuente de verdad: data, model, training, eval
│   ├── experiment/
│   │   ├── baseline_fp32.yaml        # Override: entrenamiento FP32 + ArcFace
│   │   ├── ptq_static.yaml           # Override: PTQ estático INT8
│   │   └── qat_full.yaml             # Override: QAT FX Graph Mode
│   └── quantization/
│       ├── ptq.yaml                  # Receta PTQ (engine, observer, granularidad)
│       └── qat.yaml                  # Receta QAT (qconfig_mapping)
│
├── data/
│   ├── raw/cub200/{train,test}/      # Dataset crudo (gestionado por DVC)
│   ├── processed/                    # Datos derivados
│   └── dummy/                        # 10 imágenes ruidosas para smoke-tests
│
├── notebooks/
│   ├── 00_data_exploration.ipynb     # EDA del dataset (distribución, stats)
│   ├── 01_embedding_baseline.ipynb   # t-SNE FP32 + métricas baseline
│   ├── 02_quantization_analysis.ipynb# Comparativa PTQ vs QAT (4 bloques)
│   └── 03_geometric_evaluation.ipynb # Análisis profundo bloque a bloque
│
├── outputs/
│   ├── checkpoints/                  # *.pth (FP32 / PTQ / QAT)
│   ├── embeddings/                   # *.pt (cache de embeddings)
│   ├── results/                      # *.json / *.csv / *.tex
│   └── logs/                         # experiment_YYYYMMDD_HHMMSS.log
│
├── scripts/
│   ├── train.py                      # Entrenamiento FP32 (warmup + finetuning)
│   ├── quantize.py                   # PTQ o QAT (vía --approach)
│   ├── evaluate.py                   # Evalúa los 4 bloques + reporter
│   ├── benchmark.py                  # Latencia y huella en disco (rápido)
│   └── stress_benchmark.py           # Latencia + peak RAM + FLOPs + dummy data
│
├── src/geoquant/
│   ├── __init__.py                   # __version__ = "0.2.0"
│   ├── data/
│   │   ├── dataset.py                # CUBDataset + get_dataloaders
│   │   ├── transforms.py             # TransformFactory (ImageNet stats)
│   │   ├── samplers.py               # BalancedClassSampler (n por clase)
│   │   └── dummy_generator.py        # CUB + ruido gaussiano → ImageFolder dummy
│   ├── models/
│   │   ├── backbone.py               # MobileNetV3Backbone (proyección + BN1d)
│   │   └── arcface.py                # ArcFaceHead (margen angular aditivo)
│   ├── training/
│   │   └── trainer.py                # Trainer 2-fases con k-NN val + CSV history
│   ├── quantization/
│   │   ├── ptq.py                    # apply_ptq_static (FX Graph + histogram obs.)
│   │   ├── qat.py                    # apply_qat_distillation (MSE + freeze BN ep>=5)
│   │   └── export.py                 # export_torchscript (.pt portable)
│   ├── evaluation/
│   │   ├── suite.py                  # EvaluationSuite (orquesta bloques A-D)
│   │   ├── block_a.py                # CosineDrift, ángulos
│   │   ├── block_b.py                # CKA, Alignment, Uniformity
│   │   ├── block_c.py                # Overlap@k (k=1,5,10,20)
│   │   ├── block_d.py                # EDim (entropía espectral)
│   │   ├── embeddings.py             # extract_embeddings + cache .pt
│   │   ├── latency.py                # measure_latency + disk_mb
│   │   ├── flops_counter.py          # count_flops (vía thop)
│   │   ├── memory_profiler.py        # measure_memory (peak RAM con tracemalloc)
│   │   └── reporter.py               # JSON / CSV / LaTeX
│   └── utils/
│       ├── logging.py                # get_logger + MLflowLogger opcional
│       └── reproducibility.py        # seed_everything (Python/NumPy/Torch)
│
├── tests/
│   ├── conftest.py                   # Fixtures: dummy_embeddings, small_config
│   └── unit/
│       ├── test_metrics.py           # Sanidad de bloques A-D
│       └── test_quantization.py      # PTQ shape, TorchScript export
│
├── pyproject.toml                    # Build hatchling + ruff + pytest config
├── uv.lock                           # Lockfile reproducible (uv sync)
├── Makefile                          # Targets: train, quantize-ptq/qat, evaluate, ...
├── REPRODUCE.md                      # Guía paso a paso de reproducción
├── LICENSE                           # MIT
└── README.md                         # (este archivo)
```

---

## 5. Stack tecnológico y dependencias

**Lenguaje / runtime**
- Python ≥ 3.9 (recomendado 3.10+).
- PyTorch ≥ 2.0 con `torch.ao.quantization` (FX Graph Mode).
- Torchvision ≥ 0.15 (modelos preentrenados de MobileNetV3).

**Cuantización**
- `torch.ao.quantization.quantize_fx` (`prepare_fx`, `prepare_qat_fx`, `convert_fx`).
- Backends soportados: `fbgemm` (x86 con AVX2) y `x86` (alias moderno).
- Observers: `default_histogram_observer` (activaciones), `default_weight_observer` (pesos per-channel).

**Numérico / científico**
- NumPy, SciPy, scikit-learn (t-SNE en notebooks).
- Matplotlib (visualizaciones).
- `thop` — conteo de FLOPs y parámetros (usado por `flops_counter.py`).
- `tracemalloc` (stdlib) — perfil de peak RAM en CPU (usado por `memory_profiler.py`).

**Operación**
- PyYAML (config), tqdm (progress bars).
- DVC + `dvc-gdrive` (versionado de `data/raw` y `data/processed` en Google Drive).
- MLflow (logging opcional de experimentos; degrada graciosamente si no está instalado).

**Desarrollo (opt-in `[dev]`)**
- `pytest`, `pytest-cov` — suite de tests con `tests/unit/`.
- `ruff` — linter + formateador (line-length 100, reglas E/F/I/UP).
- `mypy` — chequeo estático opcional.

**Build**
- `hatchling` como backend de build, paquete instalable como `geoquant` desde `src/geoquant`.

---

## 6. Instalación y entorno

El proyecto está estandarizado sobre **uv** (Astral) para resolución determinista de dependencias.

```bash
# 1. Clonar
git clone <url-del-repo>
cd GeoQuant-MobileNet

# 2. Sincronizar entorno (crea .venv automáticamente)
uv sync

# 3. (Opcional) extras de desarrollo
uv sync --extra dev
```

`uv.lock` fija las versiones exactas para reproducibilidad bit-a-bit. El paquete se instala en modo editable apuntando a `src/geoquant/` (configurado en `[tool.hatch.build.targets.wheel]`).

**Recursos hardware recomendados**
- GPU CUDA con ≥ 6 GB VRAM (solo necesaria para entrenamiento FP32 y QAT).
- Inferencia / cuantización / evaluación corren en **CPU x86** (forzado en el código).
- Memoria RAM mínima ≈ 8 GB; el dataloader está parametrizado para cumplir 16 GB.

---

## 7. Dataset: CUB-200-2011

**Caltech-UCSD Birds-200-2011** (`cub200`): 200 clases de aves, ~30 imágenes por clase, alta granularidad inter-clase y dispersión intra-clase elevada (un benchmark estándar para *fine-grained classification* y aprendizaje métrico).

**Layout esperado en disco** (`data/raw/cub200/`):

```
data/raw/cub200/
├── train/
│   ├── 001.Black_footed_Albatross/
│   ├── 002.Laysan_Albatross/
│   └── ...
└── test/
    ├── 001.Black_footed_Albatross/
    └── ...
```

Carga vía `torchvision.datasets.ImageFolder` (`src/geoquant/data/dataset.py:46`), por lo que **el nombre de la carpeta es la etiqueta de clase**.

**Pipeline de transforms** (`src/geoquant/data/transforms.py`):

| Fase  | Pipeline                                                                                       |
|-------|------------------------------------------------------------------------------------------------|
| Train | `RandomResizedCrop(224, scale=(0.5,1.0), bicubic)` → `RandomHorizontalFlip` → `ToTensor` → `Normalize(ImageNet)` |
| Eval  | `Resize(int(224×1.143), bicubic)` → `CenterCrop(224)` → `ToTensor` → `Normalize(ImageNet)`     |

**Estadísticas usadas** — ImageNet (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`).

**Sampler balanceado.** `BalancedClassSampler(labels, n_samples_per_class=4)` en `src/geoquant/data/samplers.py` garantiza que cada época contenga exactamente *N* muestras por clase, lo cual es **crítico** para estabilizar pérdidas métricas como ArcFace (impide que clases minoritarias colapsen).

**Datos dummy.** En `data/dummy/` se generan imágenes ruidosas (una por subdirectorio de clase) que sirven como *carga de stress* y *smoke-test* del pipeline sin requerir el dataset completo. El módulo dedicado `src/geoquant/data/dummy_generator.py` es responsable de producirlas dinámicamente (ver subsección siguiente).

### Generador de datos dummy — `dummy_generator.py`

Crea muestras ruidosas reproducibles a partir del split de CUB-200. Útil para **stress-tests de latencia/RAM** y para validar pipelines de inferencia sin cargar el dataset completo.

**API pública**

```python
from geoquant.data.dummy_generator import generate_dummy_dataset, get_dummy_loader

# 1) Generar imágenes ruidosas en data/dummy/{class_name}/{idx:06d}_noisy.png
dummy_dir = generate_dummy_dataset(
    config=config,
    output_dir="data/dummy",
    split="test",       # 'train' o 'test' del CUB-200 fuente
    sigma=0.05,         # std del ruido gaussiano en espacio [0, 1]
    n_images=500,       # tamaño del dataset dummy
    force=False,        # True regenera aunque ya exista contenido
)

# 2) Construir un DataLoader con normalización ImageNet
loader = get_dummy_loader(
    dummy_dir,
    image_size=224,
    batch_size=32,
    num_workers=0,
)
```

**Pipeline**
1. `ImageFolder` sobre `data/raw/cub200/{split}` con transform Resize→CenterCrop→ToTensor (sin normalizar).
2. Se elige aleatoriamente `n_images` con `random.sample` (semilla global controlada por `seed_everything`).
3. Para cada muestra: `noisy = clamp(tensor + N(0, sigma²), 0, 1)`.
4. Guardado como PNG en `data/dummy/{class_name}/{orig_idx:06d}_noisy.png` (mantiene la etiqueta de clase del original).
5. `get_dummy_loader` añade `Normalize(ImageNet)` al pipeline para inferencia equivalente a evaluación.

> **Nota.** El ruido se aplica **antes** de la normalización ImageNet y se clipa a `[0, 1]` para mantener píxeles válidos. La regeneración se hace solo si la carpeta está vacía o si pasas `force=True`.

**Versionado.** `data/raw/`, `data/processed/` y `data/dummy/` están excluidos del control de Git (`.gitignore`) — los datos crudos se versionan con **DVC** (Google Drive como remoto, `dvc pull` para sincronizar) y los dummy se regeneran localmente bajo demanda. El `.gitignore` también excluye `*.zip`, `*.tar.gz`, `*.rec`, `*.idx`, `*.onnx`, `*.pth` y la carpeta `outputs/`.

---

## 8. Configuración (config.yaml + overrides)

El sistema usa un patrón de **fuente única de verdad** (`configs/config.yaml`) con **overrides por experimento** (`configs/experiment/*.yaml`). El merge es superficial por sección: cada clave de nivel superior del override actualiza la sección correspondiente del config base.

### `configs/config.yaml` — base

```yaml
seed: 42

data:
  raw_dir: "data/raw/cub200"
  processed_dir: "data/processed"
  dataset: "cub200"
  num_classes: 200
  image_size: 224
  batch_size: 32
  num_workers: 4
  pin_memory: true

model:
  backbone: "mobilenet_v3_small"
  pretrained: true
  embedding_size: 512        # Dimensión del embedding (proyección Linear + BN1d)
  pooling: "avg"
  arcface:
    margin: 0.5              # m angular aditivo (radianes)
    scale: 30.0              # s factor de escala

training:
  epochs: 30
  lr: 0.01
  momentum: 0.9
  weight_decay: 5.0e-4
  scheduler: "cosine"        # CosineAnnealingLR
  label_smoothing: 0.1
  checkpoint_dir: "outputs/checkpoints"

eval:
  device: "cpu"
  quantized_backend: "fbgemm"
  output_dir: "outputs/results"
  embeddings_dir: "outputs/embeddings"
  report_formats: ["json", "csv"]
```

### Overrides

| Archivo                                  | Propósito                                                                                  |
|------------------------------------------|--------------------------------------------------------------------------------------------|
| `experiment/baseline_fp32.yaml`          | 30 epochs, `lr=0.01`, checkpoint en `outputs/checkpoints/baseline_fp32`, ArcFace `m=0.5 s=30`. |
| `experiment/ptq_static.yaml`             | `approach: ptq`, backend `x86`, observer `minmax`, 50 batches de calibración, `output_dir: outputs/checkpoints/ptq`. |
| `experiment/qat_full.yaml`               | `approach: qat`, backend `fbgemm`, 10 epochs, `lr=5e-5`, `output_dir: outputs/checkpoints/qat`. |
| `quantization/ptq.yaml`                  | Receta declarativa: `engine: torchao`, `weight: per_channel`, `activation: static`.        |
| `quantization/qat.yaml`                  | Receta declarativa QAT: `qconfig_mapping: get_default_qat_qconfig_mapping`.                |

### Lógica de merge (`scripts/train.py:21-33`, idéntica en `quantize.py`)

```python
def load_config(base, experiment=None):
    config = yaml.safe_load(open(base))
    if experiment:
        override = yaml.safe_load(open(experiment))
        for section, values in override.items():
            if isinstance(values, dict):
                config.setdefault(section, {}).update(values)  # merge superficial
            else:
                config[section] = values
    return config
```

> **Importante.** El merge **no es profundo**. Si un override redefine `model.arcface`, lo reemplaza por completo. Para extenderlo, edita la sección entera.

---

## 9. Arquitectura del modelo

### 9.1 Backbone — `MobileNetV3Backbone` (`src/geoquant/models/backbone.py`)

Wrapper sobre `torchvision.models.mobilenet_v3_small`:

1. Carga pesos preentrenados de **ImageNet1K_V1** (excepto la cabeza `classifier`).
2. Sustituye `classifier[3]` (originalmente `Linear(1024 → 1000)`) por:
   ```
   Sequential(
     Linear(1024, embedding_size, bias=False),
     BatchNorm1d(embedding_size)
   )
   ```
   donde `embedding_size = config.model.embedding_size` (default `512`).
3. La salida del forward es el **embedding pre-norm** (la proyección lineal seguida de BN1d), listo para ser L2-normalizado dentro de ArcFace.

> El uso de `BatchNorm1d` posterior a la proyección estabiliza la magnitud de los embeddings y es un truco estándar (común en SphereFace/ArcFace) para evitar que la capa lineal "explote" durante el primer warmup.

### 9.2 Cabeza ArcFace — `ArcFaceHead` (`src/geoquant/models/arcface.py`)

Implementación canónica de ArcFace (Deng et al., CVPR 2019):

```
cos(θ_y + m) · s   para la clase ground-truth
cos(θ_j) · s       para j ≠ y
```

Detalles numéricos del forward (`arcface.py:46-62`):

- L2-normaliza embeddings y pesos (vectores en la hiperesfera unitaria).
- Calcula `cos(θ + m) = cos θ · cos m − sin θ · sin m`.
- Aplica el truco de Deng para `θ + m > π`: si `cos θ ≤ cos(π − m)` se sustituye por `cos θ − sin(π − m)·m` (evita gradientes inestables).
- Escala el resultado por `s = 30.0` antes de la softmax/cross-entropy.

### 9.3 Esqueleto de inferencia post-QAT

`extract_and_save` en `src/geoquant/evaluation/embeddings.py:44` reconstruye el modelo de evaluación según el nombre del checkpoint:

- Contiene `"fp32"` → modelo FP32 directo.
- Contiene `"ptq"`  → `prepare_fx` + `convert_fx` antes de cargar `state_dict`.
- Cualquier otro → asume QAT: `prepare_qat_fx` + `convert_fx`.

Esto significa que **los nombres de archivo importan**: el discriminador es `str(ckpt_path).lower()`.

---

## 10. Pipeline de entrenamiento (FP32 baseline)

Implementado en `src/geoquant/training/trainer.py` y orquestado por `scripts/train.py`. El entrenamiento es **2 fases secuenciales**:

### Fase 1 — Warmup del head (backbone congelado)

- Duración: `training.warmup_epochs` (default `5`).
- LR: `training.lr` (default `0.01`).
- Solo se actualizan los parámetros del `classifier` (proyección + BN1d) y de `ArcFaceHead`.
- Optimizer: `SGD(momentum=0.9, weight_decay=5e-4)`.
- Scheduler: `CosineAnnealingLR(T_max=epochs)`.
- Loss: `CrossEntropyLoss(label_smoothing=0.1)` sobre los logits de ArcFace.

> **Por qué.** Si se libera el backbone desde la primera época, los gradientes ruidosos de la proyección recién inicializada destruyen los pesos preentrenados de ImageNet.

### Fase 2 — Finetuning total

- Duración: `training.epochs` (default `30`).
- LR: `base_lr * 0.1` (i.e. `0.001`) — un orden de magnitud menor.
- Todos los parámetros descongelados.
- Mismo optimizer / scheduler que Fase 1.

### Validación por época

`Trainer._val_epoch` extrae embeddings del `val_loader` y calcula:

- **1-NN accuracy** — *rigor biométrico* (nearest neighbor exacto, excluyendo self).
- **5-NN accuracy** — *cohesión por votación mayoritaria*.

```python
# trainer.py:22-44
def compute_knn_accuracy(gallery, gallery_labels, query, query_labels, k=1):
    dist = torch.cdist(query, gallery)
    if gallery is query: dist.fill_diagonal_(float('inf'))
    knn_idx = dist.topk(k, largest=False, dim=1).indices
    knn_labels = gallery_labels[knn_idx]
    preds = knn_labels.squeeze() if k == 1 else torch.mode(knn_labels, dim=1).values
    return (preds == query_labels).float().mean().item()
```

### Selección y guardado del mejor checkpoint

- Métrica de selección: **1-NN accuracy** (la más rigurosa).
- Nombrado dinámico: `best_fp32_{phase_name}_{emb_size}d.pth`, p. ej. `best_fp32_finetuning_512d.pth`.
- Historial completo en CSV: `outputs/training_history_{phase}_{emb_size}d.csv` con columnas `epoch, train_loss, val_1nn_acc, val_5nn_acc`.

### Tras entrenar

`scripts/train.py:90-96` extrae automáticamente los embeddings del *best checkpoint* en `outputs/embeddings/emb_fp32_{emb_size}d.pt`, listos para evaluación.

---

## 11. Cuantización: PTQ estática

Implementación en `src/geoquant/quantization/ptq.py`, función `apply_ptq_static`.

### Pipeline

1. **Deep-copy** del modelo FP32, `.eval()` y `.to("cpu")` (la cuantización Pytorch FX requiere CPU).
2. Detección dinámica del backend: `torch.backends.quantized.engine` (`fbgemm` en x86 con AVX2, `qnnpack` en ARM).
3. `qconfig_mapping = get_default_qconfig_mapping(backend)`.
4. **QConfig seguro y homogéneo** para capas que se fusionan:
   ```python
   safe_qconfig = QConfig(
       activation=default_histogram_observer.with_args(reduce_range=True),
       weight=default_weight_observer,
   )
   for op in [Conv2d, Linear, BatchNorm2d, ReLU, ReLU6]:
       qconfig_mapping.set_object_type(op, safe_qconfig)
   ```
   El **histogram observer** colecciona la distribución completa de activaciones (más caro pero más preciso que minmax). `reduce_range=True` mitiga overflow en hardware sin saturación.
5. `prepare_fx(model, qconfig_mapping, example_inputs=(dummy,))` — inserta observers FakeQuant.
6. **Calibración**: ~50 batches del `train_loader` pasan por el modelo con `torch.no_grad()`. Los observers acumulan estadísticas reales.
7. `convert_fx(prepared)` — transforma observers en módulos cuantizados reales (INT8 estático).
8. `torch.save(quantized_model.state_dict(), output_path)`.

### Observación crítica

Las capas **`HardSwish` y `HardSigmoid`** (presentes en MobileNetV3 small) **se dejan fuera de la fusión** intencionalmente. Esto evita errores en `prepare_fx` con FX Graph Mode, ya que la fusión automática de estos activadores con Conv-BN es frágil en el grafo trazado.

### Configuración (`configs/quantization/ptq.yaml`)

```yaml
approach: "ptq"
engine: "torchao"
backend: "x86"
dtype: "int8"
activation: "static"        # observers acumulan estadísticas globales
weight: "per_channel"       # un escalar por canal de salida (más fiel)
calibration_batches: 50
```

---

## 12. Cuantización: QAT con destilación geométrica

Implementación en `src/geoquant/quantization/qat.py`, función `apply_qat_distillation`. **No es un QAT estándar.** La función combina `prepare_qat_fx` con un esquema de **destilación pura del espacio latente** entre maestro FP32 y estudiante FakeQuant.

### Setup

- **Maestro**: copia del modelo FP32 ya entrenado, `eval()`, `requires_grad=False`.
- **ArcFaceHead**: copiado, congelado y **no participa en la pérdida**.
- **Estudiante**: copia del FP32 puesto en `train()` y envuelto con `prepare_qat_fx(student, qconfig_mapping)`.

### Pérdida y optimizador

```python
criterion_kd = nn.MSELoss()
optimizer = optim.AdamW(qat_model.parameters(), lr=lr, weight_decay=1e-4)
```

Para cada batch:

```python
with torch.no_grad():
    teacher_emb = teacher(inputs)
qat_emb = qat_model(inputs)
loss = criterion_kd(qat_emb, teacher_emb)   # distancia cuadrática vector a vector
loss.backward()
optimizer.step()
```

> **Idea clave.** Penalizar `MSE(estudiante, maestro)` en el espacio de embedding obliga al modelo cuantizado a **reproducir el espacio latente del maestro punto-por-punto**, no solo a clasificar bien. Esto preserva la topología y minimiza el *Cosine Drift* (las pruebas internas reportan ~0.22 sobre el espacio FP32).

### Truco de estabilidad

A partir de la **época 5** se congelan las estadísticas de BatchNorm:

```python
if epoch >= 5:
    qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
```

Esto detiene el running mean/var (ya cuantizado) y deja únicamente que se ajusten los pesos / escalares de FakeQuant. Sin este paso, las estadísticas oscilan y el QAT diverge.

### Validación y selección

Por época se evalúa `val_1nn_acc` y `val_5nn_acc` con el mismo helper `compute_knn_accuracy`. El **mejor estado del estudiante** se guarda según `val_1nn_acc` máxima.

### Conversión final

```python
qat_model.load_state_dict(best_qat_state)
qat_model.eval().to("cpu")
model_int8 = convert_fx(qat_model)
torch.save(model_int8.state_dict(), output_path)
```

### Configuración (`configs/quantization/qat.yaml`)

```yaml
approach: "qat"
engine: "torch.ao"
backend: "fbgemm"
dtype: "int8"
activation: "static"
weight: "per_channel"
qconfig_mapping: "get_default_qat_qconfig_mapping"
```

---

## 13. Suite de evaluación geométrica (Bloques A–D)

Punto de entrada: `EvaluationSuite.run(emb_fp32, emb_int8, labels)` en `src/geoquant/evaluation/suite.py`. Orquesta cuatro bloques que se complementan ortogonalmente.

### Bloque A — Degradación geométrica directa (`block_a.py`)

Mide cuánto se mueve **cada embedding individual** entre FP32 e INT8.

| Métrica                              | Definición                                                              | Unidad      |
|--------------------------------------|-------------------------------------------------------------------------|-------------|
| `cosine_similarity_per_sample`       | `cos(emb_fp32_i, emb_int8_i)` con L2-norm                               | [-1, 1]     |
| `cosine_drift_per_sample`            | `1 − cos_sim`                                                           | [0, 2]      |
| `cosine_drift`                       | media muestral de drift                                                 | escalar     |
| `angular_separation_per_sample_rad`  | `acos(cos_sim)`                                                         | radianes    |
| `angular_separation_per_sample_deg`  | conversión a grados                                                     | grados      |

> **Lectura.** Drift = 0.0 ⇒ embeddings idénticos. Drift > 0.05 ⇒ desviación significativa. Los datos del notebook 03 muestran un drift medio de **~7.8 × 10⁻⁵** (PTQ) y separaciones angulares medias de ~0.7°, indicando que la cuantización es prácticamente isométrica en este pipeline.

### Bloque B — Similitud de representaciones (`block_b.py`)

Mide la **estructura global** del espacio.

| Métrica           | Fórmula                                              | Interpretación                                            |
|-------------------|------------------------------------------------------|-----------------------------------------------------------|
| `cka_linear`      | `HSIC(K_fp32, K_int8) / √(HSIC·HSIC)`                | Centered Kernel Alignment lineal entre Gram matrices.     |
| `alignment_*`     | `E[‖x_i − x_j‖² | y_i = y_j]` con `α=2`              | Compactación intra-clase. Menor = mejor.                  |
| `uniformity_*`    | `log E[exp(−t·‖x_i−x_j‖²)]`, `t=2`                   | Dispersión global en la hiperesfera. Más negativo = mejor.|
| `delta_alignment` | `alignment_quant − alignment_fp32`                   | Cambio neto.                                              |
| `delta_uniformity`| `uniformity_quant − uniformity_fp32`                 | Cambio neto.                                              |

CKA = 1.0 indica espacios linealmente equivalentes.

### Bloque C — Preservación de vecindad (`block_c.py`)

Para cada muestra, calcula el set de sus *k* vecinos más próximos en FP32 y en INT8 y mide la **fracción que se preserva**:

```
overlap_at_k(i) = |NN_fp32_k(i) ∩ NN_int8_k(i)| / k
overlap_at_k    = mean_i(overlap_at_k(i))
```

`EvaluationSuite` lo invoca con `ks=(self.k,)` (default `k=10`). El módulo expone `run(... ks=(1, 5, 10, 20))` para barrido completo. Es la métrica más sensible para tareas de **retrieval**: 1.0 = vecindarios idénticos.

### Bloque D — Geometría intrínseca (`block_d.py`)

**EDim** (Effective Dimensionality) cuantifica cuántas dimensiones del espacio están realmente en uso, usando entropía espectral:

```
σ = svd(emb − mean(emb))         # valores singulares
p_i = σ_i / Σ σ
EDim = exp( -Σ p_i · log p_i )
```

- `EDim = 1` ⇒ todas las muestras colineales (colapso total).
- `EDim ≤ D` (la dimensión nominal del embedding).
- `delta_edim = edim_quant − edim_fp32` mide si la cuantización *colapsa* o *aplana* el espacio.

### Salida unificada

```python
results = {
  "block_a": {"cosine_drift": 7.8e-5, "cosine_similarity_per_sample": tensor(N), ...},
  "block_b": {"cka": 0.998, "alignment_fp32": ..., "delta_alignment": ..., ...},
  "block_c": {"overlap_at_10": 0.94},
  "block_d": {"edim_fp32": 412.3, "edim_quant": 410.8, "delta_edim": -1.5},
}
```

---

## 14. Benchmark de latencia y disco

`src/geoquant/evaluation/latency.py` provee dos funciones:

- `measure_latency(model, image_size=224, iterations=100, warmup=10)`:
  - Modelo en `cpu().eval()`.
  - Warmup de 10 iters (descarta), 100 iters cronometradas con `time.perf_counter()`.
  - Devuelve `latency_ms` (media), `latency_std_ms`, `disk_mb` (tamaño del `state_dict` en disco).
- `benchmark_models(dict, ...)`: itera sobre `{"FP32": m_fp32, "PTQ": m_ptq, "QAT": m_qat}` y agrega resultados.

`scripts/benchmark.py` imprime una tabla legible:

```
=================================================================
MÉTODO     | LATENCIA (ms)  | STD (ms)   | DISCO (MB)
=================================================================
FP32       |     XX.XX      |   XX.XX    |    9.50
PTQ        |     XX.XX      |   XX.XX    |    2.40
QAT        |     XX.XX      |   XX.XX    |    2.40
=================================================================
```

---

## 15. Stress benchmark: dummy data + FLOPs + peak RAM

`scripts/stress_benchmark.py` lleva el benchmark un paso más allá: ejecuta los modelos contra un **dataset dummy ruidoso** (creado por `dummy_generator`) y agrega tres mediciones complementarias en una sola pasada — **latencia**, **peak RAM** durante un forward de varios batches y **FLOPs / parámetros**. Está diseñado como prueba de carga realista en CPU.

### Componentes

| Módulo                                              | Función                                  | Devuelve                                                                 |
|-----------------------------------------------------|------------------------------------------|--------------------------------------------------------------------------|
| `geoquant.data.dummy_generator.generate_dummy_dataset` | Crea/reutiliza imágenes dummy            | `Path` al directorio ImageFolder                                         |
| `geoquant.data.dummy_generator.get_dummy_loader`     | DataLoader sobre las dummy               | `DataLoader` con `Normalize(ImageNet)`                                   |
| `geoquant.evaluation.latency.measure_latency`        | Tiempo medio inferencia + disco          | `latency_ms`, `latency_std_ms`, `disk_mb`                                |
| `geoquant.evaluation.flops_counter.count_flops`      | FLOPs y parámetros vía `thop.profile`    | `flops`, `params`, `flops_str` ("72.868M"), `params_str` ("1.521M")      |
| `geoquant.evaluation.memory_profiler.measure_memory` | Peak RAM de Python/PyTorch (`tracemalloc`)| `peak_ram_mb`, `current_ram_mb`                                          |

### `count_flops(model, image_size=224)` — `evaluation/flops_counter.py`

- Pone el modelo en `eval()` y CPU.
- Lanza un forward con `torch.randn(1, 3, image_size, image_size)`.
- Usa `thop.profile(model, inputs=(dummy,), verbose=False)` y `clever_format` para devolver tanto los valores crudos como cadenas legibles ("72.868M").

### `measure_memory(model, dataloader, n_batches=None)` — `evaluation/memory_profiler.py`

- Activa `tracemalloc.start()` y procesa `n_batches` (o todo el loader) con `torch.no_grad()`.
- Reporta el pico (`peak_ram_mb`) y la asignación viva al final (`current_ram_mb`).
- **No usa CUDA** — pensado únicamente para benchmarks en CPU restringido.

### Flujo del script

```
dummy_generator.generate_dummy_dataset()  →  data/dummy/{class}/*_noisy.png
                                              │
              dummy_generator.get_dummy_loader  ─┐
                                                │
       _load_model(config, ckpt) × {FP32,PTQ,QAT}│
                                                ▼
                          per modelo:
                          ├─ count_flops(model, image_size)
                          ├─ measure_latency(model, image_size, iterations)
                          └─ measure_memory(model, dummy_loader, n_batches)
                                                │
                                                ▼
                          tabla unificada por stdout
```

### Uso

```bash
# Benchmark completo de los tres modelos (genera/reusa data/dummy/)
python scripts/stress_benchmark.py \
  --fp32 outputs/checkpoints/baseline_fp32/best_fp32_finetuning_512d.pth \
  --ptq  outputs/checkpoints/ptq/model_ptq_512d.pth \
  --qat  outputs/checkpoints/qat/model_qat_512d.pth \
  --iterations 100 \
  --batch-size 32

# Forzar regeneración del set dummy (1000 imágenes, ruido más fuerte)
python scripts/stress_benchmark.py \
  --fp32 <ckpt> --regenerate --n-dummy 1000 --sigma 0.1
```

### Argumentos completos del CLI

| Flag             | Default                                              | Descripción                                                              |
|------------------|------------------------------------------------------|--------------------------------------------------------------------------|
| `--config`       | `configs/config.yaml`                                | Config base (lee `data.image_size`).                                     |
| `--fp32`         | `outputs/checkpoints/baseline/best_fp32.pth`         | Checkpoint FP32 (obligatorio en la práctica).                            |
| `--ptq`          | `None`                                               | Checkpoint PTQ (opcional).                                               |
| `--qat`          | `None`                                               | Checkpoint QAT (opcional).                                               |
| `--sigma`        | `0.05`                                               | Std del ruido gaussiano sobre la imagen normalizada en `[0, 1]`.         |
| `--n-dummy`      | `500`                                                | Número de imágenes dummy a generar.                                      |
| `--iterations`   | `100`                                                | Iteraciones de latencia (tras 10 de warmup).                             |
| `--batch-size`   | `32`                                                 | Batch size para `measure_memory` (latencia es siempre batch=1).          |
| `--dummy-dir`    | `data/dummy`                                         | Directorio de salida de las imágenes dummy.                              |
| `--split`        | `test`                                               | Split del CUB-200 fuente (`train` o `test`).                             |
| `--n-batches`    | `None`                                               | Limita los batches procesados en `measure_memory` (None = todos).        |
| `--regenerate`   | `False`                                              | Fuerza recrear el set dummy aunque exista.                               |

### Salida típica

```
===============================================================================================
MÉTODO   | LATENCIA (ms)  | STD (ms)  | DISCO (MB)  | PEAK RAM (MB)  | FLOPs      | PARAMS
===============================================================================================
FP32     |     XX.XX      |   X.XX    |    9.50     |     XXX.XX     |   72.868M  | 1.521M
PTQ      |     XX.XX      |   X.XX    |    2.40     |     XXX.XX     |   72.868M  | 1.521M
QAT      |     XX.XX      |   X.XX    |    2.40     |     XXX.XX     |   72.868M  | 1.521M
===============================================================================================

Config stress: sigma=0.05 | n_dummy=500 | iterations=100 | batch_size=32
```

> **Nota sobre FLOPs.** `thop` cuenta operaciones **del grafo FP32 declarado** y por tanto reporta el mismo valor para los tres modelos: la diferencia operativa real entre INT8 y FP32 está en *throughput* y no en número de operaciones nominales. El valor de FLOPs es útil como referencia arquitectónica, no como predictor de latencia INT8.

> **Nota sobre carga del modelo.** `_load_model` reconstruye el backbone FP32 y carga el `state_dict` con `weights_only=False` — equivalente al patrón de `benchmark.py`. Para mediciones INT8 *fieles* (no solo arquitectónicas) hay que pasar previamente por `prepare_fx`/`prepare_qat_fx` + `convert_fx` igual que en `embeddings.extract_and_save`.

---

## 16. Reportes (JSON / CSV / LaTeX)

`src/geoquant/evaluation/reporter.py` serializa el dict de `EvaluationSuite` a tres formatos en paralelo:

| Formato | Estructura                                                        | Caso de uso                |
|---------|-------------------------------------------------------------------|----------------------------|
| JSON    | árbol jerárquico bloque → métrica → valor                         | tracking + notebooks       |
| CSV     | filas `block, metric, value` (aplanado)                           | dashboards / pandas / Excel|
| LaTeX   | `\begin{table}...` con `booktabs` (`\toprule`, `\midrule`, `\bottomrule`) | papers académicos       |

`Reporter.save_all(results, stem)` escribe `<stem>.json`, `<stem>.csv` y `<stem>.tex` en `outputs/results/`. Maneja `torch.Tensor` vía `TensorEncoder` (escalares como `item()`, multidim como `tolist()`).

---

## 17. Scripts CLI

### `scripts/train.py`

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --experiment configs/experiment/baseline_fp32.yaml
```

Ejecuta `seed_everything(42)`, construye loaders, backbone y head, lanza Fase 1 (warmup) → Fase 2 (FT), guarda el mejor checkpoint y extrae embeddings FP32 al disco.

### `scripts/quantize.py`

```bash
# PTQ
python scripts/quantize.py --approach ptq \
  --config configs/config.yaml \
  --experiment configs/experiment/ptq_static.yaml \
  --checkpoint outputs/checkpoints/baseline_fp32/best_fp32_finetuning_512d.pth \
  --export

# QAT
python scripts/quantize.py --approach qat \
  --config configs/config.yaml \
  --experiment configs/experiment/qat_full.yaml \
  --checkpoint outputs/checkpoints/baseline_fp32/best_fp32_finetuning_512d.pth \
  --export
```

Carga los pesos FP32 (con fallback a `best_fp32.pth` por compatibilidad) y enruta a `apply_ptq_static` o `apply_qat_distillation` según `quantization.approach` del config merged.

### `scripts/evaluate.py`

```bash
python scripts/evaluate.py \
  --int8 outputs/checkpoints/qat/model_qat_512d.pth \
  --approach qat \
  [--fp32 outputs/checkpoints/baseline_fp32/best_fp32_finetuning.pth] \
  [--force-extract]
```

- Carga (o extrae si no existen) embeddings desde los checkpoints.
- **Cachea** los embeddings en `outputs/embeddings/emb_{key}.pt` para evitar inferencias repetidas.
- `--force-extract` ignora la caché.
- Lanza la suite y guarda `eval_{approach}.json/csv/tex`.

### `scripts/benchmark.py`

```bash
python scripts/benchmark.py \
  --fp32 outputs/checkpoints/baseline_fp32/best_fp32_finetuning_512d.pth \
  --ptq outputs/checkpoints/ptq/model_ptq_512d.pth \
  --qat outputs/checkpoints/qat/model_qat_512d.pth \
  --iterations 100
```

> **Nota.** Para los modelos cuantizados, `benchmark.py` actualmente reconstruye el backbone FP32 y carga el `state_dict` cuantizado en él, lo cual sirve como referencia de tamaño en disco; para latencia INT8 real, conviene cargar el modelo via `convert_fx` igual que en `embeddings.extract_and_save`.

### `scripts/stress_benchmark.py`

Variante de carga completa del benchmark — añade peak RAM y FLOPs y usa imágenes dummy generadas al vuelo. Ver [§15](#15-stress-benchmark-dummy-data--flops--peak-ram) para los argumentos completos y la salida.

```bash
python scripts/stress_benchmark.py \
  --fp32 outputs/checkpoints/baseline_fp32/best_fp32_finetuning_512d.pth \
  --ptq  outputs/checkpoints/ptq/model_ptq_512d.pth \
  --qat  outputs/checkpoints/qat/model_qat_512d.pth \
  --n-dummy 500 --sigma 0.05
```

---

## 18. Targets de Make

```bash
make help           # Lista todos los targets
make install        # uv sync
make train          # scripts/train.py con baseline_fp32.yaml
make quantize-ptq   # scripts/quantize.py --approach ptq
make quantize-qat   # scripts/quantize.py --approach qat
make evaluate-ptq   # scripts/evaluate.py --approach ptq
make evaluate-qat   # scripts/evaluate.py --approach qat
make benchmark      # scripts/benchmark.py con FP32+PTQ+QAT
make test           # pytest tests/ -v --tb=short
make lint           # ruff check src/ scripts/ tests/
make fmt            # ruff format src/ scripts/ tests/
make clean          # rm -rf outputs/{checkpoints,results,logs} + __pycache__
```

> En Windows con Git Bash usar `make` directamente; con PowerShell puro, ejecutar los comandos equivalentes manualmente o instalar `make` vía choco/scoop.

---

## 19. Notebooks

Los cuatro notebooks viven en `notebooks/` y exportan figuras a `notebooks/outputs/`:

| Notebook                          | Contenido                                                                                          |
|-----------------------------------|----------------------------------------------------------------------------------------------------|
| `00_data_exploration.ipynb`       | Carga CUB-200, distribución de clases, validación del pipeline de transforms.                      |
| `01_embedding_baseline.ipynb`     | Carga `emb_fp32.pt`, hace t-SNE de las primeras 20 clases, calcula `uniformity / alignment / EDim`. |
| `02_quantization_analysis.ipynb`  | Compara FP32 vs PTQ vs QAT con `EvaluationSuite`, genera reportes JSON/CSV/LaTeX.                  |
| `03_geometric_evaluation.ipynb`   | Análisis profundo bloque a bloque: histogramas, ECDF, top-k muestras con mayor drift, boxplots.    |

Los embeddings cacheados en `outputs/embeddings/*.pt` se cargan directamente en los notebooks vía `load_embeddings(path)`, evitando re-inferencia.

---

## 20. Tests

Configuración en `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

### Fixtures (`tests/conftest.py`)

- `dummy_embeddings` → `(emb (64, 576), labels (64,))` con `seed=42`.
- `dummy_embeddings_pair` → añade ruido `N(0, 0.05²)` para simular INT8.
- `small_config` → config completo de juguete para tests de modelos.

### `tests/unit/test_metrics.py`

- **Bloque A**: drift es 0 para embeddings idénticos; rango [0, 2]; claves de salida.
- **Bloque B**: CKA de un espacio consigo mismo = 1.0; rango [0, 1]; uniformity es float; alignment ≥ 0.
- **Bloque C**: `overlap_at_k(emb, emb)` = 1.0; claves correctas.
- **Bloque D**: `edim ≥ 1`, `edim ≤ D`.

### `tests/unit/test_quantization.py`

- Backbone produce shape `(B, in_features)`.
- TorchScript export crea archivo válido y modelo cargado produce inferencia.
- Tests de PTQ están con `pytest.importorskip("torchao")` (skip si torchao no está).

```bash
uv run pytest                    # toda la suite
uv run pytest tests/unit -v      # solo unit tests
uv run pytest -k "block_a"       # solo bloque A
```

---

## 21. Reproducibilidad

`src/geoquant/utils/reproducibility.py:13`:

```python
def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Esto se invoca al inicio de **todos** los scripts CLI con `config.get("seed", 42)`. La combinación de `cudnn.deterministic=True` + `benchmark=False` sacrifica algo de velocidad de convolución pero garantiza reproducibilidad bit-a-bit en la GPU.

**Logging.** Cada ejecución crea un archivo `outputs/logs/experiment_YYYYMMDD_HHMMSS.log` (`utils/logging.py:42`) con timestamps y nivel `INFO`. El logger es **idempotente**: re-llamar `get_logger(name)` devuelve el mismo handler set.

**Versionado de datos.** DVC en `data/raw/` y `data/processed/`. Comando: `dvc pull` para sincronizar desde Google Drive.

**Versionado de código.** `uv.lock` fija dependencias transitivas exactas. `pyproject.toml` define el paquete `geoquant 0.2.0` con backend `hatchling`.

---

## 22. Outputs y artefactos generados

```
outputs/
├── checkpoints/
│   ├── baseline_fp32/
│   │   ├── best_fp32_warmup_{D}d.pth          # Mejor checkpoint Fase 1
│   │   └── best_fp32_finetuning_{D}d.pth      # Mejor checkpoint Fase 2 ← maestro
│   ├── ptq/
│   │   └── model_ptq_{D}d.pth                 # state_dict cuantizado (FX Graph)
│   └── qat/
│       ├── model_qat_{D}d.pth                 # state_dict QAT convertido
│       └── training_history_qat_{D}d.csv      # epoch / loss / 1nn / 5nn
│
├── embeddings/
│   ├── emb_fp32_{D}d.pt   # dict {'embeddings': Tensor(N,D), 'labels': Tensor(N)}
│   ├── emb_ptq_{D}d.pt
│   └── emb_qat_{D}d.pt
│
├── results/
│   ├── eval_ptq.json / .csv / .tex
│   └── eval_qat.json / .csv / .tex
│
├── training_history_warmup_{D}d.csv
├── training_history_finetuning_{D}d.csv
│
└── logs/
    └── experiment_YYYYMMDD_HHMMSS.log
```

Donde `{D}` es la dimensión del embedding leída dinámicamente del modelo (típicamente `512`).

---

## 23. Glosario de métricas

| Métrica            | Bloque | Rango          | Mejor valor       | Significado                                                       |
|--------------------|--------|----------------|-------------------|-------------------------------------------------------------------|
| **Cosine Drift**   | A      | [0, 2]         | 0                 | Desplazamiento angular medio FP32→INT8.                           |
| **Angular sep (°)**| A      | [0, 180]       | 0                 | Igual que drift pero en grados.                                   |
| **CKA**            | B      | [0, 1]         | 1                 | Similitud lineal global entre dos espacios.                       |
| **Alignment**      | B      | [0, ∞)         | bajo              | Distancia media intra-clase. Menor = clústeres más compactos.     |
| **Uniformity**     | B      | (-∞, 0]        | muy negativo      | Log-densidad media; más negativo = más uniforme en la esfera.     |
| **Overlap@k**      | C      | [0, 1]         | 1                 | Fracción de vecinos k-NN preservados tras cuantización.           |
| **EDim**           | D      | [1, D]         | cercano a D       | Dimensiones efectivamente usadas (entropía espectral).            |
| **Δ Alignment**    | B      | ℝ              | ≈ 0 o negativo    | Cambio en compactación tras cuantización.                         |
| **Δ Uniformity**   | B      | ℝ              | ≈ 0               | Cambio en uniformidad.                                            |
| **Δ EDim**         | D      | ℝ              | ≈ 0               | Pérdida/ganancia de rango espectral.                              |
| **1-NN acc**       | val    | [0, 1]         | 1                 | Rigor biométrico (rank-1 retrieval).                              |
| **5-NN acc**       | val    | [0, 1]         | 1                 | Cohesión de clase por votación mayoritaria.                       |
| **Latencia (ms)**  | bench  | (0, ∞)         | bajo              | Tiempo medio de inferencia (batch=1, CPU).                        |
| **Disco (MB)**     | bench  | (0, ∞)         | bajo              | Tamaño del `state_dict` serializado.                              |
| **Peak RAM (MB)**  | stress | (0, ∞)         | bajo              | Pico de memoria viva durante un forward sobre dummy data (`tracemalloc`). |
| **FLOPs**          | stress | (0, ∞)         | bajo              | Operaciones del grafo FP32 declarado (`thop.profile`, batch=1).   |
| **Params**         | stress | (0, ∞)         | bajo              | Número de parámetros entrenables del modelo.                      |

---

## 24. Solución de problemas

**`FileNotFoundError: best_fp32_finetuning_512d.pth`**
Hay que entrenar primero (`make train`). El script `quantize.py` busca primero `best_fp32_finetuning_{D}d.pth` y como fallback `best_fp32.pth`.

**Errores de fusión `HardSwish` durante `prepare_fx`**
Verificado: `apply_ptq_static` deja fuera `HardSwish/HardSigmoid` de la fusión homogénea. Si modificas `fusion_layers`, mantén esa restricción.

**`RuntimeError: quantized engine ...`**
`torch.backends.quantized.engine` se detecta dinámicamente en el código. Si tu CPU no soporta `fbgemm` (ARM, Apple Silicon), Pytorch caerá a `qnnpack`. El código lo respeta automáticamente.

**El QAT diverge tras la época 5**
Asegúrate de no haber tocado el `qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)` — esa congelación de BN es crítica.

**Embeddings no se regeneran tras retrain**
Usa `--force-extract` en `evaluate.py`, o borra `outputs/embeddings/` manualmente.

**Tests `torchao`-skipped**
La sección `[project.optional-dependencies] torchao = ["torchao"]` es opt-in: `uv sync --extra torchao`.

**Logs vacíos (`experiment_*.log` con 0 bytes)**
Han pasado: cuando un script aborta antes del primer `logger.info`, el FileHandler queda abierto sin escribir. Los logs no nulos confirman que la fase llegó a iniciar.

**Caracteres especiales en logs (Windows)**
El módulo `logging.py` y `embeddings.py` evitan flechas Unicode (`→`) y emojis para no fallar con la codificación CP-1252 por defecto en consolas Windows.

---

## 25. Licencia y autoría

- **Licencia.** MIT — ver `LICENSE`.
- **Autor 1:** Manuel Stiven Castro Parra — *manuel.castro01@usa.edu.co*.
- **Autor 2:** Andrés Mauricio Hurtado Macias — *andres.hurtado03@usa.edu.co*.
- **Versión.** `0.2.0` (paquete `geoquant`, backend de build `hatchling`).
- **Citación sugerida.**

  ```
  Castro Parra, M. S. & Hurtado Macias, A. M. (2026). GeoQuant: El costo angular de la cuantización — Integridad geométrica vs. latencia INT8 en MobileNetV3. 
  ```

---

> **Estado del repositorio.** Framework experimental funcional para CUB-200; el conjunto Bloque A–D y benchmarks corren sobre los embeddings cacheados sin necesidad de GPU. La extensión a otros backbones (ShuffleNetV2, EfficientNet-Lite) y a otros datasets (Imagenette, ImageNet-100) se mantiene como trabajo futuro y requeriría únicamente extender `build_backbone` y los splits de `CUBDataset`.
