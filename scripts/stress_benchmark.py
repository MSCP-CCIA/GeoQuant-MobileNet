"""
scripts/stress_benchmark.py — Benchmark de stress: latencia + peak RAM + FLOPs.
Usa imágenes dummy (originales CUB-200 + ruido gaussiano) para pruebas de carga real.

Uso:
    python scripts/stress_benchmark.py --fp32 <ckpt> [--ptq <ckpt>] [--qat <ckpt>]
    python scripts/stress_benchmark.py --regenerate --n-dummy 1000 --sigma 0.1
"""

import argparse

import torch
import yaml

from geoquant.data.dummy_generator import generate_dummy_dataset, get_dummy_loader
from geoquant.evaluation.flops_counter import count_flops
from geoquant.evaluation.latency import measure_latency
from geoquant.evaluation.memory_profiler import measure_memory
from geoquant.models.backbone import build_backbone
from geoquant.utils.logging import get_logger
from geoquant.utils.reproducibility import seed_everything

logger = get_logger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_model(config, ckpt_path):
    """Carga el modelo adaptando su arquitectura según sea FP32, PTQ o QAT."""
    from geoquant.models.backbone import build_backbone
    import torch

    # 1. Construir el esqueleto original
    model = build_backbone(config)
    ckpt_str = str(ckpt_path).lower()

    # Si es el maestro de 32 bits, se carga directo
    if "fp32" in ckpt_str:
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
        return model.eval()

    # 2. Si es 8-bits (PTQ o QAT), preparamos el esqueleto FX primero
    dummy_input = torch.randn(1, 3, 224, 224)
    current_engine = torch.backends.quantized.engine

    if "ptq" in ckpt_str:
        from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
        from torch.ao.quantization import get_default_qconfig_mapping

        qconfig_mapping = get_default_qconfig_mapping(current_engine)
        prepared_model = prepare_fx(model.eval(), qconfig_mapping, example_inputs=(dummy_input,))
        eval_model = convert_fx(prepared_model)

    else:  # QAT
        from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
        from torch.ao.quantization import get_default_qat_qconfig_mapping

        qconfig_mapping = get_default_qat_qconfig_mapping(current_engine)
        prepared_model = prepare_qat_fx(model.train(), qconfig_mapping, example_inputs=(dummy_input,))
        eval_model = convert_fx(prepared_model.eval())

    # 3. Ahora sí, cargamos los pesos INT8 en el esqueleto adaptado
    eval_model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    return eval_model.eval()


def main():
    parser = argparse.ArgumentParser(description="GeoQuant — Stress Benchmark (CPU)")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--fp32", default="outputs/checkpoints/baseline/best_fp32.pth")
    parser.add_argument("--ptq", default=None, help="Checkpoint del modelo PTQ")
    parser.add_argument("--qat", default=None, help="Checkpoint del modelo QAT")
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="Std del ruido gaussiano sumado a las imágenes (default: 0.05)")
    parser.add_argument("--n-dummy", type=int, default=500,
                        help="Número de imágenes dummy a generar (default: 500)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Iteraciones para benchmark de latencia (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size para medición de memoria (default: 32)")
    parser.add_argument("--dummy-dir", default="data/dummy",
                        help="Directorio para guardar imágenes dummy (default: data/dummy)")
    parser.add_argument("--split", default="test",
                        help="Split del dataset fuente para las dummy (default: test)")
    parser.add_argument("--n-batches", type=int, default=None,
                        help="Batches a procesar en medición de memoria. None = todos")
    parser.add_argument("--regenerate", action="store_true",
                        help="Fuerza regeneración de imágenes dummy aunque ya existan")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.get("seed", 42))

    image_size = config["data"]["image_size"]

    # Generar/cargar imágenes dummy
    dummy_dir = generate_dummy_dataset(
        config=config,
        output_dir=args.dummy_dir,
        split=args.split,
        sigma=args.sigma,
        n_images=args.n_dummy,
        force=args.regenerate,
    )
    dummy_loader = get_dummy_loader(
        dummy_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # Cargar modelos
    models_to_bench: dict = {}
    models_to_bench["FP32"] = _load_model(config, args.fp32)
    if args.ptq:
        models_to_bench["PTQ"] = _load_model(config, args.ptq)
    if args.qat:
        models_to_bench["QAT"] = _load_model(config, args.qat)

    # Ejecutar benchmark por modelo
    results = {}
    for name, model in models_to_bench.items():
        logger.info(f"Benchmarking {name}...")
        results[name] = {
            **count_flops(model, image_size),
            **measure_latency(model, image_size, args.iterations),
            **measure_memory(model, dummy_loader, args.n_batches),
        }

    # Tabla de resultados
    print("\n" + "=" * 95)
    print(
        f"{'MÉTODO':<8} | {'LATENCIA (ms)':<14} | {'STD (ms)':<9} | "
        f"{'DISCO (MB)':<11} | {'PEAK RAM (MB)':<14} | {'FLOPs':<10} "
    )
    print("=" * 95)
    for name, m in results.items():
        print(
            f"{name:<8} | {m['latency_ms']:>10.2f}     | "
            f"{m['latency_std_ms']:>7.2f}   | "
            f"{m['disk_mb']:>9.2f}   | "
            f"{m['peak_ram_mb']:>12.2f}   | "
            f"{m['flops_str']:>10} | "
        )
    print("=" * 95)
    print(
        f"\nConfig stress: sigma={args.sigma} | n_dummy={args.n_dummy} | "
        f"iterations={args.iterations} | batch_size={args.batch_size}"
    )


if __name__ == "__main__":
    main()
