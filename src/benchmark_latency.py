import torch
import time
import copy
import os
import psutil
import numpy as np
import matplotlib.pyplot as plt
from src.models.factory import get_arcface_model
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.ao.quantization import get_default_qat_qconfig_mapping


def get_process_memory():
    """Retorna la memoria RAM actual del proceso en MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_disk_size(model):
    """Calcula el tamaño del estado del modelo en disco (MB)."""
    torch.save(model.state_dict(), "temp_weights.p")
    size = os.path.getsize("temp_weights.p") / (1024 * 1024)
    os.remove("temp_weights.p")
    return size


def benchmark_inference(model, input_data, iterations=100):
    """Mide latencia y uso de RAM durante la inferencia."""
    model.eval()
    # Medir RAM antes de la inferencia intensa
    mem_before = get_process_memory()

    with torch.no_grad():
        for _ in range(10): _ = model(input_data)  # Warm-up

        start = time.perf_counter()
        for _ in range(iterations):
            _ = model(input_data)  # <--- AQUÍ OCURRE LA INFERENCIA
        end = time.perf_counter()

    mem_after = get_process_memory()
    avg_lat = ((end - start) / iterations) * 1000
    return avg_lat, mem_after


def main():
    # Limitar a 1 núcleo para simular hardware restringido
    psutil.Process(os.getpid()).cpu_affinity([0])
    print("[INFO] Ejecutando en modo de capacidad reducida (1 Núcleo)...")

    dummy_input = torch.randn(1, 3, 224, 224)

    # Cargar Baseline
    base, _ = get_arcface_model('mobilenet_v3_small', num_classes=10)
    base.load_state_dict(torch.load('checkpoints/mobilenet_v3_small_best_arcface_fp32.pth', map_location='cpu'))

    # Preparar comparativa
    import torchao
    from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
    m_ptq = copy.deepcopy(base)
    quantize_(m_ptq, Int8DynamicActivationInt8WeightConfig())

    q_map = get_default_qat_qconfig_mapping("fbgemm")
    m_qat_fx = prepare_qat_fx(copy.deepcopy(base), q_map, example_inputs=(dummy_input,))
    m_qat = convert_fx(m_qat_fx)
    m_qat.load_state_dict(
        torch.load('checkpoints/quantized/mobilenet_v3_small_arcface_qat_int8.pth', map_location='cpu'))

    results = {}
    for name, m in [("FP32", base), ("PTQ_Dyn", m_ptq), ("QAT_Stat", m_qat)]:
        lat, ram = benchmark_inference(m, dummy_input)
        disk = get_disk_size(m)
        results[name] = {"lat": lat, "ram": ram, "disk": disk}

    # Mostrar Resultados
    print("\n" + "=" * 65)
    print(f"{'MÉTODO':<12} | {'LATENCIA':<10} | {'RAM (RSS)':<10} | {'DISCO':<10}")
    print("=" * 65)
    for n, d in results.items():
        print(f"{n:<12} | {d['lat']:>6.2f} ms | {d['ram']:>7.2f} MB | {d['disk']:>7.2f} MB")
    print("=" * 65)


if __name__ == "__main__":
    main()