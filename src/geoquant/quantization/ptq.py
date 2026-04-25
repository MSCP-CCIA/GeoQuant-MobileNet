import copy
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
from geoquant.utils.logging import get_logger

logger = get_logger(__name__)


def apply_ptq_static(model: torch.nn.Module, calib_loader, output_path: str, device: torch.device) -> torch.nn.Module:
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    from torch.ao.quantization import get_default_qconfig_mapping, QConfig
    from torch.ao.quantization.observer import default_histogram_observer, default_weight_observer

    cpu_device = torch.device("cpu")
    logger.info("Aplicando PTQ Estático (FX Graph Mode en CPU)...")
    ptq_model = copy.deepcopy(model).eval().to(cpu_device)

    current_engine = torch.backends.quantized.engine
    logger.info(f"Motor matemático detectado: {current_engine}")
    qconfig_mapping = get_default_qconfig_mapping(current_engine)

    safe_qconfig = QConfig(
        activation=default_histogram_observer.with_args(reduce_range=True),
        weight=default_weight_observer
    )

    # 4. Homologamos las reglas para TODAS las capas que intentan fusionarse
    # Dejamos fuera intencionalmente a HardSwish y HardSigmoid
    fusion_layers = [nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.ReLU6]
    for op in fusion_layers:
        qconfig_mapping.set_object_type(op, safe_qconfig)

    dummy_input = torch.randn(1, 3, 224, 224).to(cpu_device)

    logger.info("Insertando observadores...")
    prepared_model = prepare_fx(ptq_model, qconfig_mapping, example_inputs=(dummy_input,))

    logger.info("Calibrando tensores con datos reales...")
    with torch.no_grad():
        for inputs, _ in tqdm(calib_loader, desc="Calibración"):
            prepared_model(inputs.to(cpu_device))

    logger.info("Convirtiendo modelo a INT8 Estático...")
    quantized_model = convert_fx(prepared_model)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized_model.state_dict(), out)
    logger.info(f"Modelo PTQ Estático guardado -> {out}")

    return quantized_model