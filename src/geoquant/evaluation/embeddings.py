"""
Módulo para la extracción y guardado de embeddings.
Versión definitiva corregida para Windows y QAT FX Graph.
"""
import os
from pathlib import Path

import torch
from tqdm import tqdm

from geoquant.utils.logging import get_logger
from geoquant.models.backbone import build_backbone

logger = get_logger(__name__)


def extract_embeddings(model, dataloader, device):
    """Pasa las imágenes por el modelo y devuelve los vectores."""
    model.eval()
    all_embs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Extrayendo embeddings", leave=False):
            inputs = inputs.to(device)
            emb = model(inputs)
            all_embs.append(emb.cpu())
            all_targets.append(targets.cpu())

    return torch.cat(all_embs), torch.cat(all_targets)


def load_embeddings(emb_path: str):
    """Carga una matriz previamente extraída."""
    data = torch.load(emb_path)
    embeddings = data['embeddings']
    labels = data['labels']

    # Textos seguros para la consola de Windows (sin flechas especiales)
    logger.info(f"Embeddings cargados <- {emb_path}  [{embeddings.shape}]")
    return embeddings, labels


def extract_and_save(ckpt_path, out_path, config, dataloader, device):
    """Construye la arquitectura adecuada, carga los pesos y extrae."""

    # 1. Construir el esqueleto original FP32
    model = build_backbone(config).to(device)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    ckpt_str = str(ckpt_path).lower()

    # 2. Recrear la arquitectura exacta dependiendo del archivo
    if "fp32" in ckpt_str:
        logger.info("Cargando pesos del modelo Baseline (FP32)...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        eval_model = model

    elif "ptq" in ckpt_str:
        logger.info("Adaptando esqueleto del modelo a INT8 para PTQ (FX Graph)...")
        from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
        from torch.ao.quantization import get_default_qconfig_mapping

        current_engine = torch.backends.quantized.engine
        qconfig_mapping = get_default_qconfig_mapping(current_engine)

        prepared_model = prepare_fx(model.eval(), qconfig_mapping, example_inputs=(dummy_input,))
        eval_model = convert_fx(prepared_model)

        logger.info("Cargando pesos cuantizados PTQ...")
        eval_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        eval_model.eval().to("cpu")
        device = torch.device("cpu")

    else:
        # Asumimos QAT por defecto
        logger.info("Adaptando esqueleto del modelo a INT8 para QAT (FX Graph)...")
        from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
        from torch.ao.quantization import get_default_qat_qconfig_mapping

        current_engine = torch.backends.quantized.engine
        logger.info(f"Usando motor matemático detectado: {current_engine}")
        qconfig_mapping = get_default_qat_qconfig_mapping(current_engine)

        # La clave: Le pasamos el qconfig_mapping correcto que PyTorch estaba pidiendo
        prepared_model = prepare_qat_fx(model.train(), qconfig_mapping, example_inputs=(dummy_input,))
        eval_model = convert_fx(prepared_model.eval())

        logger.info("Cargando pesos cuantizados QAT...")
        eval_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        eval_model.eval().to("cpu")
        device = torch.device("cpu") # Forzamos inferencia en CPU para modelos INT8

    # 3. Extracción
    logger.info("Procesando dataset...")
    embeddings, labels = extract_embeddings(eval_model, dataloader, device)

    # 4. Guardado
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'embeddings': embeddings, 'labels': labels}, out)

    # Texto seguro para consola Windows
    logger.info(f"Nuevos embeddings generados y guardados -> {out} [{embeddings.shape}]")

    return embeddings, labels