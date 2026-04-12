"""
Utilidades para extracción, persistencia y carga de embeddings.

Flujo recomendado:
  1. extract_embeddings()  →  genera tensores desde un checkpoint
  2. save_embeddings()     →  serializa a disco (outputs/embeddings/)
  3. load_embeddings()     →  carga desde disco para evaluación repetida
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from geoquant.utils.logging import get_logger

logger = get_logger(__name__)


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pasa el dataloader completo por el modelo y devuelve embeddings + etiquetas.

    Args:
        model: Backbone ya cargado y en eval mode.
        dataloader: DataLoader del split deseado.
        device: Dispositivo de inferencia.

    Returns:
        (embeddings, labels) — ambos en CPU.
    """
    model.eval()
    model.to(device)
    all_emb, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Extrayendo embeddings"):
            inputs = inputs.to(device)
            emb = model(inputs).cpu()
            all_emb.append(emb)
            all_labels.append(labels.cpu())

    embeddings = torch.cat(all_emb)
    labels = torch.cat(all_labels)
    logger.info(f"Embeddings extraídos: {embeddings.shape}")
    return embeddings, labels


def save_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    path: str | Path,
) -> Path:
    """
    Guarda embeddings y etiquetas en un único archivo .pt.

    Args:
        embeddings: Tensor (N, D).
        labels: Tensor (N,).
        path: Ruta de salida (se crea el directorio si no existe).

    Returns:
        Path del archivo guardado.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings, "labels": labels}, out)
    logger.info(f"Embeddings guardados → {out}  [{embeddings.shape}]")
    return out


def load_embeddings(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Carga embeddings guardados con save_embeddings().

    Args:
        path: Ruta al archivo .pt.

    Returns:
        (embeddings, labels) en CPU.
    """
    out = Path(path)
    if not out.exists():
        raise FileNotFoundError(f"No se encontró el archivo de embeddings: {out}")
    data = torch.load(out, map_location="cpu")
    embeddings = data["embeddings"]
    labels = data["labels"]
    logger.info(f"Embeddings cargados ← {out}  [{embeddings.shape}]")
    return embeddings, labels


def extract_and_save(
        ckpt_path: str | Path,
        output_path: str | Path,
        config: dict,
        dataloader: DataLoader,
        device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Atajo que carga un checkpoint, extrae embeddings y los guarda en disco.
    """
    from geoquant.models.backbone import build_backbone

    # 1. Crear el chasis original (FP32)
    model = build_backbone(config)
    model.eval()

    ckpt_str = str(ckpt_path).lower()

    # 2. ADAPTACIÓN: Reconstruir el chasis según la técnica usada
    if "ptq" in ckpt_str:
        from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
        print("Adaptando esqueleto del modelo a INT8 para PTQ (torchao)...")
        quantize_(model, Int8DynamicActivationInt8WeightConfig())

    elif "qat" in ckpt_str:
        from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
        from torch.ao.quantization import get_default_qat_qconfig_mapping
        print("Adaptando esqueleto del modelo a INT8 para QAT (FX Graph)...")

        # QAT requiere simular un dato de entrada para trazar el grafo
        qconfig_mapping = get_default_qat_qconfig_mapping("fbgemm")
        dummy_input = torch.randn(1, 3, 224, 224)

        # Trazamos y convertimos el modelo para que tenga la misma forma que el guardado
        prepared_model = prepare_qat_fx(model, qconfig_mapping, example_inputs=(dummy_input,))
        model = convert_fx(prepared_model)

    # 3. Ahora sí, cargar los pesos en el chasis correcto
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    logger.info(f"Checkpoint cargado <- {ckpt_path}")

    embeddings, labels = extract_embeddings(model, dataloader, device)
    save_embeddings(embeddings, labels, output_path)
    return embeddings, labels