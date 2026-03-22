import torch
import os
from src.utils.logger import logger
from src.models.factory import get_arcface_model
from data.dataloaders import create_imagenette_dataloaders
from src.quantization.engine import QuantizationEngine
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)


def main():
    logger.info("==========================================================")
    logger.info("   INICIANDO EXPERIMENTO DE CUANTIZACIÓN (TESIS)          ")
    logger.info("==========================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch_name = 'mobilenet_v3_small'
    base_model_path = f'src/checkpoints/{arch_name}_best_arcface_fp32.pth'

    # 1. Verificar que el baseline exista
    if not os.path.exists(base_model_path):
        logger.error(f"¡No se encontró el modelo baseline en {base_model_path}!")
        return

    # 2. Cargar Dataloaders (Reusamos tu función de dataloaders.py)
    # Asumo que tienes una función que retorna (train_loader, val_loader)
    logger.info("Cargando dataset Imagenette...")
    train_loader, val_loader = create_imagenette_dataloaders(batch_size=64)

    # 3. Cargar el Modelo Base FP32 (Extractor + Cabeza ArcFace)
    logger.info(f"Cargando arquitectura {arch_name} con ArcFace...")
    extractor, arcface_head = get_arcface_model(arch_name, num_classes=10)
    extractor.load_state_dict(torch.load(base_model_path, map_location=device))
    extractor = extractor.to(device)

    # 4. Inicializar el Motor de Cuantización
    # Asumimos que el __init__ de tu QuantizationEngine recibe estos parámetros
    engine = QuantizationEngine(
        model=extractor,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    resultados = {}

    # =====================================================================
    # FASE 1: EVALUACIÓN DEL BASELINE FP32
    # =====================================================================
    logger.info("\n--- [FASE 1] Evaluando Espacio Latente Original (FP32) ---")
    s_fp32, intra_fp32, inter_fp32 = engine.evaluate_quantization(engine.model, "FP32 Original")
    resultados['FP32 Original'] = (s_fp32, intra_fp32, inter_fp32)

    # =====================================================================
    # FASE 2: CUANTIZACIÓN PTQ (Post-Training Quantization)
    # =====================================================================
    logger.info("\n--- [FASE 2] Aplicando Cuantización PTQ (torchao) ---")
    ptq_path = f'src/checkpoints/quantized/{arch_name}_arcface_ptq_int8.pth'

    modelo_ptq = engine.quantize_ptq(output_path=ptq_path)
    s_ptq, intra_ptq, inter_ptq = engine.evaluate_quantization(modelo_ptq, "PTQ INT8")
    resultados['PTQ INT8'] = (s_ptq, intra_ptq, inter_ptq)

    # =====================================================================
    # FASE 3: CUANTIZACIÓN QAT (Quantization-Aware Training)
    # =====================================================================
    logger.info("\n--- [FASE 3] Aplicando Cuantización QAT (5 épocas) ---")
    qat_path = f'src/checkpoints/quantized/{arch_name}_arcface_qat_int8.pth'

    # Pasamos el arcface_head para que guíe el fine-tuning
    modelo_qat = engine.quantize_qat(output_path=qat_path, arcface_head=arcface_head, epochs=5)
    s_qat, intra_qat, inter_qat = engine.evaluate_quantization(modelo_qat, "QAT INT8")
    resultados['QAT INT8'] = (s_qat, intra_qat, inter_qat)

    # =====================================================================
    # REPORTE FINAL DE LA TESIS
    # =====================================================================
    print("\n" + "=" * 85)
    print(
        f"{'MÉTODO (MobileNetV3 + ArcFace)':<32} | {'S-INDEX':<10} | {'INTRA (Dispersión)':<18} | {'INTER (Distancia)'}")
    print("=" * 85)
    for metodo, (s_idx, intra, inter) in resultados.items():
        print(f"{metodo:<32} | {s_idx:>8.4f}   | {intra:>18.4f} | {inter:>15.4f}")
    print("=" * 85)
    logger.info("Experimento finalizado con éxito.")


if __name__ == '__main__':
    main()