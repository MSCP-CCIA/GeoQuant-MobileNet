import torch
import copy
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
# 1. Imports nativos de PyTorch para QAT
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert

# 2. Imports de la nueva librería torchao para PTQ
import torchao
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
from src.topology.metrics import calculate_s_index

class QuantizationEngine:
    """Orquestador de torch.ao.quantization para PTQ y QAT."""

    def __init__(self, model, train_loader, val_loader, device):
        """
        Constructor del motor de cuantización.
        Recibe el modelo base y los datos para realizar PTQ y QAT.
        """
        from src.utils.logger import logger

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger

    def quantize_ptq(self, output_path: str):

        self.logger.info("=== Iniciando PTQ (torchao) sobre Baseline ArcFace ===")

        # Trabajamos sobre una copia para no alterar el modelo FP32 original en memoria
        ptq_model = copy.deepcopy(self.model)
        ptq_model.eval()

        self.logger.info("Aplicando compresión INT8 (torchao PTQ)...")
        quantize_(ptq_model, Int8DynamicActivationInt8WeightConfig())

        # Guardamos el modelo cuantizado
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(ptq_model.state_dict(), output_path)

        self.logger.info(f"[!] Modelo PTQ guardado exitosamente en {output_path}")
        return ptq_model

    def quantize_qat(self, output_path: str, arcface_head, epochs: int = 10):
        import copy
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from tqdm import tqdm
        import os

        # IMPORTACIONES DE VANGUARDIA: FX Graph Mode
        from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
        from torch.ao.quantization import get_default_qat_qconfig_mapping

        self.logger.info(f"=== Iniciando QAT (FX Graph Mode) sobre MobileNetV3 ({epochs} épocas) ===")

        # 1. Copiamos el modelo original
        model_to_quantize = copy.deepcopy(self.model)
        model_to_quantize.train()
        model_to_quantize.to(self.device)

        # 2. Mapeo de configuración para el motor FX
        qconfig_mapping = get_default_qat_qconfig_mapping("fbgemm")

        # 3. FX necesita un "tensor de ejemplo" para trazar la ruta de los datos en tu red
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

        # 4. Magia FX: Trazamos el grafo e insertamos los nodos de simulación INT8
        self.logger.info("Trazando el grafo matemático con FX (Superando limitación SE Block)...")
        qat_model = prepare_qat_fx(model_to_quantize, qconfig_mapping, example_inputs=(dummy_input,))

        arcface_head = arcface_head.to(self.device)
        arcface_head.train()

        criterion = nn.CrossEntropyLoss()
        params = list(qat_model.parameters()) + list(arcface_head.parameters())
        optimizer = optim.SGD(params, lr=5e-5, momentum=0.9)

        self.logger.info("Entrenando con simulación de ruido INT8 (FX FakeQuantize)...")
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"QAT Epoch {epoch}/{epochs}")
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()

                # FX maneja automáticamente la entrada FP32 -> proceso INT8 -> salida FP32
                embeddings = qat_model(inputs)

                # La cabeza ArcFace guía topológicamente el entrenamiento
                outputs = arcface_head(embeddings, targets)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

        self.logger.info("Convirtiendo grafo FX QAT a modelo INT8 definitivo...")
        qat_model.eval()
        qat_model.to('cpu')

        # Convertimos usando el motor FX
        model_int8 = convert_fx(qat_model)

        # Guardamos el estado de este grafo
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(model_int8.state_dict(), output_path)
        self.logger.info(f"[!] Extractor QAT (FX Mode) guardado en {output_path}")

        return model_int8

    def evaluate_quantization(self, eval_model, model_name="Modelo"):
        import torch
        from tqdm import tqdm
        from src.topology.metrics import calculate_s_index

        # 1. Validación estricta del dispositivo
        # Los modelos convertidos por QAT tradicional (fbgemm/qnnpack) operan EXCLUSIVAMENTE sobre CPU
        if "QAT" in model_name:
            device_to_use = torch.device('cpu')
            # Forzamos que PyTorch use el backend correcto de cuantización al evaluar
            torch.backends.quantized.engine = 'fbgemm'
        else:
            device_to_use = self.device

        eval_model.eval()
        eval_model.to(device_to_use)

        all_targets, all_embeddings = [], []

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc=f"Extrayendo {model_name}"):
                # Crucial: Mover los datos de entrada al mismo dispositivo que el modelo
                inputs = inputs.to(device_to_use)

                # Ejecutar inferencia
                embeddings = eval_model(inputs)

                all_embeddings.append(embeddings.cpu())
                all_targets.append(targets)

        emb_tensor = torch.cat(all_embeddings, dim=0)
        tgt_tensor = torch.cat(all_targets, dim=0)

        # Imprimir la dimensionalidad solo para el modelo original
        if model_name == "FP32 Original":
            self.logger.info(f"\n[INFO] Dimensión del Espacio Latente (Embeddings): {emb_tensor.shape[1]}")
            self.logger.info(f"[INFO] Volumen Total Analizado: {emb_tensor.shape[0]} imágenes\n")

        s_idx, intra, inter = calculate_s_index(emb_tensor, tgt_tensor)
        return s_idx, intra, inter
