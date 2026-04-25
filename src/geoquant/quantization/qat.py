import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from geoquant.utils.logging import get_logger
from geoquant.evaluation.block_c import overlap_at_k

logger = get_logger(__name__)


def apply_qat_distillation(
        model_fp32: torch.nn.Module,
        arcface_head: torch.nn.Module,
        train_loader,
        val_loader,
        output_path: str,
        epochs: int = 10,
        lr: float = 5e-5,
        device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
    from torch.ao.quantization import get_default_qat_qconfig_mapping

    logger.info(f"Iniciando QAT con Destilación de Espacio Latente...")

    # Preparar Maestro (FP32)
    teacher = copy.deepcopy(model_fp32).eval().to(device)
    for param in teacher.parameters():
        param.requires_grad = False

    # Preparar Estudiante (FakeQuantize)
    student = copy.deepcopy(model_fp32).train().to(device)
    qconfig_mapping = get_default_qat_qconfig_mapping("fbgemm")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    qat_model = prepare_qat_fx(student, qconfig_mapping, example_inputs=(dummy_input,))

    arcface_head = copy.deepcopy(arcface_head).to(device).train()

    criterion_arcface = nn.CrossEntropyLoss()
    criterion_kd = nn.MSELoss()
    alpha_kd = 10.0  # Prioridad a la geometría del Maestro

    params = list(qat_model.parameters()) + list(arcface_head.parameters())
    optimizer = optim.SGD(params, lr=lr, momentum=0.9)

    # --- LÓGICA DE MAXIMIZACIÓN DE OVERLAP ---
    best_overlap = 0.0
    best_qat_state = None

    for epoch in range(1, epochs + 1):
        qat_model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch}/{epochs}")

        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_emb = teacher(inputs)

            qat_emb = qat_model(inputs)
            logits = arcface_head(qat_emb, targets)

            loss_arcface = criterion_arcface(logits, targets)
            loss_kd = criterion_kd(qat_emb, teacher_emb)
            total_loss = loss_arcface + (alpha_kd * loss_kd)

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})

        # --- BUCLE DE VALIDACIÓN (Doble Extracción) ---
        qat_model.eval()
        all_student_emb, all_teacher_emb = [], []

        with torch.no_grad():
            for val_inputs, _ in val_loader:
                val_inputs = val_inputs.to(device)
                # Extraer embeddings del Estudiante ruidoso y del Maestro FP32
                s_emb = qat_model(val_inputs)
                t_emb = teacher(val_inputs)

                all_student_emb.append(s_emb.cpu())
                all_teacher_emb.append(t_emb.cpu())

        s_tensor = torch.cat(all_student_emb)
        t_tensor = torch.cat(all_teacher_emb)

        # Evaluar qué porcentaje del vecindario (k=5) sobrevivió a la cuantización
        val_overlap = overlap_at_k(t_tensor, s_tensor, k=5)

        logger.info(f"  QAT Epoch {epoch} | loss={running_loss / len(train_loader):.4f} | Overlap@5={val_overlap:.4f}")

        if val_overlap > best_overlap:
            best_overlap = val_overlap
            best_qat_state = copy.deepcopy(qat_model.state_dict())
            logger.info("  ^-- Topología mejorada, guardando estado!")

    # Cargar el pico de rendimiento y convertir
    qat_model.load_state_dict(best_qat_state)
    logger.info("Convirtiendo modelo QAT a INT8 definitivo...")
    qat_model.eval().to("cpu")
    model_int8 = convert_fx(qat_model)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_int8.state_dict(), out)
    logger.info(f"Modelo QAT (Destilado) guardado -> {out}")

    return model_int8