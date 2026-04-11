# ==============================================================================
# GeoQuant — Makefile
# Uso: make <target>
# ==============================================================================

.PHONY: help install train quantize-ptq quantize-qat evaluate benchmark test lint fmt clean

PYTHON  := python
CONFIG  := configs/config.yaml
FP32    := outputs/checkpoints/best_fp32.pth
PTQ     := outputs/checkpoints/mobilenet_v3_small_ptq_int8.pth
QAT     := outputs/checkpoints/mobilenet_v3_small_qat_int8.pth

help:
	@echo ""
	@echo "GeoQuant — Targets disponibles:"
	@echo "  install        Instalar dependencias con uv"
	@echo "  train          Entrenar baseline FP32 con ArcFace"
	@echo "  quantize-ptq   Aplicar PTQ al baseline FP32"
	@echo "  quantize-qat   Aplicar QAT al baseline FP32"
	@echo "  evaluate-ptq   Evaluación geométrica PTQ vs FP32"
	@echo "  evaluate-qat   Evaluación geométrica QAT vs FP32"
	@echo "  benchmark      Benchmark de latencia (FP32 / PTQ / QAT)"
	@echo "  test           Ejecutar suite de tests"
	@echo "  lint           Verificar código con ruff"
	@echo "  fmt            Formatear código con ruff"
	@echo "  clean          Eliminar artefactos generados"
	@echo ""

install:
	uv sync

train:
	$(PYTHON) scripts/train.py --config $(CONFIG) --experiment configs/experiment/baseline_fp32.yaml

quantize-ptq:
	$(PYTHON) scripts/quantize.py --approach ptq --config $(CONFIG) \
	    --experiment configs/experiment/ptq_static.yaml \
	    --checkpoint $(FP32) --export

quantize-qat:
	$(PYTHON) scripts/quantize.py --approach qat --config $(CONFIG) \
	    --experiment configs/experiment/qat_full.yaml \
	    --checkpoint $(FP32) --export

evaluate-ptq:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG) --fp32 $(FP32) --int8 $(PTQ) --approach ptq

evaluate-qat:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG) --fp32 $(FP32) --int8 $(QAT) --approach qat

benchmark:
	$(PYTHON) scripts/benchmark.py --config $(CONFIG) \
	    --fp32 $(FP32) --ptq $(PTQ) --qat $(QAT)

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ scripts/ tests/

fmt:
	ruff format src/ scripts/ tests/

clean:
	rm -rf outputs/checkpoints outputs/results outputs/logs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete