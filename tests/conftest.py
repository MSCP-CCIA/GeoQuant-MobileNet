"""
conftest.py — Fixtures compartidos para todos los tests de GeoQuant.
"""

import pytest
import torch


@pytest.fixture
def dummy_embeddings():
    """Embeddings sintéticos (64 muestras, 576 dims, 10 clases)."""
    torch.manual_seed(42)
    emb = torch.randn(64, 576)
    labels = torch.randint(0, 10, (64,))
    return emb, labels


@pytest.fixture
def dummy_embeddings_pair(dummy_embeddings):
    """Par de embeddings FP32/INT8 sintéticos."""
    emb_fp32, labels = dummy_embeddings
    torch.manual_seed(99)
    # INT8 simula ruido de cuantización
    emb_int8 = emb_fp32 + torch.randn_like(emb_fp32) * 0.05
    return emb_fp32, emb_int8, labels


@pytest.fixture
def small_config():
    """Configuración mínima para tests unitarios."""
    return {
        "seed": 42,
        "data": {
            "raw_dir": "data/raw/cub200",
            "processed_dir": "data/processed",
            "num_classes": 200,
            "image_size": 224,
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
        },
        "model": {
            "backbone": "mobilenet_v3_small",
            "pretrained": False,
            "arcface": {"margin": 0.5, "scale": 30.0},
        },
        "training": {
            "epochs": 1,
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "label_smoothing": 0.1,
            "checkpoint_dir": "outputs/test_checkpoints",
        },
    }