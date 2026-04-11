"""
Tests unitarios para el pipeline de cuantización PTQ/QAT.
"""

import copy
import pytest
import torch

from geoquant.models.backbone import MobileNetV3Backbone


class TestPTQ:
    def test_ptq_output_shape(self, small_config, tmp_path):
        """PTQ debe preservar la forma de salida del backbone."""
        pytest.importorskip("torchao", reason="torchao no instalado")
        from geoquant.quantization.ptq import apply_ptq

        model = MobileNetV3Backbone(pretrained=False)
        out_path = tmp_path / "ptq.pth"
        ptq_model = apply_ptq(model, str(out_path))

        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = ptq_model(dummy)
        assert out.shape == (2, model.in_features)

    def test_ptq_saves_file(self, small_config, tmp_path):
        pytest.importorskip("torchao", reason="torchao no instalado")
        from geoquant.quantization.ptq import apply_ptq

        model = MobileNetV3Backbone(pretrained=False)
        out_path = tmp_path / "ptq.pth"
        apply_ptq(model, str(out_path))
        assert out_path.exists()


class TestExport:
    def test_export_creates_file(self, tmp_path):
        """export_torchscript debe crear un archivo .pt válido."""
        from geoquant.quantization.export import export_torchscript

        model = MobileNetV3Backbone(pretrained=False)
        out_path = tmp_path / "model.pt"
        export_torchscript(model, str(out_path))
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_exported_model_runs(self, tmp_path):
        """El modelo TorchScript exportado debe producir inferencia correcta."""
        from geoquant.quantization.export import export_torchscript

        model = MobileNetV3Backbone(pretrained=False)
        out_path = tmp_path / "model.pt"
        export_torchscript(model, str(out_path))

        loaded = torch.jit.load(str(out_path))
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = loaded(dummy)
        assert out.shape[0] == 1


class TestBackbone:
    def test_backbone_output_shape(self):
        model = MobileNetV3Backbone(pretrained=False)
        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, model.in_features)

    def test_backbone_in_features(self):
        model = MobileNetV3Backbone(pretrained=False)
        assert model.in_features > 0