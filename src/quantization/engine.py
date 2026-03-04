import torch
from torch.ao.quantization import get_default_qconfig, prepare_fx, convert_fx

class QuantizationEngine:
    """Orquestador de torch.ao.quantization para PTQ y QAT."""
    def apply_ptq(self, model, calib_loader):
        pass
    
    def apply_qat(self, model, train_loader):
        pass
