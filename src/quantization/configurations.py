from torch.ao.quantization import QConfig, default_per_channel_qconfig, default_qconfig

def get_qconfig(granularity="per_channel"):
    """Mapeo de QConfigs (Per-tensor vs Per-channel) para simulación INT8."""
    pass
