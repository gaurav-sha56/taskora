import torch
from transformers import AutoModelForCausalLM

def load_model(model_id, bits):
    if bits == 16:
        # bf16 is the 16-bit option that works well on CPU (fp16 is slow/unsupported for many ops)
        return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    if bits == 8:
        # PyTorch dynamic int8 quantization of all Linear layers (CPU only, no extra install)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        return torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    if bits == 4:
        # torchao int4 weight-only (pip install torchao)
        from transformers import TorchAoConfig
        cfg = TorchAoConfig("int4_weight_only", group_size=128)
        return AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cpu", quantization_config=cfg
        )