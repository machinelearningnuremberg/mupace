"""Qwen3 dense text models (``model_type == "qwen3"``): separate q_proj / v_proj.

Adapts the meta-trained Q/V targets. Validated in research on Qwen3-32B.
"""

import torch

from mupace.integrations.base import ModelIntegration
from mupace.torch.lora import add_lora, freeze_all_but

TESTED_MODELS = ("Qwen/Qwen3-32B", "Qwen/Qwen3-0.6B")


class Qwen3Integration(ModelIntegration):
    name = "qwen3"

    def matches(self, config):
        return getattr(config, "model_type", None) == "qwen3"

    def check(self, config, model_id=None):
        if getattr(config, "num_experts", 0):
            raise ValueError("Qwen3 MoE models are not supported in this release")
        if getattr(config, "quantization_config", None):
            raise ValueError("quantized checkpoints are not supported in this release")
        if model_id is not None and model_id not in TESTED_MODELS:
            return [f"{model_id} uses the Qwen3 layout but is not in the tested list {TESTED_MODELS}"]
        return []

    def attach_lora(self, model, rank, alpha, seed):
        gen = torch.Generator().manual_seed(int(seed))
        layers = [m for n, m in model.named_modules()
                  if n.endswith("self_attn") and all(hasattr(m, t) for t in ("q_proj", "v_proj"))]
        if not layers:
            raise RuntimeError("found no self_attn module with q_proj and v_proj")
        named = []
        for i, attn in enumerate(layers):
            for t in self.targets:
                mod = add_lora(attn, t, rank, alpha, generator=gen)
                named += [(f"L{i}.{t}/A", mod.A), (f"L{i}.{t}/B", mod.B)]
        return freeze_all_but(model, named)
