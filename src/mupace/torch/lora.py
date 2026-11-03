"""LoRA building blocks shared by the built-in integrations and custom models.

Initialization matches the research code: A ~ N(0, 1/d_in), B = 0, both float32,
drawn from one seeded generator in adapter order; the adapter output is
``base(x) + (alpha / rank) * B(A(x))`` computed in float32.
"""

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "mupace_adapter.json"
FORMAT_VERSION = 1


def new_lora_pair(d_out, d_in, rank, generator, device):
    """-> (A, B) parameters: A ~ N(0, 1/d_in) of shape (rank, d_in), B = 0 of shape (d_out, rank)."""
    a = torch.randn(rank, d_in, dtype=torch.float32, generator=generator) / math.sqrt(d_in)
    return (
        nn.Parameter(a.to(device)),
        nn.Parameter(torch.zeros(d_out, rank, dtype=torch.float32, device=device)),
    )


def freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)


def lora_delta(x, a, b):
    return F.linear(F.linear(x.float(), a), b)


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` (or anything with ``.weight`` of shape (out, in))."""

    def __init__(self, base, rank, alpha, generator=None):
        super().__init__()
        self.base = base
        freeze(base)
        d_out, d_in = base.weight.shape
        self.A, self.B = new_lora_pair(d_out, d_in, rank, generator, base.weight.device)
        self.scale = float(alpha) / float(rank)

    def forward(self, x):
        out = self.base(x)
        return out + self.scale * lora_delta(x, self.A, self.B).to(out.dtype)


def add_lora(parent, attr, rank, alpha, generator=None):
    """Replace ``parent.<attr>`` with a :class:`LoRALinear`. -> the new module."""
    wrapped = LoRALinear(getattr(parent, attr), rank, alpha, generator=generator)
    setattr(parent, attr, wrapped)
    return wrapped


def freeze_all_but(model, named):
    """Freeze every parameter of ``model`` except the adapters in ``named``; verify it."""
    for p in model.parameters():
        p.requires_grad_(False)
    for _, p in named:
        p.requires_grad_(True)
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if trainable != {id(p) for _, p in named}:
        raise RuntimeError("trainable parameters are not exactly the LoRA adapters")
    return named


def save_adapters(named, path, metadata):
    """Write ``adapter_model.safetensors`` + ``mupace_adapter.json`` into ``path``."""
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    tensors = {n: p.detach().float().cpu().contiguous() for n, p in named}
    save_file(tensors, os.path.join(path, ADAPTER_WEIGHTS))
    meta = dict(metadata, format_version=FORMAT_VERSION,
                adapters={n: list(t.shape) for n, t in tensors.items()})
    with open(os.path.join(path, ADAPTER_CONFIG), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def read_adapter_config(path):
    with open(os.path.join(path, ADAPTER_CONFIG)) as f:
        meta = json.load(f)
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported adapter format {meta.get('format_version')!r}")
    return meta


def load_adapters(named, path):
    """Copy saved adapter weights into ``named``; names and shapes must match exactly."""
    from safetensors.torch import load_file

    state = load_file(os.path.join(path, ADAPTER_WEIGHTS))
    want = {n for n, _ in named}
    if set(state) != want:
        missing, extra = sorted(want - set(state)), sorted(set(state) - want)
        raise ValueError(f"adapter mismatch: missing {missing[:4]}, unexpected {extra[:4]}")
    with torch.no_grad():
        for n, p in named:
            if tuple(state[n].shape) != tuple(p.shape):
                raise ValueError(f"{n}: saved shape {tuple(state[n].shape)}, model has {tuple(p.shape)}")
            p.copy_(state[n].to(device=p.device, dtype=p.dtype))
    return named
