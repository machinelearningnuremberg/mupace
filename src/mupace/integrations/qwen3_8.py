"""Qwen3.8 text backbone (transformers ``qwen3_5``): hybrid gated-DeltaNet + attention.

Two layer kinds:

* ``full_attention``: ``q_proj`` is fused ``[query | gate]`` per head
  (rows ``num_heads * 2 * head_dim``); the adapter only touches the query half.
  ``v_proj`` is a plain linear layer.
* ``linear_attention``: ``in_proj_qkv`` is fused ``[q | k | v]``; adapters slice the
  q and v blocks. The output gates (``in_proj_z/a/b``) are never adapted.

Only the text backbone is loaded; vision and MTP weights are skipped unread.
Validated in research on Qwen/Qwen3.8-27B at the revision below with
transformers 5.12.1. Package integration tests use transformers 5.21.0.
"""

import json
import os

import torch
import torch.nn as nn

from mupace.integrations.base import ModelIntegration
from mupace.torch.lora import LoRALinear, freeze, freeze_all_but, lora_delta, new_lora_pair

MODEL_NAME = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TESTED_TRANSFORMERS = "5.21.0"
TEXT_PREFIX = "model.language_model."
UNUSED_PREFIXES = ("model.visual.", "mtp.")


class FusedQueryLoRA(nn.Module):
    """LoRA on the query half of a fused ``[query | gate]`` projection."""

    def __init__(self, base, rank, alpha, num_heads, head_dim, generator=None):
        super().__init__()
        self.base = base
        freeze(base)
        self.nh, self.hd = int(num_heads), int(head_dim)
        d_out, d_in = base.weight.shape
        if d_out != self.nh * 2 * self.hd:
            raise ValueError(f"q_proj has {d_out} rows, expected fused {self.nh}*2*{self.hd}")
        self.A, self.B = new_lora_pair(self.nh * self.hd, d_in, rank, generator, base.weight.device)
        self.scale = float(alpha) / float(rank)

    def forward(self, x):
        out = self.base(x)
        shape = out.shape[:-1]
        delta = (self.scale * lora_delta(x, self.A, self.B).to(out.dtype)).view(*shape, self.nh, self.hd)
        query, gate = out.view(*shape, self.nh, 2 * self.hd).split(self.hd, dim=-1)
        return torch.cat([query + delta, gate], dim=-1).view(*shape, -1)


class SlicedQKVLoRA(nn.Module):
    """LoRA on the q and/or v blocks of a fused ``[q | k | v]`` projection."""

    SLOT = {"q_proj": 0, "k_proj": 1, "v_proj": 2}

    def __init__(self, base, key_dim, value_dim, targets, rank, alpha, generator=None):
        super().__init__()
        self.base = base
        freeze(base)
        self.sizes = [int(key_dim), int(key_dim), int(value_dim)]
        d_out, d_in = base.weight.shape
        if d_out != sum(self.sizes):
            raise ValueError(f"in_proj_qkv has {d_out} rows, expected {sum(self.sizes)}")
        self.targets = tuple(targets)
        self.A, self.B = nn.ParameterDict(), nn.ParameterDict()
        for t in self.targets:
            self.A[t], self.B[t] = new_lora_pair(
                self.sizes[self.SLOT[t]], d_in, rank, generator, base.weight.device
            )
        self.scale = float(alpha) / float(rank)

    def forward(self, x):
        parts = list(self.base(x).split(self.sizes, dim=-1))
        for t in self.targets:
            i = self.SLOT[t]
            parts[i] = parts[i] + self.scale * lora_delta(x, self.A[t], self.B[t]).to(parts[i].dtype)
        return torch.cat(parts, dim=-1)


def _text_config(config):
    return getattr(config, "text_config", None) or config


class Qwen38Integration(ModelIntegration):
    name = "qwen3_8"

    def matches(self, config):
        return getattr(config, "model_type", None) in ("qwen3_5", "qwen3_5_text")

    def check(self, config, model_id=None):
        import transformers

        tc = _text_config(config)
        if getattr(tc, "num_experts", 0):
            raise ValueError("Qwen3.8 MoE models are not supported in this release")
        if getattr(config, "quantization_config", None):
            raise ValueError("quantized checkpoints are not supported in this release")
        warns = []
        if model_id is not None and model_id != MODEL_NAME:
            warns.append(f"{model_id} uses the Qwen3.8 layout but only {MODEL_NAME} is tested")
        if transformers.__version__ != TESTED_TRANSFORMERS:
            warns.append(f"Qwen3.8 integration was tested with transformers {TESTED_TRANSFORMERS}, "
                         f"found {transformers.__version__}")
        return warns

    def default_revision(self, model_id):
        return MODEL_REVISION if model_id == MODEL_NAME else None

    def build_meta_model(self, config):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

        with torch.device("meta"):
            return Qwen3_5ForCausalLM(_text_config(config))

    def load_model(self, model_id, *, revision=None, dtype=torch.bfloat16, device="cpu",
                   attn_implementation="sdpa"):
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig

        model_dir = model_id if os.path.isdir(model_id) else snapshot_download(
            model_id, revision=revision,
            allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"],
        )
        tcfg = _text_config(AutoConfig.from_pretrained(model_dir))
        tcfg.use_cache = False
        tcfg._attn_implementation = attn_implementation
        model = self.build_meta_model(tcfg)
        _load_text_weights(model, model_dir, torch.device(device), dtype)
        _materialize_buffers(model, torch.device(device))
        return model

    def attach_lora(self, model, rank, alpha, seed):
        gen = torch.Generator().manual_seed(int(seed))
        cfg = model.config
        layers = model.model.layers
        if len(cfg.layer_types) != len(layers):
            raise RuntimeError("layer list and config.layer_types disagree")
        named = []
        for i, (layer, kind) in enumerate(zip(layers, cfg.layer_types)):
            if kind == "full_attention":
                attn = layer.self_attn
                attn.q_proj = FusedQueryLoRA(attn.q_proj, rank, alpha, cfg.num_attention_heads,
                                             attn.head_dim, generator=gen)
                attn.v_proj = LoRALinear(attn.v_proj, rank, alpha, generator=gen)
                pairs = {t: (getattr(attn, t).A, getattr(attn, t).B) for t in self.targets}
            elif kind == "linear_attention":
                la = layer.linear_attn
                la.in_proj_qkv = SlicedQKVLoRA(la.in_proj_qkv, la.key_dim, la.value_dim,
                                               self.targets, rank, alpha, generator=gen)
                pairs = {t: (la.in_proj_qkv.A[t], la.in_proj_qkv.B[t]) for t in self.targets}
            else:
                raise RuntimeError(f"layer {i}: unknown layer type {kind!r}")
            for t in self.targets:
                a, b = pairs[t]
                named += [(f"L{i}.{t}/A", a), (f"L{i}.{t}/B", b)]
        if len(named) != 2 * len(self.targets) * len(layers):
            raise RuntimeError("adapter count does not match layers x targets")
        return freeze_all_but(model, named)


def _load_text_weights(model, model_dir, device, dtype):
    from safetensors import safe_open

    want = {n: tuple(t.shape) for n, t in model.state_dict().items()}
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    state = {}
    for shard in shards:
        with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                if name.startswith(UNUSED_PREFIXES):
                    continue
                key = "model." + name[len(TEXT_PREFIX):] if name.startswith(TEXT_PREFIX) else name
                if key not in want:
                    raise ValueError(f"{shard}:{name} is neither a text tensor nor a skipped prefix")
                if key in state:
                    raise ValueError(f"{name} appears in more than one shard")
                t = f.get_tensor(name)
                if tuple(t.shape) != want[key]:
                    raise ValueError(f"{name}: shape {tuple(t.shape)}, model expects {want[key]}")
                state[key] = t.to(device=device, dtype=dtype)
    if model.config.tie_word_embeddings and "lm_head.weight" not in state:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    missing = sorted(set(want) - set(state))
    if missing:
        raise ValueError(f"{len(missing)} text tensors missing, e.g. {missing[:5]}")
    model.load_state_dict(state, strict=True, assign=True)


def _materialize_buffers(model, device):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    # transformers 5.12 takes ``device``; later versions dropped it.
    try:
        rotary = Qwen3_5TextRotaryEmbedding(config=model.config, device=device)
    except TypeError:
        rotary = Qwen3_5TextRotaryEmbedding(config=model.config)
    model.model.rotary_emb = rotary.to(device)
    left = [n for n, b in model.named_buffers() if b.is_meta]
    left += [n for n, p in model.named_parameters() if p.is_meta]
    if left:
        raise RuntimeError(f"tensors still on the meta device after loading: {left[:5]}")
