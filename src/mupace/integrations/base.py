"""The interface between ``mupace.adapt`` and a model architecture.

A built-in integration exists for each tested layout. For another model, subclass
:class:`ModelIntegration` (see ``examples/custom_model.py``) and pass an instance
as ``mupace.adapt(..., integration=MyIntegration())``. Nothing about a model is
assumed to work until such an integration is written and checked.
"""

import torch


class ModelIntegration:
    """Locate Q/V projections, insert LoRA, freeze the rest, load and reload.

    Required: ``attach_lora``. Override ``load_model`` when ``from_pretrained`` is not
    enough, and ``metadata`` when the default width/depth rule does not apply.
    """

    name = "custom"
    targets = ("q_proj", "v_proj")

    def matches(self, config):
        """True when this integration supports a Hugging Face ``config``."""
        return False

    def check(self, config, model_id=None):
        """Raise for unsupported variants; return a list of warnings."""
        return []

    def load_model(self, model_id, *, revision=None, dtype=torch.float32, device="cpu",
                   attn_implementation="sdpa"):
        from transformers import AutoModelForCausalLM

        kw = dict(revision=revision, dtype=dtype, attn_implementation=attn_implementation,
                  low_cpu_mem_usage=True)
        if torch.device(device).type == "cuda":
            kw["device_map"] = {"": torch.device(device).index or 0}
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        return model.to(device)

    def build_meta_model(self, config):
        """The model on the meta device (no memory), for parameter counting."""
        from transformers import AutoModelForCausalLM

        with torch.device("meta"):
            return AutoModelForCausalLM.from_config(config)

    def attach_lora(self, model, rank, alpha, seed):
        """Insert adapters, freeze everything else. -> ``[(name, parameter), ...]``.

        Names must be ``L{layer}.{target}/A`` and ``.../B`` so the default width/depth
        rule counts layers; order them layer by layer, A before B.
        """
        raise NotImplementedError

    def metadata(self, named):
        """-> (width, depth) passed to the scheduler."""
        from mupace.torch import infer_width_depth

        return infer_width_depth(named)


def prepare_for_training(model, gradient_checkpointing=True):
    model.config.use_cache = False
    if gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.train()
    return model


def prepare_for_inference(model):
    if getattr(model, "is_gradient_checkpointing", False):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    return model
