"""MuPACE: fine-tune Qwen models with LoRA, no tuning needed.

Q/V LoRA adapters are trained with Muon; a frozen learned scheduler sets the learning rate.

    import mupace
    result = mupace.adapt(model="Qwen/Qwen3-0.6B",
                          dataset="Lots-of-LoRAs/task045_miscellaneous_sentence_paraphrasing")
    result.save_pretrained("./my_adapter")
    model = mupace.load("./my_adapter")

``import mupace`` stays light: the PyTorch workflow is imported on first use, and
``mupace.jax`` never imports torch.
"""

__version__ = "0.1.0"

_LAZY = {
    "adapt": "mupace.workflow",
    "load": "mupace.workflow",
    "AdaptResult": "mupace.workflow",
    "AdaptedModel": "mupace.workflow",
    "Policy": "mupace.policy",
    "is_main_process": "mupace.distributed",
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'mupace' has no attribute {name!r}")
