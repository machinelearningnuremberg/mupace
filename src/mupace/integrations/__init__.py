"""Built-in integrations for the tested layouts, plus the custom interface."""

from mupace.integrations.base import (
    ModelIntegration,
    prepare_for_inference,
    prepare_for_training,
)
from mupace.integrations.qwen3 import Qwen3Integration
from mupace.integrations.qwen3_8 import Qwen38Integration

BUILTIN = {i.name: i for i in (Qwen3Integration(), Qwen38Integration())}


def resolve(config, integration=None):
    """The integration for a Hugging Face config; a user-given one always wins."""
    if integration is not None:
        return BUILTIN[integration] if isinstance(integration, str) else integration
    for candidate in BUILTIN.values():
        if candidate.matches(config):
            return candidate
    raise ValueError(
        f"model_type {getattr(config, 'model_type', None)!r} has no built-in MuPACE "
        f"integration (built in: Qwen3 dense, Qwen3.8 text). For another architecture "
        "write a ModelIntegration (see examples/custom_model.py) and pass "
        "integration=... to mupace.adapt."
    )


__all__ = [
    "BUILTIN",
    "ModelIntegration",
    "Qwen3Integration",
    "Qwen38Integration",
    "prepare_for_inference",
    "prepare_for_training",
    "resolve",
]
