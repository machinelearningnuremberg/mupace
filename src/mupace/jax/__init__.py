"""JAX runtime: a functional, jittable MuPACE optimizer. Does not import torch.

Model loading for JAX is not part of this release; bring your own LoRA parameter
pytree and loss function.
"""

from mupace.jax.optimizer import MuPACE, MuPACEState, infer_width_depth, newton_schulz

__all__ = ["MuPACE", "MuPACEState", "infer_width_depth", "newton_schulz"]
