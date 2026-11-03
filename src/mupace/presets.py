"""Defaults matching the settings used to evaluate MuPACE."""

import numpy as np

DEFAULT_BUDGET = 200
DEFAULT_RANK = 8
DEFAULT_ALPHA = 16.0
LORA_TARGETS = ("q_proj", "v_proj")

EVAL_EVERY = 25
EVAL_EXAMPLES = 256
MAX_EXAMPLES = 4000

# The ranges the scheduler saw during meta-training. Values outside are allowed
# but untested; adapt() warns and never changes them.
META_TRAINED_RANKS = (8, 32)
META_TRAINED_ALPHAS = (8.0, 32.0)

# The shortest seq_len that fits LENGTH_COVERAGE of the measured examples, with a
# batch shape giving micro_batch * grad_accum * seq_len = 4096 padded token
# positions per update.
SEQ_LEN_SETTINGS = {
    256: dict(micro_batch=4, grad_accum=4),
    512: dict(micro_batch=2, grad_accum=4),
    1024: dict(micro_batch=1, grad_accum=4),
    2048: dict(micro_batch=1, grad_accum=2),
}
LENGTH_COVERAGE = 0.95
FALLBACK_SEQ_LEN = 512
# adapt() warns when more than this share of training examples gets cut.
MAX_TRUNCATED_SHARE = 0.05


def measured_seq_len(lengths):
    """Shortest seq_len in SEQ_LEN_SETTINGS covering LENGTH_COVERAGE of ``lengths``."""
    need = float(np.quantile(np.asarray(lengths), LENGTH_COVERAGE))
    fits = [s for s in sorted(SEQ_LEN_SETTINGS) if s >= need]
    return fits[0] if fits else max(SEQ_LEN_SETTINGS)


def resolve_batch_settings(batch_size=None, gradient_accumulation_steps=None, seq_len=None,
                           lengths=None):
    """Settings sized from the token ``lengths`` of the examples (else the fallback),
    with every explicit argument taking precedence. -> dict(seq_len, micro_batch,
    grad_accum, padded_positions_per_update, preset)."""
    if seq_len is None and lengths is not None and len(lengths):
        auto, label = measured_seq_len(lengths), "measured"
    else:
        auto, label = FALLBACK_SEQ_LEN, "fallback"
    out = dict(seq_len=auto, **SEQ_LEN_SETTINGS[auto])
    for key, value in (("micro_batch", batch_size), ("grad_accum", gradient_accumulation_steps),
                       ("seq_len", seq_len)):
        if value is not None:
            if int(value) < 1:
                raise ValueError(f"{key} must be >= 1, got {value}")
            out[key] = int(value)
    out["padded_positions_per_update"] = out["seq_len"] * out["micro_batch"] * out["grad_accum"]
    out["preset"] = label
    return out
