"""Tutorial: MuPACE on a model without built-in support (SmolLM2-135M, a Llama).

This is a tutorial integration, not a support claim: it shows the parts you write
for a new architecture and that the library then only updates the adapters. It
says nothing about MuPACE quality on Llama-style models.

The only architecture-specific code is ``attach`` (marked CHANGE). Everything else
- data, adapter init, the scheduler, its recurrent state and Muon - comes from the
library.

    python examples/custom_model.py
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mupace.torch as mt
from mupace import data
from mupace.integrations import ModelIntegration

MODEL = os.environ.get("MUPACE_EXAMPLE_MODEL", "HuggingFaceTB/SmolLM2-135M")
DATA = os.environ.get("MUPACE_EXAMPLE_DATA",
                      "Lots-of-LoRAs/task045_miscellaneous_sentence_paraphrasing")
OUT = os.environ.get("MUPACE_EXAMPLE_OUT", "./smollm2_adapter")
BUDGET = int(os.environ.get("MUPACE_EXAMPLE_BUDGET", "50"))
RANK, ALPHA = 8, 16
MICRO_BATCH, ACCUM, SEQ_LEN = 2, 4, 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- CHANGE for another model -------------------------------------------------
# Where the decoder layers live and what the Q and V projections are called.
# Llama, Mistral, Qwen2/3 and similar models have separate q_proj / v_proj inside
# model.model.layers[i].self_attn, so this function works unchanged for them.
# Fused layouts (one qkv matrix, or a query fused with a gate) need a module that
# slices the fused output instead of add_lora; see mupace/integrations/qwen3_8.py.
def attach(model, seed=0):
    gen = torch.Generator().manual_seed(seed)
    named = []
    for i, layer in enumerate(model.model.layers):
        for target in ("q_proj", "v_proj"):
            lora = mt.add_lora(layer.self_attn, target, RANK, ALPHA, generator=gen)
            named += [(f"L{i}.{target}/A", lora.A), (f"L{i}.{target}/B", lora.B)]
    return mt.freeze_all_but(model, named)  # base model frozen, adapters trainable
# -------------------------------------------------------------------------------


class SmolLM2Integration(ModelIntegration):
    """The same ``attach`` packaged for ``mupace.adapt(model, data,
    integration=SmolLM2Integration())`` when you do not need your own loop."""

    name = "smollm2_tutorial"

    def attach_lora(self, model, rank, alpha, seed):
        assert (rank, alpha) == (RANK, ALPHA)
        return attach(model, seed)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEVICE)
    model.config.use_cache = False

    named = attach(model)
    # Metadata for the scheduler: width = largest adapter dimension, depth = adapted
    # layers (inferred from the "L{i}." names). rank/alpha are the values used above.
    width, depth = mt.infer_width_depth(named)
    print(f"width={width} depth={depth} rank={RANK} alpha={ALPHA}")

    streams, info = data.prepare(DATA, tok, seq_len=SEQ_LEN, batch_size=MICRO_BATCH,
                                 max_examples=2000)
    print(f"data: {info['rows']}, scheduler feedback from {info['valid_source']}")

    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    opt = mt.MuPACE(named, budget=BUDGET, rank=RANK, alpha=ALPHA, width=width, depth=depth)
    model.train()
    while opt.num_updates < BUDGET:
        opt.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(ACCUM):
            x, y = streams["train"].next()
            loss = data.batch_loss(model, x, y, DEVICE)
            (loss / ACCUM).backward()
            train_loss += loss.item() / ACCUM
        with torch.no_grad():  # optional held-out feedback on the current parameters
            valid_loss = data.batch_loss(model, *streams["feedback"].next(), DEVICE).item()
        lr = opt.step(train_loss, valid_loss=valid_loss)  # None if the step was skipped
        if lr is not None and opt.num_updates % 10 == 0:
            print(f"update {opt.num_updates}: train {train_loss:.4f} valid {valid_loss:.4f} lr {lr:.2e}")

    same = all(torch.equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen)
    print(f"only adapters changed: {same}")

    mt.save_adapters(named, OUT, {"base_model": MODEL, "integration": SmolLM2Integration.name,
                                  "rank": RANK, "alpha": ALPHA})
    fresh = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEVICE)
    fresh_named = mt.load_adapters(attach(fresh), OUT)
    x, _ = streams["valid"].next()
    model.eval(), fresh.eval()
    with torch.no_grad():
        match = torch.allclose(model(input_ids=x.to(DEVICE)).logits,
                               fresh(input_ids=x.to(DEVICE)).logits, atol=1e-5)
    print(f"reloaded adapters match: {match} ({len(fresh_named) // 2} adapters in {OUT})")


if __name__ == "__main__":
    main()
