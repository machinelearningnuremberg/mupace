"""Minimal MuPACE example: adapt Qwen3-0.6B to a Lots-of-LoRAs task, then generate.

    python examples/quickstart.py
"""

import mupace

result = mupace.adapt(
    model="Qwen/Qwen3-0.6B",
    dataset="Lots-of-LoRAs/task045_miscellaneous_sentence_paraphrasing",
    budget=200,
    rank=8,
    alpha=16,
    batch_size=None,  # Sized from your data; an integer overrides it.
    gradient_accumulation_steps=None,
    seq_len=None,
)
result.save_pretrained("./my_adapter")
print(result.generate("Rewrite this sentence: The weather is lovely today."))

# Later, in a fresh process:
#   model = mupace.load("./my_adapter")
#   print(model.generate("Rewrite this sentence: ..."))
