# mupace

A meta-learned learning rate scheduler for Muon. No manual tuning needed.

Give it a model, a dataset, and a training budget. You get back a model ready to use. No learning rates to pick, no scheduler to configure, no training loop to write.

Under the hood, a meta-learned learning rate scheduler automatically controls the learning rate of the Muon optimizer throughout training.

We demonstrate a proof of concept with support for Qwen models using Q/V LoRA adapters. While the current implementation focuses on Qwen, the approach is not architecture-specific and could, in principle, be applied to other models.

> Only Qwen models are supported natively for now. Other models need a
> small custom integration (see [Other architectures](#other-architectures)).

## Installation

Requires Python 3.10 or newer and Transformers 5.21.0 or newer.
Install directly from GitHub.

```bash
python -m pip install "git+https://github.com/machinelearningnuremberg/mupace.git"
```

For JAX support, use the optional extra.

```bash
python -m pip install "mupace[jax] @ git+https://github.com/machinelearningnuremberg/mupace.git"
```

The scheduler weights are included in the package. Base models and Hugging Face
datasets are downloaded on first use.

## Quickstart

```python
import mupace

result = mupace.adapt(
    model="Qwen/Qwen3-0.6B",
    dataset="Lots-of-LoRAs/task045_miscellaneous_sentence_paraphrasing",
    budget=25,    # small dataset (~150 rows); use 200 for larger ones
    rank=8,
    alpha=16,
    batch_size=None,
    gradient_accumulation_steps=None,
    seq_len=None,
)
result.save_pretrained("./my_adapter")
print(result.generate("Rewrite the following sentence. The weather is lovely today."))

model = mupace.load("./my_adapter")   # later, in a fresh process
```

## What `adapt` does

1. Checks that the model is supported, measures your examples to pick the batch
   settings (see [Defaults](#defaults)), and checks that training fits in memory.
2. Picks the device (CUDA, MPS, CPU) and precision (bf16 on CUDA, float32 elsewhere).
   It uses FlashAttention-2 if installed, otherwise SDPA.
3. Freezes the base model and adds rank-`rank` LoRA to the Q and V projections
   (A ~ N(0, 1/d_in), B = 0, scale `alpha / rank`).
4. Tokenizes with the model's chat template. Only completion tokens are supervised.
5. Runs `budget` Muon updates. Before each update the scheduler sees features such as training
   statistics plus a validation-loss batch.
6. Every 25 updates, logs held-out loss on 256 examples per split. Test data is
   only reported. It is never used for scheduler feedback.

The scheduler comes pre-trained inside the package. Nothing to download, and it
never changes while you train. Your fine-tuned model runs without it.

## How the scheduler was trained

Muon still picks the direction of each update. The scheduler, a small LSTM with about
20K parameters, only sets the step size. It does this by making bounded adjustments
to a cosine-style schedule, based on how training is going. It was meta-trained
across many LoRA fine-tuning runs using a distribution of pretrained nanochat models with
8–16 layers and embedding dimensions of 128–2048, a few Lots-of-LoRAs tasks, ranks 8–32, alpha 8–32 and budgets of 100–300 steps. Its
goal was low validation loss over the later part of each run. After meta-training it
is frozen and applied one-shot to new tasks, including much deeper models such as
Qwen3-32B (64 layers), with no learning-rate search.

## Defaults

Rank **8**, alpha **16**, budget **200**, the settings used to evaluate MuPACE.
Batch size, gradient accumulation and sequence length are picked from your data.
To set them yourself, pass `batch_size` (the micro-batch), `gradient_accumulation_steps`
or `seq_len`.

## Data

Every example is a **prompt** and a **completion**. The model learns to write the
completion. `dataset` accepts the following inputs.

* **A local file** in `.jsonl`, `.json`, `.csv` or `.parquet` format.
* **A local folder** with one file per split, e.g. `train.jsonl` and `validation.jsonl`.
* **A Hugging Face dataset id**, e.g. `"Lots-of-LoRAs/task045_miscellaneous_sentence_paraphrasing"`.
* **A Python list** of `{"prompt": ..., "completion": ...}` dicts or `(prompt, completion)`
  tuples, or a dict of such lists per split (`{"train": [...], "validation": [...]}`).

The easiest start is a `.jsonl` file with one example per line.

```json
{"prompt": "Translate to German. Good morning", "completion": "Guten Morgen"}
{"prompt": "Translate to German. Thank you", "completion": "Danke"}
```

```python
result = mupace.adapt(model="Qwen/Qwen3-0.6B", dataset="my_data.jsonl")
```

Common column names like `input`/`output` or `question`/`answer` work too. For
other names, pass `column_map={"prompt": "article", "completion": "summary"}`.

If there's no validation split, ~10% of train is held out for it.

By default, `adapt` uses up to 4,000 examples per split. Pass `max_examples=0` to use all examples.

## Supported models

* **Qwen3** dense models (e.g. Qwen3-0.6B, Qwen3-32B).
* **Qwen3.8** text models (e.g. Qwen3.8-27B).

MoE, quantized and multimodal models, and splitting one model across GPUs, are
currently not supported.

### Multiple GPUs

Launch your script with `torchrun`.

```bash
torchrun --standalone --nproc_per_node=4 train.py
```

Each GPU holds the full model and trains on a share of the data. Leave `device`
unset or pass `"cuda"` for automatic assignment. Preloaded models must already
be on their worker's GPU.

Call `adapt` and `save_pretrained` on **every** rank. Only rank 0 writes the
adapter. Use `mupace.is_main_process()` only for your own prints and file writes,
never around `save_pretrained`, which would hang.

### Other architectures

Other models need a small custom integration that tells mupace where the Q and V
layers are. See `examples/custom_model.py` for a full example on a Llama model.

### JAX

```python
import mupace.jax

opt = mupace.jax.MuPACE(budget=200, rank=8, alpha=16)
state = opt.init(lora)  # {"L0.q_proj": {"A": ..., "B": ...}, ...}
lora, state, lr = opt.update(state, grads, lora, loss, valid_loss)  # jitted
```

Loading JAX models automatically is not included yet.

## Licensing

Apache-2.0, see [LICENSE](LICENSE). Copyright 2026 Jan Kobiolka, Technische
Universität Nürnberg.

The scheduler and optimizer code is a modified port of code adapted from
[`amoudgl/celo`](https://github.com/amoudgl/celo) (MIT), which in turn adapts
[`google/learned_optimization`](https://github.com/google/learned_optimization)
(Apache-2.0). See [NOTICE](NOTICE) for details.
