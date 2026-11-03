"""``mupace.adapt``: model + dataset + budget -> adapted model ready for inference."""

import json
import math
import os
import time
import warnings

import torch

from mupace import data as data_lib
from mupace import distributed, presets
from mupace.integrations import prepare_for_inference, prepare_for_training, resolve
from mupace.torch.lora import load_adapters, read_adapter_config, save_adapters
from mupace.torch.optimizer import MuPACE

HISTORY_FILE = "mupace_history.json"


# ------------------------------------------------------------------ environment


def choose_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_dtype(device, dtype=None):
    """bfloat16 on CUDA GPUs that support it, float32 elsewhere. Adapters are always float32."""
    if dtype is not None:
        return getattr(torch, dtype) if isinstance(dtype, str) else dtype
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def choose_attention(device, attn_implementation=None):
    if attn_implementation is not None:
        return attn_implementation
    if device.type == "cuda":
        try:
            from transformers.utils import is_flash_attn_2_available

            if is_flash_attn_2_available():
                return "flash_attention_2"
        except ImportError:
            pass
    return "sdpa"


def available_memory(device):
    """Free bytes on ``device`` (None when unknown)."""
    try:
        if device.type == "cuda":
            return int(torch.cuda.mem_get_info(device.index or 0)[0])
        if device.type == "mps" and hasattr(torch.mps, "recommended_max_memory"):
            return int(torch.mps.recommended_max_memory())
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError, RuntimeError):
        return None


def check_memory(integration, config, device, dtype, settings):
    """Refuse when the frozen weights cannot fit; warn when training likely will not."""
    meta = integration.build_meta_model(config)
    n_params = sum(p.numel() for p in meta.parameters())
    tc = getattr(config, "text_config", None) or config
    del meta
    elt = torch.empty((), dtype=dtype).element_size()
    weights = n_params * elt
    tokens = settings["micro_batch"] * settings["seq_len"]
    acts = tokens * (tc.num_hidden_layers * tc.hidden_size * elt  # checkpointed layer inputs
                     + 12 * tc.vocab_size                          # float32 logits, softmax, grad
                     + 16 * tc.intermediate_size * elt)            # one recomputed layer
    free = available_memory(device)
    est = {"parameters": n_params, "weight_bytes": weights, "activation_bytes": int(acts),
           "available_bytes": free}
    if free is None:
        return est
    gib = 2 ** 30
    if weights > free:
        raise MemoryError(
            f"the frozen model needs ~{weights / gib:.1f} GiB in {dtype} but only "
            f"~{free / gib:.1f} GiB is available on {device}. Use a smaller model or a "
            "larger device (quantization and model sharding are not supported yet; "
            "with torchrun every GPU holds the full model)."
        )
    if weights + acts > free:
        warnings.warn(
            f"estimated peak ~{(weights + acts) / gib:.1f} GiB exceeds the ~{free / gib:.1f} GiB "
            f"available on {device}; lower batch_size or seq_len if it runs out of memory."
        )
    return est


def _warn_outside_meta_training(rank, alpha):
    lo, hi = presets.META_TRAINED_RANKS
    if not lo <= rank <= hi:
        warnings.warn(f"rank={rank} is outside the meta-trained range [{lo}, {hi}]; "
                      "it is used as given but the scheduler is untested there")
    lo, hi = presets.META_TRAINED_ALPHAS
    if not lo <= alpha <= hi:
        warnings.warn(f"alpha={alpha} is outside the meta-trained range [{lo}, {hi}]; "
                      "it is used as given but the scheduler is untested there")


def _split_accumulation(settings, world_size):
    """Spread ``grad_accum`` micro-batches over the ranks, so one update sees the same
    number of examples on any GPU count (rounded up when it does not divide)."""
    total = settings["grad_accum"]
    per_rank = max(1, math.ceil(total / world_size))
    if per_rank * world_size != total:
        warnings.warn(f"grad_accum={total} does not split evenly over {world_size} GPUs; "
                      f"each update uses {per_rank * world_size} micro-batches instead")
    settings = dict(settings, grad_accum=per_rank * world_size, grad_accum_per_rank=per_rank,
                    world_size=world_size)
    settings["padded_positions_per_update"] = (settings["seq_len"] * settings["micro_batch"]
                                               * settings["grad_accum"])
    return settings


def _warn_truncation(lengths, seq_len):
    cut = float((lengths > seq_len).mean())
    if cut > presets.MAX_TRUNCATED_SHARE:
        warnings.warn(f"{cut:.0%} of training examples are longer than seq_len={seq_len} "
                      "tokens and will be cut; pass a larger seq_len (and a smaller "
                      "batch_size if memory runs out)")


# ------------------------------------------------------------------ results


class AdaptedModel:
    """A base model with MuPACE LoRA adapters, ready for generation."""

    def __init__(self, model, tokenizer, named, metadata, integration):
        self.model = model
        self.tokenizer = tokenizer
        self.adapters = named
        self.metadata = metadata
        self.integration = integration

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=256, **kwargs):
        """Greedy by default. ``prompt`` gets the same chat formatting as in training."""
        prepare_for_inference(self.model)
        text = data_lib.render_prompt(self.tokenizer, prompt)
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        pad = self.tokenizer.pad_token_id
        kwargs.setdefault("do_sample", False)
        if not kwargs["do_sample"]:
            kwargs.update(temperature=None, top_p=None, top_k=None)
        out = self.model.generate(
            **enc, max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id if pad is None else pad, **kwargs,
        )
        return self.tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def save_pretrained(self, path):
        """Adapters + metadata + tokenizer. Reload with ``mupace.load(path)``.

        Under ``torchrun`` call it on **every** rank, not inside an
        ``if mupace.is_main_process():`` block. Only global rank 0 writes, and then
        every rank meets at a barrier. If the call is guarded, rank 0 waits at that
        barrier for ranks that never arrive.
        """
        ctx = distributed.current()
        if ctx.is_main:
            self._write(path)
        ctx.barrier()
        return path

    def _write(self, path):
        save_adapters(self.adapters, path, self.metadata)
        self.tokenizer.save_pretrained(path)


class AdaptResult(AdaptedModel):
    """:class:`AdaptedModel` plus the training ``history`` and resolved ``settings``."""

    def __init__(self, model, tokenizer, named, metadata, integration, history, settings):
        super().__init__(model, tokenizer, named, metadata, integration)
        self.history = history
        self.settings = settings

    def _write(self, path):
        super()._write(path)
        with open(os.path.join(path, HISTORY_FILE), "w") as f:
            json.dump({"settings": self.settings, "history": self.history}, f, indent=2)


# ------------------------------------------------------------------ adapt


def adapt(
    model,
    dataset,
    budget=presets.DEFAULT_BUDGET,
    rank=presets.DEFAULT_RANK,
    alpha=presets.DEFAULT_ALPHA,
    batch_size=None,
    gradient_accumulation_steps=None,
    seq_len=None,
    *,
    tokenizer=None,
    integration=None,
    revision=None,
    column_map=None,
    format_fn=None,
    max_examples=presets.MAX_EXAMPLES,
    eval_every=presets.EVAL_EVERY,
    eval_examples=presets.EVAL_EXAMPLES,
    seed=0,
    device=None,
    dtype=None,
    attn_implementation=None,
    gradient_checkpointing=True,
    memory_check=True,
    max_consecutive_skips=10,
    policy=None,
    verbose=True,
):
    """Fine-tune Q/V LoRA adapters with MuPACE-paced Muon for ``budget`` updates.

    Args:
        model: Hugging Face id or local path, or an already loaded model (then pass
            ``tokenizer``; the base stays where it is).
        dataset: see :mod:`mupace.data` for the accepted forms.
        budget: successful optimizer updates (after gradient accumulation).
        rank, alpha: LoRA rank and alpha; passed unchanged to the scheduler.
        batch_size: micro-batch size. ``None`` sizes it from the measured example
            lengths.
        gradient_accumulation_steps, seq_len: ``None`` chooses them the same way.
        eval_every, eval_examples: periodic held-out loss on train/valid/test;
            ``eval_every=0`` disables it.

    Launched with ``torchrun --nproc_per_node=N``, training is data parallel over N
    GPUs (see :mod:`mupace.distributed`): each GPU holds the full model, the
    ``grad_accum`` micro-batches of an update are split between them, and only rank
    0 logs. Each worker trains on its own GPU (see
    :func:`mupace.distributed.rank_device`; ``device`` may be ``None`` or
    ``"cuda"``); a preloaded ``model`` must already be there.
    """
    ctx, dist_dev = distributed.setup(device)
    log = print if verbose and ctx.is_main else (lambda *a, **k: None)
    rank, alpha, budget = int(rank), float(alpha), int(budget)
    if budget < 1:
        raise ValueError("budget must be >= 1")
    _warn_outside_meta_training(rank, alpha)
    dev = dist_dev or choose_device(device)

    if isinstance(model, str):
        from transformers import AutoConfig, AutoTokenizer

        model_id = model
        with ctx.main_first():
            config = AutoConfig.from_pretrained(model_id, revision=revision)
            integ = resolve(config, integration)
            if revision is None and hasattr(integ, "default_revision"):
                revision = integ.default_revision(model_id)
            tok = tokenizer or AutoTokenizer.from_pretrained(model_id, revision=revision)
        for w in integ.check(config, model_id):
            warnings.warn(w)
        dt = choose_dtype(dev, dtype)
        attn = choose_attention(dev, attn_implementation)
    else:
        if tokenizer is None:
            raise ValueError("pass tokenizer= together with a loaded model")
        net, tok, mem = model, tokenizer, None
        model_id = getattr(net.config, "_name_or_path", None) or type(net).__name__
        integ = resolve(net.config, integration)
        for w in integ.check(net.config, None):
            warnings.warn(w)
        if ctx.enabled:
            distributed.check_model_device(net, dev)
        dev = next(net.parameters()).device
        dt = next(net.parameters()).dtype

    with ctx.main_first():
        splits, data_info = data_lib.load_splits(dataset, column_map=column_map,
                                                 format_fn=format_fn, max_examples=max_examples,
                                                 revision=None)
    lengths = data_lib.token_lengths(tok, splits["train"], seed=seed)
    settings = presets.resolve_batch_settings(batch_size, gradient_accumulation_steps, seq_len,
                                              lengths=lengths)
    settings = _split_accumulation(settings, ctx.world_size)
    _warn_truncation(lengths, settings["seq_len"])
    if isinstance(model, str):
        mem = check_memory(integ, config, dev, dt, settings) if memory_check else None
        log(f"[mupace] loading {model_id} ({dt}, {attn}) on {dev}"
            + (f" and {ctx.world_size - 1} more processes" if ctx.enabled else ""))
        with ctx.main_first():
            net = integ.load_model(model_id, revision=revision, dtype=dt, device=dev,
                                   attn_implementation=attn)

    prepare_for_training(net, gradient_checkpointing)
    named = integ.attach_lora(net, rank, alpha, seed)
    ctx.broadcast([p for _, p in named])
    width, depth = integ.metadata(named)
    streams, data_info = data_lib.build_streams(
        splits, data_info, tok, seq_len=settings["seq_len"], batch_size=settings["micro_batch"],
        seed=seed, rank=ctx.rank, world_size=ctx.world_size,
    )
    opt = MuPACE(named, budget=budget, rank=rank, alpha=alpha, width=width, depth=depth,
                 policy=policy)
    log(f"[mupace] {integ.name}: width={width} depth={depth} rank={rank} alpha={alpha} "
        f"| {sum(p.numel() for _, p in named) / 1e6:.2f}M adapter params")
    accum_text = (f"{settings['grad_accum_per_rank']} x {ctx.world_size} GPUs" if ctx.enabled
                  else str(settings["grad_accum"]))
    log(f"[mupace] batch {settings['micro_batch']} x accumulation {accum_text} "
        f"x seq_len {settings['seq_len']} ({settings['preset']} preset); "
        f"feedback from {data_info['valid_source']}; rows {data_info['rows']}")

    history = run(net, named, opt, streams, settings, budget=budget, eval_every=eval_every,
                  eval_examples=eval_examples, seed=seed, device=dev,
                  max_consecutive_skips=max_consecutive_skips, log=log, ctx=ctx)
    prepare_for_inference(net)
    metadata = {
        "base_model": model_id,
        "revision": revision,
        "integration": integ.name,
        "rank": rank,
        "alpha": alpha,
        "targets": list(integ.targets),
        "width": width,
        "depth": depth,
        "budget": budget,
        "scheduler": opt.policy.source,
    }
    run_settings = dict(settings, budget=budget, rank=rank, alpha=alpha, seed=seed,
                        device=str(dev), dtype=str(dt), data=data_info, memory=mem,
                        eval_every=eval_every, eval_examples=eval_examples)
    return AdaptResult(net, tok, named, metadata, integ, history, run_settings)


def run(model, named, opt, streams, settings, *, budget, eval_every, eval_examples, seed,
        device, max_consecutive_skips=10, log=print, ctx=distributed.SINGLE):
    """The training loop: ``budget`` successful MuPACE updates with periodic evaluation.

    With a distributed ``ctx``, losses and adapter gradients are averaged over ranks
    before each update, so every rank makes the same one.
    """
    params = [p for _, p in named]
    accum = settings.get("grad_accum_per_rank", settings["grad_accum"])
    n_eval = math.ceil(eval_examples / settings["micro_batch"]) * settings["micro_batch"]
    eval_splits = [s for s in ("train", "valid", "test") if s in streams]
    fixed = {s: streams[s].fixed(n_eval, seed + 17) for s in eval_splits} if eval_every else {}
    hist = {"step": [], **{s: [] for s in fixed}, "lr": [], "train_loss": [],
            "valid_feedback": [], "skipped": 0}

    def readout(step):
        hist["step"].append(step)
        for s, batches in fixed.items():
            hist[s].append(data_lib.eval_loss(model, batches, device, ctx))
        log(f"[mupace] update {step}/{budget}: "
            + " ".join(f"{s}={hist[s][-1]:.4f}" for s in fixed))

    if fixed:
        readout(0)
    t0, skips = time.time(), 0
    while opt.num_updates < budget:
        train_loss = accumulate_gradients(model, params, streams["train"], accum, device, ctx)
        xv, yv = streams["feedback"].next()
        with torch.no_grad():
            valid_loss = float(data_lib.batch_loss(model, xv, yv, device))
        valid_loss = ctx.from_main(valid_loss, device)
        lr = opt.step(train_loss, valid_loss=valid_loss)
        ctx.broadcast(params)
        if lr is None:
            skips += 1
            hist["skipped"] += 1
            if skips > max_consecutive_skips:
                raise FloatingPointError(f"{skips} consecutive non-finite updates; stopping")
            continue
        skips = 0
        hist["lr"].append(lr)
        hist["train_loss"].append(train_loss)
        hist["valid_feedback"].append(valid_loss)
        n = opt.num_updates
        if fixed and (n % eval_every == 0 or n == budget):
            readout(n)
    hist["seconds"] = time.time() - t0
    for p in params:
        p.grad = None
    return hist


def accumulate_gradients(model, params, stream, accum, device, ctx=distributed.SINGLE):
    """Backward over ``accum`` micro-batches from ``stream`` into fresh ``p.grad``.

    -> the mean training loss. With a distributed ``ctx`` the loss and the gradients
    are averaged over ranks, which equals one process accumulating all ranks'
    micro-batches.
    """
    for p in params:
        p.grad = None
    total = 0.0
    for _ in range(accum):
        x, y = stream.next()
        loss = data_lib.batch_loss(model, x, y, device)
        (loss / accum).backward()
        total += float(loss.detach())
    train_loss = ctx.mean(total / accum, device)
    ctx.average_grads(params)
    return train_loss


# ------------------------------------------------------------------ reload


def load(path, *, device=None, dtype=None, attn_implementation=None, integration=None):
    """Reload a saved adapter onto its base model in a fresh process.

    Adapters made with a custom integration need that ``integration`` again.
    """
    from transformers import AutoConfig, AutoTokenizer

    meta = read_adapter_config(path)
    dev = choose_device(device)
    config = AutoConfig.from_pretrained(meta["base_model"], revision=meta.get("revision"))
    if integration is None and meta["integration"] not in ("qwen3", "qwen3_8"):
        raise ValueError(f"adapter was made with integration {meta['integration']!r}; "
                         "pass integration= to mupace.load")
    integ = resolve(config, integration or meta["integration"])
    dt = choose_dtype(dev, dtype)
    model = integ.load_model(meta["base_model"], revision=meta.get("revision"), dtype=dt,
                             device=dev, attn_implementation=choose_attention(dev, attn_implementation))
    named = integ.attach_lora(model, int(meta["rank"]), float(meta["alpha"]), seed=0)
    load_adapters(named, path)
    prepare_for_inference(model)
    tok = AutoTokenizer.from_pretrained(path)
    return AdaptedModel(model, tok, named, meta, integ)
