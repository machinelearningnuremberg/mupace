"""Datasets -> (prompt, completion) pairs -> masked token rows -> batches.

Accepted ``dataset`` values:

* a registered name: ``no_robots``, ``gsm8k``, ``mbpp``, ``samsum``, ``cnn_dailymail``,
  ``xsum``, ``boolq``, ``wmt14_en_de``;
* a ``Lots-of-LoRAs/<task>`` id;
* any other Hugging Face dataset id, or a local ``.json``/``.jsonl``/``.csv``/
  ``.parquet`` file or dataset directory, with prompt/completion columns;
* in-memory data: a list of ``{"prompt": ..., "completion": ...}`` dicts or
  ``(prompt, completion)`` tuples, or a dict mapping split names to such lists.

Column names are detected for the unambiguous pairs in ``COLUMN_PAIRS``; anything
else needs ``column_map={"prompt": ..., "completion": ...}`` or
``format_fn(example) -> (prompt, completion)``.

Scheduler feedback always comes from a validation split. If the source has none, a
deterministic ~10% holdout is carved from the training data (by content hash).
Test data is only ever used for reporting.
"""

import hashlib
import os

import numpy as np
import torch

IGNORE_INDEX = -100
HOLDOUT_MOD = 10

COLUMN_PAIRS = (
    ("prompt", "completion"),
    ("input", "output"),
    ("instruction", "response"),
    ("instruction", "output"),
    ("question", "answer"),
    ("source", "target"),
)
VALID_SPLIT_NAMES = ("validation", "valid", "val", "dev")
TEST_SPLIT_NAMES = ("test",)

_CARVE = "__carve__"


def _mbpp_render(ex):
    tests = "\n".join(ex["test_list"])
    return f"{ex['text']}\nYour code should pass these tests:\n{tests}", ex["code"]


def _no_robots_render(ex):
    msgs = [m for m in ex["messages"] if m["role"] in ("user", "assistant")]
    if len(msgs) < 2 or msgs[0]["role"] != "user":
        return None
    return msgs[0]["content"], msgs[1]["content"]


# ``carve_from`` splits are divided into valid/test halves by an md5 of ``group`` (or the row index).
REGISTERED = {
    "no_robots": dict(repos=("HuggingFaceH4/no_robots",), config=None,
                      splits={"train": "train", "valid": _CARVE, "test": _CARVE},
                      carve_from="test", render=_no_robots_render, group=None),
    "gsm8k": dict(repos=("openai/gsm8k",), config="main",
                  splits={"train": "train", "valid": _CARVE, "test": _CARVE},
                  carve_from="test", render=lambda ex: (ex["question"], ex["answer"]),
                  group=None),
    "mbpp": dict(repos=("google-research-datasets/mbpp", "mbpp"), config="full",
                 splits={"train": "train", "valid": "validation", "test": "test"},
                 carve_from=None, render=_mbpp_render, group=None),
    "samsum": dict(repos=("Samsung/samsum", "knkarthick/samsum"), config=None,
                   splits={"train": "train", "valid": "validation", "test": "test"},
                   carve_from=None,
                   render=lambda ex: ("Summarize the following dialogue.\n\n" + ex["dialogue"],
                                      ex["summary"]),
                   group=None),
    "cnn_dailymail": dict(repos=("abisee/cnn_dailymail", "cnn_dailymail"), config="3.0.0",
                          splits={"train": "train", "valid": "validation", "test": "test"},
                          carve_from=None,
                          render=lambda ex: (
                              "Summarize the following news article in a few sentences.\n\n"
                              + ex["article"], ex["highlights"]),
                          group=None),
    "xsum": dict(repos=("EdinburghNLP/xsum", "knkarthick/xsum"), config=None,
                 splits={"train": "train", "valid": "validation", "test": "test"},
                 carve_from=None,
                 render=lambda ex: ("Summarize the following article in one sentence.\n\n"
                                    + (ex["document"] if "document" in ex else ex["dialogue"]),
                                    ex["summary"]),
                 group=None),
    "boolq": dict(repos=("google/boolq", "boolq"), config=None,
                  splits={"train": "train", "valid": _CARVE, "test": _CARVE},
                  carve_from="validation",
                  render=lambda ex: (f"Passage: {ex['passage']}\n\nQuestion: {ex['question']}\n"
                                     "Answer yes or no.", "yes" if ex["answer"] else "no"),
                  group=lambda ex: ex["passage"]),
    "wmt14_en_de": dict(repos=("wmt/wmt14", "wmt14"), config="de-en",
                        splits={"train": "train", "valid": "validation", "test": "test"},
                        carve_from=None, stream_train=True,
                        render=lambda ex: ("Translate the following English text to German.\n\n"
                                           f"English: {ex['translation']['en']}\nGerman:",
                                           ex["translation"]["de"]),
                        group=None),
}


def _md5_int(text):
    return int(hashlib.md5(str(text).encode()).hexdigest(), 16)


# ------------------------------------------------------------------ registered


def _load_hf_split(repos, config, split, streaming=False, revision=None):
    from datasets import load_dataset

    errors = []
    for repo in repos:
        try:
            return load_dataset(repo, name=config, split=split, streaming=streaming,
                                revision=revision)
        except Exception as e:  # try the next mirror
            errors.append(f"{repo}: {type(e).__name__}: {e}")
    raise RuntimeError("could not load dataset: " + "; ".join(errors))


def _render_rows(ds, render, group, cap):
    rows = []
    for i, ex in enumerate(ds):
        pair = render(ex)
        if pair is None:
            continue
        rows.append((pair[0], pair[1], group(ex) if group else i))
        if cap and len(rows) >= cap:
            break
    return rows


def _registered_splits(name, max_examples, revision):
    spec = REGISTERED[name]
    out = {}
    for split, hf_split in spec["splits"].items():
        if hf_split == _CARVE:
            ds = _load_hf_split(spec["repos"], spec["config"], spec["carve_from"], revision=revision)
            rows = _render_rows(ds, spec["render"], spec["group"], None)
            keep_valid = split == "valid"
            pairs = [(u, t) for u, t, key in rows if ((_md5_int(key) % 2) == 0) == keep_valid]
            out[split] = pairs[: max_examples or None]
        else:
            stream = bool(spec.get("stream_train")) and split == "train"
            ds = _load_hf_split(spec["repos"], spec["config"], hf_split, streaming=stream,
                                revision=revision)
            out[split] = [(u, t) for u, t, _ in _render_rows(ds, spec["render"], None, max_examples)]
    return out, {"valid_source": f"{name} preset"}


# ------------------------------------------------------------------ generic


def _pick_columns(columns, column_map):
    if column_map:
        missing = [c for c in (column_map.get("prompt"), column_map.get("completion"))
                   if c not in columns]
        if missing:
            raise ValueError(f"column_map names {missing}, columns are {sorted(columns)}")
        return column_map["prompt"], column_map["completion"]
    found = [pair for pair in COLUMN_PAIRS if pair[0] in columns and pair[1] in columns]
    if len(found) == 1:
        return found[0]
    why = "several candidate column pairs" if found else "no known prompt/completion columns"
    raise ValueError(
        f"cannot tell which columns are the prompt and the completion ({why}; columns: "
        f"{sorted(columns)}). Pass column_map={{'prompt': ..., 'completion': ...}} "
        "or format_fn=lambda example: (prompt, completion)."
    )


def _as_text(v):
    if isinstance(v, (list, tuple)):
        if not v:
            return ""
        v = v[0]
    return v if isinstance(v, str) else str(v)


def _pairs_from_rows(rows, column_map, format_fn, cap):
    out, cols = [], None
    for ex in rows:
        if format_fn is not None:
            pair = format_fn(ex)
        elif isinstance(ex, (tuple, list)) and len(ex) == 2:
            pair = ex
        else:
            if cols is None:
                cols = _pick_columns(set(ex), column_map)
            pair = (ex[cols[0]], ex[cols[1]])
        if pair is None:
            continue
        p, c = _as_text(pair[0]), _as_text(pair[1])
        if p.strip() and c.strip():
            out.append((p, c))
        if cap and len(out) >= cap:
            break
    return out


def _local_dataset(path):
    from datasets import load_dataset

    if os.path.isdir(path):
        return load_dataset(path)
    ext = os.path.splitext(path)[1].lower()
    builder = {".json": "json", ".jsonl": "json", ".csv": "csv", ".parquet": "parquet"}.get(ext)
    if builder is None:
        raise ValueError(f"unsupported data file {path!r}; use .json, .jsonl, .csv or .parquet")
    return load_dataset(builder, data_files=path)


def _find(names, available):
    return next((n for n in names if n in available), None)


def _carve_holdout(pairs):
    train, valid = [], []
    for p, c in pairs:
        (valid if _md5_int(p + "\x00" + c) % HOLDOUT_MOD == 0 else train).append((p, c))
    if not valid and len(train) > 1:
        valid.append(train.pop())
    return train, valid


def load_splits(dataset, *, column_map=None, format_fn=None, max_examples=0, revision=None):
    """-> ({"train": pairs, "valid": pairs[, "test": pairs]}, info)."""
    cap = max_examples or None
    if isinstance(dataset, str) and dataset in REGISTERED and not column_map and not format_fn:
        splits, info = _registered_splits(dataset, max_examples, revision)
    else:
        if isinstance(dataset, str):
            if os.path.exists(dataset):
                source = _local_dataset(dataset)
            else:
                from datasets import load_dataset

                config = "default" if dataset.startswith("Lots-of-LoRAs/") else None
                source = load_dataset(dataset, name=config, revision=revision)
            source = {k: source[k] for k in source}
        elif isinstance(dataset, dict):
            source = dict(dataset)
        else:
            source = {"train": list(dataset)}
        if "train" not in source:
            raise ValueError(f"dataset has no 'train' split (splits: {sorted(source)})")
        splits = {"train": _pairs_from_rows(source["train"], column_map, format_fn, None)}
        valid_name = _find(VALID_SPLIT_NAMES, source)
        test_name = _find(TEST_SPLIT_NAMES, source)
        if valid_name:
            splits["valid"] = _pairs_from_rows(source[valid_name], column_map, format_fn, cap)
            info = {"valid_source": f"split '{valid_name}'"}
        else:
            splits["train"], splits["valid"] = _carve_holdout(splits["train"])
            info = {"valid_source": f"deterministic 1/{HOLDOUT_MOD} holdout of train"}
        splits["train"] = splits["train"][:cap]
        splits["valid"] = splits["valid"][:cap]
        if test_name:
            splits["test"] = _pairs_from_rows(source[test_name], column_map, format_fn, cap)
    for name in ("train", "valid"):
        if not splits.get(name):
            raise ValueError(f"dataset produced no usable {name} examples")
    info["rows"] = {k: len(v) for k, v in splits.items()}
    return splits, info


# ------------------------------------------------------------------ tokenization


def has_chat_template(tokenizer):
    return bool(getattr(tokenizer, "chat_template", None))


def render_prompt(tokenizer, prompt):
    """The text a prompt becomes before the completion (chat template if any)."""
    if not has_chat_template(tokenizer):
        return prompt + "\n"
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _row_parts(tokenizer, prompt, completion):
    text = render_prompt(tokenizer, prompt)
    if prompt not in text and prompt.strip() in text:
        prompt = prompt.strip()
    head, sep, tail = text.partition(prompt)
    if not sep:
        raise ValueError("prompt not found verbatim in the rendered chat template")
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False) if s else []
    eos = tokenizer.eos_token_id
    return enc(head), enc(prompt), enc(tail), enc(completion) + ([] if eos is None else [eos])


def build_row(tokenizer, prompt, completion, seq_len):
    """-> (input_ids, labels), both length ``seq_len``; only completion tokens are
    supervised. Long prompts are cut from the left, then long completions."""
    head, user, tail, target = _row_parts(tokenizer, prompt, completion)
    markers = len(head) + len(tail)
    room = seq_len - (len(target) + markers)
    if room < 0:
        target = target[: max(1, seq_len - markers)]
        user = []
    elif room < len(user):
        user = user[len(user) - room:]
    prompt_ids = head + user + tail
    full = prompt_ids + target
    n_prompt, n = len(prompt_ids), len(full)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded = full + [pad] * (seq_len + 1 - n)
    input_ids = np.asarray(padded[:seq_len], np.int64)
    labels = np.asarray(padded[1: seq_len + 1], np.int64)
    labels[: n_prompt - 1] = IGNORE_INDEX
    labels[n - 1:] = IGNORE_INDEX
    return input_ids, labels


def token_lengths(tokenizer, pairs, max_rows=1000, seed=0):
    """Full token length (template + prompt + completion) of up to ``max_rows``
    pairs; a row longer than ``seq_len`` gets cut by :func:`build_row`."""
    if len(pairs) > max_rows:
        idx = np.random.default_rng(int(seed)).choice(len(pairs), max_rows, replace=False)
        pairs = [pairs[i] for i in idx]
    return np.asarray([sum(map(len, _row_parts(tokenizer, p, c))) for p, c in pairs], np.int64)


def tokenize_pairs(tokenizer, pairs, seq_len):
    rows = [build_row(tokenizer, p, c, seq_len) for p, c in pairs]
    x = np.stack([r[0] for r in rows])
    y = np.stack([r[1] for r in rows])
    keep = (y != IGNORE_INDEX).any(axis=1)
    return x[keep], y[keep]


class BatchStream:
    """Endless shuffled micro-batches over tokenized rows (reshuffled each epoch).

    With ``world_size > 1``, :meth:`next` only draws every ``world_size``-th row
    starting at ``rank``; :meth:`fixed` always covers all rows.
    """

    def __init__(self, x, y, batch_size, seed, rank=0, world_size=1):
        if len(x) == 0:
            raise ValueError("no rows to batch")
        if len(x) < world_size:
            raise ValueError(f"{len(x)} rows cannot be split over {world_size} processes")
        self.x, self.y = x, y
        self.batch_size = int(batch_size)
        self._rows = np.arange(len(x))[rank::world_size]
        self.rng = np.random.default_rng(int(seed))
        self._perm = self.rng.permutation(len(self._rows))
        self._i = 0

    def next(self):
        idx = []
        while len(idx) < self.batch_size:
            if self._i >= len(self._perm):
                self._perm = self.rng.permutation(len(self._rows))
                self._i = 0
            idx.append(self._rows[self._perm[self._i]])
            self._i += 1
        idx = np.asarray(idx)
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])

    def fixed(self, n_examples, seed):
        """A fixed list of micro-batches covering up to ``n_examples`` rows."""
        idx = np.random.default_rng(int(seed)).permutation(len(self.x))[: int(n_examples)]
        b = self.batch_size
        return [(torch.from_numpy(self.x[idx[i: i + b]]), torch.from_numpy(self.y[idx[i: i + b]]))
                for i in range(0, len(idx), b)]


def prepare(dataset, tokenizer, *, seq_len, batch_size, seed=0, column_map=None,
            format_fn=None, max_examples=0, revision=None):
    """Load, tokenize and batch a dataset. -> (streams, info).

    ``streams`` has ``train`` (training micro-batches), ``feedback`` (validation
    micro-batches for the scheduler), ``valid`` and optionally ``test`` (for
    periodic evaluation). Seeds follow the research runner.
    """
    splits, info = load_splits(dataset, column_map=column_map, format_fn=format_fn,
                               max_examples=max_examples, revision=revision)
    return build_streams(splits, info, tokenizer, seq_len=seq_len, batch_size=batch_size,
                         seed=seed)


def build_streams(splits, info, tokenizer, *, seq_len, batch_size, seed=0, rank=0, world_size=1):
    """Tokenize and batch splits from :func:`load_splits`. -> (streams, info).

    With ``world_size > 1`` each rank's train stream draws from its own shard of
    rows; the other streams are identical on every rank.
    """
    info = dict(info)
    xy = {k: tokenize_pairs(tokenizer, v, seq_len) for k, v in splits.items()}
    streams = {
        "train": BatchStream(*xy["train"], batch_size, seed, rank=rank, world_size=world_size),
        "feedback": BatchStream(*xy["valid"], batch_size, seed + 2),
        "valid": BatchStream(*xy["valid"], batch_size, seed + 1),
    }
    if "test" in xy and len(xy["test"][0]):
        streams["test"] = BatchStream(*xy["test"], batch_size, seed + 3)
    info["tokenized_rows"] = {k: int(len(v[0])) for k, v in xy.items()}
    info["mean_supervised_tokens"] = float((xy["train"][1] != IGNORE_INDEX).sum(1).mean())
    return streams, info


def batch_loss(model, x, y, device):
    import torch.nn.functional as F

    x, y = x.to(device), y.to(device)
    logits = model(input_ids=x).logits
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1),
                           ignore_index=IGNORE_INDEX)


@torch.no_grad()
def eval_loss(model, batches, device, ctx=None):
    """Mean batch loss. With a distributed ``ctx`` each rank evaluates its share of
    ``batches`` and the result is the mean over all of them."""
    was_training = model.training
    model.eval()
    mine = batches if ctx is None else batches[ctx.rank::ctx.world_size]
    total = sum(float(batch_loss(model, x, y, device)) for x, y in mine)
    if was_training:
        model.train()
    if ctx is not None and ctx.enabled:
        total = ctx.sum([total], device)[0]
    return total / max(len(batches), 1)
