# Copyright 2026 Jan Kobiolka, Technische Universität Nürnberg
# Licensed under the Apache License, Version 2.0 (see LICENSE).
#
# Modified port of code adapted from celo (https://github.com/amoudgl/celo,
# MIT License, Copyright (c) 2025 Abhinav Moudgil), which in turn adapts Google's
# learned_optimization (https://github.com/google/learned_optimization,
# Apache License 2.0, Copyright 2021 Google LLC). Changed by Jan Kobiolka.
# See NOTICE.

"""MuPACE for PyTorch: Muon updates on LoRA adapters, paced by the frozen scheduler.

Trust ratio is off and weight decay is zero, the settings the bundled checkpoint
was meta-trained with. The scheduler runs in
NumPy on the host (one tiny LSTM step per update); no JAX is needed.
"""

import math
import re

import numpy as np
import torch

from mupace import policy as pol

_DIGIT_RUN = re.compile(r"^(.*?\d+)")


def infer_width_depth(named_params):
    """(width, depth) as the scheduler was meta-trained to read them.

    width = largest matrix dimension among the adapters; depth = number of distinct
    layer ids, taken as the first ``<prefix><digits>`` component of each name split
    on ``/`` (``"L3.q_proj/A"`` -> ``"L3"``).
    """
    width, ids, n_mat = 1, set(), 0
    for name, p in named_params:
        if p.ndim >= 2:
            width = max(width, max(p.shape))
            n_mat += 1
        for comp in name.split("/"):
            m = _DIGIT_RUN.match(comp)
            if m:
                ids.add(m.group(1))
                break
    depth = len(ids) if ids else max(n_mat, 1)
    return int(width), int(depth)


def newton_schulz(m, coeffs, iters, eps):
    shape = m.shape
    x = m.reshape(shape[0], -1).float()
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)
    x = x / (torch.linalg.matrix_norm(x) + eps)
    a0, a1, a2 = coeffs
    for _ in range(iters):
        a = x @ x.transpose(-2, -1)
        x = a0 * x + (a1 * a + a2 * (a @ a)) @ x
    if transposed:
        x = x.transpose(-2, -1)
    return x.reshape(shape)


def _rms(x):
    return torch.sqrt(torch.mean(x.float() ** 2) + 1e-12)


def _all_rms(xs):
    total = sum(torch.sum(x.float() ** 2) for x in xs)
    return torch.sqrt(total / max(sum(x.numel() for x in xs), 1) + 1e-12)


class MuPACE(torch.optim.Optimizer):
    """Muon (Newton-Schulz orthogonalized momentum) with a learned learning-rate pace.

    Args:
        params: ``[(name, tensor), ...]`` adapter parameters (preferred: names let
            width/depth be inferred) or a list of tensors with explicit width/depth.
            Matrices get Muon updates; 1-D tensors get Adam directions.
        budget: number of optimizer updates the run will make (the schedule horizon).
        rank, alpha: the LoRA rank and alpha actually used; the scheduler reads both.
        width, depth: model width and adapted layer count. Inferred from names if None.
        policy: a :class:`mupace.policy.Policy`; defaults to the bundled checkpoint.

    Call ``step(loss, valid_loss=None)`` after ``backward()``. ``loss`` is the
    training loss of this update (mean over accumulated micro-batches);
    ``valid_loss`` is an optional held-out loss on the current parameters. A step
    with a non-finite loss or gradient is skipped and returns ``None``; otherwise it
    returns the learning rate it applied.
    """

    def __init__(self, params, *, budget, rank, alpha, width=None, depth=None, policy=None):
        params = list(params)
        if params and isinstance(params[0], tuple):
            names = [n for n, _ in params]
            tensors = [p for _, p in params]
        else:
            names, tensors = None, params
        if not tensors:
            raise ValueError("MuPACE got no parameters")
        if width is None or depth is None:
            if names is None:
                raise ValueError("pass named parameters or explicit width= and depth=")
            w, d = infer_width_depth(list(zip(names, tensors)))
            width = w if width is None else width
            depth = d if depth is None else depth
        super().__init__(tensors, defaults={})
        self.policy = policy or pol.Policy.pretrained()
        self.names = names
        self.budget, self.rank, self.alpha = int(budget), int(rank), float(alpha)
        self.width, self.depth = int(width), int(depth)
        self._theta = self.policy.theta_as(np)
        for p in tensors:
            self.state[p]["m"] = torch.zeros_like(p, dtype=torch.float32)
            self.state[p]["adam_m"] = torch.zeros_like(p, dtype=torch.float32)
            self.state[p]["v"] = torch.zeros_like(p, dtype=torch.float32)
        with torch.no_grad():
            rms_init = float(_all_rms([p.detach() for p in tensors]))
        self._setup(rms_init)
        self.sched_state = pol.initial_state(np, self.policy.config)
        self.num_updates = 0
        self.num_skipped = 0
        self.last_lr = None
        self.last_diagnostics = None

    def _setup(self, rms_init):
        self.rms_init = float(rms_init)
        self._ctx = pol.make_context(
            np, self.policy, self._theta, self.budget, self.width, self.depth,
            self.rank, self.alpha, self.rms_init,
        )

    def _params(self):
        return [p for g in self.param_groups for p in g["params"]]

    @torch.no_grad()
    def step(self, loss, valid_loss=None):
        cfg = self.policy.config
        params = self._params()
        loss = float(loss)
        valid_loss = None if valid_loss is None else float(valid_loss)
        grads = [torch.zeros_like(p, dtype=torch.float32) if p.grad is None else p.grad.float()
                 for p in params]
        finite = math.isfinite(loss) and (valid_loss is None or math.isfinite(valid_loss))
        if finite:
            finite = bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())
        if not finite:
            self.num_skipped += 1
            return None

        st = [self.state[p] for p in params]
        dot = sum(torch.sum(g * s["m"]) for g, s in zip(grads, st))
        g_sq = sum(torch.sum(g * g) for g in grads)
        m_sq = sum(torch.sum(s["m"] * s["m"]) for s in st)
        stats = {
            "loss": loss,
            "valid_loss": valid_loss,
            "cos_gm": float(dot / (torch.sqrt(g_sq * m_sq) + 1e-12)),
            "sumsq_m": float(m_sq),
            "sum_v": float(sum(torch.sum(s["v"]) for s in st)),
            "rms_p": float(_all_rms(params)),
            "rms_g": float(_all_rms(grads)),
        }
        stats = {k: (None if v is None else np.float32(v)) for k, v in stats.items()}
        lr, self.sched_state, diag = pol.step(
            np, self.policy, self._theta, self._ctx, self.sched_state, stats
        )
        lr = float(lr)

        mu = cfg.muon_momentum
        b1, b2 = float(np.float32(cfg.adam_b1)), float(np.float32(cfg.adam_b2))
        one_b1 = float(np.float32(1.0) - np.float32(cfg.adam_b1))
        one_b2 = float(np.float32(1.0) - np.float32(cfg.adam_b2))
        for p, g, s in zip(params, grads, st):
            s["m"].mul_(mu).add_(g, alpha=1.0 - mu)
            s["adam_m"].mul_(b1).add_(g, alpha=one_b1)
            s["v"].mul_(b2).add_(g * g, alpha=one_b2)
            if p.ndim >= 2:
                raw = newton_schulz(s["m"], cfg.ns_coeffs, cfg.ns_iters, cfg.ns_eps)
                direction = raw / _rms(raw)
            else:
                direction = s["adam_m"] * torch.rsqrt(s["v"] + cfg.adam_eps)
            p.add_(direction.to(p.dtype), alpha=-lr)

        self.num_updates += 1
        self.last_lr = lr
        self.last_diagnostics = {k: np.asarray(v).tolist() for k, v in diag.items()}
        self.last_diagnostics["lr"] = lr
        return lr

    # ----------------------------------------------------------- save / resume

    def state_dict(self):
        out = super().state_dict()
        out["mupace"] = {
            "budget": self.budget,
            "rank": self.rank,
            "alpha": self.alpha,
            "width": self.width,
            "depth": self.depth,
            "rms_init": self.rms_init,
            "num_updates": self.num_updates,
            "num_skipped": self.num_skipped,
            "scheduler": {k: np.asarray(v).tolist() for k, v in self.sched_state.items()},
        }
        return out

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        meta = state_dict.pop("mupace")
        for k in ("budget", "rank", "width", "depth"):
            if int(meta[k]) != getattr(self, k):
                raise ValueError(f"checkpoint {k}={meta[k]} but this optimizer has {getattr(self, k)}")
        if float(meta["alpha"]) != self.alpha:
            raise ValueError(f"checkpoint alpha={meta['alpha']} but this optimizer has {self.alpha}")
        super().load_state_dict(state_dict)
        self._setup(meta["rms_init"])
        self.num_updates = int(meta["num_updates"])
        self.num_skipped = int(meta["num_skipped"])
        sched = meta["scheduler"]
        self.sched_state = {
            k: np.asarray(sched[k], np.int32 if k == "iteration" else np.float32)
            for k in pol.STATE_KEYS
        }
