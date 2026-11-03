# Copyright 2026 Jan Kobiolka, Technische Universität Nürnberg
# Licensed under the Apache License, Version 2.0 (see LICENSE).
#
# Modified port of code adapted from celo (https://github.com/amoudgl/celo,
# MIT License, Copyright (c) 2025 Abhinav Moudgil), which in turn adapts Google's
# learned_optimization (https://github.com/google/learned_optimization,
# Apache License 2.0, Copyright 2021 Google LLC). Changed by Jan Kobiolka.
# See NOTICE.

"""MuPACE for JAX: the same scheduler and Muon update as ``mupace.torch``.

Pure JAX: no Haiku, Flax or learned_optimization. Usage::

    opt = mupace.jax.MuPACE(budget=200, rank=8, alpha=16)
    state = opt.init(lora_params)              # e.g. {"L0.q_proj": {"A": ..., "B": ...}}
    for _ in range(200):
        loss, grads = jax.value_and_grad(loss_fn)(lora_params, batch)
        lora_params, state, lr = opt.update(state, grads, lora_params, loss, valid_loss)
"""

import re
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from mupace import policy as pol

_DIGIT_RUN = re.compile(r"^(.*?\d+)")


def _path_str(path):
    return "/".join(str(getattr(e, "key", getattr(e, "idx", e))) for e in path)


def infer_width_depth(params):
    """Same rule as :func:`mupace.torch.infer_width_depth`, over pytree paths."""
    width, ids, n_mat = 1, set(), 0
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        if leaf.ndim >= 2:
            width = max(width, max(leaf.shape))
            n_mat += 1
        for comp in _path_str(path).split("/"):
            m = _DIGIT_RUN.match(comp)
            if m:
                ids.add(m.group(1))
                break
    return int(width), int(len(ids) if ids else max(n_mat, 1))


def newton_schulz(m, coeffs, iters, eps):
    shape = m.shape
    x = m.reshape(shape[0], -1).astype(jnp.float32)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.T
    x = x / (jnp.linalg.norm(x) + eps)
    a0, a1, a2 = coeffs

    def body(_, x):
        a = x @ x.T
        return a0 * x + (a1 * a + a2 * (a @ a)) @ x

    x = jax.lax.fori_loop(0, iters, body, x)
    if transposed:
        x = x.T
    return x.reshape(shape)


def _rms(x):
    return jnp.sqrt(jnp.mean(jnp.square(x)) + 1e-12)


def _all_rms(leaves):
    total = sum(jnp.sum(jnp.square(x.astype(jnp.float32))) for x in leaves)
    return jnp.sqrt(total / max(sum(int(x.size) for x in leaves), 1) + 1e-12)


class MuPACEState(NamedTuple):
    m: Any
    adam_m: Any
    v: Any
    sched: dict
    ctx: dict
    num_updates: jnp.ndarray
    num_skipped: jnp.ndarray


class MuPACE:
    """Functional MuPACE. ``update`` is jitted; the instance is a static argument."""

    def __init__(self, *, budget, rank, alpha, width=None, depth=None, policy=None):
        self.policy = policy or pol.Policy.pretrained()
        self.budget, self.rank, self.alpha = int(budget), int(rank), float(alpha)
        self.width, self.depth = width, depth
        self._theta = self.policy.theta_as(jnp)
        self.update = jax.jit(self._update)

    def init(self, params):
        width, depth = infer_width_depth(params)
        width = width if self.width is None else int(self.width)
        depth = depth if self.depth is None else int(self.depth)
        leaves = jax.tree_util.tree_leaves(params)
        ctx = pol.make_context(
            jnp, self.policy, self._theta, self.budget, width, depth,
            self.rank, self.alpha, _all_rms(leaves),
        )
        ctx = dict(ctx, depth_scale=jnp.float32(ctx["depth_scale"]),
                   num_steps=jnp.float32(ctx["num_steps"]))
        zeros = lambda: jax.tree_util.tree_map(lambda p: jnp.zeros(p.shape, jnp.float32), params)
        return MuPACEState(
            m=zeros(), adam_m=zeros(), v=zeros(),
            sched=pol.initial_state(jnp, self.policy.config), ctx=ctx,
            num_updates=jnp.asarray(0, jnp.int32), num_skipped=jnp.asarray(0, jnp.int32),
        )

    def _update(self, state, grads, params, loss, valid_loss=None):
        cfg = self.policy.config
        f32 = jnp.float32
        g_leaves, treedef = jax.tree_util.tree_flatten(
            jax.tree_util.tree_map(lambda g: g.astype(f32), grads)
        )
        p_leaves = treedef.flatten_up_to(params)
        m_leaves = treedef.flatten_up_to(state.m)
        am_leaves = treedef.flatten_up_to(state.adam_m)
        v_leaves = treedef.flatten_up_to(state.v)

        loss = jnp.asarray(loss, f32)
        finite = jnp.isfinite(loss)
        if valid_loss is not None:
            valid_loss = jnp.asarray(valid_loss, f32)
            finite &= jnp.isfinite(valid_loss)
        for g in g_leaves:
            finite &= jnp.all(jnp.isfinite(g))

        dot = sum(jnp.vdot(g, m) for g, m in zip(g_leaves, m_leaves))
        g_sq = sum(jnp.vdot(g, g) for g in g_leaves)
        m_sq = sum(jnp.vdot(m, m) for m in m_leaves)
        stats = {
            "loss": loss,
            "valid_loss": valid_loss,
            "cos_gm": dot / (jnp.sqrt(g_sq * m_sq) + 1e-12),
            "sumsq_m": m_sq,
            "sum_v": sum(jnp.sum(v) for v in v_leaves),
            "rms_p": _all_rms(p_leaves),
            "rms_g": _all_rms(g_leaves),
        }
        lr, sched, _ = pol.step(jnp, self.policy, self._theta, state.ctx, state.sched, stats)

        mu = cfg.muon_momentum
        new_p, new_m, new_am, new_v = [], [], [], []
        for p, g, m, am, v in zip(p_leaves, g_leaves, m_leaves, am_leaves, v_leaves):
            m = mu * m + (1.0 - mu) * g
            am = f32(cfg.adam_b1) * am + (f32(1.0) - f32(cfg.adam_b1)) * g
            v = f32(cfg.adam_b2) * v + (f32(1.0) - f32(cfg.adam_b2)) * g * g
            if p.ndim >= 2:
                raw = newton_schulz(m, cfg.ns_coeffs, cfg.ns_iters, cfg.ns_eps)
                direction = raw / _rms(raw)
            else:
                direction = am * jax.lax.rsqrt(v + cfg.adam_eps)
            new_p.append((p - lr * direction).astype(p.dtype))
            new_m.append(m)
            new_am.append(am)
            new_v.append(v)

        def pick(new, old):
            return jax.tree_util.tree_map(lambda a, b: jnp.where(finite, a, b), new, old)

        unflat = lambda leaves: jax.tree_util.tree_unflatten(treedef, leaves)
        out = MuPACEState(
            m=pick(unflat(new_m), state.m),
            adam_m=pick(unflat(new_am), state.adam_m),
            v=pick(unflat(new_v), state.v),
            sched=pick(sched, state.sched),
            ctx=state.ctx,
            num_updates=state.num_updates + finite.astype(jnp.int32),
            num_skipped=state.num_skipped + (~finite).astype(jnp.int32),
        )
        return pick(unflat(new_p), params), out, jnp.where(finite, lr, jnp.nan)
