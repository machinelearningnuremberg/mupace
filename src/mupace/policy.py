# Copyright 2026 Jan Kobiolka, Technische Universität Nürnberg
# Licensed under the Apache License, Version 2.0 (see LICENSE).
#
# Modified port of code adapted from celo (https://github.com/amoudgl/celo,
# MIT License, Copyright (c) 2025 Abhinav Moudgil), which in turn adapts Google's
# learned_optimization (https://github.com/google/learned_optimization,
# Apache License 2.0, Copyright 2021 Google LLC). Changed by Jan Kobiolka.
# See NOTICE.

"""The frozen MuPACE scheduler: an analytic learning-rate schedule paced by an LSTM.

Every function takes an array namespace ``xp`` (``numpy`` or ``jax.numpy``), so the
PyTorch runtime (NumPy on the host, one call per update) and the JAX runtime (jitted
``jax.numpy``) share one implementation of the scheduler math. Nothing here imports
torch or jax.

The weights and constants are bundled in ``mupace/assets/``; their source is
recorded in ``scheduler_config.json``.
"""

import dataclasses
import hashlib
import json
import math
from importlib import resources

import numpy as np

ASSET_PACKAGE = "mupace.assets"
THETA_FILE = "scheduler_theta.npz"
CONFIG_FILE = "scheduler_config.json"

THETA_SHAPES = {
    "embed/w": ("num_features", "hidden"),
    "embed/b": ("hidden",),
    "lstm/linear/w": ("2hidden", "4hidden"),
    "lstm/linear/b": ("4hidden",),
    "head/w": ("hidden", 5),
    "head/b": (5,),
}

STATE_KEYS = (
    "iteration",
    "h",
    "c",
    "loss0",
    "loss_fast",
    "loss_slow",
    "vloss0",
    "vloss_fast",
    "vloss_slow",
    "lam",
    "alp",
    "bet",
    "rho",
)


@dataclasses.dataclass(frozen=True)
class PolicyConfig:
    """Exact scheduler and optimizer constants. Loaded from the bundled JSON."""

    lstm_hidden: int
    num_features: int
    head_gain: float
    lam_warm: float
    ema: float
    lambda_0: float
    alpha_0: float
    tau_0: float
    beta_0: float
    rho_0: float
    c_lambda: float
    c_alpha: float
    c_tau: float
    c_beta: float
    c_rho: float
    normalize_shape: bool
    depth_exponent: float
    depth_ref: float
    muon_momentum: float
    adam_b1: float
    adam_b2: float
    adam_eps: float
    ns_coeffs: tuple
    ns_iters: int
    ns_eps: float
    cos_clamp: float
    x_eps: float
    shape_norm_grid: int
    peak_clamp_mult: float
    feat_eps: float
    loss_ema_fast: float
    loss_ema_slow: float
    logl0_center: float
    logl0_scale: float
    level_scale: float
    trend_scale: float
    gap_scale: float
    snr_scale: float
    snr_eps: float
    logrmsp_center: float
    logrmsp_scale: float
    logrmsg_center: float
    logrmsg_scale: float
    logrms_clip: float
    logt_center: float
    logt_scale: float
    logw_center: float
    logw_scale: float
    depth_center: float
    depth_scale: float
    alpha_center: float
    alpha_scale: float
    rank_center: float
    rank_scale: float
    logalpha_center: float
    logalpha_scale: float

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in dataclasses.fields(cls)}
        missing = sorted(names - set(d))
        if missing:
            raise ValueError(f"scheduler config is missing {missing}")
        kw = {k: d[k] for k in names}
        kw["ns_coeffs"] = tuple(float(c) for c in kw["ns_coeffs"])
        return cls(**kw)


class Policy:
    """Scheduler weights plus constants, held as float32 NumPy arrays."""

    def __init__(self, theta, config, source=None):
        self.config = config
        self.theta = {k: np.asarray(v, np.float32) for k, v in theta.items()}
        self.source = source or {}
        self._check_shapes()
        self.default_shape_rms, self.default_shape_peak = _default_shape_constants(config)

    @classmethod
    def pretrained(cls):
        """The bundled, meta-trained MuPACE scheduler."""
        files = resources.files(ASSET_PACKAGE)
        raw = json.loads(files.joinpath(CONFIG_FILE).read_text())
        blob = files.joinpath(THETA_FILE).read_bytes()
        with np.load(files.joinpath(THETA_FILE).open("rb"), allow_pickle=False) as z:
            theta = {k: z[k] for k in z.files}
        want = raw.get("source", {}).get("theta_npz_sha256")
        got = hashlib.sha256(blob).hexdigest()
        if want and want != got:
            raise RuntimeError("bundled scheduler weights do not match their recorded hash")
        return cls(theta, PolicyConfig.from_dict(raw["policy"]), source=raw.get("source"))

    def with_zero_head(self):
        """Same policy with the output head zeroed: emits the default schedule."""
        theta = dict(self.theta)
        theta["head/w"] = np.zeros_like(theta["head/w"])
        theta["head/b"] = np.zeros_like(theta["head/b"])
        return Policy(theta, self.config, source={"zero_head": True})

    def _check_shapes(self):
        c = self.config
        dims = {
            "num_features": c.num_features,
            "hidden": c.lstm_hidden,
            "2hidden": 2 * c.lstm_hidden,
            "4hidden": 4 * c.lstm_hidden,
        }
        if set(self.theta) != set(THETA_SHAPES):
            raise ValueError(f"scheduler weights have keys {sorted(self.theta)}")
        for k, spec in THETA_SHAPES.items():
            want = tuple(dims.get(d, d) for d in spec)
            if self.theta[k].shape != want:
                raise ValueError(f"{k}: shape {self.theta[k].shape}, expected {want}")

    def theta_as(self, xp):
        return {k: xp.asarray(v) for k, v in self.theta.items()}


# --------------------------------------------------------------------- schedule


def warmup_factor(xp, x, tau):
    return 1.0 - xp.exp(-x / tau)


def cos_decay(xp, cfg, x, alpha, beta):
    c = xp.clip(xp.cos(0.5 * np.pi * xp.maximum(x, cfg.x_eps) ** beta), cfg.cos_clamp, 1.0)
    return c ** alpha


def shape_factor(xp, cfg, x, alpha, tau, beta, rho):
    return warmup_factor(xp, x, tau) * ((1.0 - rho) * cos_decay(xp, cfg, x, alpha, beta) + rho)


def shape_rms(xp, cfg, alpha, tau, beta, rho):
    xs = xp.linspace(0.0, 1.0, cfg.shape_norm_grid, dtype=xp.float32)
    s = shape_factor(xp, cfg, xs, alpha, tau, beta, rho)
    return xp.sqrt(xp.mean(s * s))


def _default_shape_constants(cfg):
    f32 = np.float32
    args = (f32(cfg.alpha_0), f32(cfg.tau_0), f32(cfg.beta_0), f32(cfg.rho_0))
    xs = np.linspace(0.0, 1.0, cfg.shape_norm_grid, dtype=np.float32)
    peak = np.max(shape_factor(np, cfg, xs, args[0], args[1], args[2], args[3]))
    return f32(shape_rms(np, cfg, *args)), f32(peak)


def analytic_lr(xp, policy, x, lam, alpha, tau, beta, rho):
    """eta(x) = lam * warmup(x) * ((1 - rho) * cos(pi/2 x^beta)^alpha + rho)."""
    cfg = policy.config
    x = xp.clip(x, 0.0, 1.0)
    eta = lam * warmup_factor(xp, x, tau) * ((1.0 - rho) * cos_decay(xp, cfg, x, alpha, beta) + rho)
    if cfg.normalize_shape:
        eta = eta * (policy.default_shape_rms / shape_rms(xp, cfg, alpha, tau, beta, rho))
        eta = xp.minimum(eta, lam * (cfg.peak_clamp_mult * policy.default_shape_peak))
    return eta


def depth_scale(cfg, depth):
    if float(cfg.depth_exponent) == 0.0:
        return 1.0
    return (float(cfg.depth_ref) / max(float(depth), 1.0)) ** float(cfg.depth_exponent)


def _bound(xp, base, c, h):
    return base * xp.exp(c * xp.tanh(h))


# --------------------------------------------------------------------- features


def time_features(xp, x):
    x = xp.asarray(x, xp.float32)
    return xp.stack([x, xp.sin(np.pi * x), xp.cos(np.pi * x)])


def static_features(cfg, num_steps, width, depth, rank, alpha):
    """[log T, log width, log depth, log(alpha/rank), log rank, log alpha], normalized."""
    log_ar = float(np.log(np.float32(float(alpha) / float(rank))))
    log_rank = math.log(float(int(rank)))
    f = [
        (math.log(float(num_steps)) - cfg.logt_center) / cfg.logt_scale,
        (math.log(max(int(width), 1)) - cfg.logw_center) / cfg.logw_scale,
        (math.log(max(int(depth), 1)) - cfg.depth_center) / cfg.depth_scale,
        (log_ar - cfg.alpha_center) / cfg.alpha_scale,
        (log_rank - cfg.rank_center) / cfg.rank_scale,
        (log_ar + log_rank - cfg.logalpha_center) / cfg.logalpha_scale,
    ]
    return np.asarray(f, np.float32)


def bias_correct_snr(xp, cfg, signal_raw, total_raw, step):
    t = xp.maximum(xp.asarray(step, xp.float32), 1.0)
    c_m = 1.0 - xp.power(xp.float32(cfg.muon_momentum), t)
    c_v = 1.0 - xp.power(xp.float32(cfg.adam_b2), t)
    signal = signal_raw / xp.maximum(c_m * c_m, cfg.snr_eps)
    total = total_raw / xp.maximum(c_v, cfg.snr_eps)
    return signal, total


def dynamic_features(xp, cfg, s, stats, rms_init, has_val):
    eps = cfg.feat_eps

    def log(v):
        return xp.log(xp.maximum(v, eps))

    loss0 = xp.maximum(s["loss0"], eps)
    fast = xp.maximum(s["loss_fast"], eps)
    slow = xp.maximum(s["loss_slow"], eps)
    vloss0 = xp.maximum(s["vloss0"], eps)
    vfast = xp.maximum(s["vloss_fast"], eps)
    vslow = xp.maximum(s["vloss_slow"], eps)

    f_loss0 = xp.clip((xp.log(loss0) - cfg.logl0_center) / cfg.logl0_scale, -3.0, 3.0)
    f_level = xp.tanh(cfg.level_scale * xp.log(fast / loss0))
    f_trend = xp.tanh(cfg.trend_scale * xp.log(fast / slow))

    signal, total = stats["snr_signal"], stats["snr_total"]
    noise = xp.maximum(total - signal, 0.0)
    f_snr = xp.tanh(cfg.snr_scale * (log(signal) - log(noise + cfg.snr_eps)))
    f_snr = xp.where(total <= cfg.snr_eps, xp.float32(0.0), f_snr)

    f_growth = xp.tanh(log(stats["rms_p"]) - log(rms_init))
    clip = cfg.logrms_clip
    f_rms_p = xp.clip((log(stats["rms_p"]) - cfg.logrmsp_center) / cfg.logrmsp_scale, -clip, clip)
    f_rms_g = xp.clip((log(stats["rms_g"]) - cfg.logrmsg_center) / cfg.logrmsg_scale, -clip, clip)

    f_vlevel = has_val * xp.tanh(cfg.level_scale * xp.log(vfast / vloss0))
    f_vtrend = has_val * xp.tanh(cfg.trend_scale * xp.log(vfast / vslow))
    f_gap = has_val * xp.tanh(cfg.gap_scale * xp.log(vfast / fast))

    parts = [
        f_loss0, f_level, f_trend, stats["cos_gm"], f_snr, f_growth,
        f_rms_p, f_rms_g, f_vlevel, f_vtrend, f_gap,
    ]
    return xp.stack([xp.asarray(p, xp.float32) for p in parts])


# --------------------------------------------------------------------- network


def _sigmoid(xp, x):
    return 1.0 / (1.0 + xp.exp(-x))


def network(xp, theta, cfg, feat, h, c):
    """embed (tanh) -> LSTM (Haiku gate order i, g, f, o; forget bias +1) -> head."""
    emb = xp.tanh(feat @ theta["embed/w"] + theta["embed/b"])
    z = xp.concatenate([emb, h]) @ theta["lstm/linear/w"] + theta["lstm/linear/b"]
    n = cfg.lstm_hidden
    i, g, f, o = z[:n], z[n:2 * n], z[2 * n:3 * n], z[3 * n:]
    c = _sigmoid(xp, f + 1.0) * c + _sigmoid(xp, i) * xp.tanh(g)
    h = _sigmoid(xp, o) * xp.tanh(c)
    raw = h @ theta["head/w"] + theta["head/b"]
    warm = xp.asarray([cfg.lam_warm, 0.0, 0.0, 0.0, 0.0], xp.float32)
    return cfg.head_gain * raw + warm, h, c


def emit_tau(xp, theta, cfg, static):
    zeros = xp.zeros([cfg.num_features - static.shape[0]], xp.float32)
    feat = xp.concatenate([zeros, xp.asarray(static)])
    hz = xp.zeros([cfg.lstm_hidden], xp.float32)
    out, _, _ = network(xp, theta, cfg, feat, hz, hz)
    return _bound(xp, cfg.tau_0, cfg.c_tau, out[2])


# --------------------------------------------------------------------- stepping


def initial_state(xp, cfg):
    f = xp.float32
    z = xp.asarray(0.0, f)
    hz = xp.zeros([cfg.lstm_hidden], f)
    return {
        "iteration": xp.asarray(0, xp.int32),
        "h": hz,
        "c": hz,
        "loss0": z,
        "loss_fast": z,
        "loss_slow": z,
        "vloss0": z,
        "vloss_fast": z,
        "vloss_slow": z,
        "lam": xp.asarray(cfg.lambda_0, f),
        "alp": xp.asarray(cfg.alpha_0, f),
        "bet": xp.asarray(cfg.beta_0, f),
        "rho": xp.asarray(cfg.rho_0, f),
    }


def _ema_pair(xp, cfg, is_first, value, prev_fast, prev_slow, prev0):
    first = xp.where(is_first, value, prev0)
    pf = xp.where(is_first, value, prev_fast)
    ps = xp.where(is_first, value, prev_slow)
    fast = pf + (1.0 - cfg.loss_ema_fast) * (value - pf)
    slow = ps + (1.0 - cfg.loss_ema_slow) * (value - ps)
    return first, fast, slow


def step(xp, policy, theta, ctx, state, stats):
    """One scheduler update. -> (lr, next_state, diagnostics).

    ``ctx`` holds per-run constants: ``static`` features, ``tau``, ``depth_scale``,
    ``num_steps`` and ``rms_init``. ``stats`` holds the pre-update optimizer
    statistics (``loss``, ``valid_loss`` or None, ``cos_gm``, ``sumsq_m``,
    ``sum_v``, ``rms_p``, ``rms_g``) as float32 scalars.
    """
    cfg = policy.config
    f32 = xp.float32
    it_pre = state["iteration"]
    iteration = it_pre + 1
    x = xp.clip(xp.asarray(iteration, f32) / xp.asarray(ctx["num_steps"], f32), 0.0, 1.0)
    is_first = it_pre == 0

    loss = xp.asarray(stats["loss"], f32)
    loss0, loss_fast, loss_slow = _ema_pair(
        xp, cfg, is_first, loss, state["loss_fast"], state["loss_slow"], state["loss0"]
    )
    if stats.get("valid_loss") is None:
        has_val, vloss = f32(0.0), xp.asarray(0.0, f32)
    else:
        has_val, vloss = f32(1.0), xp.asarray(stats["valid_loss"], f32)
    vloss0, vloss_fast, vloss_slow = _ema_pair(
        xp, cfg, is_first, vloss, state["vloss_fast"], state["vloss_slow"], state["vloss0"]
    )

    signal, total = bias_correct_snr(
        xp, cfg, xp.asarray(stats["sumsq_m"], f32), xp.asarray(stats["sum_v"], f32), it_pre
    )
    nxt = {
        "loss0": loss0, "loss_fast": loss_fast, "loss_slow": loss_slow,
        "vloss0": vloss0, "vloss_fast": vloss_fast, "vloss_slow": vloss_slow,
    }
    dyn = dynamic_features(
        xp,
        cfg,
        nxt,
        {
            "snr_signal": signal,
            "snr_total": total,
            "cos_gm": xp.asarray(stats["cos_gm"], f32),
            "rms_p": xp.asarray(stats["rms_p"], f32),
            "rms_g": xp.asarray(stats["rms_g"], f32),
        },
        xp.asarray(ctx["rms_init"], f32),
        has_val,
    )
    feat = xp.concatenate([time_features(xp, x), dyn, xp.asarray(ctx["static"], f32)])
    out, h, c = network(xp, theta, cfg, feat, state["h"], state["c"])

    keep = cfg.ema
    lam = state["lam"] + (1.0 - keep) * (_bound(xp, cfg.lambda_0, cfg.c_lambda, out[0]) - state["lam"])
    alp = state["alp"] + (1.0 - keep) * (_bound(xp, cfg.alpha_0, cfg.c_alpha, out[1]) - state["alp"])
    bet = state["bet"] + (1.0 - keep) * (_bound(xp, cfg.beta_0, cfg.c_beta, out[3]) - state["bet"])
    rho = state["rho"] + (1.0 - keep) * (_bound(xp, cfg.rho_0, cfg.c_rho, out[4]) - state["rho"])
    lr = ctx["depth_scale"] * analytic_lr(xp, policy, x, lam, alp, ctx["tau"], bet, rho)

    nxt.update(iteration=iteration, h=h, c=c, lam=lam, alp=alp, bet=bet, rho=rho)
    diag = {"x": x, "lambda": lam, "alpha": alp, "beta": bet, "rho": rho, "features": feat}
    return xp.asarray(lr, f32), nxt, diag


def make_context(xp, policy, theta, num_steps, width, depth, rank, alpha, rms_init):
    """Per-run constants for :func:`step`."""
    if int(num_steps) < 1:
        raise ValueError(f"budget must be >= 1, got {num_steps}")
    if int(rank) < 1 or float(alpha) <= 0:
        raise ValueError(f"rank must be >= 1 and alpha > 0, got rank={rank} alpha={alpha}")
    cfg = policy.config
    static = static_features(cfg, num_steps, width, depth, rank, alpha)
    return {
        "static": xp.asarray(static),
        "tau": emit_tau(xp, theta, cfg, xp.asarray(static)),
        "depth_scale": float(depth_scale(cfg, depth)),
        "num_steps": int(num_steps),
        "rms_init": xp.asarray(rms_init, xp.float32),
    }


def default_schedule(policy, num_steps, depth):
    """The schedule a zero head emits: lambda_0 * shape(x) * depth factor, for x=t/T."""
    cfg = policy.config
    f32 = np.float32
    xs = (np.arange(1, int(num_steps) + 1, dtype=np.float32)) / f32(num_steps)
    eta = analytic_lr(
        np,
        policy,
        xs,
        f32(cfg.lambda_0),
        f32(cfg.alpha_0),
        _bound(np, f32(cfg.tau_0), cfg.c_tau, f32(0.0)),
        f32(cfg.beta_0),
        f32(cfg.rho_0),
    )
    return depth_scale(cfg, depth) * eta
