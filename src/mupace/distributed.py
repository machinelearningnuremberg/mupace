"""Data parallelism across GPUs, switched on by launching with ``torchrun``.

Every process holds the full frozen model and its own copy of the adapters. Each
rank trains on its own shard of the training rows; after the accumulated backward
the adapter gradients and the training loss are averaged over ranks, so every rank
feeds the scheduler the same numbers and makes the same update. Adapters are
re-broadcast from rank 0 after each update to rule out drift. The base model is
never wrapped or synchronized: it is frozen.

    torchrun --nproc_per_node=4 train.py

Without ``torchrun`` (``WORLD_SIZE`` unset or 1) everything here is a no-op.
"""

import atexit
import contextlib
import datetime
import os
import socket

import torch


class Context:
    """Rank, world size and device of this process."""

    def __init__(self, rank=0, world_size=1, local_rank=0):
        self.rank, self.world_size, self.local_rank = rank, world_size, local_rank

    @property
    def enabled(self):
        return self.world_size > 1

    @property
    def is_main(self):
        return self.rank == 0

    def barrier(self):
        if self.enabled:
            torch.distributed.barrier()

    @contextlib.contextmanager
    def main_first(self):
        """Rank 0 runs the block first (downloads, caches), then the others."""
        if not self.is_main:
            self.barrier()
        yield
        if self.is_main:
            self.barrier()

    def _flat(self, tensors):
        return torch.cat([t.detach().reshape(-1).float() for t in tensors])

    def _unflat(self, flat, tensors):
        i = 0
        for t in tensors:
            n = t.numel()
            t.copy_(flat[i: i + n].view_as(t))
            i += n

    @torch.no_grad()
    def average_grads(self, params):
        """In place: every ``p.grad`` becomes the mean over ranks."""
        if not self.enabled:
            return
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        grads = [p.grad for p in params]
        flat = self._flat(grads)
        torch.distributed.all_reduce(flat)
        flat /= self.world_size
        self._unflat(flat, grads)

    @torch.no_grad()
    def broadcast(self, tensors, src=0):
        if not self.enabled:
            return
        flat = self._flat(tensors)
        torch.distributed.broadcast(flat, src)
        self._unflat(flat, tensors)

    def mean(self, value, device):
        if not self.enabled:
            return value
        t = torch.tensor([float(value)], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(t)
        return float(t) / self.world_size

    def sum(self, values, device):
        if not self.enabled:
            return list(values)
        t = torch.tensor([float(v) for v in values], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(t)
        return t.tolist()

    def from_main(self, value, device):
        """Rank 0's ``value`` on every rank."""
        if not self.enabled:
            return value
        t = torch.tensor([float(value)], dtype=torch.float64, device=device)
        torch.distributed.broadcast(t, 0)
        return float(t)


SINGLE = Context()


def launched():
    """True when started by ``torchrun`` with more than one process."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def rank_device(device, local_rank):
    """The device a worker with ``local_rank`` trains on.

    A worker that sees exactly one GPU (visibility restricted per worker, e.g. by
    ``CUDA_VISIBLE_DEVICES``) uses ``cuda:0``; one that sees several uses
    ``cuda:LOCAL_RANK``. ``None`` and ``"cuda"`` pick that GPU (CPU when there is
    none); an explicit ``cuda:i`` must name it. Whether the workers really ended up
    on different physical GPUs is checked after joining, by :func:`check_distinct_gpus`.
    """
    if device is None:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        device = "cuda"
    dev = torch.device(device)
    if dev.type != "cuda":
        return dev
    count = torch.cuda.device_count()
    own = 0 if count == 1 else local_rank
    if own >= count:
        raise ValueError(f"worker with LOCAL_RANK={local_rank} sees {count} GPUs; start at most "
                         f"{max(count, 1)} workers per node, or give each worker one GPU")
    if dev.index is not None and dev.index != own:
        raise ValueError(
            f"device={str(device)!r} under torchrun: the worker with LOCAL_RANK={local_rank} "
            f"sees {count} GPUs and must use cuda:{own}. Pass device=None or 'cuda'."
        )
    return torch.device("cuda", own)


def gpu_identity(dev):
    """(host, physical id or None) of this worker's GPU.

    The id is the GPU UUID, else its PCI address; both name the physical card
    whatever ``CUDA_VISIBLE_DEVICES`` ordering the worker has. A CUDA index does not.
    """
    props = torch.cuda.get_device_properties(dev)
    uuid = getattr(props, "uuid", None)
    if uuid is not None:
        return socket.gethostname(), f"GPU {uuid}"
    pci = [getattr(props, k, None) for k in ("pci_domain_id", "pci_bus_id", "pci_device_id")]
    if None not in pci:
        return socket.gethostname(), "GPU at PCI {:04x}:{:02x}:{:02x}".format(*pci)
    return socket.gethostname(), None


def check_distinct_gpus(identities):
    """Raise when two ranks share a physical GPU, or when that cannot be established.

    ``identities[r]`` is :func:`gpu_identity` of global rank ``r``.
    """
    seen = {}
    for rank, (host, gpu) in enumerate(identities):
        if gpu is None:
            raise ValueError(f"cannot tell which physical GPU rank {rank} on {host} uses: "
                             "PyTorch reports neither a GPU UUID nor a PCI address. "
                             "Upgrade PyTorch to train on several GPUs.")
        if (host, gpu) in seen:
            raise ValueError(f"ranks {seen[host, gpu]} and {rank} both train on {gpu} on "
                             f"{host}. Give each worker its own GPU: with per-worker "
                             "CUDA_VISIBLE_DEVICES, set a different GPU for each; otherwise "
                             "leave it unset.")
        seen[host, gpu] = rank


def _exchange_gpu_identities(dev):
    """All ranks' :func:`gpu_identity`, gathered over a CPU (gloo) group.

    The NCCL group cannot be used for this: its collectives already run on the GPUs,
    so two ranks sharing one could fail or hang before the check sees them.
    """
    dist = torch.distributed
    group = dist.new_group(backend="gloo")
    try:
        identities = [None] * dist.get_world_size()
        dist.all_gather_object(identities, gpu_identity(dev), group=group)
    finally:
        dist.destroy_process_group(group)
    return identities


def setup(device=None):
    """Join the process group if launched with ``torchrun``. -> (Context, device or None).

    The returned device comes from :func:`rank_device`; ``None`` means "not
    distributed, choose as usual".
    """
    if not launched():
        return SINGLE, None
    dist = torch.distributed
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dev = rank_device(device, local_rank)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    if not dist.is_initialized():
        dist.init_process_group("nccl" if dev.type == "cuda" else "gloo",
                                timeout=datetime.timedelta(minutes=60))
        atexit.register(_shutdown)
    if dev.type == "cuda":
        # Before any NCCL collective. Every rank sees all assignments and raises the
        # same error, so none is left hanging.
        check_distinct_gpus(_exchange_gpu_identities(dev))
    return Context(dist.get_rank(), dist.get_world_size(), local_rank), dev


def check_model_device(model, expected):
    """Raise unless every parameter of a preloaded ``model`` is on ``expected``."""
    found = {p.device for p in model.parameters()}
    if found != {expected}:
        names = ", ".join(sorted(str(d) for d in found))
        raise ValueError(
            f"under torchrun this worker trains on {expected}, but the model passed to adapt() "
            f"is on {names}. Move it first with model.to('{expected}'), or pass the model "
            "name instead and let mupace load it on the right GPU."
        )


def _shutdown():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def current():
    """The Context of an already joined process group (single-process otherwise)."""
    dist = torch.distributed
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return Context(dist.get_rank(), dist.get_world_size(),
                       int(os.environ.get("LOCAL_RANK", "0")))
    return SINGLE


def is_main_process():
    """True on global rank 0, and always outside ``torchrun``. Guard your own prints and
    file writes with it, but not ``save_pretrained``, which every rank must call."""
    return current().is_main
