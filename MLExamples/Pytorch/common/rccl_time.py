#!/usr/bin/env python3
"""Uniform RCCL communication-time measurement via torch.profiler.

DDP exposes its all-reduce cost with ``no_sync()`` (t_sync - t_nosync), but FSDP2
communicates with all-gather + reduce-scatter that cannot be switched off the same
way. To compare communication across DDP and FSDP2 on the *same* footing, this
helper runs a few steps under ``torch.profiler`` and sums the **device time of the
RCCL collective kernels** (``ncclDevKernel_AllReduce_*`` / ``_AllGather_*`` /
``_ReduceScatter_*`` on ROCm) per step.

This is the on-GPU RCCL kernel time. It is a lower bound on the wall-clock comm
cost (it excludes launch/wait/overlap), but it is measured identically for every
example, which is what makes the cross-example table meaningful.
"""
import torch
from torch.profiler import profile, ProfilerActivity


def rccl_time_per_step(step_fn, iters, warmup=3):
    """Return (avg RCCL kernel device seconds/step, matched-kernel names).

    step_fn: a callable that performs exactly one full train step.
    """
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            step_fn()
        torch.cuda.synchronize()

    total_us = 0.0
    names = set()
    for e in prof.key_averages():
        key = getattr(e, "key", "") or ""
        low = key.lower()
        if "nccl" in low or "rccl" in low:
            t = getattr(e, "self_device_time_total", None)
            if t is None:
                t = getattr(e, "self_cuda_time_total", 0.0)
            if t:
                total_us += float(t)
                names.add(key)
    return (total_us / 1e6) / max(iters, 1), sorted(names)
