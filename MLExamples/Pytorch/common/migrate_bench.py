#!/usr/bin/env python3
"""Micro-benchmark: host->device staging cost AND HBM footprint on MI300A.

Compares three ways to make a freshly-produced host buffer usable on the GPU:

  * ``.to('cuda')``       -- classic hipMemcpy into a *separate* device
                             allocation (the batch then occupies HBM twice).
  * ``migrate(managed)``  -- alias a hipMallocManaged buffer (no copy, no
                             duplicate allocation).
  * ``register_migrate``  -- hipHostRegister an ordinary pageable buffer and
                             alias it (no copy, no duplicate; works on any
                             existing tensor).

Two measurements:
  1) transfer time per call (with a numeric-equality check), and
  2) HBM actually consumed (via torch.cuda.mem_get_info) -- the memory saving
     from not keeping a second, device-side copy of the batch.

    HSA_XNACK=1 python migrate_bench.py
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zerocopy  # noqa: E402
from zerocopy import unified_memory_available  # noqa: E402


def _elsize(dtype):
    return torch.tensor([], dtype=dtype).element_size()


def bench_transfer(ext, numel, dtype, iters):
    """Return per-call ms and GB/s for to / migrate / register, plus numeric_ok."""
    ref_cpu = (torch.arange(numel, dtype=dtype) % 97)
    gb = numel * _elsize(dtype) / 1e9
    out = {}

    def timeit(make):
        # Measure the pure cost of making the host buffer usable on the GPU. The
        # correctness read faults/reads the data back, so it doubles as a check.
        g = make()
        torch.cuda.synchronize()
        ok = bool(torch.equal(g.to(dtype).cpu(), ref_cpu))
        del g
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            g = make()
            del g              # release (also unregisters for the register path)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        return dt, (gb / dt if dt > 0 else 0.0), ok

    # .to copy (fixed pinned host buffer) -- pays a full hipMemcpy every call.
    try:
        host = torch.empty(numel, dtype=dtype, pin_memory=True)
    except Exception:
        host = torch.empty(numel, dtype=dtype)
    host.copy_(ref_cpu)
    out["to(copy)"] = timeit(lambda: host.to("cuda"))

    if ext is not None:
        # managed: alias only (prefetch=False) -- the steady-state cost when a
        # staging buffer is reused (the Stager prefetches once, then caches).
        mh = ext.managed_empty([numel], dtype)
        mh.copy_(ref_cpu)
        out["migrate:managed"] = timeit(lambda: ext.migrate(mh, False))

        # register: hipHostRegister + alias of an ordinary pageable buffer.
        ph = torch.empty(numel, dtype=dtype)
        ph.copy_(ref_cpu)
        out["migrate:register"] = timeit(lambda: ext.register_migrate(ph))
    return out


def bench_memory(ext, numel, dtype):
    """Return HBM bytes actually consumed making an N-elem host batch usable."""
    nbytes = numel * _elsize(dtype)

    def used(make):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free0, _ = torch.cuda.mem_get_info()
        keep = make()          # hold references so nothing is freed
        torch.cuda.synchronize()
        free1, _ = torch.cuda.mem_get_info()
        del keep
        torch.cuda.empty_cache()
        return free0 - free1

    res = {"nbytes": nbytes}

    def copy_make():
        try:
            h = torch.empty(numel, dtype=dtype, pin_memory=True)
        except Exception:
            h = torch.empty(numel, dtype=dtype)
        g = h.to("cuda")
        return (h, g)          # host buffer + device copy both resident
    res["to(copy)"] = used(copy_make)

    if ext is not None:
        def managed_make():
            h = ext.managed_empty([numel], dtype)
            g = ext.migrate(h, True)
            return (h, g)      # single managed buffer, aliased
        res["migrate:managed"] = used(managed_make)

        def register_make():
            h = torch.empty(numel, dtype=dtype)
            g = ext.register_migrate(h)
            return (h, g)      # single pageable buffer, aliased
        res["migrate:register"] = used(register_make)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--dtype", default="float32")
    args = p.parse_args()
    dtype = getattr(torch, args.dtype)

    print(f"HSA_XNACK={os.environ.get('HSA_XNACK')}  hip={torch.version.hip}  "
          f"device={torch.cuda.get_device_name(0)}")
    avail = unified_memory_available()
    print(f"unified_memory_available={avail}\n")
    ext = zerocopy._load_ext() if avail else None

    print("== transfer cost (per call) ==")
    sizes = [1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 28]
    modes = ["to(copy)", "migrate:managed", "migrate:register"]
    hdr = f"{'bytes':>12}" + "".join(f"{m+' ms':>22}" for m in modes) + f"{'ok':>6}"
    print(hdr)
    for n in sizes:
        r = bench_transfer(ext, n, dtype, args.iters)
        nbytes = n * _elsize(dtype)
        row = f"{nbytes:>12}"
        ok = True
        for m in modes:
            if m in r:
                dt, bw, k = r[m]
                row += f"{dt*1e3:>12.4f}/{bw:>7.0f}GB"
                ok = ok and k
            else:
                row += f"{'-':>22}"
        row += f"{str(ok):>6}"
        print(row)

    print("\n== HBM footprint (bytes actually resident to stage one batch) ==")
    print(f"{'batch_bytes':>12}{'to(copy)':>16}{'migrate:managed':>18}"
          f"{'migrate:register':>18}{'saved':>10}")
    for n in [1 << 24, 1 << 26, 1 << 28]:
        m = bench_memory(ext, n, dtype)
        c = m["to(copy)"]
        mg = m.get("migrate:managed", 0)
        rg = m.get("migrate:register", 0)
        saved = f"{100*(1 - mg/c):.0f}%" if (ext and c) else "-"
        print(f"{m['nbytes']:>12}{c:>16}{mg:>18}{rg:>18}{saved:>10}")


if __name__ == "__main__":
    main()
