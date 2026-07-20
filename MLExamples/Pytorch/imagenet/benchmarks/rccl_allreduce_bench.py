#!/usr/bin/env python3
"""Standalone RCCL/NCCL all-reduce bandwidth benchmark.

This isolates the collective-communication cost that dominates data-parallel
training. It launches one process per GPU (via torchrun), performs warm-up and
timed all-reduce iterations over a range of message sizes, and reports the
algorithm bandwidth (algbw) and bus bandwidth (busbw) per size.

    algbw = size_bytes / time
    busbw = algbw * 2 * (world_size - 1) / world_size   # ring all-reduce factor

busbw is the number to watch: it should approach the hardware link bandwidth
(xGMI / Infinity Fabric on MI300A) and stay roughly flat as GPU count grows if
the interconnect scales well. A falling busbw as ranks increase means the RCCL
collective is becoming the scaling bottleneck.

Launch (single node, all visible GPUs):

    torchrun --standalone --nproc_per_node=<N> rccl_allreduce_bench.py

Mask GPUs to sweep rank counts, e.g. 2 ranks:

    HIP_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 rccl_allreduce_bench.py
"""
import argparse
import os

import torch
import torch.distributed as dist


def human_bytes(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024.0:
            return f"{n:.0f}{unit}"
        n /= 1024.0
    return f"{n:.0f}TB"


def bench(args):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # RCCL exposes the NCCL API on ROCm; nccl backend maps to librccl.
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print(f"# world_size={world_size}  backend={dist.get_backend()}")
        print(f"# device0={torch.cuda.get_device_name(0)}")
        print(f"# {'size':>10} {'count':>12} {'time_ms':>10} "
              f"{'algbw_GBps':>12} {'busbw_GBps':>12}")

    factor = 2.0 * (world_size - 1) / world_size if world_size > 1 else 0.0

    size = args.min_bytes
    while size <= args.max_bytes:
        n_elem = size // 4  # float32
        buf = torch.ones(n_elem, dtype=torch.float32, device=device)

        for _ in range(args.warmup):
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dist.barrier()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end) / args.iters
        algbw = (size / (elapsed_ms / 1e3)) / 1e9  # GB/s
        busbw = algbw * factor

        if rank == 0:
            print(f"  {human_bytes(size):>10} {n_elem:>12} {elapsed_ms:>10.3f} "
                  f"{algbw:>12.2f} {busbw:>12.2f}")
        size *= 2

    dist.barrier()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description="RCCL/NCCL all-reduce bandwidth benchmark")
    p.add_argument("--min-bytes", type=int, default=1 << 20, help="smallest message (default 1MB)")
    p.add_argument("--max-bytes", type=int, default=1 << 30, help="largest message (default 1GB)")
    p.add_argument("--warmup", type=int, default=10, help="warm-up iterations per size")
    p.add_argument("--iters", type=int, default=50, help="timed iterations per size")
    args = p.parse_args()
    bench(args)


if __name__ == "__main__":
    main()
