#!/usr/bin/env python3
"""DDP scaling benchmark for the minGPT model, instrumented for RCCL cost.

This reuses the upstream minGPT model definition (`GPT`, `GPTConfig`) from
`pytorch/examples/distributed/minGPT-ddp` and trains it with
`DistributedDataParallel` on synthetic token batches (no dataset download, no
hydra/fsspec/boto3 dependencies). One process per GPU is launched by torchrun.

Why synthetic data: with random tokens the input pipeline is free, so a training
step is GPU compute + the DDP gradient **all-reduce** over RCCL. We measure the
all-reduce cost two ways:

  1. Directly, using DDP's `no_sync()` context, which skips the gradient
     all-reduce. Communication time per step ~= t_step(sync) - t_step(no_sync).
  2. Indirectly, via throughput scaling across GPU counts (run the sweep script).

Launch (one node, N GPUs):

    torchrun --standalone --nproc_per_node=<N> ddp_gpt_bench.py

Point at a specific upstream checkout with UPSTREAM=/path/to/minGPT-ddp/mingpt.
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def find_upstream_model():
    """Locate the upstream minGPT `mingpt` package dir and import GPT/GPTConfig."""
    candidates = []
    env = os.environ.get("UPSTREAM")
    if env:
        candidates.append(env)
    candidates += [
        os.path.expanduser("~/pytorch_examples/distributed/minGPT-ddp/mingpt"),
        os.path.expanduser("~/examples/distributed/minGPT-ddp/mingpt"),
        "./pytorch_examples/distributed/minGPT-ddp/mingpt",
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "model.py")):
            sys.path.insert(0, c)
            from model import GPT, GPTConfig  # type: ignore
            return GPT, GPTConfig, c
    raise SystemExit(
        "Could not find upstream minGPT model.py. Set UPSTREAM=/path/to/minGPT-ddp/mingpt\n"
        "  git clone --depth=1 https://github.com/pytorch/examples.git ~/pytorch_examples"
    )


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def timed_steps(model, optimizer, batch, target, iters, sync, amp=False):
    """Run `iters` train steps; return average step time in seconds.

    sync=True  -> normal DDP (gradients all-reduced every step over RCCL).
    sync=False -> model.no_sync() (all-reduce skipped) to expose the comm cost.
    amp=True   -> bf16 autocast for the forward pass.
    """
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else _null()
        if sync:
            with ctx:
                _, loss = model(batch, target)
            loss.backward()
        else:
            with model.no_sync(), ctx:
                _, loss = model(batch, target)
                loss.backward()
        optimizer.step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3


def main():
    p = argparse.ArgumentParser(description="minGPT DDP RCCL scaling benchmark")
    p.add_argument("--n-layer", type=int, default=12)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--n-embd", type=int, default=768)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--batch-size", type=int, default=8, help="per-GPU batch")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--amp", action="store_true", help="bf16 autocast forward")
    args = p.parse_args()

    GPT, GPTConfig, upstream = find_upstream_model()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    cfg = GPTConfig(
        model_type=None,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        vocab_size=args.vocab_size, block_size=args.block_size,
    )
    model = GPT(cfg).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))

    # Synthetic token batch (fixed per rank).
    torch.manual_seed(1234 + rank)
    batch = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device=device)
    target = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device=device)

    n_params = sum(p.numel() for p in model.parameters())
    grad_bytes = n_params * 4  # fp32 gradients all-reduced per step

    # Warm-up (also lets RCCL build its channels/rings).
    timed_steps(model, optimizer, batch, target, args.warmup, sync=True, amp=args.amp)

    t_sync = timed_steps(model, optimizer, batch, target, args.iters, sync=True, amp=args.amp)
    t_nosync = timed_steps(model, optimizer, batch, target, args.iters, sync=False, amp=args.amp)

    comm = max(t_sync - t_nosync, 0.0)
    comm_pct = 100.0 * comm / t_sync if t_sync > 0 else 0.0
    tokens = args.batch_size * args.block_size
    global_tokens_per_s = tokens * world_size / t_sync

    if rank == 0:
        print(f"# upstream model: {upstream}")
        print(f"# world_size={world_size}  params={n_params/1e6:.1f}M  "
              f"grad_allreduce={grad_bytes/1e6:.0f}MB/step  "
              f"per_gpu_batch={args.batch_size} block={args.block_size} "
              f"amp={'bf16' if args.amp else 'off'}")
        print(f"RESULT world_size={world_size} "
              f"step_sync_s={t_sync:.4f} step_nosync_s={t_nosync:.4f} "
              f"comm_s={comm:.4f} comm_pct={comm_pct:.1f} "
              f"tokens_per_s={global_tokens_per_s:.0f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
