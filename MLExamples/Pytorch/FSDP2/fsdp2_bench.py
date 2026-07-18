#!/usr/bin/env python3
"""FSDP2 scaling benchmark, instrumented for RCCL communication and memory.

Reuses the upstream transformer (`Transformer`, `ModelArgs`) from
`pytorch/examples/distributed/FSDP2` and shards it with the FSDP2 API
(`torch.distributed.fsdp.fully_shard`). It trains on synthetic tokens (no
dataset) and reports, per GPU count:

  * step time and global tokens/sec (throughput scaling),
  * peak memory per GPU (max_memory_allocated).

Unlike DDP (a single gradient all-reduce per step), FSDP2 shards parameters and
optimizer state across ranks, so it communicates differently:

  * forward/backward **all-gather** the sharded parameters just-in-time, and
  * backward **reduce-scatter** the gradients.

The headline FSDP2 signal is therefore two-sided: as GPU count grows, peak
memory per GPU should **drop** (params are sharded over more ranks) while the
all-gather / reduce-scatter **communication grows**. This benchmark exposes both.

Launch (one node, N GPUs, minimum 2):

    torchrun --standalone --nproc_per_node=<N> fsdp2_bench.py [--mixed-precision]

Point at a specific upstream checkout with UPSTREAM=/path/to/distributed/FSDP2.
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist


def find_upstream_model():
    """Locate upstream FSDP2 dir and import Transformer/ModelArgs."""
    candidates = []
    env = os.environ.get("UPSTREAM")
    if env:
        candidates.append(env)
    candidates += [
        os.path.expanduser("~/pytorch_examples/distributed/FSDP2"),
        os.path.expanduser("~/examples/distributed/FSDP2"),
        "./pytorch_examples/distributed/FSDP2",
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "model.py")):
            sys.path.insert(0, c)
            from model import ModelArgs, Transformer  # type: ignore
            return ModelArgs, Transformer, c
    raise SystemExit(
        "Could not find upstream FSDP2 model.py. Set UPSTREAM=/path/to/distributed/FSDP2\n"
        "  git clone --depth=1 https://github.com/pytorch/examples.git ~/pytorch_examples"
    )


def timed_steps(model, optimizer, x, iters):
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        loss = model(x).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3


def main():
    p = argparse.ArgumentParser(description="FSDP2 RCCL scaling / memory benchmark")
    p.add_argument("--n-layers", type=int, default=16)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8, help="per-GPU batch")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--mixed-precision", action="store_true")
    args = p.parse_args()

    ModelArgs, Transformer, upstream = find_upstream_model()

    try:
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    except ImportError:
        raise SystemExit("FSDP2 (torch.distributed.fsdp.fully_shard) requires PyTorch >= 2.5")

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Pass device_id so FSDP2's collectives (all-gather/reduce-scatter) bind to
    # the right device/stream; without it they can hang on some RCCL builds.
    dist.init_process_group(backend="nccl", init_method="env://", device_id=device)

    torch.manual_seed(0)
    model_args = ModelArgs(
        n_layers=args.n_layers, n_heads=args.n_heads, vocab_size=args.vocab_size,
        max_seq_len=args.seq_len, dim=args.dim, dropout_p=0.0,
    )
    with torch.device("meta"):
        model = Transformer(model_args)

    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
        )
    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    model.to_empty(device=device)
    # Initialize the (meta-created) parameters to finite values.
    for m in model.modules():
        if m is not model and hasattr(m, "reset_parameters"):
            try:
                m.reset_parameters()
            except Exception:
                pass
    model.reset_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

    timed_steps(model, optimizer, x, args.warmup)
    torch.cuda.reset_peak_memory_stats(device)
    t_step = timed_steps(model, optimizer, x, args.iters)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6

    # A full (unsharded) parameter count, for reference.
    n_params = sum(p.numel() for p in model.parameters())  # local shard count
    tokens = args.batch_size * args.seq_len
    global_tokens_per_s = tokens * world_size / t_step

    if rank == 0:
        mp = "bf16" if args.mixed_precision else "fp32"
        print(f"# upstream model: {upstream}")
        print(f"# world_size={world_size} layers={args.n_layers} dim={args.dim} "
              f"seq={args.seq_len} per_gpu_batch={args.batch_size} precision={mp}")
        print(f"RESULT world_size={world_size} step_s={t_step:.4f} "
              f"tokens_per_s={global_tokens_per_s:.0f} peak_mem_mb={peak_mb:.0f} "
              f"local_shard_params={n_params/1e6:.1f}M")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
