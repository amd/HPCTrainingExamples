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

# Optional Score-P user-region annotations (no-op unless launched via
# ../common/scorep_launch.sh, which sets SCOREP_ML=1 and runs under scorep).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from scorep_ml import region
from rccl_time import rccl_time_per_step


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


def profile_steps(model, optimizer, batch, target, amp, out_dir, rank):
    """Run a few steps under torch.profiler and dump a Kineto/TensorBoard trace.

    On ROCm the CUDA activity set captures HIP kernels *and* the RCCL collective
    kernels, so the key-averages table attributes time to compute (attention/GEMM)
    vs. communication (``nccl:all_reduce`` / ``ncclDevKernel_AllReduce_*``). The
    transformer's gradients are large, so the all-reduce is a bigger share here
    than for the ResNet in the imagenet example.
    """
    from torch.profiler import (profile, ProfilerActivity, schedule,
                                tensorboard_trace_handler)
    os.makedirs(out_dir, exist_ok=True)
    sched = schedule(wait=1, warmup=3, active=6, repeat=1)
    acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=acts, schedule=sched,
                 on_trace_ready=tensorboard_trace_handler(out_dir),
                 record_shapes=True, profile_memory=True, with_stack=False) as prof:
        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else _null()
            with ctx:
                _, loss = model(batch, target)
            loss.backward()
            optimizer.step()
            prof.step()
    if rank == 0:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        print(f"# torch.profiler trace written to {out_dir} "
              f"(view: chrome://tracing, https://ui.perfetto.dev, or TensorBoard)")


def flops_report(GPT, cfg, batch, device):
    """Report FLOPs/MACs/params of the GPT with the DeepSpeed profiler.

    ``get_model_profile`` runs on a plain ``nn.Module`` (no DeepSpeed engine
    needed), so it works on the unwrapped GPT as a compute-cost reference. The
    GPT ``forward(idx, targets=None)`` accepts just the token batch.
    """
    from deepspeed.profiling.flops_profiler import get_model_profile
    m = GPT(cfg).to(device).eval()
    flops, macs, params = get_model_profile(
        m, args=(batch,), print_profile=True, detailed=False,
        warm_up=3, as_string=True)
    print(f"FLOPS_PROFILE model=gpt n_layer={cfg.n_layer} n_embd={cfg.n_embd} "
          f"batch={batch.size(0)} block={batch.size(1)} "
          f"flops={flops} macs={macs} params={params}")
    del m
    torch.cuda.empty_cache()


def timed_steps(model, optimizer, next_batch, target, iters, sync, amp=False):
    """Run `iters` train steps; return average step time in seconds.

    sync=True  -> normal DDP (gradients all-reduced every step over RCCL).
    sync=False -> model.no_sync() (all-reduce skipped) to expose the comm cost.
    amp=True   -> bf16 autocast for the forward pass.
    next_batch -> callable returning the token batch (fixed on-GPU, or staged).
    """
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        with region("train_step_sync" if sync else "train_step_nosync"):
            batch = next_batch()
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
    p.add_argument("--compile", action="store_true",
                   help="wrap the model in torch.compile (graph capture + fusion)")
    p.add_argument("--profile", action="store_true",
                   help="run a few steps under torch.profiler and dump a trace")
    p.add_argument("--profile-dir", default="./torch_prof",
                   help="output dir for the torch.profiler trace")
    p.add_argument("--rccl-time", action="store_true",
                   help="also report per-step RCCL kernel device time (profiler)")
    p.add_argument("--flops", action="store_true",
                   help="rank 0 prints a DeepSpeed FLOPs/MACs/params report")
    p.add_argument("--migrate", action="store_true",
                   help="stage each token batch from host to GPU with zero-copy "
                        "migrate() (MI300A unified memory; needs HSA_XNACK=1). "
                        "Note: token-id batches are tiny, so the end-to-end effect "
                        "here is small; see common/migrate_bench.py for raw cost.")
    p.add_argument("--host-copy", action="store_true",
                   help="stage each token batch from host to GPU with a .to() copy "
                        "(baseline to compare against --migrate)")
    p.add_argument("--migrate-method", choices=["managed", "register"],
                   default="managed",
                   help="zero-copy method for --migrate: 'managed' aliases a "
                        "hipMallocManaged buffer; 'register' hipHostRegisters an "
                        "ordinary pageable buffer (works on any existing tensor)")
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
    if args.compile:
        # Compile the DDP module; the first (warm-up) step pays the compile cost.
        # OptimizedModule delegates attribute access, so model.no_sync() still works.
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))

    # Synthetic token batch (fixed per rank).
    torch.manual_seed(1234 + rank)
    batch = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device=device)
    target = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device=device)

    # Optional host->GPU input staging (to exercise migrate vs .to). Default is
    # the pre-resident on-GPU batch above, which leaves existing results intact.
    stage = args.migrate or args.host_copy
    if stage:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        os.pardir, "common"))
        from zerocopy import Stager
        stager = Stager(device, enabled=args.migrate, method=args.migrate_method)
        host_batch = stager.host_empty((args.batch_size, args.block_size), batch.dtype)
        host_batch.copy_(batch.cpu())
        def next_batch():
            return stager.to_device(host_batch)
    else:
        stager = None
        def next_batch():
            return batch

    n_params = sum(p.numel() for p in model.parameters())
    grad_bytes = n_params * 4  # fp32 gradients all-reduced per step

    if args.flops and rank == 0:
        flops_report(GPT, cfg, batch, device)

    if args.profile:
        timed_steps(model, optimizer, next_batch, target, args.warmup, sync=True, amp=args.amp)
        profile_steps(model, optimizer, next_batch(), target, args.amp,
                      f"{args.profile_dir}/rank{rank}", rank)
        dist.barrier()
        dist.destroy_process_group()
        return

    # Warm-up (also lets RCCL build its channels/rings).
    timed_steps(model, optimizer, next_batch, target, args.warmup, sync=True, amp=args.amp)
    if args.compile:
        # Warm the no_sync graph too; torch.compile recompiles it on first use.
        timed_steps(model, optimizer, next_batch, target, 3, sync=False, amp=args.amp)

    torch.cuda.reset_peak_memory_stats(device)
    t_sync = timed_steps(model, optimizer, next_batch, target, args.iters, sync=True, amp=args.amp)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
    t_nosync = timed_steps(model, optimizer, next_batch, target, args.iters, sync=False, amp=args.amp)

    comm = max(t_sync - t_nosync, 0.0)
    comm_pct = 100.0 * comm / t_sync if t_sync > 0 else 0.0
    tokens = args.batch_size * args.block_size
    global_tokens_per_s = tokens * world_size / t_sync

    rccl_s = -1.0
    if args.rccl_time:
        def _one_step():
            batch = next_batch()
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.amp else _null()
            with ctx:
                _, loss = model(batch, target)
            loss.backward()
            optimizer.step()
        rccl_s, _names = rccl_time_per_step(_one_step, args.iters, warmup=3)

    if rank == 0:
        print(f"# upstream model: {upstream}")
        print(f"# world_size={world_size}  params={n_params/1e6:.1f}M  "
              f"grad_allreduce={grad_bytes/1e6:.0f}MB/step  "
              f"per_gpu_batch={args.batch_size} block={args.block_size} "
              f"amp={'bf16' if args.amp else 'off'} "
              f"compile={'on' if args.compile else 'off'} "
              f"stage_input={stager.mode if stage else 'none'}")
        print(f"RESULT world_size={world_size} "
              f"step_sync_s={t_sync:.4f} step_nosync_s={t_nosync:.4f} "
              f"comm_s={comm:.4f} comm_pct={comm_pct:.1f} "
              f"tokens_per_s={global_tokens_per_s:.0f} peak_mem_mb={peak_mb:.0f} "
              f"rccl_s={rccl_s:.4f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
