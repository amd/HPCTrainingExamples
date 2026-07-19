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

    torchrun --standalone --nproc_per_node=<N> fsdp2_bench.py [--mixed-precision] [--compile]

Point at a specific upstream checkout with UPSTREAM=/path/to/distributed/FSDP2.
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

# Optional Score-P user-region annotations (no-op unless launched via
# ../common/scorep_launch.sh, which sets SCOREP_ML=1 and runs under scorep).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from scorep_ml import region
from rccl_time import rccl_time_per_step


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


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def profile_steps(model, optimizer, x, out_dir, rank):
    """Run a few steps under torch.profiler and dump a Kineto/TensorBoard trace.

    On ROCm the CUDA activity set captures HIP kernels *and* the RCCL collective
    kernels. Unlike DDP (a single ``all_reduce``), FSDP2's collectives show up as
    ``ncclDevKernel_AllGather_*`` (parameter gather, forward + backward) and
    ``ncclDevKernel_ReduceScatter_*`` (gradient reduce), so the table attributes
    time to compute vs. the two sharded-comm phases.
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
            loss = model(x).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            prof.step()
    if rank == 0:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        print(f"# torch.profiler trace written to {out_dir} "
              f"(view: chrome://tracing, https://ui.perfetto.dev, or TensorBoard)")


def flops_report(Transformer, model_args, x, device):
    """Report FLOPs/MACs/params of the *dense* (unsharded) model.

    FSDP2 stores parameters as sharded ``DTensor``s, which the DeepSpeed
    FlopsProfiler cannot introspect. So build one fresh, dense ``Transformer`` on
    rank 0 for the compute-cost reference (the full-model FLOPs are what matter;
    FSDP only changes how the work is *distributed*, not the total). For very
    large configs that do not fit on a single GPU, reduce ``--n-layers`` for this
    estimate only.
    """
    from deepspeed.profiling.flops_profiler import get_model_profile
    m = Transformer(model_args).to(device).eval()
    flops, macs, params = get_model_profile(
        m, args=(x,), print_profile=True, detailed=False,
        warm_up=3, as_string=True)
    print(f"FLOPS_PROFILE model=transformer n_layers={model_args.n_layers} "
          f"dim={model_args.dim} batch={x.size(0)} seq={x.size(1)} "
          f"flops={flops} macs={macs} params={params}")
    del m
    torch.cuda.empty_cache()


def timed_steps(model, optimizer, next_x, iters):
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        with region("train_step"):
            x = next_x()
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
    p.add_argument("--compile", action="store_true",
                   help="wrap the sharded model in torch.compile (graph capture + fusion)")
    p.add_argument("--profile", action="store_true",
                   help="run a few steps under torch.profiler and dump a trace")
    p.add_argument("--profile-dir", default="./torch_prof",
                   help="output dir for the torch.profiler trace")
    p.add_argument("--rccl-time", action="store_true",
                   help="also report per-step RCCL kernel device time (profiler): "
                        "FSDP2 all-gather + reduce-scatter")
    p.add_argument("--flops", action="store_true",
                   help="rank 0 prints a DeepSpeed FLOPs/MACs/params report (dense model)")
    p.add_argument("--migrate", action="store_true",
                   help="stage each token batch from host to GPU with zero-copy "
                        "migrate() (MI300A unified memory; needs HSA_XNACK=1). "
                        "Token-id batches are tiny; see common/migrate_bench.py "
                        "for the raw transfer cost this eliminates.")
    p.add_argument("--host-copy", action="store_true",
                   help="stage each token batch from host to GPU with a .to() copy "
                        "(baseline to compare against --migrate)")
    p.add_argument("--migrate-method", choices=["managed", "register"],
                   default="managed",
                   help="zero-copy method for --migrate: 'managed' aliases a "
                        "hipMallocManaged buffer; 'register' hipHostRegisters an "
                        "ordinary pageable buffer (works on any existing tensor)")
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

    if args.compile:
        # Compile the sharded model; the first (warm-up) step pays the compile cost.
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

    # Optional host->GPU input staging (to exercise migrate vs .to). Default is
    # the pre-resident on-GPU batch above, which leaves existing results intact.
    stage = args.migrate or args.host_copy
    if stage:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        os.pardir, "common"))
        from zerocopy import Stager
        stager = Stager(device, enabled=args.migrate, method=args.migrate_method)
        host_x = stager.host_empty((args.batch_size, args.seq_len), x.dtype)
        host_x.copy_(x.cpu())
        def next_x():
            return stager.to_device(host_x)
    else:
        stager = None
        def next_x():
            return x

    if args.flops and rank == 0:
        flops_report(Transformer, model_args, x, device)

    if args.profile:
        timed_steps(model, optimizer, next_x, args.warmup)
        profile_steps(model, optimizer, next_x(), f"{args.profile_dir}/rank{rank}", rank)
        dist.barrier()
        dist.destroy_process_group()
        return

    timed_steps(model, optimizer, next_x, args.warmup)
    torch.cuda.reset_peak_memory_stats(device)
    t_step = timed_steps(model, optimizer, next_x, args.iters)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6

    rccl_s = -1.0
    if args.rccl_time:
        def _one_step():
            xx = next_x()
            optimizer.zero_grad(set_to_none=True)
            loss = model(xx).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        rccl_s, _names = rccl_time_per_step(_one_step, args.iters, warmup=3)

    # A full (unsharded) parameter count, for reference.
    n_params = sum(p.numel() for p in model.parameters())  # local shard count
    tokens = args.batch_size * args.seq_len
    global_tokens_per_s = tokens * world_size / t_step

    if rank == 0:
        mp = "bf16" if args.mixed_precision else "fp32"
        print(f"# upstream model: {upstream}")
        print(f"# world_size={world_size} layers={args.n_layers} dim={args.dim} "
              f"seq={args.seq_len} per_gpu_batch={args.batch_size} precision={mp} "
              f"compile={'on' if args.compile else 'off'} "
              f"stage_input={stager.mode if stage else 'none'}")
        print(f"RESULT world_size={world_size} step_s={t_step:.4f} "
              f"tokens_per_s={global_tokens_per_s:.0f} peak_mem_mb={peak_mb:.0f} "
              f"local_shard_params={n_params/1e6:.1f}M "
              f"rccl_s={rccl_s:.4f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
