#!/usr/bin/env python3
"""DDP ResNet scaling benchmark, instrumented for RCCL cost.

A robust, torchrun-based replacement for driving the upstream ``imagenet/main.py
--dummy`` when studying RCCL scaling. It trains a torchvision ResNet with
``DistributedDataParallel`` on synthetic in-GPU data (like ``--dummy``, but with
no DataLoader), which avoids two problems seen with the upstream script on
MI300A nodes:

  * ``main.py`` initializes CUDA in the parent before ``mp.spawn``, which can
    poison the child CUDA contexts and hang RCCL. torchrun uses independent
    processes and avoids this.
  * FakeData + DataLoader workers add host-side noise to a comm study.

It measures the gradient all-reduce cost directly with DDP ``no_sync()`` (the
same technique as the minGPT-ddp example):

    comm_per_step ~= step_time(all-reduce) - step_time(no_sync)

Optimizations exposed as flags: ``--channels-last`` (NHWC, big win for conv on
CDNA/MI300), ``--amp`` (bf16 autocast), and ``--compile`` (``torch.compile``
graph capture + kernel fusion).

Launch (one node, N GPUs):

    torchrun --standalone --nproc_per_node=<N> ddp_resnet_bench.py -a resnet50 -b 128

Recommended environment on MI300A:

    export MIOPEN_FIND_MODE=FAST
    # Nodes must be booted with "iommu=pt" for RCCL to use direct xGMI P2P
    # (verify: grep -o 'iommu=pt' /proc/cmdline). Only if a node lacks it
    # (RCCL P2P DMA hangs) add the host-staged fallback:
    #   export NCCL_P2P_DISABLE=1
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP


def timed_steps(model, optimizer, criterion, next_x, y, iters, sync, amp):
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        x = next_x()  # fixed on-GPU tensor, or a host->GPU staged batch
        optimizer.zero_grad(set_to_none=True)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else _null()
        if sync:
            with ctx:
                loss = criterion(model(x), y)
            loss.backward()
        else:
            with model.no_sync():
                with ctx:
                    loss = criterion(model(x), y)
                loss.backward()
        optimizer.step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def profile_steps(model, optimizer, criterion, x, y, amp, out_dir, rank):
    """Run a few steps under torch.profiler and dump a Kineto/TensorBoard trace.

    On ROCm the CUDA activity set captures HIP kernels *and* the RCCL collective
    kernels, so the key-averages table attributes time to compute (conv/gemm) vs.
    communication (``nccl:all_reduce`` / ``ncclDevKernel_*``).
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
                loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            prof.step()
    if rank == 0:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        print(f"# torch.profiler trace written to {out_dir} "
              f"(view: chrome://tracing, https://ui.perfetto.dev, or TensorBoard)")


def flops_report(arch, batch, device, channels_last):
    """Report FLOPs/MACs/params/latency of the model with the DeepSpeed profiler.

    ``get_model_profile`` runs on a plain ``nn.Module`` (no DeepSpeed engine
    needed), so it works for this DDP example as a compute-cost reference.
    """
    from deepspeed.profiling.flops_profiler import get_model_profile
    m = getattr(models, arch)().to(device).eval()
    if channels_last:
        m = m.to(memory_format=torch.channels_last)
    inp = torch.randn(batch, 3, 224, 224, device=device)
    if channels_last:
        inp = inp.to(memory_format=torch.channels_last)
    flops, macs, params = get_model_profile(
        m, args=(inp,), print_profile=True, detailed=False,
        warm_up=3, as_string=True)
    print(f"FLOPS_PROFILE arch={arch} batch={batch} "
          f"flops={flops} macs={macs} params={params}")
    del m, inp
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description="DDP ResNet RCCL scaling benchmark")
    p.add_argument("-a", "--arch", default="resnet50")
    p.add_argument("-b", "--batch-size", type=int, default=128, help="per-GPU batch")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile", action="store_true",
                   help="wrap the model in torch.compile (graph capture + fusion)")
    p.add_argument("--profile", action="store_true",
                   help="run a few steps under torch.profiler and dump a trace")
    p.add_argument("--profile-dir", default="./torch_prof",
                   help="output dir for the torch.profiler trace")
    p.add_argument("--flops", action="store_true",
                   help="rank 0 prints a DeepSpeed FLOPs/params/latency report")
    p.add_argument("--migrate", action="store_true",
                   help="stage each input batch from host to GPU with zero-copy "
                        "migrate() (MI300A unified memory; needs HSA_XNACK=1)")
    p.add_argument("--host-copy", action="store_true",
                   help="stage each input batch from host to GPU with a .to() copy "
                        "(baseline to compare against --migrate)")
    p.add_argument("--migrate-method", choices=["managed", "register"],
                   default="managed",
                   help="zero-copy method for --migrate: 'managed' aliases a "
                        "hipMallocManaged buffer; 'register' hipHostRegisters an "
                        "ordinary pageable buffer (works on any existing tensor)")
    args = p.parse_args()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    model = getattr(models, args.arch)().to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = DDP(model, device_ids=[local_rank])
    if args.compile:
        # Compile the DDP module; the first (warm-up) step pays the compile cost.
        # OptimizedModule delegates attribute access, so model.no_sync() still works.
        model = torch.compile(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    x = torch.randn(args.batch_size, 3, 224, 224, device=device)
    if args.channels_last:
        x = x.to(memory_format=torch.channels_last)
    y = torch.randint(0, 1000, (args.batch_size,), device=device)

    # Optional host->GPU input staging (to exercise migrate vs .to). Default is
    # the pre-resident on-GPU tensor above, which leaves existing results intact.
    stage = args.migrate or args.host_copy
    if stage:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "common"))
        from zerocopy import Stager
        stager = Stager(device, enabled=args.migrate, method=args.migrate_method)
        host_x = stager.host_empty((args.batch_size, 3, 224, 224), torch.float32)
        host_x.copy_(torch.randn(args.batch_size, 3, 224, 224))

        def next_x():
            gx = stager.to_device(host_x)
            return gx.to(memory_format=torch.channels_last) if args.channels_last else gx
    else:
        stager = None
        def next_x():
            return x

    n_params = sum(p.numel() for p in model.parameters())

    if args.flops and rank == 0:
        flops_report(args.arch, args.batch_size, device, args.channels_last)

    if args.profile:
        timed_steps(model, optimizer, criterion, next_x, y, args.warmup, True, args.amp)
        profile_steps(model, optimizer, criterion, next_x(), y, args.amp,
                      f"{args.profile_dir}/rank{rank}", rank)
        dist.barrier()
        dist.destroy_process_group()
        return

    timed_steps(model, optimizer, criterion, next_x, y, args.warmup, True, args.amp)
    if args.compile:
        # Warm the no_sync graph too; torch.compile recompiles it on first use.
        timed_steps(model, optimizer, criterion, next_x, y, 3, False, args.amp)
    t_sync = timed_steps(model, optimizer, criterion, next_x, y, args.iters, True, args.amp)
    t_nosync = timed_steps(model, optimizer, criterion, next_x, y, args.iters, False, args.amp)

    comm = max(t_sync - t_nosync, 0.0)
    comm_pct = 100.0 * comm / t_sync if t_sync > 0 else 0.0
    global_img_s = args.batch_size * world_size / t_sync

    if rank == 0:
        opt = []
        if args.channels_last: opt.append("channels_last")
        if args.amp: opt.append("amp_bf16")
        if args.compile: opt.append("compile")
        if stage: opt.append(f"stage_input={stager.mode}")
        print(f"# arch={args.arch} world_size={world_size} per_gpu_batch={args.batch_size} "
              f"params={n_params/1e6:.1f}M grad_allreduce={n_params*4/1e6:.0f}MB "
              f"opts={','.join(opt) or 'none'}")
        print(f"RESULT world_size={world_size} step_sync_s={t_sync:.4f} "
              f"step_nosync_s={t_nosync:.4f} comm_s={comm:.4f} comm_pct={comm_pct:.1f} "
              f"img_per_s={global_img_s:.0f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
