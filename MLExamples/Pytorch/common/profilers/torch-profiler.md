# torch.profiler (Kineto) — PyTorch examples

> Shared guide for [`imagenet`](../../imagenet/profiling/PROFILING.md),
> [`minGPT-ddp`](../../minGPT-ddp/PROFILING.md), and [`FSDP2`](../../FSDP2/PROFILING.md).
> Read each example's *Ground rules* first.

`torch.profiler` is the **first stop** for an ML workload: it speaks the framework's
language (ops, `nccl:all_reduce` / `nccl:all_gather`, module names) and needs no
extra tooling — it ships in `pytorch/2.12.0`. The `--profile` flag wraps a few steps
in `torch.profiler.profile` with the `CPU` + `CUDA` activity sets; on ROCm the
`CUDA` set captures HIP kernels **and** the RCCL collective kernels, so one table
attributes time to compute vs. communication.

## Run

```bash
module load rocm openmpi pytorch
export MIOPEN_FIND_MODE=FAST     # imagenet: keep conv autotune out of the window
```

| Example | Command (2 GPUs, per-rank trace under `./torch_prof`) |
|---------|------------------------------------------------------|
| imagenet | `torchrun --standalone --nproc_per_node=2 ddp_resnet_bench.py -a resnet50 -b 128 --profile --profile-dir ./torch_prof` |
| minGPT-ddp | `torchrun --standalone --nproc_per_node=2 ddp_gpt_bench.py --profile --profile-dir ./torch_prof` |
| FSDP2 | `torchrun --standalone --nproc_per_node=2 fsdp2_bench.py --profile --profile-dir ./torch_prof` |

Rank 0 prints the key-averages table (sorted by CUDA time). Verified top rows for
**resnet50** on MI300A — convolution-bound:

```
Name                                   Self CUDA   Self CUDA %   # of Calls
DistributedDataParallel.forward        173.652ms      50.81%          6
aten::convolution_backward             151.917ms      44.45%        318
aten::miopen_convolution                68.875ms      20.15%        318
aten::miopen_batch_norm_backward        30.013ms       8.78%        318
...
Self CUDA time total: 341.746ms
```

## Finding the communication

Sort by name or search the table for `nccl`:

- **imagenet / minGPT-ddp (DDP):** `ncclDevKernel_AllReduce_*` and the
  `nccl:all_reduce` op are the gradient all-reduce. Small for resnet50 (conv
  dominates); **non-trivial** for the GPT block (larger gradients).
- **FSDP2 (sharded):** there is **no** `AllReduce`. Look for
  `ncclDevKernel_AllGather_*` / `nccl:all_gather` (parameter gather, forward **and**
  backward) and `ncclDevKernel_ReduceScatter_*` / `nccl:reduce_scatter` (gradient
  reduce). Seeing all-gather + reduce-scatter instead of all-reduce confirms a
  genuinely sharded model.

To enlarge the RCCL signal, profile a bigger model (`-a resnet152`;
`--n-layer 24 --n-embd 1024 --n-head 16`). Knobs (in each script's `profile_steps`):
`record_shapes`, `profile_memory` (both on), `with_stack=True` (source view,
adds overhead), `sort_by="self_cuda_time_total"`.

## Viewing the trace remotely

The trace is written as `*.pt.trace.json` (one per rank). Open it in a browser
inside an AAC6 graphical session:

- `man aac6_vnc` — TurboVNC desktop, then drag the JSON onto
  <https://ui.perfetto.dev> or `chrome://tracing`
- `man aac6_novnc` — the same desktop in your local browser
- `man aac6_x11` — `ssh -X` and launch a single browser window
- or load it in [TensorBoard](tensorboard.md) (§ TensorBoard page).

## See also

- [TensorBoard](tensorboard.md) — GUI over the same trace (timeline, kernel view)
- [rocprofv3](rocprofv3.md) — framework-independent kernel + RCCL trace
- [rocprofiler-systems](rocprofiler-systems.md) — the overlap timeline
- [Score-P](scorep.md) — per-rank Python training-loop regions
