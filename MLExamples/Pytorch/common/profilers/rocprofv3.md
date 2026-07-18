# rocprofv3 — PyTorch examples (kernel and RCCL trace)

> Shared guide for [`imagenet`](../../imagenet/PROFILING.md),
> [`minGPT-ddp`](../../minGPT-ddp/PROFILING.md), and [`FSDP2`](../../FSDP2/PROFILING.md).
> Read each example's *Ground rules* first.

For a **framework-independent** kernel trace — and to see the RCCL collectives as
device kernels / API calls — use `rocprofv3` from the `rocm` module. rocprofv3
traces the whole process, so bound the trace with a short `--iters`/`--warmup`
instead of `--profile`.

## Run

```bash
module load rocm openmpi pytorch
export MIOPEN_FIND_MODE=FAST      # imagenet
```

Kernel trace + stats (per-kernel totals):

| Example | Command |
|---------|---------|
| imagenet | `rocprofv3 --kernel-trace --stats --truncate-kernels -- torchrun --standalone --nproc_per_node=2 ddp_resnet_bench.py -a resnet50 -b 128 --iters 10 --warmup 5` |
| minGPT-ddp | `rocprofv3 --kernel-trace --stats --truncate-kernels -- torchrun --standalone --nproc_per_node=2 ddp_gpt_bench.py --n-layer 12 --n-embd 768 --warmup 5 --iters 10` |
| FSDP2 | `rocprofv3 --kernel-trace --stats --truncate-kernels -- torchrun --standalone --nproc_per_node=2 fsdp2_bench.py --warmup 3 --iters 10` |

The stats CSV ranks kernels by total time (MIOpen conv variants dominate resnet;
GEMM variants dominate GPT/FSDP2). To trace the collectives explicitly add the
RCCL/HIP API traces, e.g.:

```bash
rocprofv3 --hip-trace --rccl-trace --stats -- \
  torchrun --standalone --nproc_per_node=2 ddp_resnet_bench.py -a resnet50 -b 128 --iters 10
```

What the RCCL rows tell you:

- **imagenet / minGPT-ddp (DDP):** `ncclDevKernel_AllReduce_*` + RCCL API rows =
  the communication time, comparable to `comm_s`.
- **FSDP2:** `ncclDevKernel_AllGather_*` and `ncclDevKernel_ReduceScatter_*` — the
  sharded collectives, distinct from DDP's `AllReduce`.

## Viewing remotely

The `--stats` CSV/text is readable in the terminal. If you also emit a trace
(`--output-format pftrace`), open the resulting Perfetto trace in a browser inside
an AAC6 graphical session — `man aac6_vnc` (TurboVNC), `man aac6_novnc` (browser),
or `man aac6_x11` (`ssh -X`) — then drag it onto <https://ui.perfetto.dev>.

## See also

- [torch.profiler](torch-profiler.md) — the framework-native equivalent
- [rocprof-compute](rocprof-compute.md) — roofline of the same kernels
- [rocprofiler-systems](rocprofiler-systems.md) — full timeline with overlap
