# rocprofiler-systems — PyTorch examples (overlap timeline)

> Shared guide for [`imagenet`](../../imagenet/PROFILING.md),
> [`minGPT-ddp`](../../minGPT-ddp/PROFILING.md), and [`FSDP2`](../../FSDP2/PROFILING.md).
> Read each example's *Ground rules* first.

When you need to *see* the communication overlap (or stall) the compute on a
**timeline**, use `rocprofiler-systems`. It emits a Perfetto trace spanning host and
GPU streams.

## Run

```bash
module load rocprofiler-systems
```

| Example | Command |
|---------|---------|
| imagenet | `rocprof-sys-run -- torchrun --standalone --nproc_per_node=2 ddp_resnet_bench.py -a resnet50 -b 128 --iters 10 --warmup 5` |
| minGPT-ddp | `rocprof-sys-run -- torchrun --standalone --nproc_per_node=2 ddp_gpt_bench.py --warmup 5 --iters 10` |
| FSDP2 | `rocprof-sys-run -- torchrun --standalone --nproc_per_node=2 fsdp2_bench.py --warmup 3 --iters 10` |

What to look for on the timeline:

- **imagenet / minGPT-ddp (DDP):** the RCCL stream should run *concurrently* with
  the backward compute stream — that overlap is what keeps `comm_pct` low. A
  serialized all-reduce (comm after compute finishes) points to a bucketing/overlap
  problem or a host-staged transport (check `iommu=pt` and `NCCL_DEBUG=INFO`).
- **FSDP2:** you want the all-gather of layer *N+1*'s parameters to run *while*
  layer *N* computes (prefetch overlap). Serialized all-gathers (gather → compute →
  gather) indicate a prefetch/overlap problem or a host-staged transport.

## Viewing remotely

Open the resulting `perfetto-trace.proto` in a browser inside an AAC6 graphical
session:

- `man aac6_vnc` — TurboVNC desktop, then drag the trace onto <https://ui.perfetto.dev>
- `man aac6_novnc` — the same desktop in your local browser
- `man aac6_x11` — `ssh -X` and launch a single browser window

## See also

- [torch.profiler](torch-profiler.md) / [TensorBoard](tensorboard.md) — framework-native timeline
- [rocprofv3](rocprofv3.md) — kernel/RCCL trace feeding the same picture
