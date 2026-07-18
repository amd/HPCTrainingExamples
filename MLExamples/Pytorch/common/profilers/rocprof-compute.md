# rocprof-compute — PyTorch examples (roofline)

> Shared guide for [`imagenet`](../../imagenet/PROFILING.md),
> [`minGPT-ddp`](../../minGPT-ddp/PROFILING.md), and [`FSDP2`](../../FSDP2/PROFILING.md).
> Read each example's *Ground rules* first.

`rocprof-compute` proves whether the heavy kernels are **compute- or memory-bound**
on MI300A — it reports achieved HBM bandwidth and FLOP/s against peak (the
roofline). Compare the achieved TFLOP/s to the [DeepSpeed FLOP
count](deepspeed-flops.md) and the MI300A peak.

## Run

```bash
module load rocm openmpi pytorch
export MIOPEN_FIND_MODE=FAST      # imagenet
```

| Example | Profile command |
|---------|-----------------|
| imagenet | `rocprof-compute profile -n resnet_roof -- torchrun --standalone --nproc_per_node=1 ddp_resnet_bench.py -a resnet50 -b 128 --iters 10 --warmup 5` |
| minGPT-ddp | `rocprof-compute profile -n mingpt_roof -- torchrun --standalone --nproc_per_node=1 ddp_gpt_bench.py --n-layer 12 --n-embd 768 --warmup 5 --iters 10` |
| FSDP2 | `rocprof-compute profile -n fsdp2_roof -- torchrun --standalone --nproc_per_node=2 fsdp2_bench.py --warmup 3 --iters 10` |

Then analyze (text tables or the GUI):

```bash
rocprof-compute analyze -p workloads/<name>_roof/MI300A_A1            # text tables
rocprof-compute analyze -p workloads/<name>_roof/MI300A_A1 --gui      # web GUI
```

Per-example takeaway:

- **imagenet:** conv on CDNA benefits from `--channels-last` (NHWC); profile with
  and without it to quantify the win the README recommends.
- **minGPT-ddp:** bf16 autocast (`--amp`) should push the attention/MLP GEMMs
  toward the compute-bound corner.
- **FSDP2:** `--mixed-precision` (bf16 params) should do the same for the GEMMs.

## Viewing remotely

`--gui` serves a web dashboard; the text tables print to the terminal. To reach the
GUI use the AAC6 methods:

- `man aac6_x11` / SSH tunnel to the served port, browse locally
- `man aac6_vnc` — TurboVNC desktop, open the dashboard in the desktop browser
- `man aac6_novnc` — the same desktop in your local browser

## See also

- [DeepSpeed FlopsProfiler](deepspeed-flops.md) — the FLOP ceiling to compare against
- [rocprofv3](rocprofv3.md) — per-kernel time feeding the roofline
