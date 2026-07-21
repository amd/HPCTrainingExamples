# TensorBoard (torch-tb-profiler plugin) — PyTorch examples

> Shared guide for [`imagenet`](../../imagenet/profiling/PROFILING.md),
> [`minGPT-ddp`](../../minGPT-ddp/PROFILING.md), and [`FSDP2`](../../FSDP2/PROFILING.md).
> Read each example's *Ground rules* first.

The `*.pt.trace.json` files written by [`--profile`](torch-profiler.md) load
directly into TensorBoard's **PyTorch Profiler** plugin, which gives the
*Overview*, *Operator*, *Kernel*, *Trace*, and (with `with_stack=True`) *source*
views — including a GPU-utilization estimate and a step-time breakdown into
compute / communication / other. `tensorboard_trace_handler` (used by `--profile`)
already writes the exact layout the plugin expects.

## Install (venv layered on the module)

`tensorboard` and `torch-tb-profiler` are **not** in `pytorch/2.12.0`:

```bash
module load rocm openmpi pytorch
python -m venv --system-site-packages ~/venvs/tbprof
source ~/venvs/tbprof/bin/activate
pip install tensorboard torch-tb-profiler
```

## Run (point it at the `--profile` output dir)

First generate a trace with [torch.profiler](torch-profiler.md) `--profile`, then:

```bash
tensorboard --logdir ./torch_prof --port 6006
```

## Viewing remotely

TensorBoard is a web GUI on port 6006 of the compute node. Reach it with the AAC6
graphical / forwarding methods:

- `man aac6_x11` / SSH tunnel: `ssh -L 6006:<node>:6006 aac6` then browse
  <http://localhost:6006> locally
- `man aac6_vnc` — TurboVNC desktop, open <http://<node>:6006> in the desktop browser
- `man aac6_novnc` — the same desktop in your local browser

What to look for per example:

- **imagenet / minGPT-ddp (DDP):** the *Trace* view — confirm the RCCL all-reduce
  overlaps the backward pass; the *Kernel* view — the conv- (resnet) or GEMM-
  (GPT) dominated breakdown.
- **FSDP2:** the *Trace* view — whether the all-gather **prefetch** of the next
  layer's parameters overlaps the current layer's compute (the key to FSDP2
  efficiency).

## See also

- [torch.profiler](torch-profiler.md) — produces the trace TensorBoard reads
- [rocprofiler-systems](rocprofiler-systems.md) — Perfetto timeline alternative
