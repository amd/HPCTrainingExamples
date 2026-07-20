# PyTorch ML examples — per-profiler guides (shared)

These are **shared, step-by-step profiler guides** for the three PyTorch training
examples — [`imagenet`](../../imagenet/profiling/PROFILING.md),
[`minGPT-ddp`](../../minGPT-ddp/PROFILING.md), and [`FSDP2`](../../FSDP2/PROFILING.md).
Each page covers: environment/module setup, an instrumented run (with the exact
`torchrun` line per example), the **text output** to expect, and how to view any
**graphics** in a remote AAC6 session (VNC / noVNC / X11). Each example's
`PROFILING.md` keeps its own framing (what the communication looks like for that
workload) and links here for the per-tool steps.

> **Known issues & action items:** see [`../../PROFILER_ISSUES.md`](../../PROFILER_ISSUES.md)
> for the profiler problems encountered (env/module hygiene, Score-P Python + no
> GPU capture on ROCm 7.2.x, TensorBoard/DeepSpeed setup, CubeGUI viewers, the bf16
> hipBLASLt hang) with workarounds and what still needs to be addressed.

## Guides

| Profiler | Page | Scope | Notes |
|----------|------|-------|-------|
| **torch.profiler (Kineto)** | [`torch-profiler.md`](torch-profiler.md) | host + GPU | Primary ML profiler; Chrome/Perfetto trace |
| **DeepSpeed FlopsProfiler** | [`deepspeed-flops.md`](deepspeed-flops.md) | model | Compute ceiling (FLOPs/params) |
| **TensorBoard** | [`tensorboard.md`](tensorboard.md) | host + GPU | `torch-tb-profiler` plugin, port 6006 |
| **rocprofv3** | [`rocprofv3.md`](rocprofv3.md) | GPU | Kernel + RCCL trace |
| **rocprof-compute** | [`rocprof-compute.md`](rocprof-compute.md) | GPU | Roofline of GEMM/conv kernels |
| **rocprofiler-systems** | [`rocprofiler-systems.md`](rocprofiler-systems.md) | GPU + host | Perfetto overlap timeline |
| **Score-P** | [`scorep.md`](scorep.md) | Python regions | Per-rank training-loop regions → Cube/OTF2 |

For **multi-node / MPI whole-application** profiling (TAU, HPCToolkit), use the CG
example's guides — [TAU](../../../../MPI-examples/cg-solver-example/docs/profilers/tau.md)
and [HPCToolkit](../../../../MPI-examples/cg-solver-example/docs/profilers/hpctoolkit.md).

## Score-P for Python (setup note)

The `scorep` module provides the C/Fortran measurement system and the **OTF2**
Python bindings, but **not** the Score-P Python *instrumentation* bindings
(`import scorep`). Install those once into a venv layered on the PyTorch module:

```bash
module load rocm/<version> openmpi pytorch/<version> scorep/<version>
python -m venv --system-site-packages ~/scorep-venvs/ml
source ~/scorep-venvs/ml/bin/activate
pip install scorep          # builds the binding against scorep-config
```

Build the venv on a **login node** (it has network); **run** on a **compute node**
(that is where `libpapi.so.7.1`, required by the binding, is installed). Instrument
with `python -m scorep --nocompiler <script.py>`; give **each rank its own**
`SCOREP_EXPERIMENT_DIRECTORY` under `torchrun`.

## Viewing graphics remotely

- `man aac6_vnc` — TurboVNC desktop (Cube GUI, Perfetto, TensorBoard)
- `man aac6_novnc` — browser (noVNC) desktop
- `man aac6_x11` — X11 forwarding for a single GUI window
