# rocprof-compute — CG-GPU (roofline)

> Part of the [CG profiler guides](README.md). Read the shared
> [ground rules](../08-profiling.md#ground-rules-or-your-numbers-are-noise) first.
> `module load rocm/<ver>` prepends the patched `rocprof-compute` (a self-contained
> Nuitka executable) — never pipe/command-substitute `module` or it silently falls
> back to the unpatched script (see the [index note](README.md)).

`rocprof-compute` characterises GPU kernels against the hardware **roofline**: it
runs the counter passes for you and plots achieved HBM bandwidth vs. peak, proving
the SpMV and rocBLAS kernels are **memory-bound**.

## 1. Profile and analyse

The compute per rank is representative, so characterise the kernels on a single
rank:

```bash
module load rocm
cd CG-GPU && make
rocprof-compute profile -n cg_spmv -- ./cg_gpu src/Dubcova2.pm rccl 12345
rocprof-compute analyze -p workloads/cg_spmv/MI300* | less
```

Read the **roofline** and **memory chart**: the SpMV and rocBLAS kernels sit on the
**HBM-bandwidth roof** (low arithmetic intensity) — CG is memory-bound, so achieved
GB/s vs. peak is the practical compute ceiling. `--roof-only` is a faster,
fewer-pass variant.

Measured Top-Kernels on MI300A (ROCm 6.4.3): the SpMV `rocsparse::csrmvn` plus the
rocBLAS `axpy`/`dot`/`scal` kernels.

## 2. Viewing the roofline remotely

`rocprof-compute analyze` prints tables to the terminal (text). For the **graphical
roofline** it serves an interactive web (Dash) app:

```bash
rocprof-compute analyze -p workloads/cg_spmv/MI300* --gui
# serves on http://localhost:8050 by default
```

View it inside an AAC6 graphical session or over an SSH tunnel:

- `man aac6_vnc` — open `http://localhost:8050` in the TurboVNC desktop's browser
- `man aac6_novnc` — same, in your local browser via noVNC
- `man aac6_x11` — forward a browser window with `ssh -X`
- or tunnel the port: `ssh -L 8050:<node>:8050 <cluster>` then browse locally

The roofline plot places each kernel by arithmetic intensity; memory-bound CG
kernels cluster under the HBM diagonal, well below the compute ceiling.

## See also

- [rocprofv3](rocprofv3.md) — the raw kernel/counter traces this builds on
- [roofline-extractor](roofline-extractor.md) — automated *percent-of-peak* roofline
- [IntelliKit `metrix`](intellikit.md) — the same conclusion, decoded per kernel
