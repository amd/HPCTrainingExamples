# 8. Profiling: measuring communication and compute

The solver already reports a coarse **communication vs. compute** split from
`MPI_Wtime` timers (`g_halo_time`, `g_allreduce_time` in `cg.cpp`; see the
`comm total` / `halo exchange` / `dot allreduce` / `compute (rest)` lines). This
chapter shows how to go deeper with the profilers installed on the system —
attributing time to individual **kernels**, **transports** (SDMA/blit copies,
RCCL, MPI point-to-point/collective), and **hardware limits** (HBM/DRAM
bandwidth, FLOP/s) — for **both** the CPU reference and the GPU solver.

> **This chapter is now a hub.** Each profiler has its own step-by-step guide under
> [`profilers/`](profilers/README.md), including the **text output** to expect and
> how to view any **graphics** in a remote AAC6 session (VNC / noVNC / X11). The
> shared *Which profiler* and *Ground rules* material below still lives here.

## Per-profiler guides

| Profiler | Scope | Guide |
|----------|-------|-------|
| **Score-P** | MPI (+GPU/HIP) tracing → Cube/OTF2 | [`profilers/scorep.md`](profilers/scorep.md) |
| **rocprofv3** | GPU (`CG-GPU`) — kernels, transports, ATT | [`profilers/rocprofv3.md`](profilers/rocprofv3.md) |
| **rocprof-compute** | GPU — roofline (HBM-bound SpMV) | [`profilers/rocprof-compute.md`](profilers/rocprof-compute.md) |
| **rocprofiler-systems** | GPU + host — Perfetto timeline | [`profilers/rocprofiler-systems.md`](profilers/rocprofiler-systems.md) |
| **ROCm Optiq** | GPU — GUI trace/roofline viewer | [`profilers/roc-optiq.md`](profilers/roc-optiq.md) |
| **TAU** | MPI + GPU — communication matrix | [`profilers/tau.md`](profilers/tau.md) |
| **HPCToolkit** | MPI + GPU — call-path sampling | [`profilers/hpctoolkit.md`](profilers/hpctoolkit.md) |
| **likwid** | CPU (`CG-CPU`) — CPU roofline | [`profilers/likwid.md`](profilers/likwid.md) |
| **AMD uProf** | CPU — hotspots + memory | [`profilers/uprof.md`](profilers/uprof.md) |
| **Linux perf** | CPU — HW counters baseline | [`profilers/perf.md`](profilers/perf.md) |
| **Valgrind cachegrind** | CPU — simulated cache model | [`profilers/cachegrind.md`](profilers/cachegrind.md) |
| **IntelliKit** | GPU — decoded metrics / kernel isolation | [`profilers/intellikit.md`](profilers/intellikit.md) |
| **roofline-extractor** | GPU — percent-of-peak roofline | [`profilers/roofline-extractor.md`](profilers/roofline-extractor.md) |
| **rocBudAI** | GPU + host — AI assistant driving the stack | [`profilers/rocbudai.md`](profilers/rocbudai.md) |
| **perf_events security** | CPU — access-control reference | [`profilers/perf-security.md`](profilers/perf-security.md) |

The GPU-native tools (rocprofv3, rocprof-compute, rocprofiler-systems,
roofline-extractor, IntelliKit) target `CG-GPU`; the CPU tools (Linux `perf`,
Valgrind cachegrind, uProf; likwid where supported) target `CG-CPU`. **TAU** and
**HPCToolkit** are whole-application and work on either (they are the best way to
see the *MPI* communication that the ROCm GPU tools do not trace). **Score-P** also
captures MPI + GPU/HIP in one measurement. **rocBudAI** can drive any of them for you.

> **Verified on AAC6 / MI300A (ROCm 6.4.3, OpenMPI 5.0.10, 4 ranks, `Dubcova2.pm`).**
> Confirmed working: rocprofv3 (kernel + sys trace), rocprof-compute (roofline +
> Top-Kernels — SpMV `rocsparse::csrmvn` and the rocBLAS `axpy`/`dot`/`scal`
> kernels), rocprofiler-systems (Perfetto trace), TAU (MPI breakdown via `pprof`),
> HPCToolkit (`hpcrun` → `hpcstruct` → `hpcprof` database), AMD uProf (hotspots —
> top is the CSR SpMV), and Score-P (Cube profile + OTF2 trace, HIP 38% / MPI 21%).
> **Exception:** `likwid-perfctr` counters are **not supported** on the MI300A APU
> CPU in this build (see [likwid](profilers/likwid.md)).

## Ground rules (or your numbers are noise)

Everything from [chapter 3](03-correct-measurement.md) still applies under a
profiler — more so, because profiling adds overhead:

- Profile **inside an affinity-bound, `--exclusive` allocation** using
  `gpu_bind.sh` / `set_affinity_mi300a.sh`. Bad affinity dominates everything.
- **Fix the RHS** with `CG_SEED` so every profiled run solves the same system.
- Profile **one transport at a time** (`staged`, `isend`, `rccl`, `alltoallv`,
  `alltoallv_staged`, and the APU zero-copy `staged_unified` / `alltoallv_unified`
  with `HSA_XNACK=1`) and compare across runs.
- Give **each rank its own output directory** (use `$OMPI_COMM_WORLD_RANK`).

> **Launch note (verified on AAC6 / MI300A, ROCm 6.4.3, OpenMPI 5.0.10).** Use an
> `sbatch` job (or `salloc`) that requests GPUs, e.g.
> `sbatch -N1 --exclusive --gpus=4`. An `sbatch` script body runs *on the compute
> node*, which matters for the **CPU counter tools** (uProf) — they must execute
> where the ranks run, so launch them **per rank under `mpirun`**, not by wrapping
> `mpirun` itself. If `mpirun` complains about slots under a one-task step, add
> `--oversubscribe` (or `export OMPI_MCA_rmaps_base_oversubscribe=true`). The
> GPU-native tools (rocprofv3/rocprof-compute/rocprofiler-systems) run fine under
> `mpirun` and write per-rank output.

> **Tip — label the timeline.** For sharper GPU traces, wrap the halo and
> all-reduce regions in ROCTx ranges (`#include <roctracer/roctx.h>`;
> `roctxRangePush("halo")` / `roctxRangePop()`), then link `-lroctx64`. rocprofv3
> and rocprofiler-systems will show these named ranges next to the kernels.

> **`module load rocm/<ver>` pulls in the patched profilers.** The `rocm/<ver>`
> modulefile prepends `rocm-patches-<ver>/rocprof-compute/bin`, so `rocprof-compute`
> resolves to the self-contained **Nuitka single-file executable** (it bundles
> pandas/dash/matplotlib/…) rather than the plain `rocm-<ver>/bin` Python script
> that would fail on missing deps. If the site also ships a dedicated `rocm_patches`
> module, load it too. **Gotcha:** never pipe or command-substitute `module`
> (e.g. `module load rocm/7.2.3 | tail`) — a pipe runs the `module` function in a
> **subshell**, discarding its `eval`'d `PATH`/`ROCM_PATH`/`LD_LIBRARY_PATH` changes,
> after which `rocprof-compute` silently falls back to the unpatched script. Call
> `module` plainly. See the verified end-to-end driver
> [`CG-GPU/prof_all_723_test.sh`](../CG-GPU/prof_all_723_test.sh) and
> [chapter 9](09-perf-security-demo.md#do-the-rocm-723-profilers-still-work-at-paranoid2--yes-verified).

## The patched hipBLASLt (performance runs)

Some ROCm builds ship an optimised **`hipblaslt/patched`** module; load it for any
performance measurement. Behaviour differs by ROCm version — see the portable
snippet in the [profilers index](profilers/README.md#the-patched-hipblaslt-performance-runs).

## Mapping tools to the tutorial's questions

| Question | GPU (`CG-GPU`) | CPU (`CG-CPU`) |
|----------|----------------|----------------|
| Which transport is fastest for the halo exchange? | [rocprofv3](profilers/rocprofv3.md) `--sys-trace` + [rocprofiler-systems](profilers/rocprofiler-systems.md); [TAU](profilers/tau.md) MPI matrix; [Score-P](profilers/scorep.md) | [TAU](profilers/tau.md) / [HPCToolkit](profilers/hpctoolkit.md) MPI time |
| How costly is the dot-product all-reduce? | [TAU](profilers/tau.md) / [HPCToolkit](profilers/hpctoolkit.md) / [Score-P](profilers/scorep.md) `MPI_Allreduce` time | same |
| Is SpMV hitting the bandwidth ceiling? | [rocprof-compute](profilers/rocprof-compute.md) roofline; [roofline-extractor](profilers/roofline-extractor.md) % of peak; [IntelliKit `metrix`](profilers/intellikit.md) | [perf](profilers/perf.md) cache-miss rate + IPC; [uProf](profilers/uprof.md) memory |
| Which functions/lines cause cache misses? | [rocprofv3 ATT](profilers/rocprofv3.md#4-instruction-level-advanced-thread-trace-att) / [linex](profilers/intellikit.md) | [cachegrind](profilers/cachegrind.md) `cg_annotate` |
| How far from peak overall? | [roofline-extractor](profilers/roofline-extractor.md) % of peak | n/a |
| Want the workflow driven & explained end-to-end? | [rocBudAI](profilers/rocbudai.md) | [rocBudAI](profilers/rocbudai.md) |
| *Why* is SpMV memory-bound (which ISA lines stall)? | [rocprofv3 ATT](profilers/rocprofv3.md#4-instruction-level-advanced-thread-trace-att); [IntelliKit `linex`](profilers/intellikit.md) | n/a |
| Did my SpMV optimization stay correct / faster? | [IntelliKit `accordo` + `kerncap`](profilers/intellikit.md) | n/a |
| SDMA vs blit copies? | [rocprofiler-systems](profilers/rocprofiler-systems.md) timeline + `HSA_ENABLE_SDMA` sweep | n/a |
| Where does each rank stall (load imbalance)? | [HPCToolkit](profilers/hpctoolkit.md) call-path | [HPCToolkit](profilers/hpctoolkit.md) / [TAU](profilers/tau.md) |

> Flag names and event lists vary across releases. If a flag is rejected, check
> `rocprofv3 --help`, `rocprof-compute --help`, `rocprof-sys-run --help`,
> `tau_exec -help`, `hpcrun -L`, `likwid-perfctr -a`, and
> `AMDuProfCLI collect --help` on the loaded module.
