# CG solver — per-profiler guides

This directory holds **one step-by-step guide per profiler** for the CG solver
example (`CG-GPU` and `CG-CPU`). It refines the single combined chapter
[`../08-profiling.md`](../08-profiling.md) into focused, copy-pasteable walkthroughs,
each with: setup/module load, an instrumented run, the **text output** you should
expect, and how to view any **graphics** in a remote session
(VNC / noVNC / X11 — see [`../../../../docs/graphics/README.md`](#viewing-graphics-remotely)).

The combined [`../08-profiling.md`](../08-profiling.md) is now a **hub**: it keeps
the shared *Which profiler* / *Ground rules* / *Mapping* material and links here.

> **Known issues & action items:** see [`../PROFILER_ISSUES.md`](../PROFILER_ISSUES.md)
> for the profiler problems encountered (Score-P GPU on ROCm 7.2.x, `cube_dump`,
> CubeGUI viewers/headless graphics, hipBLASLt) with workarounds and what still
> needs to be addressed.

## Guides

| Profiler | Target | Scope | Page |
|----------|--------|-------|------|
| **Score-P** | `CG-GPU` / `CG-CPU` | MPI (+GPU/HIP) tracing → Cube/OTF2 | [`scorep.md`](scorep.md) |
| **rocprofv3** | `CG-GPU` | Per-kernel + RCCL/HIP + ATT | [`rocprofv3.md`](rocprofv3.md) |
| **rocprof-compute** | `CG-GPU` | Roofline (HBM-bound SpMV) | [`rocprof-compute.md`](rocprof-compute.md) |
| **rocprofiler-systems** | `CG-GPU` + host | Perfetto timeline / overlap | [`rocprofiler-systems.md`](rocprofiler-systems.md) |
| **TAU** | either | MPI comm view | [`tau.md`](tau.md) |
| **HPCToolkit** | either | Call-path sampling | [`hpctoolkit.md`](hpctoolkit.md) |
| **likwid** | `CG-CPU` | CPU roofline | [`likwid.md`](likwid.md) |
| **AMD uProf** | `CG-CPU` | CPU hotspots | [`uprof.md`](uprof.md) |
| **Linux perf** | `CG-CPU` | HW counters baseline | [`perf.md`](perf.md) |
| **Valgrind cachegrind** | `CG-CPU` | Simulated cache model | [`cachegrind.md`](cachegrind.md) |
| **IntelliKit** | `CG-GPU` | Decoded metrics / kernel isolation | [`intellikit.md`](intellikit.md) |
| **roofline-extractor** | `CG-GPU` | Percent-of-peak roofline | [`roofline-extractor.md`](roofline-extractor.md) |
| **rocBudAI** | `CG-GPU` + host | AI assistant driving the stack | [`rocbudai.md`](rocbudai.md) |
| **perf_events security** | `CG-CPU` | Access-control reference | [`perf-security.md`](perf-security.md) |

See also [`../09-perf-security-demo.md`](../09-perf-security-demo.md) — the runnable
perf-security demo.

## The patched hipBLASLt (performance runs)

Some ROCm builds ship an optimised **`hipblaslt/patched`** module. Load it for
any performance measurement so GEMM-heavy kernels use the tuned library. Behaviour
differs by ROCm version, so use this portable snippet:

```bash
module load rocm/<version>
# Load the tuned hipBLASLt if this ROCm build provides one.
if module avail hipblaslt/patched 2>&1 | grep -q 'hipblaslt/patched'; then
  module load hipblaslt/patched
fi
module list 2>&1 | grep -q 'hipblaslt/patched' \
  && echo '[ok] hipblaslt/patched active' \
  || echo '[info] no patched hipBLASLt for this ROCm build'
```

Observed on AAC6 / MI300A: ROCm **7.2.0/7.2.3/7.2.4** auto-load it with the `rocm`
module; ROCm **7.13.0** has it but needs the explicit load above; ROCm **6.4.x /
7.0.x** do not provide it.

## Viewing graphics remotely

Cube/Perfetto/roofline GUIs and rendered PNG figures are viewed inside a remote
graphical session on AAC6. See the cluster man pages and the shared graphics guide:

- `man aac6_vnc` — TurboVNC desktop
- `man aac6_novnc` — browser (noVNC) desktop
- `man aac6_x11` — X11 forwarding for a single GUI window
