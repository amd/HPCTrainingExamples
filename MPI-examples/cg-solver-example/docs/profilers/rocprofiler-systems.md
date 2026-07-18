# rocprofiler-systems — CG-GPU (timeline of overlap)

> Part of the [CG profiler guides](README.md). Read the shared
> [ground rules](../08-profiling.md#ground-rules-or-your-numbers-are-noise) first.

`rocprofiler-systems` (formerly Omnitrace) records a **Perfetto timeline** across
the host thread, the GPU-kernel stream, and the memory-copy engines — the best way
to *see* halo-exchange overlap and SDMA-vs-blit copies.

## 1. Collect a per-rank trace

```bash
module load rocprofiler-systems
cd CG-GPU && make
CG_SEED=12345 mpirun -n 4 ./gpu_bind.sh \
  rocprof-sys-run -- ./cg_gpu src/Dubcova2.pm rccl
```

This writes a per-rank Perfetto trace. Line up the GPU-kernel row, the memory-copy
row, and the host row to see:

- the **on-proc SpMV overlapping** the in-flight halo messages (the "exposed wait"
  the README mentions for `isend`/`rccl`/`alltoallv`),
- **SDMA vs blit** copies for GPU-Aware transports — rerun with
  `HSA_ENABLE_SDMA=0` (blit) vs `=1` (SDMA) and compare the copy rows, tying the
  timeline back to the [SDMA vs blit chapter](../05-sdma-vs-blit.md).

> **Tip — label the timeline.** Wrap the halo and all-reduce regions in ROCTx
> ranges (`#include <roctracer/roctx.h>`; `roctxRangePush("halo")` /
> `roctxRangePop()`), then link `-lroctx64`. The named ranges then appear next to
> the kernels in the trace.

## 2. Viewing the timeline remotely

Open the resulting `*.proto` / `perfetto-trace` at <https://ui.perfetto.dev>. Since
the trace is on the cluster, run a browser inside a graphical session:

- `man aac6_vnc` — TurboVNC desktop, open the trace in Firefox/Chromium
- `man aac6_novnc` — the same desktop in your local browser
- `man aac6_x11` — `ssh -X` and launch a single browser window

Drag the `.proto` file onto the Perfetto UI; zoom to one CG iteration and align the
kernel, copy, and host rows.

## See also

- [rocprofv3](rocprofv3.md) `--sys-trace` — a lighter per-process Perfetto trace
- [rocprof-compute](rocprof-compute.md) — the roofline for the kernels you see here
