# roofline-extractor — CG-GPU (percent-of-peak roofline)

> Part of the [CG profiler guides](README.md). Read the shared
> [ground rules](../08-profiling.md#ground-rules-or-your-numbers-are-noise) first.

`roofline-extractor` (module `roofline-extractor`, load after `rocm`) is an
automated **percent-of-peak** roofline analyser. It drives `rocprofv3` for you
(four counter passes + one kernel-trace pass), then reports, per kernel, arithmetic
intensity at each memory level (HBM/L2/L1/LDS), the active limiter,
achieved-vs-peak throughput, and % of roofline reached — with an optional
interactive HTML plot. It is the turnkey complement to
[rocprof-compute](rocprof-compute.md): less to drive, and it prints the headline
number directly.

## 1. Run

```bash
module load rocm
module load roofline-extractor
cd CG-GPU && make
# End-to-end: profiles cg_gpu and prints the per-kernel roofline breakdown.
roofline-extractor-profile -o rex_out -- ./cg_gpu src/Dubcova2.pm rccl 12345
```

Verified on MI300A (gfx942, ROCm 7.13.0) against `cg_gpu`:

```
<rocBLAS kernel>  contribution 10.7% GPU time
  Arithmetic intensity (HBM): 0.5752    limiter: HBM_BW (MI300A)
  Achieved 129.4 GFLOPS/s vs 2121 peak  →  6.1% of linear roofline
  ...
Average percent of linear roofline achieved: 17.2 %
Total application GPU time on MI300A: 7.48e6 ns
```

The low arithmetic intensity (~0.58 FLOP/byte) and the `HBM_BW` limiter are the
percent-of-peak restatement of the memory-bound result: CG lives on the bandwidth
roof.

> **Arch note.** The tool needs an MI-series arch it recognizes
> (MI250/MI300A/MI300X/MI325X/MI350X/…). On a **CPX**-mode node the virtual GPUs may
> not match a known arch string — run on **SPX**, or pin `--arch MI300A`.

## 2. Viewing the roofline remotely

Outputs (`counters.csv`, `*_EXTRACTED*.csv`, `counters.html`) land in `rex_out/`.
Open `counters.html` to see each kernel plotted on the roofline:

- `man aac6_vnc` — TurboVNC desktop, open `rex_out/counters.html` in a browser
- `man aac6_novnc` — the same desktop in your local browser
- `man aac6_x11` — `ssh -X` and open a single browser window
- or `scp` the self-contained HTML to your workstation.

For a multi-rank/phase view, collect per-run bundles and use
`roofline-extractor-extract -D <bundle_dir>`. See `man roofline-extractor` and the
`METRICS_SUMMARY.md` / `METRICS_DETAILED.md` under the install root.

## See also

- [rocprof-compute](rocprof-compute.md) — the interactive roofline GUI
- [IntelliKit `metrix`](intellikit.md) — decoded per-kernel bandwidth
