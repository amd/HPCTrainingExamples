# Figures for the FSDP2 profiler guides

These PNGs are **generated**, not authored by hand. Run
[`../submit_timeline_traces.sbatch`](../submit_timeline_traces.sbatch) on a
`--gpus>=4` node to produce them (and the large trace binaries they come from,
which are intentionally not committed). See each guide's *Capturing the
screenshots* section for the standalone command. All four below are **real
captures** from an MI300A / 4-GPU / fp32 run (ROCm 7.2.3, PyTorch 2.12).

| File | Guide | What it shows |
|------|-------|---------------|
| `fsdp2_torch_allgather.png` | [torch-profiler.md](../torch-profiler.md) | Perfetto (Kineto): the fused `ncclDevKernel_Generic_2` collective on stream 7, concurrent with GEMM on stream 4 |
| `fsdp2_rocprofv3_timeline.png` | [rocprofv3.md](../rocprofv3.md) / [rocprofiler-systems.md](../rocprofiler-systems.md) | Perfetto (`rocprofv3 --sys-trace`): the collective on STREAM 0 with host + memory-copy counter rows |
| `fsdp2_tb_overview.png` | [tensorboard.md](../tensorboard.md) | TensorBoard PyTorch-Profiler Overview: 4 workers, MI300A, step-time breakdown (comm reads ~0 — the fused name isn't recognized) |
| `fsdp2_tracelens_collective.png` | [tracelens.md](../tracelens.md) | TraceLens: `gpu_timeline` split, per-collective bus bandwidth, and skew-vs-latency (rendered from the CSVs) |

**Not committed (capture interactively):** a true `rocprofiler-systems` `.proto`
timeline (`fsdp2_rocsys_timeline.png`) and the TensorBoard *Distributed / Trace /
Kernel* views — headless capture of these is unreliable on this stack (see the
respective guides).
