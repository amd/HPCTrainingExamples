# IntelliKit — CG-GPU (metrix / kerncap / accordo / linex)

> Part of the [CG profiler guides](README.md). Read the shared
> [ground rules](../08-profiling.md#ground-rules-or-your-numbers-are-noise) first.

[IntelliKit](https://github.com/AMDResearch/intellikit) (module `intellikit/main`,
loaded **after** a `rocm` module) is an "agent-first" ROCm profiling and validation
toolkit built on rocprofiler-sdk. Where the raw tools emit counters and traces,
IntelliKit gives **decoded, human-readable** metrics and **kernel
isolation/validation** — and every tool also ships as an MCP server so an AI agent
can drive it. Installed: `metrix`, `kerncap`, `accordo` (CLIs); `linex`, `nexus`
(SQTT / HSA, MCP-driven); `rocm_mcp`, `uprof_mcp` (MCP servers).

> **Invocation note (this build).** Only `accordo` is installed as a `bin` script.
> Invoke the others through the module's Python entrypoints:
>
> ```bash
> module load rocm/7.2.4 intellikit/main
> kerncap(){ python -c "from kerncap.cli import main; main()" "$@"; }
> metrix(){  python -c "from metrix.cli.main import main; main()" "$@"; }
> # accordo is on PATH directly
> ```

## metrix — GPU profiling, *decoded*

```bash
# Decoded HBM/L2 bandwidth for just the SpMV kernel, top 5 dispatches
metrix profile -p memory_bandwidth -k 'csrmv' --top 5 \
  "./cg_gpu src/Dubcova2.pm rccl 12345"
# profiles: quick | memory | memory_bandwidth | memory_cache | compute
```

Verified on MI300A (ROCm 7.2.4) against the `rocsparse::csrmvn` SpMV:

```
Dispatch #1420: csrmvn_general_kernel   Duration avg=8.84 μs
  HBM Bandwidth Utilization      4.75 Percent
  HBM Read Bandwidth           245.55 GB/s
  L2 Cache Bandwidth Util.     668.79 GB/s
```

For this small test matrix the SpMV is largely **L2-resident** — L2 bandwidth
(~669 GB/s) far exceeds HBM (~245 GB/s). Scale the matrix up and the working set
spills to HBM, pushing the kernel onto the HBM roof — the decoded restatement of the
[rocprof-compute roofline](rocprof-compute.md).

## kerncap — isolate the SpMV kernel

```bash
kerncap profile -- ./cg_gpu src/Dubcova2.pm rccl 12345      # rank kernels by time
kerncap extract csrmvn_general --cmd './cg_gpu src/Dubcova2.pm rccl 12345' \
  --language hip -o spmv_repro/                              # standalone reproducer
kerncap replay  spmv_repro/                                 # VA-faithful re-dispatch
kerncap validate spmv_repro/                                # outputs match capture?
```

`kerncap profile` ranks the CG kernels by time — the `rocsparse::csrmvn` SpMV plus
the rocBLAS `axpy`/`dot`/`scal` and rocclr fill/copy buffers.

## accordo — correctness validation of an optimization

```bash
accordo validate --kernel-name csrmvn_general_kernel \
  --ref-binary ./cg_gpu_ref --opt-binary ./cg_gpu_opt \
  --atol 1e-8 --rtol 1e-5
```

The guard-rail for the optimization path in
[chapter 7](../07-optimization-path.md): prove a faster kernel still computes the
same result before trusting its speedup.

## linex / nexus and the MCP servers

- **`linex`** — *source-level SQTT*: maps GPU cycles back to your **source lines**;
  the decoded companion to the raw ATT/SQTT trace in
  [rocprofv3 §4](rocprofv3.md#4-instruction-level-advanced-thread-trace-att). Needs
  the same `rocprof-trace-decoder` library.
- **`nexus`** — HSA packet / kernel source extractor.
- **MCP servers** — `linex-mcp`, `nexus-mcp`, `kerncap-mcp`, `metrix-mcp`,
  `accordo-mcp`, plus `rocm_mcp` and `uprof_mcp` — expose the tools to an MCP client
  so an agent can profile, extract, and validate CG kernels conversationally.

> **Verified on PPAC / MI300A (gfx942, ROCm 7.2.4, `intellikit/main`).** `metrix
> profile` and `kerncap profile`/`extract` run against `cg_gpu` using
> rocprofiler-sdk counters directly (no extra decoder); `linex` source mapping needs
> the `rocprof-trace-decoder` library.

## Viewing the results

`metrix`/`kerncap` output is text; `linex` source mapping and the MCP servers are
driven from an editor/agent. No standalone GUI is required. For remote graphical
work (e.g. an MCP-enabled editor), use `man aac6_vnc` / `man aac6_novnc` /
`man aac6_x11`.

## See also

- [rocprof-compute](rocprof-compute.md) / [roofline-extractor](roofline-extractor.md) — plotted roofline
- [rocprofv3 ATT](rocprofv3.md#4-instruction-level-advanced-thread-trace-att) — the raw SQTT trace behind `linex`
