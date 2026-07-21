# rocBudAI — CG-GPU (AI profiling assistant)

> Part of the [CG profiler guides](README.md). Read the shared
> [ground rules](../08-profiling.md#ground-rules-or-your-numbers-are-noise) first.

`rocBudAI` (module `rocbudai`) is an AI performance-engineering assistant that
**drives the whole stack for you**. It wires the OpenCode TUI to a locally-hosted
model (Ollama) pre-seeded with the AMD profiling toolchain (rocprofv3,
rocprof-compute, rocprof-sys, rocpd, rocm-smi), then runs a
build → profile → analyse → optimise → re-profile loop, writing a running
`report.md` whose claims are tagged `[FACT]` / `[INFERENCE]` / `[OPINION]`.

It runs **only on a Slurm-allocated compute node**, and the `--comment=ollama`
allocation is air-gapped (source, prompts, and model output stay on the node):

```bash
# 1. Allocate an SPX MI300A node with the local model daemon started for you:
salloc -p PPAC_MI300A_SPX --exclusive --comment=ollama --time=02:00:00
# 2. Point it at this example:
cd ~/HPCTrainingExamples/MPI-examples/cg-solver-example/CG-GPU
# 3. Launch the assistant (auto-starts the TUI, pre-configured for AMD profiling):
module load rocbudai
```

A seven-question interview (app type, ROCm version, modules, build command, run
command, figure of merit) configures the run. For this example answer roughly:
*HIP/C++ + MPI*; `rocm/7.13.0` + `openmpi`; build `make`; run
`mpirun -n 4 ./gpu_bind.sh ./cg_gpu src/Dubcova2.pm rccl`; FoM = *CG solve time*
(or *% of roofline*). The agent then profiles with the tools in these guides and
proposes optimizations. Use it when you want the workflow orchestrated and explained
rather than run by hand — but still check its conclusions against the raw tool
output.

See `man rocbudai` for airgap enforcement, session management, and ASK vs AUTO-RUN
permission modes; verify the sandbox with `rocbudai-airgap-check`.

## Viewing the results

rocBudAI runs in a terminal TUI and writes a text `report.md`; no graphical session
is required. Any plots it produces via the underlying tools
([rocprof-compute](rocprof-compute.md), [roofline-extractor](roofline-extractor.md))
are viewed as described on those pages.

## See also

- All the tools it drives: [rocprofv3](rocprofv3.md),
  [rocprof-compute](rocprof-compute.md), [rocprofiler-systems](rocprofiler-systems.md)
