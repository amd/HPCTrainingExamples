# 5. Tuning the copy engines (SDMA vs blit kernels)

Once you've chosen a GPU-Aware transport, there's a lower-level knob that changes *how* ROCm physically moves
bytes between GPU memory and the fabric — and it can shift communication time by 20–30 % with **zero code
changes**.

## The two mechanisms

When GPU-Aware MPI (or the staging `hipMemcpy`) moves data, ROCm can use either:

| mechanism | `HSA_ENABLE_SDMA` | how it works |
|---|---|---|
| **SDMA engines** | `1` (default) | dedicated hardware DMA controllers move data independently of the shader engines — no compute resources consumed |
| **Blit kernels** | `0` | ROCm dispatches a small compute shader ("blit") that copies using the shader engines — no dedicated DMA hardware |

A second flag gangs multiple SDMA engines onto one transfer:

```bash
export HSA_ENABLE_SDMA=1
export HSA_ENABLE_SDMA_GANG=1
```

Because this is purely a runtime setting, **the binary is identical across configurations** — you only change
the environment. That makes it a clean controlled experiment: anything that moves is the copy path, and
**compute should stay flat** (a built-in sanity check).

## Run the sweep

Switch it by hand:

```bash
HSA_ENABLE_SDMA=1 mpirun -n 4 ./gpu_bind.sh ./cg_gpu src/Dubcova2.pm isend   # SDMA (default)
HSA_ENABLE_SDMA=0 mpirun -n 4 ./gpu_bind.sh ./cg_gpu src/Dubcova2.pm isend   # blit kernels
```

…or use the sweep harness (builds once, sweeps blit / sdma / sdma+gang across all methods, min of `REPEATS`):

```bash
sbatch --export=ALL,ROCM_VER=6.4.3,REPEATS=5 submit_sdma_sweep.sbatch
```

The maintainer's `run_test_7.13.sh` sweeps `HSA_ENABLE_SDMA` ∈ {1,0} for the GPU-Aware variants (`isend`,
`alltoallv`) and labels each run `(sdma)` / `(blit_kernel)` in the log.

## Results (ROCm 6.4.3, 4 ranks, seed 12345, min of 5 runs)

Halo-exchange time and its change vs. blit:

| method | blit | sdma | sdma_gang | halo Δ vs blit |
|--------|------|------|-----------|----------------|
| staged           | 0.1215 | 0.1116 | 0.1129 | **−8 %** |
| isend            | 0.0154 | 0.0119 | 0.0113 | **−27 %** |
| rccl             | 0.0143 | 0.0129 | 0.0129 | −10 % |
| alltoallv_staged | 0.0480 | 0.0418 | 0.0394 | **−18 %** |
| alltoallv        | 0.0116 | 0.0140 | 0.0121 | +21 % (see below) |

*(full solve/comm/compute table in [`STUDY_REPORT.md`](../CG-GPU/STUDY_REPORT.md) §5.2. The Cray/ROCm 7.0.3
sweep in [`STUDY_REPORT_PrgEnv-amd.md`](../CG-GPU/STUDY_REPORT_PrgEnv-amd.md) §5 shows the same pattern:
`isend` −24 %, `alltoallv` −23 %.)*

## How to read it

- **SDMA helps the copy-heavy paths**, exactly as expected. Disabling it (blit) is slower on `staged` (−8 %),
  `alltoallv_staged` (−13…−18 %), and GPU-Aware `isend` (−23…−27 %). Blit copies burn CU/shader cycles and
  contend with the SpMV kernels; SDMA offloads them to dedicated hardware.
- **Ganging adds a small extra win** on the copy-bound variants (`isend` −23 %→−27 %, `alltoallv_staged`
  −13 %→−18 %) and is within noise elsewhere.
- **Compute stays flat** across all three settings — the effect is cleanly isolated to data transport, which is
  the confirmation that you're measuring the copy path and not something else.
- **The one exception** (`alltoallv` marginally *better* with blit) shows the general rule: for very small,
  latency-sensitive transfers, the SDMA engine's start-up cost can exceed a quick shader copy, especially on
  MI300A's unified memory. The effect is small and near the noise floor.

## When does each win?

- **SDMA** wins for **larger / copy-heavy** transfers where keeping the shader engines free to overlap with
  compute matters more than raw copy latency.
- **Blit kernels** can win for **small, latency-sensitive** transfers where the shaders are otherwise idle and
  the lower dispatch overhead beats the DMA engine's start-up.

The right choice is workload- and hardware-dependent, which is the real lesson: **it's a knob to sweep, not a
constant to assume.** The default (`HSA_ENABLE_SDMA=1`) is the right starting point; forcing blit costs up to
~25 % of halo time on the staged and GPU-Aware point-to-point paths. Turn on `HSA_ENABLE_SDMA_GANG=1` when the
halo volume is large.

Next: [6. ROCm version & the SpMV API →](06-rocm-version-and-spmv.md)
