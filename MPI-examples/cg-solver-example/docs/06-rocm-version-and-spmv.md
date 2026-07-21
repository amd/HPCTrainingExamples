# 6. ROCm version & the SpMV API

The transport and copy-engine chapters optimize **communication**. But the comm/compute breakdown from
Chapter 3 has a second column — **compute** — and it turns out to be the dominant, and most surprising, part of
the story at this problem size. This chapter is a worked example of using the same measurement discipline to
chase down a *compute*-side regression, and of not being fooled by a plausible-sounding fix.

## Sweeping ROCm versions

Because the binary must be rebuilt per toolchain, the sweep harness rebuilds `cg_gpu` under each module and runs
one method with warm-up + repeats, all in one exclusive allocation (so results are directly comparable):

```bash
sbatch --export=ALL,VERSIONS="6.4.1 7.0.2 7.2.4 7.13.0",REPEATS=5,METHOD=isend \
       submit_rocm_sweep.sbatch
```

### Result: a compute regression at the 6.x → 7.x boundary

`isend`, 4 ranks, seed 12345, min of 5 runs (compute = solve − comm):

| ROCm | solve (s) | comm (s) | compute (s) |
|------|-----------|----------|-------------|
| 6.4.1  | 0.0480 | 0.0178 | **0.0302** (baseline) |
| 7.0.2  | 0.0717 | 0.0148 | 0.0569 |
| 7.2.4  | 0.0700 | 0.0141 | 0.0559 |
| 7.13.0 | 0.0713 | 0.0146 | 0.0567 |

Two things jump out:

- **Communication is as good or *better* on 7.x** (comm 0.018 → 0.014 s) — the MPI/UCX path did not regress.
- **Compute roughly doubled** (0.030 → 0.056 s) and is **flat across the whole 7.x line**, i.e. introduced at
  the 6.x→7.x boundary and not since recovered. Since compute now dominates, total solve time regressed ~1.5×.

This is only visible *because* the measurement splits comm from compute. A blind "solve time" comparison would
have wrongly implicated the communication stack.

## The plausible fix that isn't: `rocsparse_v2_spmv`

On ROCm 7.x the classic `rocsparse_spmv` is deprecated in favour of `rocsparse_v2_spmv` (a persistent SpMV
descriptor + an explicit one-time analysis stage). The obvious hypothesis: the regression is API/host overhead,
and the new API will fix it. The examples let you test that hypothesis directly:

```bash
make SPMV_V2=1     # compile the rocsparse_v2_spmv path (auto-falls back to v1 on ROCm 6.x)
# compare both paths across the 7.x line:
sbatch --export=ALL,SPMV="v1 v2",VERSIONS="6.4.1 7.0.2 7.2.4 7.13.0",REPEATS=5,METHOD=isend \
       submit_rocm_sweep.sbatch
```

`SPMV_V2=1` also folds in a second micro-optimization applied to *both* paths: the dense-vector descriptors are
created once and rebound with `rocsparse_dnvec_set_values` each iteration instead of being created/destroyed
per SpMV.

### Result: v2 does **not** recover the regression

compute (s), min of 5 runs:

| ROCm | v1 (`rocsparse_spmv`) | v2 (`rocsparse_v2_spmv`) |
|------|-----------------------|---------------------------|
| 6.4.1  | 0.0302 (baseline) | — (v2 not available) |
| 7.0.2  | 0.0569 | 0.0547 |
| 7.2.4  | 0.0559 | 0.0558 |
| 7.13.0 | 0.0567 | 0.0552 |

v2 improves 7.x compute by only **0–3 % (within noise)** and leaves the ~1.85× gap vs 6.4.1 intact. Since the
per-call descriptor fix is now in the v1 numbers too and v1 is still slow, **the regression is in the rocSPARSE
CSR SpMV compute kernel itself**, not in the API surface or host/descriptor overhead. (Migrating to
`rocsparse_v2_spmv` is still worthwhile for API longevity — v1 is deprecated — just not as a performance
remedy.)

The methodology lesson: **a controlled experiment can disprove a plausible fix.** Because both paths were
measured under identical conditions with the comm/compute split, the "it's the deprecated API" hypothesis was
cleanly falsified. The next step to localize it is a one-iteration `rocprofv3` kernel trace on 6.4.1 vs 7.2.4
(Chapter 7).

## Cross-toolchain: OpenMPI vs cray-mpich

The same solver was also built with **PrgEnv-amd / cray-mpich** (ROCm 7.0.3) and measured identically. `isend`,
4 ranks, same node class & matrix:

| stack | ROCm | solve | comm | halo | dot allreduce | compute |
|-------|------|-------|------|------|---------------|---------|
| OpenMPI 5.0.10 | 6.4.1 | 0.051 | 0.028 | 0.014 | 0.014 | 0.024 |
| OpenMPI 5.0.10 | 7.2.4 | 0.111 | 0.020 | — | 0.003 | 0.091 |
| cray-mpich 9.1.0 | 7.0.3 | 0.097 | 0.018 | 0.015 | 0.004 | 0.079 |

- **The MPI stack is not the differentiator here.** cray-mpich's GPU-Aware communication is on par with (a
  touch better than) GPU-Aware OpenMPI; the total-solve difference between rows is driven by the **ROCm
  compute version**, not the MPI implementation. ROCm 7.0.3 lands, as expected, between the 7.0.2 and 7.2.4
  compute points.
- **The affinity cliff did not reproduce on Cray** (Chapter 3): an `--exclusive` `srun` already binds
  NUMA-locally, so explicit binding bought only ~3–8 %. Same solver, different launcher defaults.

Full details: [`STUDY_REPORT.md`](../CG-GPU/STUDY_REPORT.md) §5/§5.1 (OpenMPI) and
[`STUDY_REPORT_PrgEnv-amd.md`](../CG-GPU/STUDY_REPORT_PrgEnv-amd.md) (cray-mpich).

## Takeaways

- Always split comm vs compute before blaming the network — here the network was fine and the *library kernel*
  regressed.
- Rebuild per toolchain and compare in one allocation; binaries are not runtime-compatible across major ROCm
  versions anyway (a 7.x-linked binary against a 6.x runtime fails with an undefined HSA symbol).
- A new API is not automatically a faster API — measure it the same way you measure everything else.

Next: [7. The optimized configuration →](07-optimization-path.md)
