# 7. The optimized configuration

You've now measured every axis. This chapter assembles them into a recommended setup, a decision checklist, and
the directions worth pursuing next.

## The optimization path at a glance

Performance progression as each optimization is applied (method = `isend`, 4 ranks, `Dubcova2.pm`,
`CG_SEED=12345`, min of runs on one MI300A node):

| Step | What changed | Config | Solve (s) | vs prev | vs naive GPU |
|------|--------------|--------|-----------|---------|--------------|
| 0 | *CPU reference (baseline)* | `CG-CPU`, 4 ranks | 0.159 | — | — |
| 1 | Naive GPU: default transport, **no affinity** | `staged`, unbound/non-exclusive | 9.27 | — | 1× |
| 2 | **Fix CPU/GPU affinity** (exclusive + `gpu_bind.sh`) | `staged`, bound | 0.148 | **63×** | 63× |
| 3 | **Switch to GPU-Aware transport** | `isend`, bound | 0.051 | 2.9× | 182× |
| 4 | **Enable SDMA + gang copy engines** | `isend`, bound, `HSA_ENABLE_SDMA_GANG=1` | 0.045 | 1.13× | **206×** |

The optimized GPU run (0.045 s) is ~206× faster than the naive GPU run and ~3.5× faster than the CPU reference
— and **~90 % of that gain is affinity alone** (step 2). Steps 1–3 are ROCm 6.4.1; step 4 is 6.4.3.

Per-axis detail behind each step:

**Step 2 — affinity** (solve time, bound vs unbound):

| method | unbound, not exclusive | `--exclusive` + `gpu_bind.sh` | speedup |
|--------|------------------------|-------------------------------|---------|
| staged     | 9.27 s  | 0.15 s | 62× |
| isend      | 10.58 s | 0.05 s | 212× |
| rccl       | 7.39 s  | 0.05 s | 148× |
| alltoallv  | 8.46 s  | 0.05 s | 169× |

**Step 3 — transport** (bound, ROCm 6.4.1):

| method | solve | comm | halo exch | dot allreduce | compute | comm % |
|--------|-------|------|-----------|---------------|---------|--------|
| staged           | 0.1477 | 0.1243 | 0.1095 | 0.0148 | 0.0234 | 84 % |
| isend            | 0.0513 | 0.0282 | 0.0142 | 0.0140 | 0.0231 | 55 % |
| rccl             | 0.0505 | 0.0252 | 0.0160 | 0.0092 | 0.0253 | 50 % |
| alltoallv_staged | 0.0823 | 0.0560 | 0.0422 | 0.0138 | 0.0263 | 68 % |
| alltoallv        | 0.0512 | 0.0250 | 0.0129 | 0.0121 | 0.0262 | 49 % |

**Step 4 — SDMA vs blit** (ROCm 6.4.3, halo-exchange time):

| method | blit | sdma | sdma_gang | best Δ vs blit |
|--------|------|------|-----------|----------------|
| staged           | 0.1215 | 0.1116 | 0.1129 | −8 % |
| isend            | 0.0154 | 0.0119 | 0.0113 | **−27 %** |
| rccl             | 0.0143 | 0.0129 | 0.0129 | −10 % |
| alltoallv_staged | 0.0480 | 0.0418 | 0.0394 | −18 % |
| alltoallv        | 0.0116 | 0.0140 | 0.0121 | +21 % |

**Separate axis — ROCm version / SpMV API** (`compute` time, `isend`). Not a speedup you "apply" but a
version choice: staying on ROCm 6.4.x avoids a ~1.85× compute regression that `rocsparse_v2_spmv` does *not*
recover:

| ROCm | v1 `rocsparse_spmv` | v2 `rocsparse_v2_spmv` |
|------|---------------------|-------------------------|
| 6.4.1  | 0.0302 (baseline) | — |
| 7.0.2  | 0.0569 | 0.0547 |
| 7.2.4  | 0.0559 | 0.0558 |
| 7.13.0 | 0.0567 | 0.0552 |

Caveats: single node / single matrix / latency-bound regime; the cumulative path mixes ROCm 6.4.1 (steps 2–3)
and 6.4.3 (step 4); for the overlap variants the `halo` figure is exposed wait, not raw wire time.

## The recommended launch

For a distributed GPU CG solve on MI300A, in priority order:

1. **Exclusive allocation** that owns whole NUMA nodes (`sbatch --exclusive`).
2. **Per-rank GPU + NUMA-local CPU binding** (`gpu_bind.sh` or `set_affinity_mi300a.sh`, launched under
   `--bind-to none` / `--cpu-bind=none`). *This is worth up to ~100× — nothing else comes close.*
3. **A GPU-Aware transport** — `isend`, `alltoallv`, or `rccl`. Avoid the `staged` variants unless you have no
   GPU-Aware MPI; the staging copies cost ~7× the halo time. **If you lack GPU-Aware MPI but are on an APU
   (MI300A), use `staged_unified` / `alltoallv_unified` with `HSA_XNACK=1`** — they drop the staging copies via
   the single address space and recover most of the win over `staged` while staying on the host MPI path.
4. **SDMA copy engines on** (the default), with `HSA_ENABLE_SDMA_GANG=1` if halo volume is large.
5. **The best available ROCm compute version** — on this workload ROCm 6.4.x compute is ~1.85× faster than
   7.x; check whether that still holds on your version before locking it in.

```bash
cd CG-GPU
make                       # (or: make SPMV_V2=1 on ROCm 7.x for API longevity)

CG_SEED=12345 \
HSA_ENABLE_SDMA=1 HSA_ENABLE_SDMA_GANG=1 \
mpirun -n 4 --bind-to none ./gpu_bind.sh ./cg_gpu src/Dubcova2.pm isend
```

Reproduce any of the studies end-to-end:

```bash
sbatch submit_cg_gpu_test.sbatch                                  # all 5 methods + convergence + timing
sbatch --export=ALL,REPEATS=5 submit_sdma_sweep.sbatch           # SDMA vs blit, all methods
sbatch --export=ALL,SPMV="v1 v2",REPEATS=5 submit_rocm_sweep.sbatch  # ROCm versions × SpMV API
sbatch --export=ALL,CG_SEED=12345 submit_cg_gpu_cray.sbatch      # PrgEnv-amd / cray-mpich
```

## Decision checklist: "which transport should I use?"

```
Do you have GPU-Aware MPI?
├─ no  → staged / alltoallv_staged (host-buffered) are your only correct options.
│         Expect halo to be dominated by D↔H copies. Turn SDMA on.
└─ yes → use a GPU-resident transport:
         ├─ mostly neighbour exchange, few peers  → isend
         ├─ dense/irregular all-to-all pattern    → alltoallv
         └─ want GPU-native, hipGraph-friendly,    → rccl
            or plan to overlap on a side stream
         (On one node over XGMI these are within noise — pick on ergonomics,
          then measure on your real pattern/scale.)
```

## Where the time actually goes — and what to optimize next

Once affinity is fixed and a GPU-Aware transport is chosen, the remaining per-iteration cost splits into three
buckets. Attack them in ROI order:

### A. Dot-product synchronization (highest ROI)

Each iteration does **two** `MPI_Allreduce`s, each gated by a full `hipDeviceSynchronize`. That's up to ~27 % of
solve on this problem and it is *independent of the halo transport*, so no amount of transport tuning touches
it. Options:

- **Fuse the two dot products** (`r·r` and `p·Ap`) into a single allreduce where the algorithm allows.
- Adopt a **pipelined / communication-avoiding CG** (Ghysels & Vanroose pipelined CG, or s-step CG) to overlap
  the global reduction with the SpMV and cut the number of synchronization points.
- Replace `hipDeviceSynchronize` with **stream/event-scoped** sync so unrelated GPU work isn't stalled.

### B. Kernel-launch overhead (small per-GPU problems)

At ~16k rows/GPU the loop is launch-latency bound. Capture the per-iteration kernel/BLAS sequence in a **HIP
graph**, or fuse the rocBLAS `axpy`/`scal`/`dot` chain into custom kernels, to cut launch count.

### C. SpMV / comm overlap

- In `rccl`, replace the full `hipDeviceSynchronize` after the gather with an **event + `hipStreamWaitEvent`**
  so only the RCCL stream waits.
- For the point-to-point variants, split the on-proc SpMV to better hide `MPI_Waitall` latency.

### D. The compute-kernel regression (Chapter 6)

If you're on ROCm 7.x and compute-bound, the biggest single factor may be the rocSPARSE CSR SpMV regression.
Localize it with a **one-iteration `rocprofv3` kernel trace** on 6.4.1 vs your version to identify the kernel
whose duration grew, try `rocsparse_spmv_alg_csr_rowsplit` for this near-uniform-nnz matrix, and file an issue
with the reproducer if confirmed.

## Analysis directions to deepen the study

1. **Strong & weak scaling** — sweep 1→4→8 GPUs, then multi-node, to expose intra-node (XGMI) vs inter-node
   behaviour, where `rccl` and GPU-Aware `alltoallv` should start to diverge.
2. **Problem-size sweep** — larger matrices per GPU shift the balance from latency-bound to bandwidth/compute
   bound and shrink the exposed halo of the overlap variants; report a compute-bound regime to complement this
   latency-bound one.
3. **Raw vs exposed communication** — add non-overlapped timing or profiler traces to separate wire time from
   exposed wait for the overlap variants.
4. **Profiler-level attribution** — `rocprofv3` / `rocprofiler-systems` to attribute time to SpMV, dot,
   allreduce, and copies, and to confirm the NUMA-locality benefit at the hardware-counter level.
5. **Communication-volume model** — correlate measured halo time with actual per-rank ghost byte counts (from
   `form_comm`) to build a latency + bandwidth model and predict behaviour at scale.

## The one-paragraph summary

The path to an optimized, *comparable* performance measurement on AMD multi-GPU nodes is: **bind first**
(affinity dominates everything), **fix the seed and take warm minimums** (so runs are comparable), **split comm
from compute** (so you optimize the right thing), then compare **transport** (use GPU-Aware MPI; the GPU-resident
variants tie on one node), **copy engine** (SDMA on, gang for big halos), and **toolchain/version** (watch for
compute-side regressions the network numbers won't reveal). The examples in this repo are instrumented and
scripted so that each of these is a controlled, reproducible experiment rather than a guess.

← Back to the [tutorial index](README.md) · In-depth data:
[`STUDY_REPORT.md`](../CG-GPU/STUDY_REPORT.md) ·
[`STUDY_REPORT_PrgEnv-amd.md`](../CG-GPU/STUDY_REPORT_PrgEnv-amd.md)
