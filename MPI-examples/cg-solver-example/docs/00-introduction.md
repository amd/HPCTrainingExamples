# 0. Introduction & setup

## The problem

Both examples solve the same linear system **A x = b** with the **Conjugate Gradient (CG)** method, where
`A` is a large sparse symmetric positive-definite (SPD) matrix. The test matrix `Dubcova2.pm` is
65 536 × 65 536 and SPD. The matrix is distributed across MPI ranks by rows; each rank owns a contiguous block
of rows and the corresponding slice of the solution vector.

CG is an iterative solver. Each iteration needs:

1. one **sparse matrix–vector product** (SpMV), `Ap` — this is where inter-rank **communication** happens,
   because a rank needs "ghost" values of `p` owned by its neighbours;
2. two **dot products** (`r·r` and `p·Ap`) — each requires an `MPI_Allreduce` across all ranks;
3. a few **vector updates** (`axpy`, `scal`) — purely local.

So a distributed CG iteration has exactly two kinds of communication:

- **halo exchange** (neighbour-to-neighbour) inside the SpMV, and
- **global reductions** (`MPI_Allreduce`) for the dot products.

This tutorial is about measuring and optimizing the **halo exchange** — the part where you get to *choose* the
transport — while keeping everything else fixed.

## The two examples

```
cg-solver-example/
├── CG-CPU/     reference: plain C++/MPI, std::vector, hand-written loops (no GPU)
└── CG-GPU/     GPU port: rocSPARSE / rocBLAS / RCCL + GPU-Aware MPI, 5 comm variants
```

`CG-GPU/` is a line-by-line evolution of `CG-CPU/`: same algorithm, same matrix, same MPI comm pattern setup.
Only the numerical kernels move to the GPU and the halo exchange gains four alternative implementations. That
shared lineage is what makes the comparison meaningful.

## The seven communication variants (preview)

| method | transport | buffers | collective? |
|--------|-----------|---------|-------------|
| `staged` | `MPI_Isend/Irecv` | **host** (D→H, H→D copies) | no |
| `isend` | `MPI_Isend/Irecv` | **GPU** (GPU-Aware MPI) | no |
| `staged_unified` | `MPI_Isend/Irecv` | **host `malloc`, 0 copies** (APU + XNACK) | no |
| `rccl` | `ncclSend/ncclRecv` | **GPU** (RCCL) | no |
| `alltoallv_staged` | `MPI_Alltoallv` | **host** | yes |
| `alltoallv` | `MPI_Alltoallv` | **GPU** (GPU-Aware MPI) | yes |
| `alltoallv_unified` | `MPI_Alltoallv` | **host `malloc`, 0 copies** (APU + XNACK) | yes |

All seven produce **identical numerical results** — they differ only in how ghost values move. The two
`*_unified` variants exploit the MI300A APU's single address space: MPI uses the **host** path on ordinary
`malloc`'d buffers (not GPU-Aware MPI), yet the GPU packs/reads them in place via XNACK, so there are **no
staging copies**. They require `HSA_XNACK=1` and an APU.

## Hardware & software prerequisites

**Hardware:** an AMD Instinct **MI300A** node (this tutorial's reference). MI300A is an APU: CPU + GPU share
HBM, and each of the 4 (or 8, depending on partitioning) GPU dies is local to one NUMA node. The affinity
lessons in Chapter 3 apply to any multi-GPU AMD node.

**Software:**

- **ROCm** ≥ 6.3 (provides `hipcc`, rocSPARSE, rocBLAS, RCCL). This tutorial shows results across 6.4.1–7.13.
- A **GPU-Aware MPI**: either OpenMPI/UCX built `--with-rocm`, or HPE cray-mpich with `craype-accel-amd-gfx942`.
- A batch scheduler (SLURM) for reproducible, exclusive allocations.

Two module setups appear throughout:

```bash
# OpenMPI + ROCm (AAC6 / MI300A Ubuntu)
module load rocm/6.4.1
module load openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4

# HPE Cray EX + cray-mpich (PrgEnv-amd)
module swap PrgEnv-cray PrgEnv-amd/8.7.0
module load craype-accel-amd-gfx942
```

The `CG-GPU/Makefile` auto-detects which of these you are in (it keys off the Cray `PE_ENV`), so `make` works
in both without edits.

## What "done" looks like

By the end you will be able to produce a table like this — and, more importantly, **trust it**:

| method | solve (s) | comm (s) | compute (s) | comm % |
|--------|-----------|----------|-------------|--------|
| staged            | 0.148 | 0.124 | 0.023 | 84 % |
| isend             | 0.051 | 0.028 | 0.023 | 55 % |
| rccl              | 0.051 | 0.025 | 0.025 | 50 % |
| alltoallv_staged  | 0.082 | 0.056 | 0.026 | 68 % |
| alltoallv         | 0.051 | 0.025 | 0.026 | 49 % |

*(AAC6, ROCm 6.4.1, 4 ranks, `CG_SEED=12345`. Your absolute numbers will differ; the methodology is what
transfers.)*

Next: [1. The CPU reference →](01-cpu-reference.md)
