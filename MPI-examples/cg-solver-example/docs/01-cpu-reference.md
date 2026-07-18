# 1. The CPU reference (`CG-CPU/`)

Start here even if you only care about the GPU. The CPU version is small, has no GPU or math-library moving
parts, and establishes the algorithm and the distributed communication pattern that the GPU version inherits
unchanged.

## Build & run

```bash
cd CG-CPU
module load openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4   # any MPI with mpicxx
make
mpirun -n 4 ./cg_cpu src/Dubcova2.pm
```

Expected output:

```
172 iterations to converge
2-norm of residual: 0.000190
CG solve time:      0.1588 s  (0.0009 s/iter)
```

The iteration count and residual will wobble slightly run-to-run because the right-hand side `b` is built from
a **random** `x₀`. Hold that thought — reproducibility is the subject of Chapter 3.

## What's in `cg.cpp`

| function | role |
|---|---|
| `spmv(Mat&, ...)` | serial CSR sparse matrix–vector product (single block) |
| `spmv(ParMat&, ...)` | **distributed** SpMV: post receives, pack+send, on-proc SpMV, wait, off-proc SpMV |
| `inner_product` | local dot product + `MPI_Allreduce` |
| `axpy` / `scale` | `x += αy` / `x *= α` (local loops) |
| `main` | read matrix, form comm pattern, run CG |

## The distributed SpMV — the pattern that matters

This is the halo exchange, and its structure is identical in every GPU variant later:

```
1. Post non-blocking receives for ghost values   (MPI_Irecv)
2. Pack and post non-blocking sends              (loop + MPI_Isend)
3. Compute the on-proc SpMV                       (overlaps with in-flight messages)
4. Wait for receives                             (MPI_Waitall)
5. Compute the off-proc SpMV using the ghosts
6. Wait for sends                                (MPI_Waitall)
```

Two ideas are baked in here and carry through the whole tutorial:

- **The matrix is split into an on-proc block and an off-proc block.** The on-proc block only touches locally
  owned vector entries, so it can be computed *while messages are in flight*. The off-proc block needs the
  ghost values, so it runs after the receives complete. This on-proc/off-proc split is what makes
  communication/computation **overlap** possible.
- **The comm pattern is set up once.** `sparse_mat.hpp` (`form_comm`) figures out, per rank, exactly which
  indices to send to which neighbour and how many to receive. This setup is done on the CPU and is *identical*
  in `CG-GPU/` — only the data movement changes.

## The CG algorithm

```
r₀ = b − A x₀
p₀ = r₀
for i = 0, 1, 2, …:
    α  = (rᵢ, rᵢ) / (Apᵢ, pᵢ)      ← needs Apᵢ (SpMV) and two dot products
    x  += α pᵢ
    r  = b − A x                    ← residual recomputed every 8 iters for stability
    β  = (rᵢ₊₁, rᵢ₊₁) / (rᵢ, rᵢ)
    p  = r + β p
```

Per iteration: **1 SpMV** (halo exchange) + **2 dot products** (`MPI_Allreduce`) + a few local vector ops.
That fixed structure is the "denominator" of every measurement later — the work per iteration never changes,
so if the iteration count is held constant (Chapter 3), differences in solve time come only from the transport
and the hardware/software stack.

## Why keep the CPU version around?

- It is the **correctness oracle**: the GPU variants must converge to the same residual in the same number of
  iterations (for a fixed seed).
- It is a **baseline** for "how much did the GPU actually buy us?" — and a reminder that for a small per-rank
  problem the answer is subtle (the CPU solve here, ~0.16 s, is in the same ballpark as some *unoptimized* GPU
  launches, which is exactly why measurement discipline matters).

Next: [2. Porting to the GPU →](02-gpu-port.md)
