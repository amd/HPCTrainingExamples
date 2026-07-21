# CPU Conjugate Gradient — Reference Implementation

A distributed, CPU-only Conjugate Gradient (CG) solver.  This is the starting point for the tutorial: every operation is written in plain C++
with MPI, using only `std::vector` and hand-written loops.  No GPU, no external math libraries.

Note: This implementation was written and provided by Amanda Bienz, UNM. Her main repo is located at: https://github.com/bienz2/CG.

`CG-GPU/` is built directly on top of this implementation — comparing the two side by side shows exactly what changes when the solver is
ported to the GPU.

---

## File layout

```
CG-CPU/
├── Makefile
├── README.md
└── src/
    ├── cg.cpp            ← solver (the interesting file)
    ├── sparse_mat.hpp    ← CPU structs + MPI comm setup
    ├── par_binary_IO.hpp ← parallel binary matrix reader
    └── Dubcova2.pm       ← test matrix (65 536 × 65 536, SPD)
```

---

## Environment Setup

```bash
module load openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4
```

Any MPI build that provides `mpicxx` works; the solver has no GPU dependency.

---

## How to build

```bash
cd CG-CPU
make
```

---

## How to run

```bash
mpirun -n 4 ./cg_cpu src/Dubcova2.pm
```

Expected output:

```
172 iterations to converge
2-norm of residual: 0.000190
CG solve time:      0.1588 s  (0.0009 s/iter)
```

The iteration count, residual, and solve time vary slightly between runs because the right-hand side vector `b` is generated from a random
`x₀`. The solve time covers the CG loop only and is reported as the maximum across all MPI ranks.

---

## What's in `cg.cpp`

| Function | Description |
|---|---|
| `spmv(Mat&, ...)` | Serial sparse matrix–vector product (CSR row loop) |
| `spmv(ParMat&, ...)` | Distributed SpMV: non-blocking Isend/Irecv, then on-proc + off-proc SpMV |
| `inner_product` | Local dot product reduced across ranks with `MPI_Allreduce` |
| `axpy` | `x += alpha * y` (CPU loop) |
| `scale` | `x *= alpha` (CPU loop) |
| `main` | Reads the matrix, forms the comm pattern, runs CG |

### Distributed SpMV outline

```
1. Post non-blocking receives for ghost values  (MPI_Irecv)
2. Pack and post non-blocking sends             (CPU loop + MPI_Isend)
3. Compute on-proc SpMV                         (overlaps with in-flight messages)
4. Wait for receives to complete                (MPI_Waitall)
5. Compute off-proc SpMV using received ghosts
6. Wait for sends to complete                   (MPI_Waitall)
```

### CG algorithm

```
r₀ = b − A x₀
p₀ = r₀
for i = 0, 1, 2, …:
    α  = (rᵢ, rᵢ) / (Apᵢ, pᵢ)
    x  += α pᵢ
    r  = b − A x        (recomputed every 8 iterations for numerical stability)
    β  = (rᵢ₊₁, rᵢ₊₁) / (rᵢ, rᵢ)
    p  = r + β p
```

---

## What changes in `CG-GPU/`

| This file (CPU)                      | `CG-GPU/` (GPU)                                       |
|--------------------------------------|-------------------------------------------------------|
| `std::vector<double>` for vectors    | `hipMalloc` / `hipMemcpy` for device vectors          |
| Hand-written SpMV loop               | `rocsparse_spmv` (rocSPARSE generic API)              |
| Hand-written pack loop (CPU)         | `rocsparse_dgthr` (GPU gather)                        |
| Hand-written dot product             | `rocblas_ddot` (rocBLAS)                              |
| Hand-written axpy / scale            | `rocblas_daxpy` / `rocblas_dscal` (rocBLAS)           |
| `MPI_Irecv/Isend` on host buffers    | Choice of 5 GPU communication variants                |
