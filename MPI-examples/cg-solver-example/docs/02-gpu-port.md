# 2. Porting to the GPU (`CG-GPU/`)

`CG-GPU/` keeps the algorithm and the MPI comm-pattern setup from `CG-CPU/` and moves every numerical operation
to the GPU. The two headers (`sparse_mat.hpp`, `par_binary_IO.hpp`) are the same — file I/O and comm setup stay
on the host.

## What changed

| `CG-CPU/` | `CG-GPU/` |
|---|---|
| `std::vector<double>` | `hipMalloc` / `hipMemcpy` device arrays |
| hand-written SpMV loop | `rocsparse_spmv` (rocSPARSE generic API) |
| hand-written pack/gather | `rocsparse_dgthr` (GPU gather) |
| hand-written dot | `rocblas_ddot` |
| hand-written axpy/scal | `rocblas_daxpy` / `rocblas_dscal` |
| `MPI_Isend/Irecv` on host buffers | **choice of 5 communication variants** |

## Build & run

```bash
cd CG-GPU
module load rocm/6.4.1
module load openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4
make
mpirun -n 4 ./cg_gpu src/Dubcova2.pm isend
```

The `Makefile` auto-detects the toolchain (OpenMPI+ROCm vs Cray PE) and the GPU arch. Useful knobs:

```bash
make GPU_ARCH=gfx90a   # target a specific arch (default: auto-detected, fallback gfx942)
make SPMV_V2=1         # use rocsparse_v2_spmv (ROCm 7.x); see Chapter 6
make run               # quick 4-rank sanity run over four methods
```

A first run prints something like:

```
method=isend  ranks=4  gpus_visible=1
RHS seed: 12345  (CG_SEED)
Initial residual: 195.873
174 iterations to converge
2-norm of residual: 0.000188555
CG solve time:      0.0513 s  (0.0003 s/iter)
  comm total:       0.0282 s  (55.0% of solve)
  halo exchange:    0.0142 s  (0.0001 s/iter, 27.6%)
  dot allreduce:    0.0140 s  (27.3%)
  compute (rest):   0.0231 s  (45.0%)
```

That last block — the **communication breakdown** — is instrumentation added for this tutorial and is the key
to everything that follows. We'll dissect it in Chapter 3.

## The three ports you should actually look at

### 1. SpMV: a library call instead of a loop

```cpp
// CPU:
for (int i = 0; i < A.n_rows; i++)
    for (int j = A.rowptr[i]; j < A.rowptr[i+1]; j++)
        b[i] += alpha * A.data[j] * x[A.col_idx[j]];

// GPU:
rocsparse_spmv(handle, rocsparse_operation_none,
               &alpha, A.descr, vec_x, &beta, vec_b, ...);
```

The rocSPARSE descriptor wraps the CSR arrays uploaded once with `hipMemcpy`. The on-proc and off-proc blocks
each get their own descriptor, preserving the overlap structure from Chapter 1.

### 2. Halo exchange: the send buffer is packed *on the GPU*

```cpp
// Pack the send buffer on the GPU — no host copy
rocsparse_dgthr(handle, send_size, d_x, d_sendbuf, d_send_idx, ...);
hipDeviceSynchronize();                 // gather must finish before MPI reads d_sendbuf

// GPU-Aware MPI: hand GPU pointers straight to MPI
MPI_Isend(d_sendbuf + off, count, MPI_DOUBLE, dest, ...);
MPI_Irecv(d_recvbuf + off, count, MPI_DOUBLE, src,  ...);
```

Without GPU-Aware MPI you would copy device→host before the send and host→device after the receive. The
`staged` variant does exactly those copies; the `isend` variant skips them. Comparing the two (Chapter 4) is
how you *measure* the value of GPU-Aware MPI.

### 3. Dot products: `rocblas_ddot` then one sync then `MPI_Allreduce`

```cpp
rocblas_ddot(blas_handle, n, d_a, 1, d_b, 1, &local_sum);
hipDeviceSynchronize();                 // make the scalar visible on the host
MPI_Allreduce(&local_sum, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
```

Every dot product forces a device→host sync. Two dots per iteration means two synchronization points per
iteration — a latency cost that shows up as `dot allreduce` in the breakdown and is a prime optimization target
(Chapter 7).

## One GPU per rank

```cpp
hipSetDevice(rank % num_gpus);
```

Under `mpirun` this hands ranks 0–3 distinct GPUs. That is *correct* but not the whole story: **which** cores a
rank runs on relative to its GPU is what makes or breaks performance. That is Chapter 3, and it is the single
most important chapter in this tutorial.

Next: [3. Measuring performance correctly →](03-correct-measurement.md)
