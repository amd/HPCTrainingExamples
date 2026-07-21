# Distributed Conjugate Gradient GPU-Based Example

A GPU-accelerated, MPI-distributed Conjugate Gradient (CG) solver.

This directory is a direct companion to `CG-CPU/` — it solves the same problem with the same algorithm but moves every numerical
operation to the GPU and uses GPU-Aware MPI so communication buffers live in GPU memory the whole time.

---

## What changed from `CG-CPU/`

| `CG-CPU/`                            | `CG-GPU/`                                             |
|--------------------------------------|-------------------------------------------------------|
| `std::vector<double>` for vectors    | `hipMalloc` / `hipMemcpy` for device vectors          |
| Hand-written SpMV loop               | `rocsparse_spmv` (rocSPARSE generic API)              |
| Hand-written gather (pack send buf)  | `rocsparse_dgthr` (rocSPARSE gather)                  |
| Hand-written dot product             | `rocblas_ddot` (rocBLAS)                              |
| Hand-written axpy / scale            | `rocblas_daxpy` / `rocblas_dscal` (rocBLAS)           |
| `MPI_Irecv/Isend` with host buffers  | `MPI_Irecv/Isend` with **GPU pointers** (GPU-Aware)   |
| `MPI_Allreduce` on host scalar       | Same (result copied from GPU → host, then allreduced) |

The two header files (`sparse_mat.hpp`, `par_binary_IO.hpp`) are **identical** to the originals — file I/O and comm pattern setup
stay on the CPU.

---

## File layout

```
CG-GPU/
├── Makefile
├── README.md
├── set_affinity_mi300a.sh   ← per-rank GPU + CPU affinity wrapper (8-GPU MI300A, ROCR_VISIBLE_DEVICES + taskset)
├── run_test_7.13.sh         ← SLURM job: sweeps SDMA vs blit kernels for isend/alltoallv
├── gpu_bind.sh              ← alternative per-rank GPU + NUMA binding wrapper (numactl)
├── test_cg_gpu.sh           ← builds + runs all 5 methods, checks convergence + timing
├── submit_cg_gpu_test.sbatch← SLURM wrapper for the test (build + run on a GPU node)
├── sweep_rocm_versions.sh   ← build + benchmark one method across ROCm versions (v1/v2 SpMV)
├── submit_rocm_sweep.sbatch ← SLURM wrapper for the ROCm version sweep
├── sweep_sdma.sh            ← SDMA vs blit sweep across all methods from one build
├── submit_sdma_sweep.sbatch ← SLURM wrapper for the SDMA sweep
├── check_affinity.cpp       ← diagnostic: prints per-rank GPU (PCI id) + CPU affinity
├── submit_affinity_check.sbatch ← SLURM wrapper for the affinity diagnostic
├── STUDY_REPORT.md / .pdf   ← communication & affinity performance study
└── src/
    ├── cg.cpp            ← GPU solver (the interesting file)
    ├── sparse_mat.hpp    ← CPU structs + MPI comm setup  (same as CG-CPU/)
    ├── par_binary_IO.hpp ← parallel binary matrix reader (same as CG-CPU/)
    └── Dubcova2.pm       ← test matrix (65 536 × 65 536, SPD)
```

---

## Environment Setup

```bash
module load rocm/6.4.1
module load openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4
```

ROCm provides `hipcc`, rocSPARSE, rocBLAS, and RCCL.  The OpenMPI build must be compiled with ROCm/UCX support so that MPI can send and
receive GPU pointers directly (GPU-Aware MPI).

---

## How to build

```bash
cd CG-GPU
make
```

---

## How to run

Pass the matrix file as the first argument and the communication method as the second (default: `staged`):

```bash
mpirun -n 4 ./cg_gpu src/Dubcova2.pm staged            # (default) Isend/Irecv through CPU host buffers
mpirun -n 4 ./cg_gpu src/Dubcova2.pm isend             # Isend/Irecv with GPU buffers (GPU-Aware)
mpirun -n 4 ./cg_gpu src/Dubcova2.pm rccl              # RCCL ncclSend/ncclRecv
mpirun -n 4 ./cg_gpu src/Dubcova2.pm alltoallv_staged  # MPI_Alltoallv through CPU host buffers
mpirun -n 4 ./cg_gpu src/Dubcova2.pm alltoallv         # MPI_Alltoallv with GPU buffers (GPU-Aware)
HSA_XNACK=1 mpirun -n 4 ./cg_gpu src/Dubcova2.pm staged_unified     # zero-copy host-path Isend/Irecv via MI300A single address space
HSA_XNACK=1 mpirun -n 4 ./cg_gpu src/Dubcova2.pm alltoallv_unified  # zero-copy host-path MPI_Alltoallv via MI300A single address space
```

All seven methods produce the same numerical result (the CG algorithm is identical; only the spmv data exchange differs).

By default the right-hand side is built from a **random** vector seeded by wall-clock time, so each invocation solves a slightly different
system and the iteration count varies from run to run.  To make runs reproducible (identical system across methods and invocations), fix the
seed with the `CG_SEED` environment variable or an optional third argument:

```bash
CG_SEED=12345 mpirun -n 4 ./cg_gpu src/Dubcova2.pm rccl     # via environment
mpirun -n 4 ./cg_gpu src/Dubcova2.pm rccl 12345            # via argv[3]
```

The chosen seed is echoed as `RHS seed: <n>` at the top of the output.  With a fixed seed every method solves the same system, so their
iteration counts match (up to tiny floating-point ordering differences).

Expected output (any method):

```
method=staged  ranks=4  gpus_visible=1
RHS seed: 12345  (CG_SEED)
Initial residual: 195.873
174 iterations to converge
2-norm of residual: 0.000188555
CG solve time:      0.1544 s  (0.0009 s/iter)
  comm total:       0.1260 s  (81.7% of solve)
  halo exchange:    0.1161 s  (0.0007 s/iter, 75.2%)
  dot allreduce:    0.0100 s  (0.0001 s/iter, 6.5%)
  compute (rest):   0.0283 s  (18.3%)
```

The iteration count, residual, and solve time vary slightly between runs because the right-hand side is seeded from a random initial guess.
The solve time covers the CG loop only (matrix read, upload, and setup are excluded) and is reported as the maximum across all MPI ranks.

### Communication timing

Besides the total `CG solve time`, the solver reports how that time splits between computation and inter-rank communication (all values are the max across ranks, accumulated over the CG loop only):

- **halo exchange** — the SpMV ghost exchange: staging copies (D↔H), the GPU gather/pack, and the MPI/RCCL `send`/`recv`/`wait`/`alltoallv` calls (excludes the rocSPARSE SpMV compute itself).
- **dot allreduce** — the `MPI_Allreduce` in the CG dot products.
- **comm total** = halo exchange + dot allreduce; **compute (rest)** = solve − comm total.

For the overlap variants (`isend`, `rccl`, `alltoallv`) the halo figure is mostly the *exposed* wait, since the on-proc SpMV runs concurrently with the in-flight messages.

### GPU assignment

The solver assigns one GPU per MPI rank using:

```cpp
hipSetDevice(rank % num_gpus);
```

On a node with 4 MPI ranks and 4 visible GPUs launched with `mpirun`, each rank gets a distinct GPU (ranks 0–3 → GPUs 0–3).  You can verify
this with the bundled diagnostic, which prints each rank's selected GPU **PCI bus id** and CPU affinity mask and flags any two ranks that share a GPU:

```bash
mpicxx -O2 -std=c++17 check_affinity.cpp -o check_affinity \
       -I$ROCM_PATH/include -L$ROCM_PATH/lib -lamdhip64
mpirun -n 4 ./check_affinity
```

You do **not** need to set `ROCR_VISIBLE_DEVICES` per rank for correctness under `mpirun` — the modulo already yields distinct GPUs.  Two caveats:

- **`srun` without PMIx** launches each task as its own MPI world of size 1, so every task computes `rank % num_gpus = 0` and they **all collide on GPU 0**.  Launch with `srun --mpi=pmix` (so one communicator forms) or use the `gpu_bind.sh` wrapper below.
- **Multi-node** runs use the global rank in the modulo; a per-node local rank is more robust.  The `gpu_bind.sh` wrapper handles this too.

### CPU / NUMA affinity (important for performance)

On MI300A each GPU is local to one NUMA node (**GPU _i_ ↔ NUMA node _i_**, 24 cores each).  If ranks are not bound to their GPU-local
CPUs, performance collapses.  A default `--gpus=4 --ntasks=4` allocation (no `--exclusive`/`--cpus-per-task`) hands the whole job only a
handful of hardware threads, and `mpirun --bind-to none` lets all ranks share them — remote from most GPUs and heavily contended.  Measured
on a single MI300A node (4 ranks, `Dubcova2.pm`), fixing affinity gave a **~100× speedup**:

| launch | GPU/rank | CPU affinity | staged | isend | rccl | alltoallv |
|--------|----------|--------------|--------|-------|------|-----------|
| `--bind-to none`, not exclusive | distinct | 4 hwthreads shared by all ranks | 9.27 s | 10.58 s | 7.39 s | 8.46 s |
| `--exclusive` + `gpu_bind.sh`   | distinct | each rank → its GPU-local NUMA (24 cores) | **0.15 s** | **0.05 s** | **0.05 s** | **0.05 s** |

**Recommended launch** — pin each rank to one GPU and its NUMA-local CPUs/memory with the `gpu_bind.sh` wrapper (requires an allocation that
owns whole NUMA nodes, e.g. `sbatch --exclusive`):

```bash
mpirun -n 4 ./gpu_bind.sh ./cg_gpu src/Dubcova2.pm rccl
```

`gpu_bind.sh` sets `ROCR_VISIBLE_DEVICES=<local_rank>` (so each rank sees exactly its own GPU) and `numactl --cpunodebind=<local_rank>
--membind=<local_rank>`.  A pure-`mpirun` alternative (no wrapper) is `mpirun --map-by numa --bind-to core` inside an `--exclusive`
allocation.  See the [Affinity part 2 blog post](https://rocm.blogs.amd.com/software-tools-optimization/affinity/part-2/README.html)
for more on matching MPI ranks to GPU NUMA domains.

---

## Testing

`test_cg_gpu.sh` builds `cg_gpu`, runs every communication variant, checks that each converges to a small residual, and prints a timing
summary (total wall time, CG solve time, and the communication breakdown per method).  It skips gracefully (exit 0) on a node with no GPU.

Run it inside a GPU allocation, or submit the SLURM wrapper (which uses `--exclusive` + `gpu_bind.sh` for correct affinity):

```bash
# On a GPU node / inside an allocation:
GPU_BIND=./gpu_bind.sh ./test_cg_gpu.sh

# Or submit to a compute node:
sbatch submit_cg_gpu_test.sbatch
```

Useful overrides: `MATRIX`, `NUM_RANKS`, `RES_TOL`, `METHODS`, `GPU_BIND`, `CG_SEED` (fix the RHS seed so iteration counts are comparable across methods).

---

## Communication variants

### Method 1 — `staged` (CPU-buffered Isend/Irecv)

The simplest baseline: all packing, communication, and unpacking go through the CPU.  MPI operates on host pointers.

1. Copy the full local vector from GPU to host (`hipMemcpy D→H`)
2. Pack send values into a pinned host buffer (CPU loop)
3. `MPI_Irecv` / `MPI_Isend` on host pointers
4. After wait, copy received values back to GPU (`hipMemcpy H→D`)

### Method 2 — `isend` (GPU-Aware Isend/Irecv)

Same point-to-point structure as `staged`, but ghost values never touch the CPU.  `rocsparse_dgthr` packs the send buffer directly on the
GPU, and `MPI_Isend` / `MPI_Irecv` operate on GPU pointers.  On-proc SpMV overlaps with in-flight messages.

Comparing `staged` vs. `isend` shows the cost of the D→H + H→D copies that GPU-Aware MPI eliminates.

### Method 3 — `rccl` (RCCL ncclSend / ncclRecv)

RCCL (ROCm Collective Communications Library) provides GPU-native communication that can exploit Infinity Fabric / NVLink paths without any CPU
involvement.  Sends and receives are grouped with `ncclGroupStart` / `ncclGroupEnd` and launched on a dedicated `rccl_stream`.  On-proc SpMV
runs on the default GPU stream concurrently — true GPU-GPU overlap.

### Method 4 — `alltoallv_staged` (CPU-buffered MPI_Alltoallv)

Replaces the many point-to-point messages with one collective, but uses explicit host staging:

1. `rocsparse_dgthr` packs into `d_sendbuf` on the GPU
2. `hipMemcpy D→H` → `h_sendbuf`
3. `MPI_Alltoallv` on host pointers
4. `hipMemcpy H→D` → `d_recvbuf`
5. On-proc SpMV overlaps with step 3 (CPU blocks in MPI, GPU computes)

### Method 5 — `alltoallv` (GPU-Aware MPI_Alltoallv)

Same collective as `alltoallv_staged` but passes GPU pointers directly to `MPI_Alltoallv` — no host copies needed.  On-proc SpMV is submitted 
to the GPU before the collective blocks the CPU, so both run in parallel.

Comparing `alltoallv_staged` vs. `alltoallv` isolates the PCIe round-trip cost (D→H + H→D) that GPU-Aware MPI avoids.

### Method 6 — `staged_unified` (MI300A single address space, zero-copy plain MPI)

Exploits the **APU's unified address space**: the send/recv buffers are ordinary
**`malloc`'d host memory**, so MPI sees *host* pointers and takes the **host
transport (plain, non-GPU-Aware MPI)** — yet there are still **zero staging
copies**, because on MI300A the GPU reads/writes those same host buffers directly
via XNACK page faulting. The send buffer is packed on the GPU with
`rocsparse_dgthr` (GPU writing host memory); a single `hipDeviceSynchronize` makes
it visible to MPI; MPI then `Isend`/`Irecv`s on the host pointers, and off-proc
SpMV reads the received ghosts in place on the GPU.

This is the key difference from `isend`: `isend` hands **device** pointers to
GPU-Aware MPI, whereas `staged_unified` only ever gives MPI **host** pointers
(`malloc`, *not* `hipMalloc`/`hipHostMalloc`) — a device or pinned pointer would
push UCX back onto the GPU-Aware path.

**Requires `HSA_XNACK=1`** in the environment (set before HSA init) so the GPU can
access the `malloc`'d host memory via page faults:

```bash
HSA_XNACK=1 mpirun -n 4 ./cg_gpu src/Dubcova2.pm staged_unified
```

It is APU-specific — on a discrete GPU the same code degrades to slow migration.
Comparing `staged` vs. `isend` vs. `staged_unified` shows three points on the
spectrum: copies+host MPI, zero-copy+GPU-Aware MPI, and zero-copy+host MPI on an
APU. Verified on MI300A (gfx942): `staged_unified` converges bit-for-bit with
`staged`, runs ~2× faster (no copies), and sits between `staged` and `isend`
because it uses the host transport rather than GPU-Aware MPI.

### Method 7 — `alltoallv_unified` (MI300A single address space, zero-copy host MPI_Alltoallv)

The **collective analogue of `staged_unified`**. `MPI_Alltoallv` runs on the same
**`malloc`'d host buffers** (host transport, *not* GPU-Aware MPI), while the GPU
packs `u_sendbuf` and reads `u_recvbuf` in place via XNACK — **zero staging
copies**. It completes the 3×2 matrix of communication choices:

| | 2 copies + host MPI | 0 copies + GPU-Aware MPI | 0 copies + host MPI (APU) |
|---|---|---|---|
| **point-to-point** | `staged` | `isend` | `staged_unified` |
| **collective** | `alltoallv_staged` | `alltoallv` | `alltoallv_unified` |

**Requires `HSA_XNACK=1`**:

```bash
HSA_XNACK=1 mpirun -n 4 ./cg_gpu src/Dubcova2.pm alltoallv_unified
```

Verified on MI300A (gfx942): converges bit-for-bit with the other variants; like
`staged_unified`, it lands between the staged (copies) and GPU-Aware collectives.

---

## Key concepts

### 1. One GPU per MPI rank

```cpp
hipSetDevice(rank % num_gpus);
```

Ranks round-robin across available GPUs.  On a node with 4 MI300A partitions and 4 MPI ranks, each rank gets its own GPU.

### 2. rocSPARSE SpMV instead of a CPU loop

```cpp
// CPU (original):
for (int i = 0; i < A.n_rows; i++)
    for (int j = A.rowptr[i]; j < A.rowptr[i+1]; j++)
        b[i] += alpha * A.data[j] * x[A.col_idx[j]];

// GPU (this file):
rocsparse_spmv(handle, rocsparse_operation_none,
               &alpha, A.descr, vec_x, &beta, vec_b, ...);
```

The rocSPARSE descriptor wraps the CSR arrays that were uploaded with `hipMemcpy`.  rocSPARSE picks the best SpMV algorithm for the hardware.

### 3. GPU-Aware MPI — sending/receiving GPU buffers directly

```cpp
// Pack the send buffer on the GPU (no host copy needed)
rocsparse_dgthr(handle, send_size, d_x, d_sendbuf, d_send_idx, ...);
hipDeviceSynchronize();   // gather must finish before MPI reads d_sendbuf

// Send and receive GPU pointers — the MPI runtime handles the rest
MPI_Isend(d_sendbuf + offset, count, MPI_DOUBLE, dest, ...);
MPI_Irecv(d_recvbuf + offset, count, MPI_DOUBLE, src,  ...);
```

Without GPU-Aware MPI you would have to copy from GPU→host before sending and host→GPU after receiving.  With GPU-Aware MPI you skip both copies.

### 4. rocBLAS for dot / axpy / scale

```cpp
rocblas_ddot (blas_handle, n, d_a, 1, d_b, 1, &local_sum);
rocblas_daxpy(blas_handle, n, &alpha, d_y, 1, d_x, 1);   // d_x += alpha*d_y
rocblas_dscal(blas_handle, n, &alpha, d_x, 1);            // d_x *= alpha
```

After `rocblas_ddot`, a single `hipDeviceSynchronize()` makes the result visible on the CPU before `MPI_Allreduce` reduces it across ranks.

---

## SDMA engines vs. blit kernels (`HSA_ENABLE_SDMA`)

When GPU-Aware MPI (`isend`, `alltoallv`) passes a GPU pointer to the MPI runtime, UCX must physically move the data out of device memory.
ROCm gives it two mechanisms to do that:

| Mechanism | `HSA_ENABLE_SDMA` | How it works |
|---|---|---|
| **SDMA engines** | `1` (default) | Dedicated hardware DMA controllers transfer data between GPU memory and the fabric.
They run independently of the shader engines and do not consume compute resources. |
| **Blit kernels** | `0` | ROCm dispatches a small compute shader (a "blit") to copy the data using the GPU's shader engines.
No dedicated DMA hardware is used. |

Set the variable before `mpirun` to switch between them:

```bash
# SDMA engines (default)
HSA_ENABLE_SDMA=1 mpirun -n 8 --bind-to none bash set_affinity_mi300a.sh ./cg_gpu src/Dubcova2.pm isend

# Blit kernels
HSA_ENABLE_SDMA=0 mpirun -n 8 --bind-to none bash set_affinity_mi300a.sh ./cg_gpu src/Dubcova2.pm isend
```

`run_test_7.13.sh` sweeps both values automatically for `isend` and `alltoallv` and labels each run in the log output:

```
=== isend  HSA_ENABLE_SDMA=1  (sdma) ===
=== isend  HSA_ENABLE_SDMA=0  (blit_kernel) ===
=== alltoallv  HSA_ENABLE_SDMA=1  (sdma) ===
=== alltoallv  HSA_ENABLE_SDMA=0  (blit_kernel) ===
```

### When does each win?

- **SDMA** tends to win for **large, infrequent transfers** where having the shader engines free to overlap computation matters more
  than raw copy throughput.
- **Blit kernels** tend to win for **small, latency-sensitive transfers** where the shader engines are otherwise idle and the lower
  software overhead of a compute dispatch beats the DMA engine's start-up cost.  On MI300A unified memory, blit kernels often outperform
  SDMA for the modest ghost-zone message sizes typical of sparse solvers.

The right choice is workload- and hardware-dependent; the sweep in `run_test_7.13.sh` surfaces the difference empirically.

---

## Requirements

- ROCm ≥ 6.3 (rocSPARSE, rocBLAS, hipcc)
- GPU-Aware MPI (OpenMPI/UCX built with ROCm support — see module list on
  the login banner)
