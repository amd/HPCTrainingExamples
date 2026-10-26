# rocprofiler-systems: OpenMP allocations and memory pools on MI300A

This exercise uses the Fortran OpenMP code from
`Rocprofv3/OpenMP/Allocations_and_MemoryPool_MI300A/Fortran` and profiles it with
`rocprof-sys-run`. It uses the same problem as the
[`rocprofv3` exercise](../../Rocprofv3/OpenMP/Allocations_and_MemoryPool_MI300A/Fortran), but uses the rocprofiler-systems HPC/OpenMP presets and unified memory profiling.

This exercise was last tested on MI300A AAC7 cluster, using ROCm 10.1.0. For the visualization, we relied on Perfetto traces, but the same can be achieved with the [rocpd traces](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocpd-output-format.html) visualized using [ROCm Optiq](https://rocm.docs.amd.com/projects/roc-optiq/en/latest/what-is-optiq.html).

The goal is to show how repeated unified-memory allocations create expensive first
touch/page-fault behavior on MI300A. This exercise shows how to identify the issue and several ways how to solve this.
This is a common optimization for codes using the allocator ALLOCATE for memory touched in GPU kernels when relying on unified shared memory (needs `HSA_XNACK=1`). 

**Note**: the same is true for `malloc` or `new` in C/C++ codes. C/C++ codes can be optimized analogous to this Fortran example.

## Environment

Run the examples on a compute node with one GPU:

on aac6:
```bash
salloc -N 1 --gpus=1 -p PPAC_MI300A_CPX
```
on aac7:
```bash
salloc  -N 1 --gpus=1 --time=48:00:0 --partition 192C4G1H_MI300A_RHEL9_A1_CPX --mem=20GB
```

Load ROCm 10.1.0 and use an Umpire build that matches that ROCm/compiler stack for
the memory-pool versions:

```bash
module load rocm/10.1.0 # On aac6
module load rocm-new/10.1.0 # On aac7
# Needed for Solution 2:
module load umpire/v2025.12.0-rocm-new-10.1.0
```

MI300A unified shared memory examples require XNACK:

```bash
export HSA_XNACK=1
```

Start this exercise from this directory:

```bash
cd rocprofiler-systems/MemoryPool
```

The rocprofiler-systems presets used below are:

```bash
rocprof-sys-run --list-presets
rocprof-sys-run --explain=trace-openmp
```

`trace-openmp` enables OpenMP target runtime tracing through OMPT plus ROCm kernel
dispatch and memory copy domains. It also records markers if the application already
has them.
The examples below use `trace-openmp` for the focused OpenMP timeline and add
`--use-unified-memory-profiling` when the trace should include unified-memory
page-fault and migration reports. This flag automatically enables the required KFD tracing.

## Version 1: Problem, no code changes

Build the local copy of the unmodified problem version:

```bash
cd problem
amdflang -fopenmp --offload-arch=gfx942 \
  -o test_mempool \
  test_mempool.f90
```

Verify that the application runs without profiling by simply executing:

```
./test_mempool
# Expected output (ignore the performance):
# Final result:   1.000000
# Runtime is: 1.009241 secs
```

Now let's start with profiling. Run it directly with the OpenMP preset:

```bash
rocprof-sys-run --preset=trace-openmp --output problem-out -- ./test_mempool
```

Look at the *.proto file from the output folder with Perfetto (https://ui.perfetto.dev/).
This produces the first view of the problem. Look for the long first kernel in each iteration and the gaps between
GPU work. With unified shared memory, those long first-touch regions are a symptom
that allocation and page placement costs are being paid repeatedly.

<img width="1918" height="1017" alt="v1_rocprof-sys" src="https://github.com/user-attachments/assets/5024d089-b635-4212-a34b-feb4dc674838" />

Zoom in:

<img width="1635" height="762" alt="v1_rocprof-sys_Zoom" src="https://github.com/user-attachments/assets/deb2ce6f-de24-4594-bbf8-bd43c451fa42" />

## Add memory profiling option

Run the binary with the HPC/OpenMP preset and unified-memory event domain enabled:

```bash
rocprof-sys-run \
  --preset=trace-openmp \
  --use-unified-memory-profiling \
  --output problem-usm-profile \
  -- ./test_mempool
```

Look at the *.proto file from the output folder with Perfetto (https://ui.perfetto.dev/).
The unified-memory profiling report is useful when explaining why the first kernel
in each iteration is expensive: page faults and migration/placement events show
that unified-memory pages are being made resident for GPU access. This is the
evidence that motivates avoiding repeated dynamic allocation or adding a memory
pool.

<img width="1622" height="587" alt="image" src="https://github.com/user-attachments/assets/93be269a-cd4c-42dc-8940-599772c7b011" />


## Solution 1, move allocations out of the loop

This version keeps the allocations for the whole iterative region instead of
allocating and deallocating every iteration:

The solution sources below are the existing files from the Rocprofv3 exercise. Some
of them already contain ROCTx calls, so the compile lines link ROCTx where required
by that source. The rocprofiler-systems workflow does not depend on those markers;
they are extra confirmation if present in the trace.

```bash
cd ../solution1
amdflang -fopenmp --offload-arch=gfx942 \
  -o test_mempool \
  test_mempool.f90

rocprof-sys-run \
  --preset=trace-openmp \
  --use-unified-memory-profiling \
  --output solution1-out \
  -- ./test_mempool
```

Look at the *.proto file from the output folder with Perfetto (https://ui.perfetto.dev/).
Compare this trace with the problem trace. The first iteration still pays the initial
placement cost, but the later iterations should no longer show the same repeated
first-touch/page-fault pattern.

## Solution 2, Umpire memory pool

Use the memory-pool solution when the application cannot easily hoist temporary
allocations out of loops or subroutines. The Umpire module must match the ROCm
10.1.0 compiler/runtime environment.

```bash
cd ../solution2
amdflang -fopenmp --offload-arch=gfx942 \
  -I${UMPIRE_PATH}/include/umpire \
  -L${UMPIRE_PATH}/lib64 -L${UMPIRE_PATH}/lib \
  -lumpire -lcamp -lfmt \
  -o test_mempool \
  test_mempool.f90

rocprof-sys-run \
  --preset=trace-openmp \
  --use-unified-memory-profiling \
  --output solution2-out \
  -- ./test_mempool
```

Look at the *.proto file from the output folder with Perfetto (https://ui.perfetto.dev/).
The memory pool may still warm up in the first iterations, but later iterations
should avoid the repeated allocation/page-fault behavior seen in the problem. This is
the practical solution when dynamic temporaries are difficult to remove from the
application structure. Look at the *.proto file in the output folder with perfetto.


## Solution 3, compiler flag to allocate on device automatically
This example uses a compiler flag which is still experimental in ROCm 10.1 and that automatically allocates all ALLOCATE as device allocations (omp_target_alloc) with zero code changes. This can be useful if all computations are running on GPU. The `-fopenmp-default-allocate=target` flag works per file, so CPU and GPU allocations could be separated by files if there are still CPU allocations needed. Note that this feature is relatively new and still under development. It is intended for easier porting of codes requiring unified_shared_memory to discrete GPUs.

```bash
cd ../problem

amdflang -fopenmp --offload-arch=gfx942 \
 -fopenmp-default-allocate=target \
  -o test_mempool \
  test_mempool.f90

rocprof-sys-run \
  --preset=trace-openmp \
  --use-unified-memory-profiling \
  --output solution3-out \
  -- ./test_mempool
```

Look at the *.proto file from the output folder with Perfetto (https://ui.perfetto.dev/). You will see the allocations on the device between each of the repeated kernel calls, but much shorter kernel times as no first touch penalty happens since the allocation is on the device already. If you would touch the data from the CPU allocated on GPU, you will see a first touch penalty again. With unified memory profiling and this flag it is easy to find out where one unintentionally mixes CPU and GPU allocations and computations: something important to understand for performance on discrete GPUs when using the USM programming model and avoiding first touch penalties with frequent reallocations on MI300A.

## Next steps

Try other rocprofiler-system options with this small example or try other examples from this repository.

**Finally, try to profile your own application!**
