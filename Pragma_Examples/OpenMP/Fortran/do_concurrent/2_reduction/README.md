## `do concurrent` with `REDUCE` (OpenMP-style reduction)

### Exercise overview

| Path | Description |
|------|-------------|
| `0_port_yourself/` | Serial version: sum 100,000 array elements (each 1.0) with plain `do` loops — port to `do concurrent` and `REDUCE(+)`, then use `1_do_concurrent_reduce/`. |
| `1_do_concurrent_reduce/` | Reference solution: `freduce.F08` with GPU offload of initialization and of the reduction. |

The reference program uses an array of length 100,000, then sums the elements in parallel on the device. The reduction is expressed as a `do concurrent` loop with **`REDUCE`**.

```fortran
do concurrent (j = 1:n) REDUCE (+: sum2)
   sum2 = sum2 + array(j)
end do
```

This is lowered to OpenMP target code, similar in spirit to a `REDUCE` clause on a parallel `do` loop.


### Toolchain and environment
This example requires at least Fortran Drop 23.2.0 (April 2026, beta release). There is no official ROCm version yet which enables REDUCE correctly.
See [here](#beta-compiler-release-which-enables-reduce) for more details on how to install this version.

```bash
module load rocm/therock-23.2.0
export FC=amdflang
```
Either use `HSA_XNACK=1` or `HSA_XNACK=0`, you can also experiment with that.


### Required compiler flags
As in `1_do_concurrent_reduce/Makefile`, you need to pass additional flags to the compiler to enable `do concurrent` on the GPU:

- `-fdo-concurrent-to-openmp=device`: map `do concurrent` to OpenMP *device* regions  
- `-fopenmp --offload-arch=<arch>`: enable OpenMP offload compile and link  

The `Makefile` sets `ROCM_GPU` to the first `rocminfo` line that contains a `gfx` token (see `1_do_concurrent_reduce/Makefile`). On CPU login nodes this value is empty. In that case pass **`ROCM_GPU`** manually, e.g. `make ROCM_GPU=gfx942` for MI300-series GPUs (use the arch that matches your GPU).


### Build and run the example
First, build the serial starting point (any Fortran compiler would do, but use the latest pre-release Fortran Drop 23.2.0 (April 2026) as required for the next step):
```bash
module load rocm/therock-23.2.0
cd 0_port_yourself
make
./freduce
```

Expected result: `sum=   100000.0` (or similar formatting).

Next, compare the code changes you made to the solution. Run the solution with:
```bash
cd 1_do_concurrent_reduce
module load rocm/therock-23.2.0  
export FC=amdflang
make   # or:  make ROCM_GPU=gfx942 if not on a compute node
./freduce # needs to run on a compute node!
```

It should print the same sum, `100000.0` (summing 100,000 values of 1.0).

### Optional: confirm `do concurrent` is leveraging OpenMP offload 
Set the `LIBOMPTARGET_KERNEL_TRACE=1` environment variable to enable additional output of the OpenMP runtime:

```bash
cd 1_do_concurrent_reduce
LIBOMPTARGET_KERNEL_TRACE=1 ./freduce
```

You should see traces for kernels whose names include `__omp_offloading`, indicating the `do concurrent` (including `REDUCE`) path was lowered to OpenMP target code.


## Beta compiler release which enables `REDUCE`:
This feature was enabled very recently in the compiler. Today (April 2026) it only works with this pre-release version:

- (https://repo.radeon.com/rocm/misc/flang/therock-afar-23.2.0-gfxX-7.13.0-663ad81964a.txt)

Read that file for download locations and install notes for your GPU architecture (`gfx*`).
New pre-release Fortran Drops are published infrequently here: (https://repo.radeon.com/rocm/misc/flang)
