
## Fortran `do concurrent` on the GPU

README.md from `HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/do_concurrent/1_saxpy` in the HPC Training Examples repository.

**Note:** Fortran do concurrent enabled by flag `-fdo-concurrent-to-openmp` requires at least ROCm 7.2.

```
module load rocm # or rocm-new/7.2.0 depending on your system
export FC=amdflang
```
### Why is this under OpenMP/Fortran?

The `amdflang` compiler converts `do concurrent` loops into OpenMP constructs under the hood.
The compiler flag `-fdo-concurrent-to-openmp=host` maps them to host-parallel OpenMP, while
`-fdo-concurrent-to-openmp=device` maps them to GPU-offloaded OpenMP target regions.
Because the generated code is OpenMP, you still need the OpenMP offload flags (`-fopenmp --offload-arch=<gpu>`)
for device version linking and also for compilation.
You can verify that do concurrent under the hood behaves like OpenMP `!$omp target teams distribute parallel do` by running the executable with `LIBOMPTARGET_KERNEL_TRACE=1` (see below).

### Exercise overview

There are three folders:

| Folder | Description |
|--------|-------------|
| `0_port_yourself` | Serial CPU code — replace `do` loops with `do concurrent` on your own |
| `1_saxpy_doconcurrent_host` | Solution: `do concurrent` compiled for host-parallel execution |
| `2_saxpy_doconcurrent_device` | Solution: `do concurrent` compiled for GPU offload |

### Part 0: The serial starting point

```
cd 0_port_yourself
```

This is a plain serial saxpy. The task is to replace the `do` loops with `do concurrent` loops. You may also want to replace the initialization with such loops and compile with the appropriate `-fdo-concurrent-to-openmp=` flag.

Build and run the serial version (only `-fopenmp` is needed for `omp_get_wtime`):

```
make
./saxpy
```


### Part 1: Solution — `do concurrent` mapped to host OpenMP

```
cd ../1_saxpy_doconcurrent_host
```

The `do` loops have been replaced by `do concurrent` loops. The key compiler flag is:

```
-fdo-concurrent-to-openmp=host
```

Build and run:

```
make
./saxpy
```
It is recommended to set `OMP_NUM_THREADS=24` (or another number; 24 makes sense for 1 MI300A) to run in parallel. For best performance, affinity should be set (system dependent), for example `OMP_PROC_BIND=close numactl -C 0-23 -m 0 ./saxpy`.



### Part 2: Solution — `do concurrent` mapped to GPU via OpenMP offload

```
cd ../2_saxpy_doconcurrent_device
```

Same source code, but now compiled with:

```
-fdo-concurrent-to-openmp=device
```

together with the offload flags (`-fopenmp --offload-arch=<gpu>`).

Build and run:

```
make
./saxpy
```

### Verify that OpenMP offload is used: `LIBOMPTARGET_KERNEL_TRACE=1`

To confirm that `do concurrent` is indeed transformed into OpenMP target offload by the
compiler, run the device version with the OpenMP target runtime tracing enabled:

```
cd 2_saxpy_doconcurrent_device
make
LIBOMPTARGET_KERNEL_TRACE=1 ./saxpy
```

You will see kernel launch traces from the OpenMP offload runtime, proving that the
`do concurrent` loops were lowered to OpenMP target regions by the compiler:

```
DEVID:  0 SGN:2 ConstWGSize:256  args: 7 teamsXthrds:( 152X 256) reqd:(   0X   0) lds_usage:0B scratch:0B sgpr_count:34 vgpr_count:20 agpr_count:0 sgpr_spill_count:0 vgpr_spill_count:0 tripcount:10000000 rpc:0 md:0 md_LB:-1 md_UB:-1 Max Occupancy: 8 Achieved Occupancy: 50% n:__omp_offloading_34_5e243c__QQmain_l48
DEVID:  0 SGN:2 ConstWGSize:256  args: 8 teamsXthrds:( 152X 256) reqd:(   0X   0) lds_usage:2048B scratch:0B sgpr_count:45 vgpr_count:19 agpr_count:0 sgpr_spill_count:0 vgpr_spill_count:0 tripcount:10000000 rpc:0 md:0 md_LB:-1 md_UB:-1 Max Occupancy: 8 Achieved Occupancy: 50% n:__omp_offloading_34_5e243c__QMsaxpymodPsaxpy_l22
Time of kernel: 0.000410
 plausibility check:
y(1) 4.000000
y(n) 4.000000
```

The kernel names contain `__omp_offloading`, confirming the compiler transformation.
Two kernels are launched: one for the initialization `do concurrent` loop (`_QQmain_l48`)
and one for the saxpy computation `do concurrent` loop (`_QMsaxpymodPsaxpy_l22`).

Note: For best performance, setting affinity is important (and system dependent!), for example: `ROCR_VISIBLE_DEVICES=0 OMP_PROC_BIND=close numactl -C 0 -m 0 ./saxpy`

### Running with `HSA_XNACK=0` and `HSA_XNACK=1`

`HSA_XNACK` controls whether the GPU uses page migration (unified shared memory) or
explicit data copies. On an APU such as MI300A, both settings work because host and device
share the same memory. On a discrete GPU (e.g. MI300X, MI200 series), `HSA_XNACK=1` enables
managed memory so that the runtime can page-fault and migrate data on demand, while
`HSA_XNACK=0` requires explicit data mapping.

Try both settings with the device version:

```
cd 2_saxpy_doconcurrent_device
make
```

With unified shared memory (APU programming model):

```
export HSA_XNACK=1
LIBOMPTARGET_KERNEL_TRACE=1 ./saxpy
```

Without unified shared memory (discrete GPU programming model):

```
export HSA_XNACK=0
LIBOMPTARGET_KERNEL_TRACE=1 ./saxpy
```

On MI300A you should see both runs succeed. Compare the kernel times — with `HSA_XNACK=1`
the runtime only needs to map pointers rather than copy entire arrays, which can result in
faster execution after the first touch of the data. For the `HSA_XNACK=0` case you will see that data migration relies on the default implicit mapping in OpenMP.

This makes `do concurrent` a portable, pragma-free way to express parallelism in
standard Fortran while still leveraging GPU hardware through the OpenMP offload infrastructure.
