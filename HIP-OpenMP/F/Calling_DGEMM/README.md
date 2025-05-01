## Calling GEMM from Fortran

README.md in `HPCTrainingExamples/HIP-OpenMP/F/Calling_DGEMM` from the Training Examples repository

The files in this directory show how to call a rocblas dgemm function from an OpenMP application code written in Fortran. If the `amdclang` module is not available in your system, set `FC=amdflang` or to the next generation AMD Fortran compiler.

### Explicit Memory Management

In this explicit memory management example, a target data region is created, from which a wrapper to the rocblas dgemm is called. Pay particular attention to the items passed to the wrapper call. Also notice the use `use_device_addr(A,B,C)` before the call to the wrapper.

What happens if you instead use `use_device_ptr(A,B,C)`? Check the output by setting `OMPLIBTARGET_INFO=-1`. Remember that the behavior of OpenMP directives may be different across languages, such as Fortran and C++.

To compile and run:

```
module load rocm
module load amdclang
make
```

### Unified Shared Memory

In the `usm` directory, we are showing how the code can be simplified rather dramatically by removing all the explicit data management due to the use of unified shared memory, setting HSA_XNACK=1. To compile and run:

```
module load rocm
module load amdclang
make
```

Try to use hipfort to avoid having to include the explicit rocm_interface that we are using in this example.
