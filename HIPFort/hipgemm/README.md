# Use the hipfort interface to make hip calls from Fortran

In this example, we are leveraging the hipfort interface to make a call to the C++ hipblas function `hipblasZgemm`.
Namely, in the `gemm_mod.f90` Fortran file, you can see the call to this C++ function made possible by the inclusion of the `iso_c binding` Fortran module and the `hipfort` and `hipfort_hipblas` Fortran modules. Note also the calls to the C++ HIP function `hipDeviceSynchronize()` made directly from Fortran.

It is important to note that the hipblas function `hipblasZgemm` requires arrays that exist on the device (GPU). For this, appropriate OpenMP target map directives are used to move the arrays from host (CPU) to device (GPU). This is done in `gemm_prog.f90` the file.

Think about other ways you could have moved the memory from the host to the device, such as HIP for instance.
Try to create a similar example where instead of a call from hipblas you are making a call from hipfft.

To run the example in this directory:

```
module load rocm
module load amdclang
make
```

The make process will create four executables: `gemm_global`, `gemm_local`, `gemm_global_sd`, `gemm_local_sd`.
The `gemm_global` has the matrices allocated outside of the matrix multiply function, and they are then supplied to the function as arguments, whereas `gemm_local` performs the allocation within the function call for the matrix multiplication. 
The variants with `_sd` use a one line openmp directive to perform the mapping of the data from the host to the device, whereas those without the `_sd` split the directives into multiple lines.

