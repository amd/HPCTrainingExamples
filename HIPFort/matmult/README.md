# Matrix multiplication with hipfort and hip

With this example, we intend to show how hipfort works by letting users create their own interface to call a HIP kernel from Fortran and perform a matrix mult
iplication. What hipfort provides are such interfaces, hence users can better understand how hipfort works by creating an interface themselves.

The code initializes two matrices `A` and `B`, and then computes `C = A*B` in two ways: the first way is by calling a hipblas function from within Fortran, using the hipfort interface. The second way is by using a HIP kernel called from Fortran through a user defined interface.

Try to not look at the files called `matmult_interface.f90` and `matmult_hip.cpp`, and create these yourself: the first one provides the fortran interface to
call the C++ function that launches the HIP kernel. Such a function will be defined in `matmult_hip.cpp`: this file will contain the definition of the HIP kernel to do the matrix multiplication as well as the C++ function that invokes it. Remember to enclose these definitions with `extern "C" {`. Note that when passing Fortran two dimensional arrays to C++ as pointers, one has to remember that Fortran uses a column-major data layout, whereas C++ uses row-major.

To run the code in this example:

```
module load rocm
module load amdclang
make
./matmult_hipfort
```
