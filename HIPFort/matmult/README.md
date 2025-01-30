# Matrix multiplication with hipfort and hip

With this example, we intend to show how hipfort works by letting users create their own interface to call a HIP kernel from Fortran and perform a matrix mult
iplication. What hipfort provides are such interfaces, hence users can better understand how hipfort works by creating an interface themselves.

The code initializes two matrices A and B and then computes C = A*B in two ways, first calling a hipblas routine from within Fortran, using the hipblas interface, and then using a HIP kernel called from Fortran through a user defined interface.

Try to not look at the files called `matmult_interface.f90` and `matmult_hip.cpp`, and create these yourself: the first one provides the fortran interface to
call the C++ function that launches the HIP kernel. Such a function will be defined in `matmult_hip.cpp`. Note that when passing arrays from Fortran to C++, y
ou have to remember the different types of memory allocations for arrays (column-major for Fortran vs row-major for C++).
