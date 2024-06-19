# Finite difference method - Laplacian blog series

Accompanying code examples for the following blogs:

[Finite difference method - Laplacian part 1](https://rocm.blogs.amd.com/high-performance-computing/finite-difference/laplacian-part-1/README.html)

[Finite difference method - Laplacian part 2](https://rocm.blogs.amd.com/high-performance-computing/finite-difference/laplacian-part-2/README.html)

[Finite difference method - Laplacian part 3](https://rocm.blogs.amd.com/high-performance-computing/finite-difference/laplacian-part-3/README.html)

[Finite difference method - Laplacian part 4](https://rocm.blogs.amd.com/high-performance-computing/finite-difference/laplacian-part-4/README.html)

## Build instructions

All kernels can be compiled with the following command:

```bash
make
```

Additional instructions:

1. Build individual kernels only: `make kernel1`
2. Generate register files:  `make TEMPS=true`
3. Compile with single precision: `make DOUBLE=false`
4. View the theoretical fetch and write sizes: `make THEORY=true`

## Kernel execution

The program can be executed as follows:

```bash
./laplacian_dp_kernel1 <nx> <ny> <nz> <bx> <by> <bz>
```

where `nx`, `ny`, and `nz` corresponding to the grid sizes in the x, y, and z directions, respectively,
and `bx`, `by`, and `bz` correspond to the number of threads per thread block.

The default is a 512 x 512 x 512 grid with a 256 x 1 x 1 thread block configuration
