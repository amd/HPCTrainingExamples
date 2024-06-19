# Finite difference method - Laplacian blog series

Accompanying code examples for the following blogs:

[https://rocm.blogs.amd.com/high-performance-computing/finite-difference/laplacian-part1/README.html](https://rocm.blogs.amd.com/high-performance-computing/finite-difference/laplacian-part1/README.html)
                                                                                                    
[https://rocm.blogs.amd.com/high-performance-computing/finite-difference/laplacian-part2/README.html](https://rocm.blogs.amd.com/high-performance-computing/finite-difference/laplacian-part2/README.html)
                                                                                                    
[https://rocm.blogs.amd.com/high-performance-computing/finite-difference/laplacian-part3/README.html](https://rocm.blogs.amd.com/high-performance-computing/finite-difference/laplacian-part3/README.html)
                                                                                                    
[https://rocm.blogs.amd.com/high-performance-computing/finite-difference/laplacian-part4/README.html](https://rocm.blogs.amd.com/high-performance-computing/finite-difference/laplacian-part4/README.html)

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

Configuring the loop tiling/unrolling factors `m`, launch bounds `LB`, or
number of subdomain blocks `BY` is done manually inside each kernel header file.

## Kernel execution

The program can be executed as follows:

```bash
./laplacian_dp_kernel1 nx ny nz bx by bz
```

where `nx`, `ny`, and `nz` corresponding to the grid sizes in the x, y, and z directions, respectively,
and `bx`, `by`, and `bz` correspond to the number of threads per thread block.

The default values are `512`, `512`, `512`, `256`, `1`, and `1`, respectively.

Note that if `bx` * `by` * `bz` exceeds the launch bounds size `LB`, the program will automatically reset
`bx` * `by` * `bz` to `LB` * 1 * 1

Can also execute with the provided rocprof input file:

```bash
rocprof -i rocprof_input.txt -o kernel1.csv ./laplacian_dp_kernel1 ...
```
