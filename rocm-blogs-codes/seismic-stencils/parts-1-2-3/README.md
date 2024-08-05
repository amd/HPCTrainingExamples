# Build instructions

This program executes all GPU kernels and can be compiled with the following command:

```bash
make
```

This builds the `float4` and 8th order finite difference by default.
You can build the CPU version of the sliding window technique with `make sliding_window`

Additional instructions:

1. Adjust the finite difference order with `make radius=<1,2,3,4>`
2. Specify the vectorization level with `make vec=<0,1,2>`

## Kernel execution

The program can be executed as follows:

```bash
./stencils_R_X_vec_Y.x <nx> <ny> <nz> <its> <window> <align>
```

The defaults for `<nx> <ny> <nz> <its> <window> <align>` are 512, 512, 512 100, 1, and 0, respectively
