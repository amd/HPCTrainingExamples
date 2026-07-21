# Function calls from host and device code

The code in this directory serves as an example on how to define a function that can be called both from host code and from device code. The example computes an approximation of the value of the exponential function at a given point, using a truncated Taylor series expansion. The terms of the Taylor series are compute in parallel on the host with OpenMP and on the device with a HIP kernel. Then, the approximation is obtained with a reduction operations that sums these terms up, done in parallel on the host with OpenMP (for simplicity). Play with this code to familiarize with some of the aspects of HIP kernels and function qualifiers.

To compile and run:

```bash
module load rocm
make
./exp
```

1. First, see what happens if you remove the `__host__` decoration or the `__device__` decoration from the definition of the `compute_term` function.
2. Next, modify the `gpu_get_Taylor_terms` kernel declaration and assign a non void return value: observe the compilation error.
3. Then, use only one block of threads by modifying the declaration of `blocks_in_grid` as `int blocks_in_grid=1`. Notice that yo get a "FAIL on the GPU" message. The reason is because we are only using 64 threads, but we are requesting 100 terms in the Taylor expansion, hence there are not enough threads to do all the operations that we set ourselves to do.
4. The next thing to try is to keep `int blocks_in_grid=1` and use striding: we will be reusing the threads in the `gpu_get_Taylor_terms` kernel so that even if we only have one block, the result will still be correct. Think about how to implement this and make sure you get a "PASS on the GPU message". What stride size did oyou pick? If you do not want to do it yourself, check the implementation of the `gpu_get_Taylor_terms_stride` kernel, and run that one instead of the `gpu_get_Taylor_terms` that does not have striding.
5. Last item in this list, try to replace the definition of `id` in the any of the kernels to `int id = threadIdx.x * gridDim.x + blockIdx.x`. Does the code still work? The code will still work as this new definition is also valid, though it is not optimal for performance. This example is too small though to see any difference.
6. Bonus point: try to write a HIP kernel to perform the reduction currently done with OpenMP.
