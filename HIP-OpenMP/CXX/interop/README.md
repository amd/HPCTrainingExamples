# Synchronization of HIP kernels and OpenMP target kernels

This example is aimed at exploring the use of the `interop` construct from OpenMP to make sure that the execution of a HIP kernel happens before the execution of an OpenMP target kernel. Note that kernel launching is non blocking for HIP and therefore without proper synchronization there might be an overlap of computation which would likely invalidate the results. The `interop` construct has been introduced among other reasons to make sure that foreign runtimes (foreing to OpenMP, such as HIP for instance) are enqueued according to  the user's desire, to make sure proper interoperability with OpenMP is achieved.

In this particular example, we have a HIP kernel, called `hip_square` that squares the entries of an input array, followed by an OpenMP tartget kernel that adds one unit to the same array. If $x$ is any element of the input array after initialization, the desired outcome would then be $x^2+\pi$. Because $x^2+\pi \neq (x+\pi)^2$, we can easily determine if a race condition has happened and the right order of operations (i.e. the right enqueuing of the kernels) has not been achieved. Asynchronos memory copies from host to device and back are also performed before and after the two GPU kernels.

We are using an object of `interop_t` type to create a HIP stream which is then used to enqueue the HIP kernel and the two async copies.
According to the intended usage of the `interop use` construct, when `interop use` is invoked, proper enqueuing of HIP kernels and OpenMP target kernels should be achieved. It currently seem that this is not the case however for the AMD compiler `amdclang++` and we still had to call a `hipStreamSynchronize(stream)` in place of `#pragma omp interop use` to make sure that the HIP kernel was not overlapping execution with the OpenMP target kernel.

To compile and run:

```
module load rocm
module load amdclang
make
./interop
```

If the `amdclang` module is not available in your system, please do: `export CXX=$(which amdclang++)` after loading the rocm module.  
