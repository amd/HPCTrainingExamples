# Allocators and kernel execution time

With this example, we explore how different kinds of memory allocators affect the execution time of a vector add kernel:

```
for all i:
c[i] = a[i] + b[i]
```

where the above operation could be performed on the CPU or on the GPU.
The allocators in consideration are `malloc`, `hipHostMalloc`, `hipMalloc` and `hipMallocManaged`. To compile and run, just do:

```
module load rocm
make
./test_allocators
```

Try this example on different generations of hardware and observe the output of the example, showing the execution time on CPU and GPU for each alllocator. Note that the parallelism on CPU is obtained with OpenMP. Try also to do:

```
export HSA_XNACK=1
```

Do the results change?
