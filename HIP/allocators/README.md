# Allocators and kernel execution time

With this example, we explore how different kinds of memory allocators affect the execution time of a vector add kernel:

```
for all i:
c[i] = a[i] + b[i]
```

where the above operation could be performed on the CPU or on the GPU.
The allocators in consideration are `malloc`, `hipHostMalloc`, `hipMalloc` and `hipMallocManaged`. To compile and run, just do:

```
export HSA_XNACK=1
module load rocm
make
./test_allocators
```

Try this example on different generations of hardware and observe the output of the example, showing the execution time on CPU and GPU for each alllocator. Note that the parallelism on CPU is obtained with OpenMP. Try also to do:

```
unset HSA_XNACK
```

You will notice that when the GPU compute kernel is called passing arrays allocated with `malloc` an error similar to this one shows:
```
Memory access fault by GPU node-8 (Agent handle: 0x516040) on address 0x7fafd2203000. Reason: Write access to a read-only page.
GPU core dump created: gpucore.12974
```

This is because system allocators are visible on the GPU only if automatic page migration is enabled by setting `HSA_XNACK=1` as done initially. Notice that the arrys allocated with HIP are by default initialized with a GPU kernel call: `init_data_gpu`. Alternatively, you can initialize these arrays with a `hipMemcpy`, by running the executable with:

```
./test_allocators --init-with-memcpy
```

You will then see that the memory access fault that was showing before does not show anymore, and the GPU compute kernel works even if we passed arrays allocated with `malloc` to it. The reason for this behavior is that the `hipMemcpy` made the data allocated with `malloc` visible on the GPU.
