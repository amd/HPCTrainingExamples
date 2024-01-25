# Optimizing DAXPY HIP

In this exercise, we will progressively make changes to optimize the DAXPY kernel on GPU. Any AMD GPU can be used to test this.

DAXPY Problem:
```
Z = aX + Y
```
where `a` is a scalar, `X`, `Y` and `Z` are arrays of double precision values.

In DAXPY, we load 2 FP64 values (8 bytes each) and store 1 FP64 value (8 bytes). We can ignore the scalar load because it is constant. We have 1 multiplication and 1 addition operation for the 12 bytes moved per element of the array. This yields a low arithmetic intensity of 2/24. So, this kernel is not compute bound, so we will only measure the achieved memory bandwith instead of FLOPS.

# Inputs
- `N`, the number of elements in `X`, `Y` and `Z`. `N` may be reset to suit some optimizations.
   Choose a sufficiently large array size to see some differences in performance.

# Build Code
```
make
```

# Run exercises
```
./daxpy_1 10000000
./daxpy_2 10000000
./daxpy_3 10000000
./daxpy_4 10000000
./daxpy_5 10000000
```

# Things to ponder about

## `daxpy_1`
This shows a naive implementation of the daxpy problem on the GPU where only 1 wavefront is launched and the 64 work-items in that wavefront loop over the entire array and process 64 elements at a time. We expect this kernel to perform very poorly because it simply utilizes a part of 1 CU, and leaves the rest of the GPU unutilized.

## `daxpy_2`
This time, we are launching multiple wavefronts, each work-item now processing only 1 element of each array. This launches `N/64` wavefronts, enough to be scheduled on all CUs. We see a big improvement in performance here.

## `daxpy_3`
In this experiment, we check to see if launching larger workgroups can help lower our kernel launch overhead because we launch fewer workgroups if each workgroup has 256 work-items. In this case too, an improvement in measured bandwidth achieved is seen.

## `daxpy_4`
If we ensured that the array has a multiple of `BLOCK_SIZE` elements so that all work-items in each workgroup have an element to process, then we can avoid the conditional statement in the kernel. This could reduce some instructions in the kernel.. Do we see any improvement? In this trivial case, this does not matter. Nevertheless, it is something we could keep in mind.

Question: What happens if `BLOCK_SIZE` is `1024`? Why?

## `daxpy_5`
In this experiment, we will use double2 type in the kernel to see if the compiler can generate `global_load_dwordx4` instructions instead of `global_load_dwordx2` instructions. So, with same number of load and store instructions, we are able to process two elements in each thread. This should help amortize on the cost of index calculation. This kernel does not perform as well.. and we need to understand why. Adding `__launch_bounds__` to the kernel does not help as we did not have register pressure to begin with.

First, to get the assembly for these kernels, ensure that the `-g --save-temps` flags are passed to `hipcc`. Then you can find the assembly code in `daxpy_*-host-x86_64-unknown-linux-gnu.s` files. Examining `daxpy_3` and `daxpy_5`, we see the two cases:

`daxpy_3`:
```
    global_load_dwordx2 v[2:3], v[2:3], off
    v_mov_b32_e32 v6, s5
    global_load_dwordx2 v[4:5], v[4:5], off
    v_add_co_u32_e32 v0, vcc, s4, v0
    v_addc_co_u32_e32 v1, vcc, v6, v1, vcc
    s_waitcnt vmcnt(0)
    v_fmac_f64_e32 v[4:5], s[6:7], v[2:3]
    global_store_dwordx2 v[0:1], v[4:5], off
```

`daxpy_5`:
```
    global_load_dwordx4 v[0:3], v[0:1], off
    v_mov_b32_e32 v10, s5
    global_load_dwordx4 v[4:7], v[4:5], off
    s_waitcnt vmcnt(0)
    v_fmac_f64_e32 v[4:5], s[6:7], v[0:1]
    v_add_co_u32_e32 v0, vcc, s4, v8
    v_fmac_f64_e32 v[6:7], s[6:7], v[2:3]
    v_addc_co_u32_e32 v1, vcc, v10, v9, vcc
    global_store_dwordx4 v[0:1], v[4:7], off
```
We observe that in the `daxpy_5` case, there are two `v_fmac_f64_e32` instructions, one for each element being processed.

To understand why `daxpy_5` does not perform better, we can profile the two kernels and compare. Here are the steps:

```
omniperf profile -n daxpy_3 --no-roof -- ./daxpy_3 10000000
omniperf profile -n daxpy_5 --no-roof -- ./daxpy_5 10000000
omniperf analyze -p workloads/daxpy_3/mi200 -p workloads/daxpy_5/mi200 --dispatch 5
```
In the analyze command above, we are skipping the warmup runs and the first few iterations and analyzing only the 5th kernel dispatch in each case. The command compares the profiles and shows how each metric changes between the two cases. We leave the reader to ponder about the reasons why `daxpy_5` is slower than `daxpy_3`. Our thoughts are that due to the increased latency of read and write instructions now, the compute operations cannot be scheduled as fast as in the `daxpy_3` case. This optimization does not quite help this case, but could be useful in `SAXPY`, for instance, where processing 2 FP32 values could be done with one packed FP64 FMA instruction.

# Notes
- Before timing kernels, it is best to launch the kernel at least once as warmup so that those initial GPU launch latencies do not affect your timing measurements.
- The timing loop is typically several hundred iterations.


