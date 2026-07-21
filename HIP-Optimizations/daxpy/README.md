
# Advanced HIP

README.md from `HPCTrainingExamples/HIP-Optimizations/daxpy` from the Training Examples repository.

Set up your environment so that ROCm compilers are available.
```bash
module load rocm  # or rocm-new, depending on the system you are on
```
The exercises here were tested for systems with an AMD software stack, loading a ROCm version is sufficient as `hipcc` is used for the compiler.

Note: In a CPE environment you can use `CC -x hip` instead of `hipcc`. This can be important for MPI applications so that you link the correct libraries for MPI. This is not relevant for this exercise here, but something to keep in mind depending on the system you work on.

## Optimizing DAXPY HIP

In this exercise, we will progressively make changes to optimize the DAXPY kernel on GPU. Any AMD GPU can be used to test this.

DAXPY Problem:
```
Z = aX + Y
```
where `a` is a scalar, and `X`, `Y` and `Z` are arrays of double precision values.

In DAXPY, we load 2 FP64 values (8 bytes each) and store 1 FP64 value (8 bytes). We can ignore the scalar load because it is constant. We have 1 multiplication and 1 addition operation for the 24 bytes moved per element of the array. This yields a low arithmetic intensity of `2/24 FLOP/byte`. Since this kernel is not compute bound, we will only measure the achieved memory bandwidth instead of FLOPS.

### Inputs
- `N`, the number of elements in `X`, `Y` and `Z`. `N` may be reset to suit some optimizations.
   Choose a sufficiently large array size to see some differences in performance.

### Build Code
```bash
git clone https://github.com/amd/HPCTrainingExamples.git
cd HPCTrainingExamples/HIP-Optimizations/daxpy
make
```

### Run exercises
```bash
./daxpy_1 10000000
./daxpy_2 10000000
./daxpy_3 10000000
./daxpy_4 10000000
./daxpy_5 10000000
```
Note: Depending on your allocated node configuration you may get better/more consistent performance numbers if you set affinity, for example: `ROCR_VISIBLE_DEVICES=0 numactl -C 0 -m 0 ./daxpy_<num>`. See the [Affinity exercises](https://github.com/amd/HPCTrainingExamples/tree/main/Affinity) for more details.

### Things to ponder about
Note: Comparing the code versions using `vimdiff daxpy_<ver1>.hip daxpy_<ver2>.hip` may help you understand the differences in each implementation.
#### `daxpy_1`
This shows a naive implementation of the daxpy problem on the GPU where only 1 wavefront is launched and the 64 work-items in that wavefront loop over the entire array and process 64 elements at a time. We expect this kernel to perform very poorly because it simply utilizes a part of 1 CU, and leaves the rest of the GPU unutilized.

#### `daxpy_2`
This time, we are launching multiple wavefronts, with each work-item now processing only 1 element of each array. This launches `N/64` wavefronts, enough to be scheduled on all CUs. We see a big improvement in performance here.

#### `daxpy_3`
In this experiment, we check to see if launching larger workgroups can help lower our kernel launch overhead because we launch fewer workgroups if each workgroup has 256 work-items. In this case too, an improvement in measured bandwidth achieved is seen.

#### `daxpy_4`
If we ensured that the array has a multiple of `BLOCK_SIZE` elements so that all work-items in each workgroup have an element to process, then we can avoid the conditional statement in the kernel. This could reduce some instructions in the kernel. Do we see any improvement? In this trivial case, this does not matter. Nevertheless, it is something we could keep in mind.

Question: What happens if `BLOCK_SIZE` is `1024`? Why?

#### `daxpy_5`
In this experiment, we will use the `double2` type in the kernel to see if the compiler can generate `global_load_dwordx4` instructions instead of `global_load_dwordx2` instructions. So, with the same number of load and store instructions, we are able to read/write two elements from each array in each thread. This should help amortize the cost of index calculations.

To show this difference, we need to generate the assembly for these two kernels. To generate the assembly code for these kernels, ensure that the `-g --save-temps` flags are passed to `hipcc`. Then you can find the assembly code in `daxpy_*-hip-amdgcn-amd-amdhsa-gfx*.s` files. Examining `daxpy_3` and `daxpy_5`, we see the two cases (edited here for clarity; exact assembly may differ based on hardware and compiler version, the important thing is to note the differences in `dwordx*`):

`daxpy_3`:
```asm
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
```asm
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
We observe that in the `daxpy_5` case, there are two `v_fmac_f64_e32` instructions as expected, one for each element being processed.

### Notes
- Before timing kernels, it is best to launch the kernel at least once as warmup so that those initial GPU launch latencies do not affect your timing measurements.
- You may find that the various optimizations work differently on MI210 vs MI300A devices, and this may be due to differences in hardware architecture.
- **CPX mode note:** On MI300A systems configured in CPX mode (where each XCD acts as a logical GPU), the speedups from daxpy_2 to daxpy_3 may not be visible because the single XCD's memory bandwidth (~660 GB/s) is already saturated with 64-thread workgroups.
