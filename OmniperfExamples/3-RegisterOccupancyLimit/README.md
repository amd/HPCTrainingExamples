## Exercise 3: Register Occupancy Limiter

More complex yAx implementation to demonstrate a register limited kernel using an innocuous looking
function call.

**Note:** This exercise was tested on a system with MI210s, on omniperf version `1.0.10` and ROCm 5.7.0

### Initial Roofline Analysis
This kernel is slightly different from the one we used in previous exercises. Let's see how well it measures up in the roofline:

| Roofline Type | Roofline Legend                                    | Roofline Plot                                        |
|---------------|----------------------------------------------------|------------------------------------------------------|
|FP32           |<img src="exercise1_problem_kernelName_legend.png"/>|<img src="exercise3_problem_roofline_fp32.png"/>      |
|FP16/INT8      |<img src="exercise1_problem_kernelName_legend.png"/>|<img src="exercise3_problem_roofline_int8_fp16.png"/> |

You can generate these plots by running:
```
omniperf profile -n problem_roof_only --roof-only --kernel-names -- ./problem.exe
```
The plots will appear as PDF files in the `./workloads/problem_roof_only/mi200` directory, if generated on MI200 hardware.

We see that the kernel is still a considerable amount below the maximum achievable bandwidth, so there should still be room for improvement.

### Exercise Instructions:
Let's get an idea of the runtime of this code:

```
make
./problem.exe
```
(*simulated output*)
```
yAx time 79 ms
```
We see that this kernel seems to be on par with some of our other exercises, but let's see what omniperf shows us:

```
omniperf profile -n problem --no-roof -- ./problem.exe
```
(*lots of output from this command*)
```
omniperf analyze -p workloads/problem/mi200 --dispatch 1 --metric 2.1.26 6.2.5 7.1.5 7.1.6 7.1.7
```
- `2.1.26` Shows Wavefront occupancy
- `6.2.5` Shows Insufficient SIMD VGPRs -- indicating if this kernel is occupancy limited by VGPR usage
- `7.1.5-7` Shows the register usage: VGPRs, SGPRs, and AGPRs
```
--------
Analyze
--------


--------------------------------------------------------------------------------
0. Top Stat
╒════╤══════════════════════════════════════════╤═════════╤═════════════╤═════════════╤══════════════╤════════╕
│    │ KernelName                               │   Count │     Sum(ns) │    Mean(ns) │   Median(ns) │    Pct │
╞════╪══════════════════════════════════════════╪═════════╪═════════════╪═════════════╪══════════════╪════════╡
│  0 │ yax(double*, double*, double*, int, int, │    1.00 │ 76983902.00 │ 76983902.00 │  76983902.00 │ 100.00 │
│    │  double*)                                │         │             │             │              │        │
╘════╧══════════════════════════════════════════╧═════════╧═════════════╧═════════════╧══════════════╧════════╛


--------------------------------------------------------------------------------
2. System Speed-of-Light
2.1 Speed-of-Light
╒═════════╤════════════════╤═════════╤════════════╤═════════╤═══════╕
│ Index   │ Metric         │   Value │ Unit       │    Peak │   PoP │
╞═════════╪════════════════╪═════════╪════════════╪═════════╪═══════╡
│ 2.1.26  │ Wave Occupancy │  438.00 │ Wavefronts │ 3328.00 │ 13.16 │
╘═════════╧════════════════╧═════════╧════════════╧═════════╧═══════╛


--------------------------------------------------------------------------------
6. Shader Processor Input (SPI)
6.2 SPI Resource Allocation
╒═════════╤═════════════════════════╤═════════════╤═════════════╤═════════════╤════════╕
│ Index   │ Metric                  │         Avg │         Min │         Max │ Unit   │
╞═════════╪═════════════════════════╪═════════════╪═════════════╪═════════════╪════════╡
│ 6.2.5   │ Insufficient SIMD VGPRs │ 13733460.00 │ 13733460.00 │ 13733460.00 │ Simd   │
╘═════════╧═════════════════════════╧═════════════╧═════════════╧═════════════╧════════╛


--------------------------------------------------------------------------------
7. Wavefront
7.1 Wavefront Launch Stats
╒═════════╤══════════╤════════╤════════╤════════╤═══════════╕
│ Index   │ Metric   │    Avg │    Min │    Max │ Unit      │
╞═════════╪══════════╪════════╪════════╪════════╪═══════════╡
│ 7.1.5   │ VGPRs    │  92.00 │  92.00 │  92.00 │ Registers │
├─────────┼──────────┼────────┼────────┼────────┼───────────┤
│ 7.1.6   │ AGPRs    │ 132.00 │ 132.00 │ 132.00 │ Registers │
├─────────┼──────────┼────────┼────────┼────────┼───────────┤
│ 7.1.7   │ SGPRs    │  48.00 │  48.00 │  48.00 │ Registers │
╘═════════╧══════════╧════════╧════════╧════════╧═══════════╛
```
Looking at this data, we see:
- Insufficient SIMD VGPRs (`6.2.5`) shows a large number (on the order of 10 million), which indicates our kernel is occupancy limited by VGPR register usage.
- VGPRs (`7.1.5`) shows we are using a lot of VGPRs and AGPRs (`7.1.6`) shows we are using a lot of AGPRs

This is due to a call to `assert` that checks if our result is zeroed out on device.
We need to check this hypothesis, let's look at the solution code:

```
cd solution
make
./solution.exe
```
(*simulated output*)
```
yAx time: 70 ms
```

Our runtime gets better from removing the `assert`, but we should also check that omniperf reports that our limiters are gone:

```
omniperf profile -n solution --no-roof -- ./solution.exe
```
(*omitted output*)
```
omniperf analyze -p workloads/solution/mi200 --dispatch 1 --metric 2.1.26 6.2.5 7.1.5 7.1.6 7.1.7
```
The output of this command should look something like:

```
--------
Analyze
--------


--------------------------------------------------------------------------------
0. Top Stat
╒════╤══════════════════════════════════════════╤═════════╤═════════════╤═════════════╤══════════════╤════════╕
│    │ KernelName                               │   Count │     Sum(ns) │    Mean(ns) │   Median(ns) │    Pct │
╞════╪══════════════════════════════════════════╪═════════╪═════════════╪═════════════╪══════════════╪════════╡
│  0 │ yax(double*, double*, double*, int, int, │    1.00 │ 69815871.00 │ 69815871.00 │  69815871.00 │ 100.00 │
│    │  double*)                                │         │             │             │              │        │
╘════╧══════════════════════════════════════════╧═════════╧═════════════╧═════════════╧══════════════╧════════╛


--------------------------------------------------------------------------------
2. System Speed-of-Light
2.1 Speed-of-Light
╒═════════╤════════════════╤═════════╤════════════╤═════════╤═══════╕
│ Index   │ Metric         │   Value │ Unit       │    Peak │   PoP │
╞═════════╪════════════════╪═════════╪════════════╪═════════╪═══════╡
│ 2.1.26  │ Wave Occupancy │  444.10 │ Wavefronts │ 3328.00 │ 13.34 │
╘═════════╧════════════════╧═════════╧════════════╧═════════╧═══════╛


--------------------------------------------------------------------------------
6. Shader Processor Input (SPI)
6.2 SPI Resource Allocation
╒═════════╤═════════════════════════╤═══════╤═══════╤═══════╤════════╕
│ Index   │ Metric                  │   Avg │   Min │   Max │ Unit   │
╞═════════╪═════════════════════════╪═══════╪═══════╪═══════╪════════╡
│ 6.2.5   │ Insufficient SIMD VGPRs │  0.00 │  0.00 │  0.00 │ Simd   │
╘═════════╧═════════════════════════╧═══════╧═══════╧═══════╧════════╛


--------------------------------------------------------------------------------
7. Wavefront
7.1 Wavefront Launch Stats
╒═════════╤══════════╤═══════╤═══════╤═══════╤═══════════╕
│ Index   │ Metric   │   Avg │   Min │   Max │ Unit      │
╞═════════╪══════════╪═══════╪═══════╪═══════╪═══════════╡
│ 7.1.5   │ VGPRs    │ 32.00 │ 32.00 │ 32.00 │ Registers │
├─────────┼──────────┼───────┼───────┼───────┼───────────┤
│ 7.1.6   │ AGPRs    │  0.00 │  0.00 │  0.00 │ Registers │
├─────────┼──────────┼───────┼───────┼───────┼───────────┤
│ 7.1.7   │ SGPRs    │ 96.00 │ 96.00 │ 96.00 │ Registers │
╘═════════╧══════════╧═══════╧═══════╧═══════╧═══════════╛
```
Looking at this data, we see:
- Insufficient SIMD VGPRs (`6.2.5`) shows we are no longer occupancy limited by VGPR usage.
- VGPRs (`7.1.5`) and AGPRs (`7.1.6`) show considerably fewer vector registers.
- SGPRs (`7.1.7`) shows a 2x increase over the previous implementation.
- Wave Occupancy (`2.1.26`) shows our occupancy increased only slightly from the previous implementation.

### Solution Roofline
Let's see how the solution stacks up in the roofline:

| Roofline Type | Roofline Legend                                    | Roofline Plot                                        |
|---------------|----------------------------------------------------|------------------------------------------------------|
|FP32           |<img src="exercise1_problem_kernelName_legend.png"/>|<img src="exercise3_solution_roofline_fp32.png"/>      |
|FP16/INT8      |<img src="exercise1_problem_kernelName_legend.png"/>|<img src="exercise3_solution_roofline_int8_fp16.png"/> |


You can generate these plots with:

```
omniperf profile -n problem_roof_only --roof-only --kernel-names -- ./problem.exe
```
The plots will appear as PDF files in the `./workloads/problem_roof_only/mi200` directory, if generated on MI200 hardware.

We see there is still room for improvement in the solution, as this kernel is not getting the maximum achievable bandwidth.

### Roofline Comparison

| Roofline Type | Problem Roofline                                     | Solution Roofline                                      |
|---------------|------------------------------------------------------|--------------------------------------------------------|
| FP32          | <img src="exercise3_problem_roofline_fp32.png"/>     | <img src="exercise3_solution_roofline_fp32.png"/>      |
| FP16/INT8     | <img src="exercise3_problem_roofline_int8_fp16.png"/>| <img src="exercise3_solution_roofline_int8_fp16.png"/> |

The most notable change between these rooflines is that the L1/L2 arithmetic intensity spread is more pronounced in the problem, which shows that the call to `assert` was causing more data to be moved to the L1, while not adding floating-point operations.

**Note:** Arithmetic Intensity is computed as `(total floating point operations)/(total data movement)`

### Summary and Take-aways

Function calls inside kernels can have surprisingly adverse performance side-effects. Calling assert, printf and even excessive use of math functions (e.g. pow, sin, cos) can limit performance in difficult-to-predict ways. If you see unexpected resource usage, try eliminating these sorts of function calls.
