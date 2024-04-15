## Exercise 2: LDS Occupancy Limiter

Simple kernel implementing a version of yAx, to demonstrate the downside of allocating a large 
amount of LDS, and the benefit of using a smaller amount of LDS due to occupancy limits.

**Note:** This exercise was tested on a system with MI210s, on omniperf version `1.0.10` and ROCm 5.7.0
<details>
<summary><h3>Background: Acronyms and terms used in this exercise</h3></summary>
     <ul>
          <li><strong>Wavefront:</strong> A collection of threads, usually 64.</li>
          <li><strong>Workgroup:</strong> A collection of Wavefronts (at least 1), which can be scheduled on a Compute Unit (CU)</li>
          <li><strong>LDS:</strong> Local Data Store is Shared Memory that is accessible to the entire workgroup on a Compute Unit (CU)</li>
          <li><strong>CU:</strong> The Compute Unit is responsible for executing the User's kernels</li>
          <li><strong>SPI:</strong> Shader Processor Input, also referred to as the Workgroup Manager, is responsible for scheduling workgroups on Compute Units</li>
          <li><strong>Occupancy:</strong> A measure of how many wavefronts are executing on the GPU on average through the duration of the kernel</li>
          <li><strong>PoP:</strong> Percent of Peak refers to the ratio of an achieved value and a theoretical or actual maximum. In terms of occupancy, it is how many wavefronts on average were on the device divided by how many can fit on the device.
          <li><strong>yAx:</strong> a vector-matrix-vector product, y*A*x, where y and x are vectors, and A is a matrix</li>
          <li><strong>FP(32/16):</strong> 32- or 16-bit Floating Point numeric types</li>
          <li><strong>FLOPs:</strong> Floating Point Operations Per second</li>
          <li><strong>HBM:</strong> High Bandwidth Memory is globally accessible from the GPU, and is a level of memory above the L2 cache</li>
     </ul>
</details>

### Initial Roofline Analysis
In this exercise we're using a problem code that is slightly different than where we left off in Exercise 1. 
Regardless, to get started we need to get a roofline by running:

```
omniperf profile -n problem_roof_only --roof-only --kernel-names -- ./problem.exe
```
The plots will appear as PDF files in the `./workloads/problem_roof_only/mi200` directory, if generated on MI200 hardware.

For convenience, the resulting plots on a representative system are below:
| Roofline Type | Roofline Legend                                    | Roofline Plot                                        |
|---------------|----------------------------------------------------|------------------------------------------------------|
|FP32           |<img src="exercise1_problem_kernelName_legend.png"/>|<img src="exercise2_problem_roofline_fp32.png"/>      |
|FP16/INT8      |<img src="exercise1_problem_kernelName_legend.png"/>|<img src="exercise2_problem_roofline_int8_fp16.png"/> |

We see that there looks to be room for improvement here. We'll use omniperf to see what the current limiters are.

### Exercise Instructions:
First, we should get an idea of the code's runtime:

```
make
./problem.exe
```
(*simulated output*)
```
yAx time: 140 ms
```
This problem.cpp uses LDS allocations to move the x vector closer to the compute resources, a common optimization.
However, we see that it ends up slower than the previous solution that didn't use LDS at all.
In kernels that request a lot of LDS, it is common to see that the LDS usage limits the occupancy of the kernel.
That is, more wavefronts cannot be resident on the device, because all of them need more LDS than is available.
We need to confirm this hypothesis, let's start by running:

```
omniperf profile -n problem --no-roof -- ./problem.exe
```
The usage of `omniperf profile` arguments can be found [here](https://amdresearch.github.io/omniperf/profiling.html#omniperf-profiling), or by running `omniperf profile --help`.

This `omniperf profile` command will take a minute or two to run, as omniperf must run your code a few times to collect all the hardware counters.

>**Note:** For large scientific codes, it can be useful to profile a small representative workload if possible, as profiling a full run may take prohibitively long.

Once the profiling run completes, let's take a look at the occupancy stats related to LDS allocations:

```
omniperf analyze -p workloads/problem/mi200 --dispatch 1 --metric 2.1.26 6.2.7
```
The metrics we're looking at are:
- `2.1.26` Wavefront occupancy -- a measure of how many wavefronts, on average, are active on the device
- `6.2.7` SPI: Insufficient CU LDS -- indicates whether wavefronts are not able to be scheduled due to insufficient LDS

The SPI section (`6.2`) generally shows what resources limit occupancy, while Wavefront occupancy (`2.1.26`) shows how severely occupancy is limited in general. 
The SPI 'insufficient' fields are typically either zero or very large numbers (on the order of 1 million), with large numbers indicating some resource preventing wavefronts from scheduling.
If more than one field is nonzero, the relative magnitude of the nonzero fields correspond to how severely the resources are limiting occupancy, but if only one field is nonzero it is difficult to say how severely that field is limiting occupancy.

The output of the `omniperf analyze` command should look similar to this:

```
--------
Analyze
--------


--------------------------------------------------------------------------------
0. Top Stat
╒════╤══════════════════════════════════════════╤═════════╤══════════════╤══════════════╤══════════════╤════════╕
│    │ KernelName                               │   Count │      Sum(ns) │     Mean(ns) │   Median(ns) │    Pct │
╞════╪══════════════════════════════════════════╪═════════╪══════════════╪══════════════╪══════════════╪════════╡
│  0 │ yax(double*, double*, double*, int, int, │    1.00 │ 175427205.00 │ 175427205.00 │ 175427205.00 │ 100.00 │
│    │  double*)                                │         │              │              │              │        │
╘════╧══════════════════════════════════════════╧═════════╧══════════════╧══════════════╧══════════════╧════════╛


--------------------------------------------------------------------------------
2. System Speed-of-Light
2.1 Speed-of-Light
╒═════════╤════════════════╤═════════╤════════════╤═════════╤═══════╕
│ Index   │ Metric         │   Value │ Unit       │    Peak │   PoP │
╞═════════╪════════════════╪═════════╪════════════╪═════════╪═══════╡
│ 2.1.26  │ Wave Occupancy │  102.70 │ Wavefronts │ 3328.00 │  3.09 │
╘═════════╧════════════════╧═════════╧════════════╧═════════╧═══════╛


--------------------------------------------------------------------------------
6. Shader Processor Input (SPI)
6.2 SPI Resource Allocation
╒═════════╤═════════════════════╤═══════════════╤═══════════════╤═══════════════╤════════╕
│ Index   │ Metric              │           Avg │           Min │           Max │ Unit   │
╞═════════╪═════════════════════╪═══════════════╪═══════════════╪═══════════════╪════════╡
│ 6.2.7   │ Insufficient CU LDS │ 6015745446.00 │ 6015745446.00 │ 6015745446.00 │ Cu     │
╘═════════╧═════════════════════╧═══════════════╧═══════════════╧═══════════════╧════════╛
```
Looking through this data we see:
- Wavefront occupancy (`2.1.26`) is 3%, which is very low
- Insufficient CU LDS (`6.2.7`) contains a very large number, which indicates our occupancy is currently limited by LDS allocations.

There are two solution directories, which correspond to two ways that this occupancy limit can be addressed.
First, we have `solution-no-lds`, which completely removes the LDS usage. Let's build and run this solution:

```
cd solution-no-lds
make
./solution.exe
```
(*simulated output*)
```
yAx time: 70 ms
```

We see that the runtime is much better for this solution than the problem, let's see if removing LDS did indeed increase occupancy:

```
omniperf profile -n solution --no-roof -- ./solution.exe
```
(*output omitted*)

Once the profile command completes, run:
```
omniperf analyze -p workloads/solution/mi200 --dispatch 1 --metric 2.1.26 6.2.7
```

The output should look something like:

```
--------
Analyze
--------


--------------------------------------------------------------------------------
0. Top Stat
╒════╤══════════════════════════════════════════╤═════════╤═════════════╤═════════════╤══════════════╤════════╕
│    │ KernelName                               │   Count │     Sum(ns) │    Mean(ns) │   Median(ns) │    Pct │
╞════╪══════════════════════════════════════════╪═════════╪═════════════╪═════════════╪══════════════╪════════╡
│  0 │ yax(double*, double*, double*, int, int, │    1.00 │ 70512671.00 │ 70512671.00 │  70512671.00 │ 100.00 │
│    │  double*)                                │         │             │             │              │        │
╘════╧══════════════════════════════════════════╧═════════╧═════════════╧═════════════╧══════════════╧════════╛


--------------------------------------------------------------------------------
2. System Speed-of-Light
2.1 Speed-of-Light
╒═════════╤════════════════╤═════════╤════════════╤═════════╤═══════╕
│ Index   │ Metric         │   Value │ Unit       │    Peak │   PoP │
╞═════════╪════════════════╪═════════╪════════════╪═════════╪═══════╡
│ 2.1.26  │ Wave Occupancy │  445.33 │ Wavefronts │ 3328.00 │ 13.38 │
╘═════════╧════════════════╧═════════╧════════════╧═════════╧═══════╛


--------------------------------------------------------------------------------
6. Shader Processor Input (SPI)
6.2 SPI Resource Allocation
╒═════════╤═════════════════════╤═══════╤═══════╤═══════╤════════╕
│ Index   │ Metric              │   Avg │   Min │   Max │ Unit   │
╞═════════╪═════════════════════╪═══════╪═══════╪═══════╪════════╡
│ 6.2.7   │ Insufficient CU LDS │  0.00 │  0.00 │  0.00 │ Cu     │
╘═════════╧═════════════════════╧═══════╧═══════╧═══════╧════════╛
```
Looking through this data we see:
- Wave occupancy (`2.1.26`) is 10% higher than in problem.cpp
- Insufficient CU LDS (`6.2.7`) is now zero, indicating solution-no-lds is not occupancy limited by LDS allocations.

Can we get some runtime advantage from using smaller LDS allocations?

This is the solution implemented in the `solution` directory:
```
cd ../solution
make
./solution.exe
```
(*simulated output*)
```
yAx time: 50 ms
```
This solution, rather than removing the LDS allocation, simply reduces the amount of LDS requested to address the occupancy limit.
This gives us the benefit of having some data pulled closer than it was in `solution-no-lds` which is validated through the speedup we see.
But is this solution still occupancy limited by LDS?

```
omniperf profile -n solution --no-roof -- ./solution.exe
```
(*output omitted*)

Once the profile command completes, run:
```
omniperf analyze -p workloads/solution/mi200 --dispatch 1 --metric 2.1.26 6.2.7
```
The output should look something like:

```
--------
Analyze
--------


--------------------------------------------------------------------------------
0. Top Stat
╒════╤══════════════════════════════════════════╤═════════╤═════════════╤═════════════╤══════════════╤════════╕
│    │ KernelName                               │   Count │     Sum(ns) │    Mean(ns) │   Median(ns) │    Pct │
╞════╪══════════════════════════════════════════╪═════════╪═════════════╪═════════════╪══════════════╪════════╡
│  0 │ yax(double*, double*, double*, int, int, │    1.00 │ 50366185.00 │ 50366185.00 │  50366185.00 │ 100.00 │
│    │  double*)                                │         │             │             │              │        │
╘════╧══════════════════════════════════════════╧═════════╧═════════════╧═════════════╧══════════════╧════════╛


--------------------------------------------------------------------------------
2. System Speed-of-Light
2.1 Speed-of-Light
╒═════════╤════════════════╤═════════╤════════════╤═════════╤═══════╕
│ Index   │ Metric         │   Value │ Unit       │    Peak │   PoP │
╞═════════╪════════════════╪═════════╪════════════╪═════════╪═══════╡
│ 2.1.26  │ Wave Occupancy │  487.32 │ Wavefronts │ 3328.00 │ 14.64 │
╘═════════╧════════════════╧═════════╧════════════╧═════════╧═══════╛


--------------------------------------------------------------------------------
6. Shader Processor Input (SPI)
6.2 SPI Resource Allocation
╒═════════╤═════════════════════╤═══════╤═══════╤═══════╤════════╕
│ Index   │ Metric              │   Avg │   Min │   Max │ Unit   │
╞═════════╪═════════════════════╪═══════╪═══════╪═══════╪════════╡
│ 6.2.7   │ Insufficient CU LDS │  0.00 │  0.00 │  0.00 │ Cu     │
╘═════════╧═════════════════════╧═══════╧═══════╧═══════╧════════╛
```
Looking at this data we see:
- Wave Occupancy (`2.1.26`) is even higher than before
- Insufficient CU LDS (`6.2.7`) shows we are not occupancy limited by LDS allocations.

Pulling some data from global device memory to LDS can be an effective optimization strategy, if occupancy limits are carefully avoided.

### Solution Roofline
Let's take a look at the roofline for `solution`, which can be generated with:

```
omniperf profile -n solution_roof_only --roof-only -- ./solution.exe
```
The plots will appear as PDF files in the `./workloads/problem_roof_only/mi200` directory, if generated on MI200 hardware.

The plots are shown here:
| Roofline Type | Roofline Legend                                    | Roofline Plot                                        |
|---------------|----------------------------------------------------|------------------------------------------------------|
|FP32           |<img src="exercise1_problem_kernelName_legend.png"/>|<img src="exercise2_solution_roofline_fp32.png"/>      |
|FP16/INT8      |<img src="exercise1_problem_kernelName_legend.png"/>|<img src="exercise2_solution_roofline_int8_fp16.png"/> |

We see that there is still room to move the solution roofline up towards the bandwidth limit.

### Roofline Comparison
| Roofline Type | Problem Roofline                                     | Solution Roofline                                      |
|---------------|------------------------------------------------------|--------------------------------------------------------|
| FP32          | <img src="exercise2_problem_roofline_fp32.png"/>     | <img src="exercise2_solution_roofline_fp32.png"/>      |
| FP16/INT8     | <img src="exercise2_problem_roofline_int8_fp16.png"/>| <img src="exercise2_solution_roofline_int8_fp16.png"/> |

Again, we see that the solution's optimizations have resulted in the kernel moving up in the roofline, meaning the solution executes more efficiently than the problem.

### Summary and Take-aways

Using LDS can be very helpful in reducing global memory reads where you have repeated use of the same data. 
However, large LDS allocations can also negatively impact performance by limiting the amount of 
wavefronts that can be resident in the device at any given time. Be wary of LDS usage, and check 
the SPI stats to ensure your LDS usage is not negatively impacting occupancy.
