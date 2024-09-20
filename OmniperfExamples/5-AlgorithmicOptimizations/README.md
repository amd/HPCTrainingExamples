## Exercise 5: Algorithmic Optimizations

A simple yAx kernel, and more efficient, but more complex yAx kernel to demonstrate algorithmic improvements.

**Note:** This exercise was tested on a system with MI210s, on omniperf version `2.0.0` and ROCm `6.1.2`
**Omniperf `2.0.0` is incompatible with ROCm versions lesser than `6.0.0`**

<details>
<summary><h3>Background: Acronyms and terms used in this exercise</h3></summary>
     <ul>
          <li><strong>L1:</strong> Level 1 Cache, the first level cache local to the Compute Unit (CU). If requested data is not found in the L1, the request goes to the L2</li>
          <li><strong>L2:</strong> Level 2 Cache, the second level cache, which is shared by all Compute Units (CUs) on a GPU. If requested data is not found in the L2, the request goes to HBM</li>
          <li><strong>HBM:</strong> High Bandwidth Memory is globally accessible from the GPU, and is a level of memory above the L2 cache</li>
          <li><strong>CU:</strong> The Compute Unit is responsible for executing the User's kernels </li> 
          <li><strong>yAx:</strong> a vector-matrix-vector product, y*A*x, where y and x are vectors, and A is a matrix</li>
          <li><strong>FP(32/16):</strong> 32- or 16-bit Floating Point numeric types</li>
     </ul>
</details>

<details>
<summary><h3>Background: yAx Algorithmic Improvement Explanation</h3></summary>
 Our approach up to this point could be described as having each thread sum up a row, as illustrated below:
 <img src="threadrows.PNG"/>
 However, this is not efficient in the way the parallelism is expressed. Namely, we could add up all the partial sums for each row in parallel.
 This would make our approach to be: give a rows to wavefronts, and have the threads inside each wavefront sum up partial sums in parallel.
 Then, we reduce the partial sums atomically with shared memory, before completing the computation and reducing the final answer using global atomics.
 This approach expresses more of the parallelism that is available, and would look something like the figure below:
 <img src="wavefrontrow.PNG"/>
 The expressed parallelism in each approach roughly corresponds to the number of red arrows in each figure.
</details>

### Initial Roofline Analysis
We should start by doing a roofline to see where the problem executable stands.
These plots can be generated with:

```
omniperf profile -n problem_roof_only --roof-only --kernel-names -- ./problem.exe
```
The plots will appear as PDF files in the `./workloads/problem_roof_only/MI200` directory, if generated on MI200 hardware.

They are also provided below for easy reference:

| Roofline Type | Roofline Legend                                    | Roofline Plot                                        |
|---------------|----------------------------------------------------|------------------------------------------------------|
|FP32/FP64      |<img src="exercise1_problem_kernelName_legend.png"/>|<img src="exercise5_problem_roofline_fp32.png"/>      |
|FP16/INT8      |<img src="exercise1_problem_kernelName_legend.png"/>|<img src="exercise5_problem_roofline_int8_fp16.png"/> |

The performance of this kernel looks pretty close to being HBM bandwidth bound.
In the case of algorithmic optimizations, there may not be obvious evidence other than a suspicion that poor 
usage of hardware resources may be improved by changing the overall approach. 
In this case, we should be able to make better usage of both L1 and L2 resources by using wavefronts more efficiently 
to better parallelize our computation.

### Exercise Instructions:

To start, let's profile `problem.exe`:

```
make
./problem.exe
```
(*simulated output*)
```
yAx time 12 ms
```

This should be in line with our last solution. From the last exercise, we saw this output from `omniperf analyze` for this kernel:

```
  ___                  _                  __
 / _ \ _ __ ___  _ __ (_)_ __   ___ _ __ / _|
| | | | '_ ` _ \| '_ \| | '_ \ / _ \ '__| |_
| |_| | | | | | | | | | | |_) |  __/ |  |  _|
 \___/|_| |_| |_|_| |_|_| .__/ \___|_|  |_|
                        |_|

Analysis mode = cli
[analysis] deriving Omniperf metrics...

--------------------------------------------------------------------------------
0. Top Stats
0.1 Top Kernels
╒════╤══════════════════════════════════════════╤═════════╤═════════════╤═════════════╤══════════════╤════════╕
│    │ Kernel_Name                              │   Count │     Sum(ns) │    Mean(ns) │   Median(ns) │    Pct │
╞════╪══════════════════════════════════════════╪═════════╪═════════════╪═════════════╪══════════════╪════════╡
│  0 │ yax(double*, double*, double*, int, int, │    1.00 │ 12364156.00 │ 12364156.00 │  12364156.00 │ 100.00 │
│    │  double*) [clone .kd]                    │         │             │             │              │        │
╘════╧══════════════════════════════════════════╧═════════╧═════════════╧═════════════╧══════════════╧════════╛
0.2 Dispatch List
╒════╤═══════════════╤═══════════════════════════════════════════════════════════════╤══════════╕
│    │   Dispatch_ID │ Kernel_Name                                                   │   GPU_ID │
╞════╪═══════════════╪═══════════════════════════════════════════════════════════════╪══════════╡
│  0 │             1 │ yax(double*, double*, double*, int, int, double*) [clone .kd] │        8 │
╘════╧═══════════════╧═══════════════════════════════════════════════════════════════╧══════════╛


--------------------------------------------------------------------------------
16. Vector L1 Data Cache
16.1 Speed-of-Light
╒═════════════╤═════════════╤═══════╤═════════════╕
│ Metric_ID   │ Metric      │   Avg │ Unit        │
╞═════════════╪═════════════╪═══════╪═════════════╡
│ 16.1.0      │ Hit rate    │ 49.98 │ Pct of peak │
├─────────────┼─────────────┼───────┼─────────────┤
│ 16.1.1      │ Bandwidth   │ 12.29 │ Pct of peak │
├─────────────┼─────────────┼───────┼─────────────┤
│ 16.1.2      │ Utilization │ 98.12 │ Pct of peak │
├─────────────┼─────────────┼───────┼─────────────┤
│ 16.1.3      │ Coalescing  │ 25.00 │ Pct of peak │
╘═════════════╧═════════════╧═══════╧═════════════╛


--------------------------------------------------------------------------------
17. L2 Cache
17.1 Speed-of-Light
╒═════════════╤═══════════════════════════════╤════════╤════════╕
│ Metric_ID   │ Metric                        │    Avg │ Unit   │
╞═════════════╪═══════════════════════════════╪════════╪════════╡
│ 17.1.0      │ Utilization                   │  98.56 │ Pct    │
├─────────────┼───────────────────────────────┼────────┼────────┤
│ 17.1.1      │ Bandwidth                     │  10.03 │ Pct    │
├─────────────┼───────────────────────────────┼────────┼────────┤
│ 17.1.2      │ Hit Rate                      │   0.52 │ Pct    │
├─────────────┼───────────────────────────────┼────────┼────────┤
│ 17.1.3      │ L2-Fabric Read BW             │ 694.86 │ Gb/s   │
├─────────────┼───────────────────────────────┼────────┼────────┤
│ 17.1.4      │ L2-Fabric Write and Atomic BW │   0.00 │ Gb/s   │
╘═════════════╧═══════════════════════════════╧════════╧════════╛

```
Looking at this data again, we see:
- L1 Cache Hit (`16.1.0`) is about 50%, which is fairly low for a "well performing" kernel.
- L2 Cache Hit (`17.1.2`) is about 0%, which is very low to consider this kernel "well performing".

This data indicates that we should be able to make better usage of our memory system, so let's apply the algorithmic optimization present in `solution.cpp`:

```
cd solution
make
./solution.exe
```
(*simulated output*)
```
yAx time: 7.7 ms
```

It should be noted again that algorithmic optimizations are usually the most expensive optimizations to implement, as they usually entail
re-conceptualizing the problem in a way that allows for a more efficient solution. However, as we see here, algorithmic optimization _can_
result in impressive speedups. A better runtime is not proof that we are using our caches more efficiently, we have to profile the solution:

```
omniperf profile -n solution --no-roof -- ./solution.exe
```
(*output omitted*)
```
omniperf analyze -p workloads/solution/MI200 --dispatch 1 --block 16.1 17.1
```
The output for the solution should look something like:
```
  ___                  _                  __
 / _ \ _ __ ___  _ __ (_)_ __   ___ _ __ / _|
| | | | '_ ` _ \| '_ \| | '_ \ / _ \ '__| |_
| |_| | | | | | | | | | | |_) |  __/ |  |  _|
 \___/|_| |_| |_|_| |_|_| .__/ \___|_|  |_|
                        |_|

   INFO Analysis mode = cli
   INFO [analysis] deriving Omniperf metrics...

--------------------------------------------------------------------------------
0. Top Stats
0.1 Top Kernels
╒════╤══════════════════════════════════════════╤═════════╤════════════╤════════════╤══════════════╤════════╕
│    │ Kernel_Name                              │   Count │    Sum(ns) │   Mean(ns) │   Median(ns) │    Pct │
╞════╪══════════════════════════════════════════╪═════════╪════════════╪════════════╪══════════════╪════════╡
│  0 │ yax(double*, double*, double*, int, int, │    1.00 │ 7774568.00 │ 7774568.00 │   7774568.00 │ 100.00 │
│    │  double*) [clone .kd]                    │         │            │            │              │        │
╘════╧══════════════════════════════════════════╧═════════╧════════════╧════════════╧══════════════╧════════╛
0.2 Dispatch List
╒════╤═══════════════╤═══════════════════════════════════════════════════════════════╤══════════╕
│    │   Dispatch_ID │ Kernel_Name                                                   │   GPU_ID │
╞════╪═══════════════╪═══════════════════════════════════════════════════════════════╪══════════╡
│  0 │             1 │ yax(double*, double*, double*, int, int, double*) [clone .kd] │        2 │
╘════╧═══════════════╧═══════════════════════════════════════════════════════════════╧══════════╛


--------------------------------------------------------------------------------
16. Vector L1 Data Cache
16.1 Speed-of-Light
╒═════════════╤═════════════╤═══════╤═════════════╕
│ Metric_ID   │ Metric      │   Avg │ Unit        │
╞═════════════╪═════════════╪═══════╪═════════════╡
│ 16.1.0      │ Hit rate    │ 71.52 │ Pct of peak │
├─────────────┼─────────────┼───────┼─────────────┤
│ 16.1.1      │ Bandwidth   │ 39.06 │ Pct of peak │
├─────────────┼─────────────┼───────┼─────────────┤
│ 16.1.2      │ Utilization │ 97.85 │ Pct of peak │
├─────────────┼─────────────┼───────┼─────────────┤
│ 16.1.3      │ Coalescing  │ 25.00 │ Pct of peak │
╘═════════════╧═════════════╧═══════╧═════════════╛


--------------------------------------------------------------------------------
17. L2 Cache
17.1 Speed-of-Light
╒═════════════╤═══════════════════════════════╤═════════╤════════╕
│ Metric_ID   │ Metric                        │     Avg │ Unit   │
╞═════════════╪═══════════════════════════════╪═════════╪════════╡
│ 17.1.0      │ Utilization                   │   91.55 │ Pct    │
├─────────────┼───────────────────────────────┼─────────┼────────┤
│ 17.1.1      │ Bandwidth                     │   20.44 │ Pct    │
├─────────────┼───────────────────────────────┼─────────┼────────┤
│ 17.1.2      │ Hit Rate                      │   21.23 │ Pct    │
├─────────────┼───────────────────────────────┼─────────┼────────┤
│ 17.1.3      │ L2-Fabric Read BW             │ 1110.67 │ Gb/s   │
├─────────────┼───────────────────────────────┼─────────┼────────┤
│ 17.1.4      │ L2-Fabric Write and Atomic BW │    0.00 │ Gb/s   │
╘═════════════╧═══════════════════════════════╧═════════╧════════╛

```
Looking at this data, we see:
- L1 Cache Hit (`16.1.0`) shows 71.52%, which is an increase of 1.43x over 49.98% for problem.
- L2 Cache Hit (`17.1.2`) shows 21.23%, which is an increase of 40x over 0.52% for problem.
- L2-Fabric Read BW (`17.1.3`) shows 1110.67 Gb/s, an increase of 1.6x over 694.86 Gb/s for problem.

Notice that the ratio between the runtimes in this case: 12/7.7 = 1.56x, which aligns closely with the L2-Fabric Read BW increases, suggesting this kernel is bounded primarily by memory bandwidth.

### Solution Roofline Analysis
As a final step, we should check how this new implementation stacks up with the roofline.
These plots can be generated with:

```
omniperf profile -n solution_roof_only --roof-only --kernel-names -- ./solution.exe
```
The plots will appear as PDF files in the `./workloads/solution_roof_only/MI200` directory, if generated on MI200 hardware.

They are also provided below for easy reference:

| Roofline Type | Roofline Legend                                    | Roofline Plot                                        |
|---------------|----------------------------------------------------|------------------------------------------------------|
|FP32/FP64      |<img src="exercise1_problem_kernelName_legend.png"/>|<img src="exercise5_solution_roofline_fp32.png"/>      |
|FP16/INT8      |<img src="exercise1_problem_kernelName_legend.png"/>|<img src="exercise5_solution_roofline_int8_fp16.png"/> |

As the Omniperf stats indicate, we are moving most data through the L1, which shows in the roofline as a decrease in Arithmetic Intensity for that cache layer.
We have a high hit rate in L1, with a fairly low hit rate in L2, and we end up having to go to HBM much less frequently than we did previously, 
thus our HBM bandwidth has decreased as a result of more efficient usage of our memory system.

### Roofline Comparison

The comparison of these two rooflines is confusing, due to the fact that these algorithms use the memory system very differently.
It is important to keep in mind that our solution runs **29x** faster than the problem.

| Roofline Type | Problem Roofline                                     | Solution Roofline                                      |
|---------------|------------------------------------------------------|--------------------------------------------------------|
| FP32/FP64     | <img src="exercise5_problem_roofline_fp32.png"/>     | <img src="exercise5_solution_roofline_fp32.png"/>      |
| FP16/INT8     | <img src="exercise5_problem_roofline_int8_fp16.png"/>| <img src="exercise5_solution_roofline_int8_fp16.png"/> |

We see a significant speedup from problem to solution, but on the roofline it is difficult to determine which implementation is using the hardware more efficiently. The problem seems to be better, as the HBM point is very close to the achievable bandwidth, while the performance of the solution points seem to decrease.
The roofline, though useful for estimating efficiencies of kernels, still only shows one perspective of performance.

### Summary and Take-aways

This algorithmic optimization is able to work more efficiently out of the L1, generating far fewer 
L2 requests that require expensive memory operations. Algorithmic optimizations are all but guaranteed
to have significant development overhead, but finding a more efficient algorithm can have large impacts
to performance. If profiling reveals inefficient use of the memory hardware, it could be worth thinking
about alternative algorithms. 
