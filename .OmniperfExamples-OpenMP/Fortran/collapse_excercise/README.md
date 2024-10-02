# Exercise 1: Increase workload by collapse
Client-side installation instructions are available in the official omniperf documentation, and provide all functionality demonstrated here.

<details>
<summary><h3>Background: Acronyms and terms used in this exercise</h3></summary>
     <ul>
          <li>yAx: a vector-matrix-vector product, y*A*x, where y and x are vectors, and A is a matrix</li>
          <li>FP(32/16): 32- or 16-bit Floating Point numeric types</li>
          <li>FLOPs: Floating Point Operations Per second</li>
          <li>HBM: High Bandwidth Memory is globally accessible from the GPU, and is a level of memory above the L2 cache</li>
     </ul>
</details>
## Results on MI300A

In this section, we show results obtained running this exercise on a system with MI300A, using ROCm `6.2.1` and the associated Omniperf, version `6.2.1`.

### Roofline Analysis:

At present (September 28th 2024), rooflines are disabled on MI300A.

### Exercise Instructions:
Inspect the code: it is a prototype of a red-black smoother. Note that the innermost loop has strides such that the data is acceced in a checkerboard pattern.

Build and run the problem code:

```
make
./problem
```
Output:
```
time 181.2760829925537 ms
```

We launch the following command:

```
omniperf profile -n problem --no-roof -- ./problem
```

Followed by:

```
omniperf analyze -p workloads/problem/MI300A_A1 --block 0
```

This lists all the Top Stats of the kernels in problem.
We are interested in the kernel in the last kernel, as the other two are initialization and a warmup loop and hence not representative for performance meassurements.
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
╒════╤═════════════════════════════════════════╤═════════╤══════════════╤═════════════╤══════════════╤═══════╕
│    │ Kernel_Name                             │   Count │      Sum(ns) │    Mean(ns) │   Median(ns) │   Pct │
╞════╪═════════════════════════════════════════╪═════════╪══════════════╪═════════════╪══════════════╪═══════╡
│  0 │ __omp_offloading_32_6140__QQmain_l54.kd │    2.00 │ 180791491.00 │ 90395745.50 │  90395745.50 │ 33.88 │
├────┼─────────────────────────────────────────┼─────────┼──────────────┼─────────────┼──────────────┼───────┤
│  1 │ __omp_offloading_32_6140__QQmain_l37.kd │    2.00 │ 179446724.00 │ 89723362.00 │  89723362.00 │ 33.63 │
├────┼─────────────────────────────────────────┼─────────┼──────────────┼─────────────┼──────────────┼───────┤
│  2 │ __omp_offloading_32_6140__QQmain_l17.kd │    2.00 │ 173409738.00 │ 86704869.00 │  86704869.00 │ 32.50 │
╘════╧═════════════════════════════════════════╧═════════╧══════════════╧═════════════╧══════════════╧═══════╛
0.2 Dispatch List
╒════╤═══════════════╤═════════════════════════════════════════╤══════════╕
│    │   Dispatch_ID │ Kernel_Name                             │   GPU_ID │
╞════╪═══════════════╪═════════════════════════════════════════╪══════════╡
│  0 │             0 │ __omp_offloading_32_6140__QQmain_l17.kd │        4 │
├────┼───────────────┼─────────────────────────────────────────┼──────────┤
│  1 │             1 │ __omp_offloading_32_6140__QQmain_l17.kd │        4 │
├────┼───────────────┼─────────────────────────────────────────┼──────────┤
│  2 │             2 │ __omp_offloading_32_6140__QQmain_l37.kd │        4 │
├────┼───────────────┼─────────────────────────────────────────┼──────────┤
│  3 │             3 │ __omp_offloading_32_6140__QQmain_l37.kd │        4 │
├────┼───────────────┼─────────────────────────────────────────┼──────────┤
│  4 │             4 │ __omp_offloading_32_6140__QQmain_l54.kd │        4 │
├────┼───────────────┼─────────────────────────────────────────┼──────────┤
│  5 │             5 │ __omp_offloading_32_6140__QQmain_l54.kd │        4 │
╘════╧═══════════════╧═════════════════════════════════════════╧══════════╛

```
This lists the time the execution of each kernel took in ```0.1 Top Kernels``` we can see that three kernels were launched and each of them was launced two times (´´´Count´´´). The last column shows how much of the overall time was spent in each of the kernels. The first kernel is the initialization, the second the warm-up loop. The third kernel is the one we are interested in. This is the kernel in line 54 in our source code and is thus marked as __omp_offloading_32_6140__QQmain_l54.kd. Note the l54 at the end provides the line.
In ```0.2 Dispatch List``` we see the dispatch ID of each of the kernels. The ones we are interested in are the the ones with ´´´Dispatch_ID´´´ 3 and 4.
For now, lets choos dispatch 3 and choose a few blocks of interest (you can view all availa:
```
omniperf analyze -p workloads/problem/MI300A_A1 --dispatch 3 --block 7.1.0 7.1.1 7.1.2 2.1.7 2.1.15
```

Then inspect the output:

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
╒════╤══════════════════════════════════════════╤═════════╤══════════════╤══════════════╤══════════════╤════════╕
│    │ Kernel_Name                              │   Count │      Sum(ns) │     Mean(ns) │   Median(ns) │    Pct │
╞════╪══════════════════════════════════════════╪═════════╪══════════════╪══════════════╪══════════════╪════════╡
│  0 │ yax(double*, double*, double*, int, int, │    1.00 │ 541264224.00 │ 541264224.00 │ 541264224.00 │ 100.00 │
│    │  double*) [clone .kd]                    │         │              │              │              │        │
╘════╧══════════════════════════════════════════╧═════════╧══════════════╧══════════════╧══════════════╧════════╛
0.2 Dispatch List
╒════╤═══════════════╤═══════════════════════════════════════════════════════════════╤══════════╕
│    │   Dispatch_ID │ Kernel_Name                                                   │   GPU_ID │
╞════╪═══════════════╪═══════════════════════════════════════════════════════════════╪══════════╡
│  0 │             1 │ yax(double*, double*, double*, int, int, double*) [clone .kd] │        4 │
╘════╧═══════════════╧═══════════════════════════════════════════════════════════════╧══════════╛


--------------------------------------------------------------------------------
7. Wavefront
7.1 Wavefront Launch Stats
╒═════════════╤══════════════════╤════════╤════════╤════════╤════════════╕
│ Metric_ID   │ Metric           │    Avg │    Min │    Max │ Unit       │
╞═════════════╪══════════════════╪════════╪════════╪════════╪════════════╡
│ 7.1.0       │ Grid Size        │ 256.00 │ 256.00 │ 256.00 │ Work items │
├─────────────┼──────────────────┼────────┼────────┼────────┼────────────┤
│ 7.1.1       │ Workgroup Size   │  64.00 │  64.00 │  64.00 │ Work items │
├─────────────┼──────────────────┼────────┼────────┼────────┼────────────┤
│ 7.1.2       │ Total Wavefronts │   4.00 │   4.00 │   4.00 │ Wavefronts │
╘═════════════╧══════════════════╧════════╧════════╧════════╧════════════╛

```

As for the MI210 case, the workgroup size is 64 and the number of Wavefronts launched is 4.

To see improved performance, we turn to the code in the `solution` directory:

```
cd solution
make
./solution.exe
```
(*simulated output*)
```
yAx time: 9.7 ms
```

For the MI210 case, the compute time was about 42 times smaller when going from `problem` to `solution`. For the MI300A case, we see it is about 70 times smaller.

To visually confirm the updated launch parameters in the `solution` code, run:

```
omniperf profile -n solution --no-roof -- ./solution.exe
omniperf analyze -p workloads/solution/MI300A_A1 --dispatch 1 --block 7.1.0 7.1.1 7.1.2
```

Then see the number of Wavefronts now being 2048:

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
│  0 │ yax(double*, double*, double*, int, int, │    1.00 │ 9482864.00 │ 9482864.00 │   9482864.00 │ 100.00 │
│    │  double*) [clone .kd]                    │         │            │            │              │        │
╘════╧══════════════════════════════════════════╧═════════╧════════════╧════════════╧══════════════╧════════╛
0.2 Dispatch List
╒════╤═══════════════╤═══════════════════════════════════════════════════════════════╤══════════╕
│    │   Dispatch_ID │ Kernel_Name                                                   │   GPU_ID │
╞════╪═══════════════╪═══════════════════════════════════════════════════════════════╪══════════╡
│  0 │             1 │ yax(double*, double*, double*, int, int, double*) [clone .kd] │        4 │
╘════╧═══════════════╧═══════════════════════════════════════════════════════════════╧══════════╛


--------------------------------------------------------------------------------
7. Wavefront
7.1 Wavefront Launch Stats
╒═════════════╤══════════════════╤═══════════╤═══════════╤═══════════╤════════════╕
│ Metric_ID   │ Metric           │       Avg │       Min │       Max │ Unit       │
╞═════════════╪══════════════════╪═══════════╪═══════════╪═══════════╪════════════╡
│ 7.1.0       │ Grid Size        │ 131072.00 │ 131072.00 │ 131072.00 │ Work items │
├─────────────┼──────────────────┼───────────┼───────────┼───────────┼────────────┤
│ 7.1.1       │ Workgroup Size   │     64.00 │     64.00 │     64.00 │ Work items │
├─────────────┼──────────────────┼───────────┼───────────┼───────────┼────────────┤
│ 7.1.2       │ Total Wavefronts │   2048.00 │   2048.00 │   2048.00 │ Wavefronts │
╘═════════════╧══════════════════╧═══════════╧═══════════╧═══════════╧════════════╛

```

### Omniperf Command Line Comparison Feature:

We can compare the performance of `problem` and `solution` using `omniperf analyze`:

```
omniperf analyze -p workloads/problem/MI300A_A1/ -p solution/workloads/solution/MI300A_A1/ --dispatch 1 --block 7.1.0 7.1.1 7.1.2
```

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
╒════╤══════════════════════════════════════════╤═════════╤════════════╤════════════╤══════════════╤═════════════════════╤══════════════╤═════════════════════╤══════════════╤═════════════════════╤════════╤══════════════╕
│    │ Kernel_Name                              │   Count │ Count      │   Abs Diff │      Sum(ns) │ Sum(ns)             │     Mean(ns) │ Mean(ns)            │   Median(ns) │ Median(ns)          │    Pct │ Pct          │
╞════╪══════════════════════════════════════════╪═════════╪════════════╪════════════╪══════════════╪═════════════════════╪══════════════╪═════════════════════╪══════════════╪═════════════════════╪════════╪══════════════╡
│  0 │ yax(double*, double*, double*, int, int, │    1.00 │ 1.0 (0.0%) │       0.00 │ 541264224.00 │ 9482864.0 (-98.25%) │ 541264224.00 │ 9482864.0 (-98.25%) │ 541264224.00 │ 9482864.0 (-98.25%) │ 100.00 │ 100.0 (0.0%) │
│    │  double*) [clone .kd]                    │         │            │            │              │                     │              │                     │              │                     │        │              │
╘════╧══════════════════════════════════════════╧═════════╧════════════╧════════════╧══════════════╧═════════════════════╧══════════════╧═════════════════════╧══════════════╧═════════════════════╧════════╧══════════════╛
0.2 Dispatch List
╒════╤═══════════════╤═══════════════════════════════════════════════════════════════╤══════════╕
│    │   Dispatch_ID │ Kernel_Name                                                   │   GPU_ID │
╞════╪═══════════════╪═══════════════════════════════════════════════════════════════╪══════════╡
│  0 │             1 │ yax(double*, double*, double*, int, int, double*) [clone .kd] │        4 │
╘════╧═══════════════╧═══════════════════════════════════════════════════════════════╧══════════╛


--------------------------------------------------------------------------------
7. Wavefront
7.1 Wavefront Launch Stats
╒═════════════╤══════════════════╤════════╤═════════════════════╤════════════╤════════╤═════════════════════╤════════╤═════════════════════╤════════════╕
│ Metric_ID   │ Metric           │    Avg │ Avg                 │   Abs Diff │    Min │ Min                 │    Max │ Max                 │ Unit       │
╞═════════════╪══════════════════╪════════╪═════════════════════╪════════════╪════════╪═════════════════════╪════════╪═════════════════════╪════════════╡
│ 7.1.0       │ Grid Size        │ 256.00 │ 131072.0 (51100.0%) │  130816.00 │ 256.00 │ 131072.0 (51100.0%) │ 256.00 │ 131072.0 (51100.0%) │ Work items │
├─────────────┼──────────────────┼────────┼─────────────────────┼────────────┼────────┼─────────────────────┼────────┼─────────────────────┼────────────┤
│ 7.1.1       │ Workgroup Size   │  64.00 │ 64.0 (0.0%)         │       0.00 │  64.00 │ 64.0 (0.0%)         │  64.00 │ 64.0 (0.0%)         │ Work items │
├─────────────┼──────────────────┼────────┼─────────────────────┼────────────┼────────┼─────────────────────┼────────┼─────────────────────┼────────────┤
│ 7.1.2       │ Total Wavefronts │   4.00 │ 2048.0 (51100.0%)   │    2044.00 │   4.00 │ 2048.0 (51100.0%)   │   4.00 │ 2048.0 (51100.0%)   │ Wavefronts │
╘═════════════╧══════════════════╧════════╧═════════════════════╧════════════╧════════╧═════════════════════╧════════╧═════════════════════╧════════════╛

```

Note that the new execution time for `solution` is about 1.75% of the original execution time for `problem`.

### More Kernel Filtering:

Run the following command to once again see a ranking of the top kernels that take up most of the kernel runtime:

```
cd ..
omniperf analyze -p workloads/problem/MI300A_A1/ --list-stats
```

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
Detected Kernels (sorted descending by duration)
╒════╤═══════════════════════════════════════════════════════════════╕
│    │ Kernel_Name                                                   │
╞════╪═══════════════════════════════════════════════════════════════╡
│  0 │ yax(double*, double*, double*, int, int, double*) [clone .kd] │
╘════╧═══════════════════════════════════════════════════════════════╛

--------------------------------------------------------------------------------
Dispatch list
╒════╤═══════════════╤═══════════════════════════════════════════════════════════════╤══════════╕
│    │   Dispatch_ID │ Kernel_Name                                                   │   GPU_ID │
╞════╪═══════════════╪═══════════════════════════════════════════════════════════════╪══════════╡
│  0 │             0 │ yax(double*, double*, double*, int, int, double*) [clone .kd] │        4 │
├────┼───────────────┼───────────────────────────────────────────────────────────────┼──────────┤
│  1 │             1 │ yax(double*, double*, double*, int, int, double*) [clone .kd] │        4 │
╘════╧═══════════════╧═══════════════════════════════════════════════════════════════╧══════════╛

```

To see aggregated stats for the `yax` kernel, run

```
omniperf analyze -p workloads/problem/MI300A_A1/ -k 0 --block 7.1.0 7.1.1 7.1.2

```

Which will show an output similar to this one:

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
╒════╤══════════════════════════════════════════╤═════════╤═══════════════╤══════════════╤══════════════╤════════╤═════╕
│    │ Kernel_Name                              │   Count │       Sum(ns) │     Mean(ns) │   Median(ns) │    Pct │ S   │
╞════╪══════════════════════════════════════════╪═════════╪═══════════════╪══════════════╪══════════════╪════════╪═════╡
│  0 │ yax(double*, double*, double*, int, int, │    2.00 │ 1083496775.00 │ 541748387.50 │ 541748387.50 │ 100.00 │ *   │
│    │  double*) [clone .kd]                    │         │               │              │              │        │     │
╘════╧══════════════════════════════════════════╧═════════╧═══════════════╧══════════════╧══════════════╧════════╧═════╛
0.2 Dispatch List
╒════╤═══════════════╤═══════════════════════════════════════════════════════════════╤══════════╕
│    │   Dispatch_ID │ Kernel_Name                                                   │   GPU_ID │
╞════╪═══════════════╪═══════════════════════════════════════════════════════════════╪══════════╡
│  0 │             0 │ yax(double*, double*, double*, int, int, double*) [clone .kd] │        4 │
├────┼───────────────┼───────────────────────────────────────────────────────────────┼──────────┤
│  1 │             1 │ yax(double*, double*, double*, int, int, double*) [clone .kd] │        4 │
╘════╧═══════════════╧═══════════════════════════════════════════════════════════════╧══════════╛


--------------------------------------------------------------------------------
7. Wavefront
7.1 Wavefront Launch Stats
╒═════════════╤══════════════════╤════════╤════════╤════════╤════════════╕
│ Metric_ID   │ Metric           │    Avg │    Min │    Max │ Unit       │
╞═════════════╪══════════════════╪════════╪════════╪════════╪════════════╡
│ 7.1.0       │ Grid Size        │ 256.00 │ 256.00 │ 256.00 │ Work items │
├─────────────┼──────────────────┼────────┼────────┼────────┼────────────┤
│ 7.1.1       │ Workgroup Size   │  64.00 │  64.00 │  64.00 │ Work items │
├─────────────┼──────────────────┼────────┼────────┼────────┼────────────┤
│ 7.1.2       │ Total Wavefronts │   4.00 │   4.00 │   4.00 │ Wavefronts │
╘═════════════╧══════════════════╧════════╧════════╧════════╧════════════╛

```
