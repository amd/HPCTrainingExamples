# Exercise 1: 
Note: Client-side installation instructions are available in the official omniperf documentation (https://rocm.docs.amd.com/projects/omniperf/en/latest/install/core-install.html), and provide all functionality demonstrated here.

This example was tested on MI300A.
```
module load  amdflang-new
module load rocm/6.2.1
module load omniperf/6.2.1
export HSA_XNACK=1
```

In this section, we show results obtained running this exercise on a system with MI300A, using ROCm `6.2.1` and the associated Omniperf, version `6.2.1`.

### Exercise Instructions:
Inspect the code in ```problem.F```: it is a prototype of a red-black smoother. Note that the innermost loop has strides such that the data is acceced in a checkerboard pattern.

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
This lists the runtime of each kernel in ```0.1 Top Kernels```. We can see that three kernels were launched and each of them was launced two times (```Count```) for the red and the black iteration. The last column shows how much of the overall time was spent in each of the kernels. The first kernel is the initialization, the second the warm-up loop. The third kernel is the one we are interested in. This is the kernel in line 54 in our source code and is thus marked as __omp_offloading_32_6140__QQmain_l54.kd. Note the l54 at the end provides the line.
In ```0.2 Dispatch List``` we see the dispatch ID of each of the kernels. The ones we are interested in are the the ones with ```Dispatch_ID``` 3 and 4.
For now, lets choose dispatch 3 and choose a few blocks of interest:
```
omniperf analyze -p workloads/problem/MI300A_A1 --dispatch 3 --block 7.1.0 7.1.1 7.1.2 2.1.7 2.1.15
```

Then inspect the output:

```
 
2. System Speed-of-Light
2.1 Speed-of-Light
╒═════════════╤═════════════════════╤═══════╤════════════╤═════════╤═══════════════╕
│ Metric_ID   │ Metric              │   Avg │ Unit       │    Peak │   Pct of Peak │
╞═════════════╪═════════════════════╪═══════╪════════════╪═════════╪═══════════════╡
│ 2.1.7       │ Active CUs          │ 12.00 │ Cus        │  228.00 │          5.26 │
├─────────────┼─────────────────────┼───────┼────────────┼─────────┼───────────────┤
│ 2.1.15      │ Wavefront Occupancy │ 15.82 │ Wavefronts │ 7296.00 │          0.22 │
╘═════════════╧═════════════════════╧═══════╧════════════╧═════════╧═══════════════╛


--------------------------------------------------------------------------------
7. Wavefront
7.1 Wavefront Launch Stats
╒═════════════╤══════════════════╤════════╤════════╤════════╤════════════╕
│ Metric_ID   │ Metric           │    Avg │    Min │    Max │ Unit       │
╞═════════════╪══════════════════╪════════╪════════╪════════╪════════════╡
│ 7.1.0       │ Grid Size        │ 512.00 │ 512.00 │ 512.00 │ Work items │
├─────────────┼──────────────────┼────────┼────────┼────────┼────────────┤
│ 7.1.1       │ Workgroup Size   │  32.00 │  32.00 │  32.00 │ Work items │
├─────────────┼──────────────────┼────────┼────────┼────────┼────────────┤
│ 7.1.2       │ Total Wavefronts │  16.00 │  16.00 │  16.00 │ Wavefronts │
╘═════════════╧══════════════════╧════════╧════════╧════════╧════════════╛

```
We see that the ``` 2. System Speed-of-Light``` shows only 5.26% of the CUs are active and the wavefront occupancy is 0.22%. This means that there is not enough paralellism in our kernel to use the GPU efficiently.
A simple way to introduce more parallelism is using the collapse clause on the nested loops. The compiler does not allow collapsing of the innermost loop with strides. Hence, the red-black scheme was implemented in an other way to be able to use collapse(3) to have enough iterations to improve the occupancy.
To see improved performance, we turn to the code in the `solution` directory:
```
cd solution
```
Inspect the code in solution.F.
```
make
./solution
```
(*simulated output*)
```
time 5.27191162109375 ms
```
The compute time is now about 34 times smaller when going from `problem` to `solution`.

To visually confirm the hypothesis that this improvement is due to improved usage of CUs and improved occupancy of the `solution` code, run:

```
omniperf profile -n solution --no-roof -- ./solution
omniperf analyze -p workloads/solution/MI300A_A1 --dispatch 3 --block 7.1.0 7.1.1 7.1.2 2.1.7 2.1.15
```
The output shows:
```
 2. System Speed-of-Light
2.1 Speed-of-Light
╒═════════════╤═════════════════════╤═════════╤════════════╤═════════╤═══════════════╕
│ Metric_ID   │ Metric              │     Avg │ Unit       │    Peak │   Pct of Peak │
╞═════════════╪═════════════════════╪═════════╪════════════╪═════════╪═══════════════╡
│ 2.1.7       │ Active CUs          │  220.00 │ Cus        │  228.00 │         96.49 │
├─────────────┼─────────────────────┼─────────┼────────────┼─────────┼───────────────┤
│ 2.1.15      │ Wavefront Occupancy │ 3444.16 │ Wavefronts │ 7296.00 │         47.21 │
╘═════════════╧═════════════════════╧═════════╧════════════╧═════════╧═══════════════╛

--------------------------------------------------------------------------------
7. Wavefront
7.1 Wavefront Launch Stats
╒═════════════╤══════════════════╤═══════════╤═══════════╤═══════════╤════════════╕
│ Metric_ID   │ Metric           │       Avg │       Min │       Max │ Unit       │
╞═════════════╪══════════════════╪═══════════╪═══════════╪═══════════╪════════════╡
│ 7.1.0       │ Grid Size        │ 233472.00 │ 233472.00 │ 233472.00 │ Work items │
├─────────────┼──────────────────┼───────────┼───────────┼───────────┼────────────┤
│ 7.1.1       │ Workgroup Size   │    256.00 │    256.00 │    256.00 │ Work items │
├─────────────┼──────────────────┼───────────┼───────────┼───────────┼────────────┤
│ 7.1.2       │ Total Wavefronts │   3648.00 │   3648.00 │   3648.00 │ Wavefronts │
╘═════════════╧══════════════════╧═══════════╧═══════════╧═══════════╧════════════╛

```
The number of active CUs is now close to 96.49% and the wavefront occupancy is improved to 47.21%. Note that also the workgroup size the grid size and the total number of wavefronts changed significantly through the improved usage of the device through the introduction of more parallelism.

### Omniperf Command Line Comparison Feature:

We can compare the performance of `problem` and `solution` using `omniperf analyze`:

```
cd ..
omniperf analyze -p workloads/problem/MI300A_A1/ -p solution/workloads/solution/MI300A_A1/ --dispatch 3 --block 7.1.0 7.1.1 7.1.2 2.1.7 2.1.15
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
╒════╤═════════════════════════════════════════╤═════════╤════════════╤════════════╤═════════════╤════════════════════╤═════════════╤════════════════════╤══════════════╤════════════════════╤════════╤══════════════╕
│    │ Kernel_Name                             │   Count │ Count      │   Abs Diff │     Sum(ns) │ Sum(ns)            │    Mean(ns) │ Mean(ns)           │   Median(ns) │ Median(ns)         │    Pct │ Pct          │
╞════╪═════════════════════════════════════════╪═════════╪════════════╪════════════╪═════════════╪════════════════════╪═════════════╪════════════════════╪══════════════╪════════════════════╪════════╪══════════════╡
│  0 │ __omp_offloading_32_6140__QQmain_l37.kd │    1.00 │ 1.0 (0.0%) │       0.00 │ 90245554.00 │ 849281.0 (-99.06%) │ 90245554.00 │ 849281.0 (-99.06%) │  90245554.00 │ 849281.0 (-99.06%) │ 100.00 │ 100.0 (0.0%) │
╘════╧═════════════════════════════════════════╧═════════╧════════════╧════════════╧═════════════╧════════════════════╧═════════════╧════════════════════╧══════════════╧════════════════════╧════════╧══════════════╛
0.2 Dispatch List
╒════╤═══════════════╤═════════════════════════════════════════╤══════════╕
│    │   Dispatch_ID │ Kernel_Name                             │   GPU_ID │
╞════╪═══════════════╪═════════════════════════════════════════╪══════════╡
│  0 │             3 │ __omp_offloading_32_6140__QQmain_l37.kd │        4 │
╘════╧═══════════════╧═════════════════════════════════════════╧══════════╛


--------------------------------------------------------------------------------
2. System Speed-of-Light
2.1 Speed-of-Light
╒═════════════╤═════════════════════╤═══════╤═════════════════════╤════════════╤════════════╤═════════╤═══════════════╤═══════════════╤═══════════════════╕
│ Metric_ID   │ Metric              │   Avg │ Avg                 │   Abs Diff │ Unit       │    Peak │ Peak          │   Pct of Peak │ Pct of Peak       │
╞═════════════╪═════════════════════╪═══════╪═════════════════════╪════════════╪════════════╪═════════╪═══════════════╪═══════════════╪═══════════════════╡
│ 2.1.7       │ Active CUs          │ 12.00 │ 220.0 (1733.33%)    │      91.23 │ Cus        │  228.00 │ 228.0 (0.0%)  │          5.26 │ 96.49 (1733.37%)  │
├─────────────┼─────────────────────┼───────┼─────────────────────┼────────────┼────────────┼─────────┼───────────────┼───────────────┼───────────────────┤
│ 2.1.15      │ Wavefront Occupancy │ 15.82 │ 3444.16 (21677.33%) │      46.99 │ Wavefronts │ 7296.00 │ 7296.0 (0.0%) │          0.22 │ 47.21 (21677.65%) │
╘═════════════╧═════════════════════╧═══════╧═════════════════════╧════════════╧════════════╧═════════╧═══════════════╧═══════════════╧═══════════════════╛


--------------------------------------------------------------------------------
7. Wavefront
7.1 Wavefront Launch Stats
╒═════════════╤══════════════════╤════════╤═════════════════════╤════════════╤════════╤═════════════════════╤════════╤═════════════════════╤════════════╕
│ Metric_ID   │ Metric           │    Avg │ Avg                 │   Abs Diff │    Min │ Min                 │    Max │ Max                 │ Unit       │
╞═════════════╪══════════════════╪════════╪═════════════════════╪════════════╪════════╪═════════════════════╪════════╪═════════════════════╪════════════╡
│ 7.1.0       │ Grid Size        │ 512.00 │ 233472.0 (45500.0%) │  232960.00 │ 512.00 │ 233472.0 (45500.0%) │ 512.00 │ 233472.0 (45500.0%) │ Work items │
├─────────────┼──────────────────┼────────┼─────────────────────┼────────────┼────────┼─────────────────────┼────────┼─────────────────────┼────────────┤
│ 7.1.1       │ Workgroup Size   │  32.00 │ 256.0 (700.0%)      │     224.00 │  32.00 │ 256.0 (700.0%)      │  32.00 │ 256.0 (700.0%)      │ Work items │
├─────────────┼──────────────────┼────────┼─────────────────────┼────────────┼────────┼─────────────────────┼────────┼─────────────────────┼────────────┤
│ 7.1.2       │ Total Wavefronts │  16.00 │ 3648.0 (22700.0%)   │    3632.00 │  16.00 │ 3648.0 (22700.0%)   │  16.00 │ 3648.0 (22700.0%)   │ Wavefronts │
╘═════════════╧══════════════════╧════════╧═════════════════════╧════════════╧════════╧═════════════════════╧════════╧═════════════════════╧════════════╛

```

Note that the new execution time for `solution` is reduced by 99.06% of the original execution time for `problem`.

### More Kernel Filtering:

Run the following command to once again see a ranking of the top kernels that take up most of the runtime:

```
omniperf analyze -p workloads/problem/MI300A_A1/ --list-stats
```

```

 / _ \ _ __ ___  _ __ (_)_ __   ___ _ __ / _|
| | | | '_ ` _ \| '_ \| | '_ \ / _ \ '__| |_
| |_| | | | | | | | | | | |_) |  __/ |  |  _|
 \___/|_| |_| |_|_| |_|_| .__/ \___|_|  |_|
                        |_|

   INFO Analysis mode = cli
   INFO [analysis] deriving Omniperf metrics...

--------------------------------------------------------------------------------
Detected Kernels (sorted descending by duration)
╒════╤═════════════════════════════════════════╕
│    │ Kernel_Name                             │
╞════╪═════════════════════════════════════════╡
│  0 │ __omp_offloading_32_6140__QQmain_l54.kd │
├────┼─────────────────────────────────────────┤
│  1 │ __omp_offloading_32_6140__QQmain_l37.kd │
├────┼─────────────────────────────────────────┤
│  2 │ __omp_offloading_32_6140__QQmain_l17.kd │
╘════╧═════════════════════════════════════════╛

--------------------------------------------------------------------------------
Dispatch list
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
This helps you to see where optimization will be most helpful for optimizing time to solution in larger apps.
