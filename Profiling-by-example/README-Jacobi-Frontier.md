# Profiling the HIP Jacobi code

Here, we follow a sequence of steps to understand the Jacobi code better, each time
using a specific profiler tool because we seek a particular type of information. We
have at our disposal three types of tools:

- `rocprofv3` that can help get GPU hotspots, traces or counters
- `rocprof-sys` that can help us get GPU and CPU traces
- `rocprof-compute` that can help us understand GPU kernel performance 

The Jacobi example can be found in this repo at
[https://github.com/gsitaram/HPCTrainingExamples/tree/main/HIP/jacobi](https://github.com/gsitaram/HPCTrainingExamples/tree/main/HIP/jacobi). It uses HIP for offloading compute
to GPUs. We know that the Jacobi code uses MPI for halo exchanges. Let's assume that
we don't know yet the characteristics of the application or the limiters of the hotspots.

## Set up environment on Frontier

First, set up your environment to get a newer version of ROCm, and tools. Since rocprofiler-systems is not yet installed in rocm/6.3.1 on Frontier, we are going to install it by ourselves using an installer script into our home directory.

```
module load rocm/6.3.1
module load rocprofiler-compute/3.0.0

wget https://github.com/ROCm/rocprofiler-systems/releases/download/rocm-6.3.3/rocprofiler-systems-0.1.2-opensuse-15.6-ROCm-60300-PAPI-OMPT-Python3.sh
chmod +x rocprofiler-systems-0.1.2-opensuse-15.6-ROCm-60300-PAPI-OMPT-Python3.sh
mkdir -p ${HOME}/rocprofiler-systems
./rocprofiler-systems-0.1.2-opensuse-15.6-ROCm-60300-PAPI-OMPT-Python3.sh --exclude-subdir --prefix=${HOME}/rocprofiler-systems
source ${HOME}/rocprofiler-systems/share/rocprofiler-systems/setup-env.sh
```

Also, for ease of use, set up your project name in Frontier in an environment variable.

```
export PROJ=<proj>
```

## Build and Run Jacobi (single rank and dual rank)

Check if you can clone and build the Jacobi example. On Frontier, run the commands:

```
cd ${HOME}
git clone git@github.com:amd/HPCTrainingExamples.git
cd HPCTrainingExamples/HIP/jacobi
make -f Makefile.cray
srun -N1 -n1 -c7 -A ${PROJ} -t 02:00 ./Jacobi_hip -g 1 1
```

That should show an output similar to the following:

```
Topology size: 1 x 1
Local domain size (current node): 4096 x 4096
Global domain size (all nodes): 4096 x 4096
Rank 0 selecting device 0 on host frontier04031
Starting Jacobi run.
Iteration:   0 - Residual: 0.022108
Iteration: 100 - Residual: 0.000625
Iteration: 200 - Residual: 0.000371
Iteration: 300 - Residual: 0.000274
Iteration: 400 - Residual: 0.000221
Iteration: 500 - Residual: 0.000187
Iteration: 600 - Residual: 0.000163
Iteration: 700 - Residual: 0.000145
Iteration: 800 - Residual: 0.000131
Iteration: 900 - Residual: 0.000120
Iteration: 1000 - Residual: 0.000111
Stopped after 1000 iterations with residue 0.000111
Total Jacobi run time: 1.3129 sec.
Measured lattice updates: 12.78 GLU/s (total), 12.78 GLU/s (per process)
Measured FLOPS: 217.24 GFLOPS (total), 217.24 GFLOPS (per process)
Measured device bandwidth: 1.23 TB/s (total), 1.23 TB/s (per process)
```

That was a successful run. Now, we can try running a job with 2 processes. This time,
add the Slurm option `--gpu-bind=closest` to ensure that each process gets a
different GPU device and one that is closest to the CPU core that it runs on. Notice
that we increased the number of processes in `-n2` and modified the Jacobi grid
in `-g 2 1`.

```
srun -N1 -n2 -c7 --gpu-bind=closest -A ${PROJ} -t 02:00 ./Jacobi_hip -g 2 1
```

## Is this application GPU bound or CPU bound?

To understand whether we spend most of the application runtime on the GPU or on the host,
getting the GPU kernel hotspots using `rocprofv3` is a quick method. Using the total
time spent in the most expensive kernel, we can calculate how much of the application
time is spent running GPU compute kernels. We will use the single rank run for this
experiment. Notice that we added the tool and its options,
`rocprofv3 --kernel-trace --stats --`
before calling the application in the srun command. You can do similar experiments
to get HIP API traces using `--hip-trace -stats` or memory copy stats using
`--memory-copy-trace --stats` or a combination of these.

```
srun -N1 -n1 -c7 -A ${PROJ} -t 02:00 rocprofv3 --kernel-trace --stats -- ./Jacobi_hip -g 1 1
```

Of the output files, there will be one called `XXXXX_kernel_stats.csv`. A `cat` of that
file should show a list of GPU kernel hot spots as seen below:

```
"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"JacobiIterationKernel(int, double, double, double const*, double const*, double*, double*)",1000,517647434,517647.434000,42.56,510404,527365,2907.325924
"NormKernel1(int, double, double, double const*, double*)",1001,412893767,412481.285714,33.95,401603,423364,2862.057355
"LocalLaplacianKernel(int, int, int, double, double, double const*, double*)",1000,269619091,269619.091000,22.17,263842,281762,1763.747497
"HaloLaplacianKernel(int, int, int, double, double, double const*, double const*, double*)",1000,13295466,13295.466000,1.09,12320,15360,316.811361
"NormKernel2(int, double const*, double*)",1001,2869948,2867.080919,0.2360,2720,3840,135.996465
```

Here, we see that the `JacobiIterationKernel` is the most expensive one on the MI250X GCD
that this job ran. Taking the total duration of 517ms in this kernel which was
42.56% of the total run time, we get 1214.8 ms of time spent in GPU kernels during
this run. Given the total elapsed time of this run of 1.2545 seconds, we can quickly
conclude that 97% of elapsed time of this run was spent running GPU kernels.

To avoid seeing all the kernel arguments in the hotspot list and make it more readable,
use the `--truncate-kernels` option to `rocprofv3`.

```
srun -N1 -n1 -c7 -A ${PROJ} -t 02:00 rocprofv3 --kernel-trace --stats --truncate-kernels -- ./Jacobi_hip -g 1 1
```

To truly understand the overhead induced by halo exchanges, we should do this experiment
when we run multiple ranks and collect hotspots per process. This exercise will be
left to the reader.

## Getting application trace on host and device

In order to get a more holistic view of the application, use `rocprof-sys` to collect
a trace. Run with one process first, and then progress to running with multiple processes.

Before we start using `rocprof-sys` tools, it is best to create a runtime config file.
Then, in order to get a less cluttered trace, edit some options to turn off CPU frequency
sampling for all CPU cores. These config options can be edited directly in the file
you create, or via environment variables as shown below:

``` 
rocprof-sys-avail -G ${HOME}/.rocprofsys.cfg
export ROCPROFSYS_CONFIG_FILE=${HOME}/.rocprofsys.cfg
export ROCPROFSYS_SAMPLING_CPUS=none
```

If you know which GPU device this process is going to run on, then you can turn on sampling
for the characteristics of that GPU device only. For instance, if we are going to 
run on GPU 0, then we can do something like the following:

```
export ROCR_VISIBLE_DEVICES=0
export ROCPROFSYS_SAMPLING_GPUS=0
```

Now, instrument the code and collect a trace.

```
rocprof-sys-instrument -o ./Jacobi_hip.inst -- ./Jacobi_hip 
srun -N1 -n1 -c7 -A ${PROJ} -t 02:00 rocprof-sys-run -- ./Jacobi_hip.inst -g 1 1
```

This should output a `.proto` file. Copy that over to your laptop to view with Perfetto UI,
[https://ui.perfetto.dev](https://ui.perfetto.dev).

When you run with multiple ranks, it may be best to use the Slurm option `--gpu-bind=closest` and sample all GPUs because we don't know which GPUs the processes are going to run on.

```
export ROCPROFSYS_SAMPLING_GPUS=all
srun -N1 -n2 -c7 --gpu-bind=closest -A ${PROJ} -t 02:00 rocprof-sys-run -- ./Jacobi_hip.inst -g 2 1
```

You will observe that a `.proto` file is created for each rank. You can simply concatenate
those `.proto` files to create a merged trace for viewing in Perfetto.

```
cat rocprofsys-Jacobi_hip.inst-output/<timestamp>/perfetto-trace-*.proto > merged.proto
```

## Dive deep in to kernel performance

Use `rocprof-compute` to first get a roofline plot. This roofline plot can help us
understand whether the kernels are memory bound, compute bound or latency bound.

```
srun -N1 -n1 -c7 -A ${PROJ} -t 02:00 rocprof-compute profile -n roofline --roof-only --device 0 --kernel-names  -- ./Jacobi_hip -g 1 1
```

The above command results in a few PDF files being created in the 
`workloads/roofline/MI200` directory. The file `empirRoof_gpu-0_fp32_fp64.pdf`
contains the roofline plot itself, and the file `kernelName_legend.pdf` contains
the legend for the plot. When you view the roofline
plot, you will observe that all kernels are either memory bound or latency bound.

Next collect kernel performance metrics. You will notice that this command runs your
application multiple times to collect different batches of hardware counters. For
this reason, we recommend running with only 1 rank if possible, and increase
the time required for this run.

```
srun -N1 -n1 -c7 -A ${PROJ} -t 10:00 rocprof-compute profile -n test --no-roof -- ./Jacobi_hip -g 1 1
```

Next, analyze and look at kernel stats to get an idea of either the kernel ID or
the dispatch ID to analyze further.
All hardware counter values are saved in the workloads directory `workloads/test/MI200`.
Supply this path to the analyze command as shown below.

```
srun -N1 -n1 -c7 -A ${PROJ} -t 02:00 rocprof-compute analyze -p workloads/test/MI200 --list-stats >& stats_output.log
```

You should see something like the following:

```
--------------------------------------------------------------------------------
Detected Kernels (sorted descending by duration)
╒════╤════════════════════════════════════════════════════════════════════════════════════════════════════════╕
│    │ Kernel_Name                                                                                            │
╞════╪════════════════════════════════════════════════════════════════════════════════════════════════════════╡
│  0 │ JacobiIterationKernel(int, double, double, double const*, double const*, double*, double*) [clone .kd] │
├────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  1 │ NormKernel1(int, double, double, double const*, double*) [clone .kd]                                   │
├────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  2 │ LocalLaplacianKernel(int, int, int, double, double, double const*, double*) [clone .kd]                │
├────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  3 │ HaloLaplacianKernel(int, int, int, double, double, double const*, double const*, double*) [clone .kd]  │
├────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  4 │ NormKernel2(int, double const*, double*) [clone .kd]                                                   │
├────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  5 │ __amd_rocclr_fillBufferAligned.kd                                                                      │
╘════╧════════════════════════════════════════════════════════════════════════════════════════════════════════╛

--------------------------------------------------------------------------------
Dispatch list
╒══════╤═══════════════╤════════════════════════════════════════════════════════════════════════════════════════════════════════╤══════════╕
│      │   Dispatch_ID │ Kernel_Name                                                                                            │   GPU_ID │
╞══════╪═══════════════╪════════════════════════════════════════════════════════════════════════════════════════════════════════╪══════════╡
│    0 │             0 │ __amd_rocclr_fillBufferAligned.kd                                                                      │        4 │
├──────┼───────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│    1 │             1 │ NormKernel1(int, double, double, double const*, double*) [clone .kd]                                   │        4 │
├──────┼───────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│    2 │             2 │ NormKernel2(int, double const*, double*) [clone .kd]                                                   │        4 │
├──────┼───────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│    3 │             3 │ LocalLaplacianKernel(int, int, int, double, double, double const*, double*) [clone .kd]                │        4 │
├──────┼───────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│    4 │             4 │ HaloLaplacianKernel(int, int, int, double, double, double const*, double const*, double*) [clone .kd]  │        4 │
├──────┼───────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│    5 │             5 │ JacobiIterationKernel(int, double, double, double const*, double const*, double*, double*) [clone .kd] │        4 │
├──────┼───────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│    6 │             6 │ NormKernel1(int, double, double, double const*, double*) [clone .kd]                                   │        4 │
```

We are interested in the JacobiIterationKernel, so let's analyze just dispatch number 5.

```
srun -N1 -n1 -c7 -A ${PROJ} -t 02:00 rocprof-compute analyze -p workloads/test/MI200 -d 5 >& dispatch5_output.log
```

This output log file now contains all the metrics that should help you understand this
invocation's Speed Of Light (SOL), HBM Read Bandwidth, whether the wavefront launches
were limited by any resources such as registers or shared memory, and other things
such as your launch parameters and instruction mix in the kernel.

Exploring this output is left as an exercise for the reader. Some snapshots of
this output are shown below as a teaser.

```
0. Top Stats
0.1 Top Kernels
╒════╤══════════════════════════════════════════╤═════════╤═══════════╤════════════╤══════════════╤════════╕
│    │ Kernel_Name                              │   Count │   Sum(ns) │   Mean(ns) │   Median(ns) │    Pct │
╞════╪══════════════════════════════════════════╪═════════╪═══════════╪════════════╪══════════════╪════════╡
│  0 │ JacobiIterationKernel(int, double, doubl │    1.00 │ 503043.00 │  503043.00 │    503043.00 │ 100.00 │
│    │ e, double const*, double const*, double* │         │           │            │              │        │
│    │ , double*) [clone .kd]                   │         │           │            │              │        │
╘════╧══════════════════════════════════════════╧═════════╧═══════════╧════════════╧══════════════╧════════╛
0.2 Dispatch List
╒════╤═══════════════╤══════════════════════════════════════════════════════════════════════════════════╤══════════╕
│    │   Dispatch_ID │ Kernel_Name                                                                      │   GPU_ID │
╞════╪═══════════════╪══════════════════════════════════════════════════════════════════════════════════╪══════════╡
│  0 │             5 │ JacobiIterationKernel(int, double, double, double const*, double const*, double* │        4 │
│    │               │ , double*) [clone .kd]                                                           │          │
╘════╧═══════════════╧══════════════════════════════════════════════════════════════════════════════════╧══════════╛
```

SOL info:

```
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.13      │ VALU Active Threads       │ 64.0    │ Threads          │ 64.0     │ 100.0         │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.14      │ IPC                       │ 0.22    │ Instr/cycle      │ 5.0      │ 4.33          │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.15      │ Wavefront Occupancy       │ 2102.73 │ Wavefronts       │ 3520.0   │ 59.74         │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.16      │ Theoretical LDS Bandwidth │ 0.0     │ Gb/s             │ 23936.0  │ 0.0           │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.17      │ LDS Bank Conflicts/Access │         │ Conflicts/access │ 32.0     │               │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.18      │ vL1D Cache Hit Rate       │ 50.0    │ Pct              │ 100.0    │ 50.0          │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.19      │ vL1D Cache BW             │ 2668.12 │ Gb/s             │ 11968.0  │ 22.29         │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.20      │ L2 Cache Hit Rate         │ 49.03   │ Pct              │ 100.0    │ 49.03         │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.21      │ L2 Cache BW               │ 2094.37 │ Gb/s             │ 3481.6   │ 60.16         │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.22      │ L2-Fabric Read BW         │ 800.45  │ Gb/s             │ 1638.4   │ 48.86         │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
│ 2.1.23      │ L2-Fabric Write BW        │ 528.23  │ Gb/s             │ 1638.4   │ 32.24         │
├─────────────┼───────────────────────────┼─────────┼──────────────────┼──────────┼───────────────┤
```

Occupancy limiters:

```
├─────────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
│ 6.2.4       │ Insufficient SIMD Waveslots            │  0.00 │  0.00 │  0.00 │ Pct    │
├─────────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
│ 6.2.5       │ Insufficient SIMD VGPRs                │  0.00 │  0.00 │  0.00 │ Pct    │
├─────────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
│ 6.2.6       │ Insufficient SIMD SGPRs                │  0.00 │  0.00 │  0.00 │ Pct    │
├─────────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
│ 6.2.7       │ Insufficient CU LDS                    │  0.00 │  0.00 │  0.00 │ Pct    │
├─────────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
```

Wavefront launch stats:

```
7.1 Wavefront Launch Stats
╒═════════════╤═════════════════════╤═════════════╤═════════════╤═════════════╤════════════════╕
│ Metric_ID   │ Metric              │         Avg │         Min │         Max │ Unit           │
╞═════════════╪═════════════════════╪═════════════╪═════════════╪═════════════╪════════════════╡
│ 7.1.0       │ Grid Size           │ 16777216.00 │ 16777216.00 │ 16777216.00 │ Work items     │
├─────────────┼─────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ 7.1.1       │ Workgroup Size      │      512.00 │      512.00 │      512.00 │ Work items     │
├─────────────┼─────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ 7.1.2       │ Total Wavefronts    │        0.00 │        0.00 │        0.00 │ Wavefronts     │
├─────────────┼─────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ 7.1.3       │ Saved Wavefronts    │        0.00 │        0.00 │        0.00 │ Wavefronts     │
├─────────────┼─────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ 7.1.4       │ Restored Wavefronts │        0.00 │        0.00 │        0.00 │ Wavefronts     │
├─────────────┼─────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ 7.1.5       │ VGPRs               │       32.00 │       32.00 │       32.00 │ Registers      │
├─────────────┼─────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ 7.1.6       │ AGPRs               │        0.00 │        0.00 │        0.00 │ Registers      │
├─────────────┼─────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ 7.1.7       │ SGPRs               │       32.00 │       32.00 │       32.00 │ Registers      │
├─────────────┼─────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ 7.1.8       │ LDS Allocation      │        0.00 │        0.00 │        0.00 │ Bytes          │
├─────────────┼─────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ 7.1.9       │ Scratch Allocation  │        0.00 │        0.00 │        0.00 │ Bytes/workitem │
╘═════════════╧═════════════════════╧═════════════╧═════════════╧═════════════╧════════════════╛
```

Wavefront runtime stats:

```
7.2 Wavefront Runtime Stats
╒═════════════╤════════════════════════════╤═══════════╤═══════════╤═══════════╤═════════════════╕
│ Metric_ID   │ Metric                     │       Avg │       Min │       Max │ Unit            │
╞═════════════╪════════════════════════════╪═══════════╪═══════════╪═══════════╪═════════════════╡
│ 7.2.0       │ Kernel Time (Nanosec)      │ 503043.00 │ 503043.00 │ 503043.00 │ Ns              │
├─────────────┼────────────────────────────┼───────────┼───────────┼───────────┼─────────────────┤
│ 7.2.1       │ Kernel Time (Cycles)       │ 910207.00 │ 910207.00 │ 910207.00 │ Cycle           │
├─────────────┼────────────────────────────┼───────────┼───────────┼───────────┼─────────────────┤
│ 7.2.2       │ Instructions per wavefront │     73.00 │     73.00 │     73.00 │ Instr/wavefront │
├─────────────┼────────────────────────────┼───────────┼───────────┼───────────┼─────────────────┤
│ 7.2.3       │ Wave Cycles                │   7273.37 │   7273.37 │   7273.37 │ Cycles per wave │
├─────────────┼────────────────────────────┼───────────┼───────────┼───────────┼─────────────────┤
│ 7.2.4       │ Dependency Wait Cycles     │   5526.10 │   5526.10 │   5526.10 │ Cycles per wave │
├─────────────┼────────────────────────────┼───────────┼───────────┼───────────┼─────────────────┤
│ 7.2.5       │ Issue Wait Cycles          │   1925.98 │   1925.98 │   1925.98 │ Cycles per wave │
├─────────────┼────────────────────────────┼───────────┼───────────┼───────────┼─────────────────┤
│ 7.2.6       │ Active Cycles              │    312.00 │    312.00 │    312.00 │ Cycles per wave │
├─────────────┼────────────────────────────┼───────────┼───────────┼───────────┼─────────────────┤
│ 7.2.7       │ Wavefront Occupancy        │   2102.73 │   2102.73 │   2102.73 │ Wavefronts      │
╘═════════════╧════════════════════════════╧═══════════╧═══════════╧═══════════╧═════════════════╛
```

Instruction mix in kernel:

```
├─────────────┼──────────┼───────┼───────┼───────┼────────────────┤
│ 10.1.0      │ VALU     │ 54.00 │ 54.00 │ 54.00 │ Instr per wave │
├─────────────┼──────────┼───────┼───────┼───────┼────────────────┤
│ 10.1.1      │ VMEM     │  5.00 │  5.00 │  5.00 │ Instr per wave │
├─────────────┼──────────┼───────┼───────┼───────┼────────────────┤
│ 10.1.2      │ LDS      │  0.00 │  0.00 │  0.00 │ Instr per wave │
├─────────────┼──────────┼───────┼───────┼───────┼────────────────┤
│ 10.1.3      │ MFMA     │  0.00 │  0.00 │  0.00 │ Instr per wave │
├─────────────┼──────────┼───────┼───────┼───────┼────────────────┤
│ 10.1.4      │ SALU     │  4.00 │  4.00 │  4.00 │ Instr per wave │
├─────────────┼──────────┼───────┼───────┼───────┼────────────────┤
│ 10.1.5      │ SMEM     │  4.00 │  4.00 │  4.00 │ Instr per wave │
├─────────────┼──────────┼───────┼───────┼───────┼────────────────┤
```

VALU FLOPs:

```
├─────────────┼──────────┼───────┼───────┼───────┼────────────────┤
│ 10.1.0      │ VALU     │ 54.00 │ 54.00 │ 54.00 │ Instr per wave │
├─────────────┼──────────┼───────┼───────┼───────┼────────────────┤
│ 10.1.1      │ VMEM     │  5.00 │  5.00 │  5.00 │ Instr per wave │
├─────────────┼──────────┼───────┼───────┼───────┼────────────────┤
│ 10.1.2      │ LDS      │  0.00 │  0.00 │  0.00 │ Instr per wave │
├─────────────┼──────────┼───────┼───────┼───────┼────────────────┤
│ 10.1.3      │ MFMA     │  0.00 │  0.00 │  0.00 │ Instr per wave │
├─────────────┼──────────┼───────┼───────┼───────┼────────────────┤
│ 10.1.4      │ SALU     │  4.00 │  4.00 │  4.00 │ Instr per wave │
├─────────────┼──────────┼───────┼───────┼───────┼────────────────┤
│ 10.1.5      │ SMEM     │  4.00 │  4.00 │  4.00 │ Instr per wave │
├─────────────┼──────────┼───────┼───────┼───────┼────────────────┤
```

and

```
11.3 Arithmetic Operations
╒═════════════╤═══════════════╤═════════╤═════════╤═════════╤══════════════╕
│ Metric_ID   │ Metric        │     Avg │     Min │     Max │ Unit         │
╞═════════════╪═══════════════╪═════════╪═════════╪═════════╪══════════════╡
│ 11.3.0      │ FLOPs (Total) │ 3008.00 │ 3008.00 │ 3008.00 │ Ops per wave │
├─────────────┼───────────────┼─────────┼─────────┼─────────┼──────────────┤
│ 11.3.1      │ IOPs (Total)  │  704.00 │  704.00 │  704.00 │ Ops per wave │
├─────────────┼───────────────┼─────────┼─────────┼─────────┼──────────────┤
│ 11.3.2      │ F16 OPs       │    0.00 │    0.00 │    0.00 │ Ops per wave │
├─────────────┼───────────────┼─────────┼─────────┼─────────┼──────────────┤
│ 11.3.3      │ BF16 OPs      │    0.00 │    0.00 │    0.00 │ Ops per wave │
├─────────────┼───────────────┼─────────┼─────────┼─────────┼──────────────┤
│ 11.3.4      │ F32 OPs       │    0.00 │    0.00 │    0.00 │ Ops per wave │
├─────────────┼───────────────┼─────────┼─────────┼─────────┼──────────────┤
│ 11.3.5      │ F64 OPs       │ 3008.00 │ 3008.00 │ 3008.00 │ Ops per wave │
├─────────────┼───────────────┼─────────┼─────────┼─────────┼──────────────┤
│ 11.3.6      │ INT8 OPs      │    0.00 │    0.00 │    0.00 │ Ops per wave │
╘═════════════╧═══════════════╧═════════╧═════════╧═════════╧══════════════╛
```

Traffic to HBM:

```
17.2 L2 - Fabric Transactions
╒═════════════╤═══════════════════════════════════╤═════════╤═════════╤═════════╤════════════════╕
│ Metric_ID   │ Metric                            │ Avg     │ Min     │ Max     │ Unit           │
╞═════════════╪═══════════════════════════════════╪═════════╪═════════╪═════════╪════════════════╡
│ 17.2.0      │ Read BW                           │ 1536.03 │ 1536.03 │ 1536.03 │ Bytes per wave │
├─────────────┼───────────────────────────────────┼─────────┼─────────┼─────────┼────────────────┤
│ 17.2.1      │ HBM Read Traffic                  │ 100.0   │ 100.0   │ 100.0   │ Pct            │
├─────────────┼───────────────────────────────────┼─────────┼─────────┼─────────┼────────────────┤
│ 17.2.2      │ Remote Read Traffic               │ 0.0     │ 0.0     │ 0.0     │ Pct            │
├─────────────┼───────────────────────────────────┼─────────┼─────────┼─────────┼────────────────┤
│ 17.2.3      │ Uncached Read Traffic             │ 0.0     │ 0.0     │ 0.0     │ Pct            │
├─────────────┼───────────────────────────────────┼─────────┼─────────┼─────────┼────────────────┤
│ 17.2.4      │ Write and Atomic BW               │ 1013.66 │ 1013.66 │ 1013.66 │ Bytes per wave │
├─────────────┼───────────────────────────────────┼─────────┼─────────┼─────────┼────────────────┤
│ 17.2.5      │ HBM Write and Atomic Traffic      │ 100.0   │ 100.0   │ 100.0   │ Pct            │
├─────────────┼───────────────────────────────────┼─────────┼─────────┼─────────┼────────────────┤
```

Happy optimizing!


 





