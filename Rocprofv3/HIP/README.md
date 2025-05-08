
# Rocprofv3

## Jacobi


### Setup environment
```
module load rocm/6.2.1
module load amdclang
module load openmpi/5.0.5-ucc1.3.0-ucx1.17.0-xpmem2.7.3
```

* Download examples repo (if necessary) and navigate to the `jacobi` exercises
```
cd ~/HPCTrainingExamples/HIP/jacobi
```

### Compile and run one case

```
make clean
make
mpirun -np 2 ./Jacobi_hip -g 2 1
```

### Let's profile HIP

```
mpirun -np 2 rocprofv3 --hip-trace -- ./Jacobi_hip -g 2 1
```

Now you have two files per MPI process, one with the HW information (`XXXXX_agent_info.csv`) and for the HIP API `XXXXX_hip_api_trace.csv` where XXXXX are numbers.

```
"Domain","Function","Process_Id","Thread_Id","Correlation_Id","Start_Timestamp","End_Timestamp"
"HIP_COMPILER_API","__hipRegisterFatBinary",1389712,1389712,1,4762229062888604,4762229062892624
"HIP_COMPILER_API","__hipRegisterFunction",1389712,1389712,2,4762229062903414,4762229062910744
"HIP_COMPILER_API","__hipRegisterFatBinary",1389712,1389712,3,4762229062911814,4762229062911924
...
"HIP_RUNTIME_API","hipGetDeviceCount",1389712,1389712,9,4762229067837299,4762229201986925
"HIP_RUNTIME_API","hipStreamCreate",1389712,1389712,10,4762229253999055,4762229484333519
"HIP_RUNTIME_API","hipStreamCreate",1389712,1389712,11,4762229484352199,4762229502251764
"HIP_RUNTIME_API","hipEventCreateWithFlags",1389712,1389712,12,4762229502311284,4762229502317444
"HIP_RUNTIME_API","hipEventCreateWithFlags",1389712,1389712,13,4762229502318894,4762229502319244
"HIP_RUNTIME_API","hipEventCreateWithFlags",1389712,1389712,14,4762229502320134,4762229502320454
...
```

Correlation_Id: Unique identifier for correlation between HIP and HSA async calls during activity tracing.

Start_Timestamp: Begin time in nanoseconds (ns) when the kernel begins execution.

End_Timestamp: End time in ns when the kernel finishes execution.

### Let's create statistics

```
mpirun -np 2 rocprofv3 --stats --hip-trace -- ./Jacobi_hip -g 2 1
```

Now there is extra one file per MPI process call `XXXXX_hip_stats.csv` and the content are:

```
"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"hipMemcpy",1005,567684477,564860.176119,47.99,277461,4165053,163501.123978
"hipStreamCreate",4,257540667,64385166.750000,21.77,79530,226259882,108165143.720195
"hipStreamSynchronize",2000,143870836,71935.418000,12.16,6990,173191,64446.616580
"hipGetDeviceCount",2,139874248,69937124.000000,11.82,830,139873418,98904855.476912
"hipMalloc",7,18917455,2702493.571429,1.60,1520,4763843,2484579.981128
...
```

The column Percentage means how much percentage of the execution time this command takes, in this case we have all the calls of a specific HIP API in the same row, as you can see the column Calls of how times this HIP command was called.

### Where are the kernels?

```
mpirun -np 2 rocprofv3 --stats --kernel-trace --hip-trace -- ./Jacobi_hip -g 2 1
```

We have one extra file per MPI process, called `XXXXX_kernel_stats.csv`:

```
"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"NormKernel1(int, double, double, double const*, double*)",1001,358330487,357972.514486,53.00,354720,384680,1797.388416
"JacobiIterationKernel(int, double, double, double const*, double const*, double*, double*)",1000,172351563,172351.563000,25.49,165120,206241,7074.371069
"LocalLaplacianKernel(int, int, int, double, double, double const*, double*)",1000,133229404,133229.404000,19.71,122561,168920,8710.277191
"HaloLaplacianKernel(int, int, int, double, double, double const*, double const*, double*)",1000,9922658,9922.658000,1.47,9080,12160,315.466626
"NormKernel2(int, double const*, double*)",1001,2269165,2266.898102,0.3356,2000,3560,168.847901
"__amd_rocclr_fillBufferAligned",1,3640,3640.000000,5.384e-04,3640,3640,0.00000000e+00
```

In order to have information for each Kernel call, remove the `--stats`

### Create pftrace file for Perfetto and Visualize

 `mpirun -np 2 rocprofv3 --kernel-trace --hip-trace --output-format pftrace  -- ./Jacobi_hip -g 2 1`
 
 Now we have only pftrace files, one per MPI process.
 
 * Merge the pftraces, if you want: `cat *_results.pftrace > jacobi.pftrace`
 * Download the trace on your laptop and load the file on Perfetto.
 `scp -P 7002 aac6.amd.com:<path_to_file>/jacobi.pftrace jacobi.pftrace`

1. Open a browser and go to [https://ui.perfetto.dev/](https://ui.perfetto.dev/).
2. Click on `Open trace file` in the top left corner.
3. Navigate to the `jacobi.pftrace` or the file before the merging, that you just downloaded.
4. Use the keystrokes W,A,S,D to zoom in and move right and left in the GUI

```
Navigation
w/s	Zoom in/out
a/d	Pan left/right
```

Feel free to use various flags as they were presented in the presentation

### Hardware Counters

Read about hardware counters available for the GPU on this system (look for gfx90a section)
```
less $ROCM_PATH/lib/rocprofiler/gfx_metrics.xml
```
Create a `rocprof_counters.txt` file with the counters you would like to collect
```
vi rocprof_counters.txt
```
Content for `rocprof_counters.txt`:
```
pmc: VALUUtilization VALUBusy FetchSize WriteSize MemUnitStalled
pmc: GPU_UTIL CU_OCCUPANCY MeanOccupancyPerCU MeanOccupancyPerActiveCU
```
Execute with the counters we just added:
```
 mpirun -np 2 rocprofv3 -i rocprof_counters.txt --kernel-trace --hip-trace -- ./Jacobi_hip -g 2 1
 ```
You'll notice that `rocprofv3` runs 2 passes, one for each set of counters we have in that file.
Now the data are in two different folders, one for each MPI process, pmc_1 and pmc_2.

Explore the content of the pmc_* directories. 

Try to use the `--hsa-trace` option also.


### Tips

Do not forget for OMP Offloading information to declare the `--kernel-trace`

