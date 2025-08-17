
# Rocprofv3 Exercises for HIP

## Jacobi


### Setup environment
```
module load rocm
module load amdclang
module load openmpi
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

Note that there is an output message showing that the output csv files are placed into a subdirectory.
There are two files per MPI process: one with the HW information (`XXXXX_agent_info.csv`) and for the HIP API `XXXXX_hip_api_trace.csv` where XXXXX are numbers.

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

Now there are two extra files per MPI process. The first is called `XXXXX_domain_stats.csv`. The contents
are

```
"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"HIP_API",24044,1660103043,69044.378764,100.00,79,297454413,1941729.969641
```

The second is called `XXXXX_hip_api_stats.csv` and the contents are:

```
"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"hipMemcpy",1005,1248355080,1242144.358209,75.20,427157,16960295,594564.020099
"hipMemset",1,297454413,297454413.000000,17.92,297454413,297454413,0.00000000e+00
"hipStreamSynchronize",2000,62408983,31204.491500,3.76,14160,11567558,259729.446877
"hipStreamCreate",2,13571635,6785817.500000,0.8175,6588338,6983297,279278.187191
"hipInit",1,10837633,10837633.000000,0.6528,10837633,10837633,0.00000000e+00
"hipLaunchKernel",5002,9285550,1856.367453,0.5593,1030,501227,10801.917865
"hipMemcpy2DAsync",1000,7495461,7495.461000,0.4515,1500,5522446,174573.470155
"hipEventRecord",2000,3031773,1515.886500,0.1826,750,7250,627.097396
"hipMemcpyAsync",1000,2597421,2597.421000,0.1565,2040,36110,1201.163919
"hipFree",4,1466101,366525.250000,0.0883,3380,1380181,676012.011385
"hipDeviceSynchronize",1001,713007,712.294705,0.0429,540,3060,200.535668
"hipEventElapsedTime",1000,603200,603.200000,0.0363,449,3090,174.077112
"__hipPushCallConfiguration",5002,572690,114.492203,0.0345,80,15670,223.429644
"__hipPopCallConfiguration",5002,535744,107.105958,0.0323,79,14360,265.450993
"hipHostMalloc",3,497417,165805.666667,0.0300,92599,233828,70757.088651
"hipMalloc",7,336148,48021.142857,0.0202,1820,171208,57008.652464
"hipHostFree",2,294098,147049.000000,0.0177,118099,175999,40941.482631
"__hipRegisterFatBinary",3,26819,8939.666667,1.616e-03,80,26599,15293.460705
"__hipRegisterFunction",5,8520,1704.000000,5.132e-04,170,7530,3257.518995
"hipGetDeviceCount",1,8380,8380.000000,5.048e-04,8380,8380,0.00000000e+00
"hipEventCreate",2,1690,845.000000,1.018e-04,260,1430,827.314934
"hipSetDevice",1,1280,1280.000000,7.710e-05,1280,1280,0.00000000e+00
```

The column Percentage means how much percentage of the execution time this command takes, in this case we have all the calls of a specific HIP API in the same row, as you can see the column Calls of how times this HIP command was called.

### Where are the kernels?

```
mpirun -np 2 rocprofv3 --stats --kernel-trace --hip-trace -- ./Jacobi_hip -g 2 1
```

We have two more files per MPI process. The first is called `XXXXX_kernel_stats.csv`:

```
"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"JacobiIterationKernel(int, double, double, double const*, double const*, double*, double*)",1000,545275480,545275.480000,43.03,526082,709763,8821.418627
"NormKernel1(int, double, double, double const*, double*)",1001,410964270,410553.716284,32.43,401121,421282,2773.931233
"LocalLaplacianKernel(int, int, int, double, double, double const*, double*)",1000,285486734,285486.734000,22.53,278561,291521,2040.331200
"HaloLaplacianKernel(int, int, int, double, double, double const*, double const*, double*)",1000,13996851,13996.851000,1.10,12800,17120,424.909547
"__amd_rocclr_copyBuffer",1001,7823550,7815.734266,0.6173,6720,9280,672.661144
"NormKernel2(int, double const*, double*)",1001,3754094,3750.343656,0.2962,3520,4320,105.963559
"__amd_rocclr_fillBufferAligned",1,5920,5920.000000,4.671e-04,5920,5920,0.00000000e+00
```

The second file is called `XXXXX_kernel_trace.csv`. It has detailed information on each kernel dispatch.

```
"Kind","Agent_Id","Queue_Id","Thread_Id","Dispatch_Id","Kernel_Id","Kernel_Name","Correlation_Id","Start_Timestamp","End_Timestamp","Private_Segment_Size","Group_Segment_Size","Workgroup
_Size_X","Workgroup_Size_Y","Workgroup_Size_Z","Grid_Size_X","Grid_Size_Y","Grid_Size_Z"
"KERNEL_DISPATCH",8,1,252734,1,10,"__amd_rocclr_fillBufferAligned",15,4484384343929154,4484384343935074,0,0,256,1,1,8192,1,1
"KERNEL_DISPATCH",8,2,252734,2,18,"NormKernel1(int, double, double, double const*, double*)",33,4484384527705139,4484384528106260,0,1024,128,1,1,16384,1,1
"KERNEL_DISPATCH",8,2,252734,3,17,"NormKernel2(int, double const*, double*)",36,4484384528106260,4484384528109780,0,1024,128,1,1,128,1,1
"KERNEL_DISPATCH",8,1,252734,4,13,"__amd_rocclr_copyBuffer",37,4484384528126420,4484384528135540,0,0,512,1,1,512,1,1
...
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

