
# rocpd

In this series of examples, we will demonstrate the capabilities of rocprofv3 combined with rocpd for profiling two versions of Jacobi application. rocpd is often referred to both as a format (SQLite3 database) and as a command line tool for manipulating and analyzing that database generated through rocprofv3 profiling. For a full list of rocpd options, please check `rocpd --help` and the [rocpd documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocpd-output-format.html).

Note that the focus of this exercise is on the rocprofv3 and rocpd, not on how to achieve optimal performance on MI300A. This exercise was last tested with ROCm 7.1.1 on the MI300A AAC6 cluster.

The examples are based on [Fortran+OpenMP Jacobi](https://github.com/amd/HPCTrainingExamples/tree/main/Pragma_Examples/OpenMP/Fortran/8_jacobi/1_jacobi_usm) and [MPI HIP Jacobi](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/jacobi) porting examples from HPCTrainingExamples.

## rocpd database format for rocprov3 and rocprof-sys

For more details about rocprofv3 options check its [documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html), and for examples of typical profiling and analysis with rocprofv3 check [HIP](https://github.com/amd/HPCTrainingExamples/tree/main/Rocprofv3/HIP) and [OpenMP](https://github.com/amd/HPCTrainingExamples/tree/main/Rocprofv3/OpenMP). If you are used in rocprofv3 behavior prior to ROCm 7.0 and commands that existed before rocpd was introduced, you can still typically achieve the same with newer rocprofv3 just by adding the option `--output-format` (e.g., `--output-format csv` for analysis of the text files in the terminal or `--output-format pftrace` for Perfetto traces).

In the rest of this example, we focus on rocprofv3, but similar things apply for rocprof-sys. Rocprof-sys also supports for rocpd database output with the `ROCPROFSYS_USE_ROCPD` configuration setting. For more info check [Understanding the Systems Profiler output — ROCm Systems Profiler 1.2.1 documentation](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/how-to/understanding-rocprof-sys-output.html#generating-rocpd-output).

## rocpd dependencies

Note that rocpd tool for ROCm 7.0 and 7.1 should be used with the default Python that was used for its installation (e.g., Python 3.6). This strict dependency was removed starting from ROCm 7.2.0. Additionally, rocpd tool has a dependency on pandas, so make user might need to install it using virtual environment (this is not needed on AAC6).

## Main rocpd functionalities

The rocpd tool provides three main subcommands for different analysis workflows:
    1. convert - Transform rocpd databases to alternative formats (csv, OTF2, pftrace)
    2. summary - Generate statistical analysis reports equivalent to rocprofv3 summary functionality
    3. query - Execute SQL queries against rocpd databases with flexible output options

While option to query rocpd database is very powerful, it is not commonly used for the basic analysis of HPC profiles. Hence, this option won't be discussed in this example, but we refer the curios user to the [rocpd query documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocpd-output-format.html#query) for more info.

Note that there also a couple of convenient helper scripts which mimic the bahavior of the original rocpd with most common options:
     - `rocpd2csv` == `rocpd convert --output-format csv`
     - `rocpd2pftrace` == `rocpd convert --output-format pftrace`
     - `rocpd2summary` == `rocpd summary`

## Example 1: Fortran OpenMP Jacobi
### Setup, build and run

Download the examples repository and navigate to the Fortran+OpenMP Jacobi example exercises:

```
git clone https://github.com/amd/HPCTrainingExamples.git
cd HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/8_jacobi/1_jacobi_usm
```

Load the necessary modules, including amdflang (a.k.a. flang-new) compiler. Note that the module name for the amdflang compiler on your system might differ, check for `rocm-afar-drop`, `amd-llvm`, `amdflang-new` or something similar.

```
module load rocm
module load amdflang-new
```

For now, unset the `HSA_XNACK` environment variable:

```
export HSA_XNACK=0
```

No profiling yet, just check that the code compiles and runs correctly:

```
make clean
make FC=amdflang
./jacobi -m 1024
```

This run should show output that looks like this:

```
Domain size:  1024 x  1024
Starting Jacobi run
Iteration:    0 - Residual: 4.42589E-02
Iteration:  100 - Residual: 1.25109E-03
Iteration:  200 - Residual: 7.43407E-04
Iteration:  300 - Residual: 5.48292E-04
Iteration:  400 - Residual: 4.41773E-04
Iteration:  500 - Residual: 3.73617E-04
Iteration:  600 - Residual: 3.25807E-04
Iteration:  700 - Residual: 2.90186E-04
Iteration:  800 - Residual: 2.62490E-04
Iteration:  900 - Residual: 2.40262E-04
Iteration: 1000 - Residual: 2.21976E-04
Stopped after 1000 iterations with residue: 2.21976E-04
Total Jacobi run time: ***** sec.
Measured lattice updates: 0.087 LU/s
Effective Flops:   1.5 GFlops
Effective device bandwidth: 0.008 TB/s
Effective AI=0.177
```

### Profile application only once

Collect the first profile about the GPU kernels used by the Fortran+OpenMP Jacobi application. Do not forget `--` between rocprofv3 options and application binary. For a more consistent output path, we will use the optional arguments `--output-directory` and `--output-file`. Rocpd database format is a default one starting from ROCm 7.0, but you can also explicitly specify it by adding `--output-format rocpd`.

```
rocprofv3 --kernel-trace --output-directory omp_output --output-file omp -- ./jacobi -m 1024
```

rocprofv3 should generate a single rocpd database file `omp_output/omp_results.db`. Profiling data is now available and there is no longer need to additionally profile the code with rocprofv3 - just do different analysis using rocpd tool!

### Conversion to csv

First to generate csv outputs, there are two simple alternatives:

```
rocpd2csv -i omp_output/omp_results.db
# or
rocpd convert --output-format csv -i omp_output/omp_results.db
```

One can now observe newly generated `rocpd-output-data/out_agent_info.csv` and `rocpd-output-data/out_kernel_trace.csv`, which can be opened in the terminal or with the sheet viewer and analyzed following the same instructions as in the other [example](https://github.com/amd/HPCTrainingExamples/tree/main/Rocprofv3/OpenMP).

```
head rocpd-output-data/out_kernel_trace.csv

"Guid","Kind","Agent_Id","Queue_Id","Stream_Id","Thread_Id","Dispatch_Id","Kernel_Id","Kernel_Name","Correlation_Id","Start_Timestamp","End_Timestamp","Lds_Block_Size","Scratch_Size","Vgpr_Count","Accum_Vgpr_Count","Sgpr_Count","Workgroup_Size_X","Workgroup_Size_Y","Workgroup_Size_Z","Grid_Size_X","Grid_Size_Y","Grid_Size_Z"
"00006409-00d1-70d1-a5cb-b5ab88214544","KERNEL_DISPATCH","Agent 4",1,0,714789,1,3,"__omp_offloading_30_57b9f__QMnorm_modPnorm_l23",1,1678314002982197,1678314003082277,8456,0,16,0,32,256,1,1,233472,1,1
"00006409-00d1-70d1-a5cb-b5ab88214544","KERNEL_DISPATCH","Agent 4",1,0,714789,2,2,"__omp_offloading_30_57b8c__QMlaplacian_modPlaplacian_l22",2,1678314008879169,1678314008930489,0,0,32,0,32,256,1,1,233472,1,1
"00006409-00d1-70d1-a5cb-b5ab88214544","KERNEL_DISPATCH","Agent 4",1,0,714789,3,1,"__omp_offloading_30_57b7d__QMboundary_modPboundary_conditions_l24",3,1678314009761051,1678314009790891,0,0,20,4,64,32,1,1,4096,1,1
"00006409-00d1-70d1-a5cb-b5ab88214544","KERNEL_DISPATCH","Agent 4",1,0,714789,4,4,"__omp_offloading_30_57ba0__QMupdate_modPupdate_l22",4,1678314012698497,1678314012750017,0,0,24,0,48,256,1,1,233472,1,1
"00006409-00d1-70d1-a5cb-b5ab88214544","KERNEL_DISPATCH","Agent 4",1,0,714789,5,3,"__omp_offloading_30_57b9f__QMnorm_modPnorm_l23",5,1678314013691619,1678314013782299,8456,0,16,0,32,256,1,1,233472,1,1
"00006409-00d1-70d1-a5cb-b5ab88214544","KERNEL_DISPATCH","Agent 4",1,0,714789,6,2,"__omp_offloading_30_57b8c__QMlaplacian_modPlaplacian_l22",6,1678314014399380,1678314014451780,0,0,32,0,32,256,1,1,233472,1,1
"00006409-00d1-70d1-a5cb-b5ab88214544","KERNEL_DISPATCH","Agent 4",1,0,714789,7,1,"__omp_offloading_30_57b7d__QMboundary_modPboundary_conditions_l24",7,1678314015208622,1678314015237622,0,0,20,4,64,32,1,1,4096,1,1
"00006409-00d1-70d1-a5cb-b5ab88214544","KERNEL_DISPATCH","Agent 4",1,0,714789,8,4,"__omp_offloading_30_57ba0__QMupdate_modPupdate_l22",8,1678314016416905,1678314016468345,0,0,24,0,48,256,1,1,233472,1,1
"00006409-00d1-70d1-a5cb-b5ab88214544","KERNEL_DISPATCH","Agent 4",1,0,714789,9,3,"__omp_offloading_30_57b9f__QMnorm_modPnorm_l23",9,1678314017403787,1678314017498787,8456,0,16,0,32,256,1,1,233472,1,1
```

### Conversion to pftrace

Similarly to csv conversion, one can generate pftrace files that can be opened in Perfetto:

```
rocpd2pftrace -i omp_output/omp_results.db
# or
rocpd convert --output-format pftrace -i omp_output/omp_results.db
```

Then, the same way as in [example](https://github.com/amd/HPCTrainingExamples/tree/main/Rocprofv3/OpenMP), one should copy the `rocpd-output-data/out_results.pftrace` file to the local machine and open it in a browser using Perfetto.

Note that additional options are available related to buffers (e.g., `--perfetto-buffer-size`) or hardware queues (`--group-by-queue`), which can be interesting when analyzing large executions or the ones using multiple streams (e.g., see [stream example](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/Stream_Overlap).

It is also possible to convert traces to the otf2 format, suitable for visualization by different 3rd party tools.

### Filtering

From the original large database file, one can use filtering to generate a csv/pftrace output for only a subsection of the whole execution. This allows users to exclude initialization/finalization phases or to focus only on a handful of iterations. It is achieved either by specifying the absolute timestamps in nanoseconds or using the percentage notation. Note that we will additionally specify a new output file name using `--output-file out_filtered`, in order not to override the existing csv output files:

```
rocpd2csv -i omp_output/omp_results.db --output-file out_filtered --start 1678314981964256 --end 1678317483667894
# or
rocpd2csv -i omp_output/omp_results.db --output-file out_filtered --start 25% --end 75%
```

The output should look like this:

```
#  Initial time bounds: 1678314002982197 : 1678318478629607 nsec (delta=4475647410 nsec)
# Windowed time bounds: 1678315122323119 : 1678317339084483 nsec (delta=2216761364 nsec)
# Time windowing reduced the duration by  50.47%
Exported to: <PATH TO EXAMPLE>/rocpd-output-data/out_filtered_agent_info.csv

Exported to: <PATH TO EXAMPLE>/rocpd-output-data/out_filtered_kernel_trace.csv
```

For the application that contains markers, there is additionally a method to get a time-windowed trace export is using `--start-marker` and `--end-marker`.

### Basic summary

One can use SQL queries or Python API to generate custom summaries. However, rocpd tool already provides several useful pre-defined analyses, and more are coming in the near future.

By default, `rocpd2summary` provides a useful summary overview of the execution:

```
rocpd2summary -i omp_output/omp_results.db
# or
rocpd summary -i omp_output/omp_results.db
```

This should generate an output similar to:

```
KERNELS_SUMMARY:
                                                             Name  Calls  DURATION (nsec)  AVERAGE (nsec)  PERCENT (INC)  MIN (nsec)  MAX (nsec)     STD_DEV
                   __omp_offloading_30_57b9f__QMnorm_modPnorm_l23   1001         92043050    91951.098901      42.085668       85520      105080 3300.146442
         __omp_offloading_30_57b8c__QMlaplacian_modPlaplacian_l22   1000         49900815    49900.815000      22.816596       26440       58200 2455.666999
               __omp_offloading_30_57ba0__QMupdate_modPupdate_l22   1000         48184981    48184.981000      22.032050       31920       57200 4369.884229
__omp_offloading_30_57b7d__QMboundary_modPboundary_conditions_l24   1000         28575181    28575.181000      13.065686       14560       34921 2319.276042
```

rocpd summary can also generate data analysis reports of the same data in different formats (csv, html,):

```
rocpd2summary -i omp_output/omp_results.db --format html --output-path reports/
```

### Comparing multiple executions

Let's collect a profile for a second, slightly different execution, and then do comparative analysis. For this example, the easiest is to change `HSA_XNACK` variable, and then rerun the profiling:

```
export HSA_XNACK=1
rocprofv3 --kernel-trace --output-directory omp_output --output-file omp2 -- ./jacobi -m 1024
```

One can now observe 2 databases from two profile runs using a single `-i` (`-i omp_output/omp_results.db omp_output/omp2_results.db`). First, we can merge them into a single, new csv/pftrace output:

```
rocpd convert -i omp_output/omp_results.db omp_output/omp2_results.db --output-format csv pftrace --output-path combined_folder --output-file combined_profile
```

We can also generate a unified_summary report:

```
rocpd summary -i omp_output/omp_results.db -i omp_output/omp2_results.db --output-path combined_folder
```

That generates an output similar to:

```
KERNELS_SUMMARY:
                                                             Name  Calls  DURATION (nsec)  AVERAGE (nsec)  PERCENT (INC)  MIN (nsec)  MAX (nsec)       STD_DEV
               __omp_offloading_30_57ba0__QMupdate_modPupdate_l22   2000        304492165   152246.082500      34.275096       31920     4364040 199740.702362
         __omp_offloading_30_57b8c__QMlaplacian_modPlaplacian_l22   2000        292131291   146065.645500      32.883697       26440    25579602 625033.260297
                   __omp_offloading_30_57b9f__QMnorm_modPnorm_l23   2002        235146210   117455.649351      26.469183       84200    15000641 375054.076868
__omp_offloading_30_57b7d__QMboundary_modPboundary_conditions_l24   2000         56607620    28303.810000       6.372025       14560       34921   2088.160445

```

The number of kernel calls has doubled, which clearly indicates that the two runs were merged.

Let's now compare two runs by adding the `--summary-by-rank` option (the name of this option comes from the most common use case which compares output of different MPI ranks):

```
rocpd summary -i omp_output/omp_results.db omp_output/omp2_results.db --output-path combined_folder --summary-by-rank
```

This should additionally print:

```
KERNELS_SUMMARY_BY_RANK:
 ProcessID        Hostname                                                              Name  Calls  DURATION (nsec)  AVERAGE (nsec)  PERCENT (INC)  MIN (nsec)  MAX (nsec)       STD_DEV
    714789 ppac-pl1-s24-26                    __omp_offloading_30_57b9f__QMnorm_modPnorm_l23   1001         92043050    91951.098901      42.085668       85520      105080   3300.146442
    714789 ppac-pl1-s24-26          __omp_offloading_30_57b8c__QMlaplacian_modPlaplacian_l22   1000         49900815    49900.815000      22.816596       26440       58200   2455.666999
    714789 ppac-pl1-s24-26                __omp_offloading_30_57ba0__QMupdate_modPupdate_l22   1000         48184981    48184.981000      22.032050       31920       57200   4369.884229
    714789 ppac-pl1-s24-26 __omp_offloading_30_57b7d__QMboundary_modPboundary_conditions_l24   1000         28575181    28575.181000      13.065686       14560       34921   2319.276042
    715613 ppac-pl1-s24-26                __omp_offloading_30_57ba0__QMupdate_modPupdate_l22   1000        256307184   256307.184000      38.273469       40520     4364040 241110.829207
    715613 ppac-pl1-s24-26          __omp_offloading_30_57b8c__QMlaplacian_modPlaplacian_l22   1000        242230476   242230.476000      36.171442       45440    25579602 873615.657583
    715613 ppac-pl1-s24-26                    __omp_offloading_30_57b9f__QMnorm_modPnorm_l23   1001        143103160   142960.199800      21.369102       84200    15000641 529300.132811
    715613 ppac-pl1-s24-26 __omp_offloading_30_57b7d__QMboundary_modPboundary_conditions_l24   1000         28032439    28032.439000       4.185988       24720       33200   1788.496474
```

Hostnames are the same, but two profiling runs had different process id. One can now analyze the difference in kernel behavior depending on the `HSA_XNACK` variable.

To explore slightly more advanced rocpd summary options, let's use another example that parallelizes application execution using multiple MPI ranks.

## Example 2: MPI HIP Jacobi
### Setup, build and run

Download the examples repository and navigate to the Fortran+OpenMP Jacobi example exercises:

```
git clone https://github.com/amd/HPCTrainingExamples.git
cd HPCTrainingExamples/HIP/jacobi
```

Load the necessary modules, including openmpi.

```
module load rocm
module load openmpi
```

No profiling yet, just check that the code compiles and runs correctly with 2 MPI ranks:

```
make clean
make
mpirun -np 2 ./Jacobi_hip -g 2 1
```

This run should show output that looks like this:

```
Topology size: 2 x 1
Local domain size (current node): 4096 x 4096
Global domain size (all nodes): 8192 x 4096
Rank 0 selecting device 0 on host ppac-pl1-s24-26
Rank 1 selecting device 1 on host ppac-pl1-s24-26
Starting Jacobi run.
Iteration:   0 - Residual: 0.031741
Iteration: 100 - Residual: 0.001755
Iteration: 200 - Residual: 0.001045
Iteration: 300 - Residual: 0.000771
Iteration: 400 - Residual: 0.000621
Iteration: 500 - Residual: 0.000526
Iteration: 600 - Residual: 0.000459
Iteration: 700 - Residual: 0.000408
Iteration: 800 - Residual: 0.000370
Iteration: 900 - Residual: 0.000338
Iteration: 1000 - Residual: 0.000313
Stopped after 1000 iterations with residue 0.000313
Total Jacobi run time: 0.9922 sec.
Measured lattice updates: 33.82 GLU/s (total), 16.91 GLU/s (per process)
Measured FLOPS: 574.93 GFLOPS (total), 287.46 GFLOPS (per process)
Measured device bandwidth: 3.25 TB/s (total), 1.62 TB/s (per process)
Percentage of MPI traffic hidden by computation: 87.7
```

### MPI application performance analysis of different ranks

Let's first collect a profile for this MPI+HIP Jacobi application, running with 4 MPI ranks:

```
mpirun -np 4 rocprofv3 --hip-trace --output-directory mpi_output -- ./Jacobi_hip -g 2 2
```

Generate unified summary for overall application performance:

```
rocpd summary -i mpi_output/ppac-pl1-s24-26/*_results.db --output-path mpi_output --output-file application_overview
```

Now we can use the `--summary-by-rank` option to identify load-balancing issues with rank-by-rank comparison:

```
rocpd summary -i mpi_output/ppac-pl1-s24-26/*_results.db --output-path mpi_output --output-file load_balance_analysis --summary-by-rank
```

It is important to note that while rocprofv3 can be used to run MPI application with multiple ranks, the tool does not profile MPI communications. rocprof-sys tool is much more suited for that type of analysis. 

### GPU scaling study

Similar approach can be use for analysis of weak/strong scaling performance, e.g.:

```
for gpus in 1 2 4; do
    mpirun -np $gpus rocprofv3 --kernel-trace --output-format rocpd --output-directory scaling_${gpus} -- ./Jacobi_hip -g $gpus 1
done
```

Then to compare the GPU kernel performance for these three different runs, again use `--summary-by-rank`:

```
rocpd summary -i scaling_1/ppac-pl1-s24-26/718729_results.db scaling_2/ppac-pl1-s24-26/718764_results.db scaling_4/ppac-pl1-s24-26/718835_results.db --summary-by-rank
```

This provides an output similar to:

```
KERNELS_SUMMARY:
                                                                                      Name  Calls  DURATION (nsec)  AVERAGE (nsec)  PERCENT (INC)  MIN (nsec)  MAX (nsec)      STD_DEV
                                  NormKernel1(int, double, double, double const*, double*)   3003       1163863212   387566.837163      53.158347      350681    44723133 1.053328e+06
JacobiIterationKernel(int, double, double, double const*, double const*, double*, double*)   3000        558300618   186100.206000      25.499851      164360    49084273 8.931972e+05
               LocalLaplacianKernel(int, int, int, double, double, double const*, double*)   3000        430398154   143466.051333      19.658027      138040      927873 1.451155e+04
 HaloLaplacianKernel(int, int, int, double, double, double const*, double const*, double*)   3000         25618372     8539.457333       1.170095        7560       14160 5.868638e+02
                                                  NormKernel2(int, double const*, double*)   3003          7978572     2656.867133       0.364414         920       69000 1.264590e+03
                                                                   __amd_rocclr_copyBuffer   3003          3256244     1084.330336       0.148726         560        5680 4.841719e+02
                                                            __amd_rocclr_fillBufferAligned      3            11840     3946.666667       0.000541        3320        4960 8.857389e+02

KERNELS_SUMMARY_BY_RANK:
 ProcessID        Hostname                                                                                       Name  Calls  DURATION (nsec)  AVERAGE (nsec)  PERCENT (INC)  MIN (nsec)  MAX (nsec)      STD_DEV
    718729 ppac-pl1-s24-26                                   NormKernel1(int, double, double, double const*, double*)   1001        355961654   355606.047952      52.060841      350681      456680 4.282216e+03
    718729 ppac-pl1-s24-26 JacobiIterationKernel(int, double, double, double const*, double const*, double*, double*)   1000        170921199   170921.199000      24.997921      165160      182880 3.118681e+03
    718729 ppac-pl1-s24-26                LocalLaplacianKernel(int, int, int, double, double, double const*, double*)   1000        145242140   145242.140000      21.242254      140560      152441 2.258880e+03
    718729 ppac-pl1-s24-26  HaloLaplacianKernel(int, int, int, double, double, double const*, double const*, double*)   1000          8215060     8215.060000       1.201486        7560       10240 2.957950e+02
    718729 ppac-pl1-s24-26                                                   NormKernel2(int, double const*, double*)   1001          2451842     2449.392607       0.358592        1920        3640 2.245723e+02
    718729 ppac-pl1-s24-26                                                                    __amd_rocclr_copyBuffer   1001           944802      943.858142       0.138181         560        5680 4.251742e+02
    718729 ppac-pl1-s24-26                                                             __amd_rocclr_fillBufferAligned      1             4960     4960.000000       0.000725        4960        4960          NaN
    718764 ppac-pl1-s24-26                                   NormKernel1(int, double, double, double const*, double*)   1001        357497980   357140.839161      52.515908      352321      450880 4.450820e+03
    718764 ppac-pl1-s24-26 JacobiIterationKernel(int, double, double, double const*, double const*, double*, double*)   1000        168958012   168958.012000      24.819674      164960      177601 2.162844e+03
    718764 ppac-pl1-s24-26                LocalLaplacianKernel(int, int, int, double, double, double const*, double*)   1000        142193735   142193.735000      20.888043      138040      149441 1.616942e+03
    718764 ppac-pl1-s24-26  HaloLaplacianKernel(int, int, int, double, double, double const*, double const*, double*)   1000          8391209     8391.209000       1.232656        7640       12720 3.670044e+02
    718764 ppac-pl1-s24-26                                                   NormKernel2(int, double const*, double*)   1001          2652846     2650.195804       0.389699        1960        4280 3.601542e+02
    718764 ppac-pl1-s24-26                                                                    __amd_rocclr_copyBuffer   1001          1045161     1044.116883       0.153533         560        3960 4.468414e+02
    718764 ppac-pl1-s24-26                                                             __amd_rocclr_fillBufferAligned      1             3320     3320.000000       0.000488        3320        3320          NaN
    718835 ppac-pl1-s24-26                                   NormKernel1(int, double, double, double const*, double*)   1001        450403578   449953.624376      54.598139      362401    44723133 1.823412e+06
    718835 ppac-pl1-s24-26 JacobiIterationKernel(int, double, double, double const*, double const*, double*, double*)   1000        218421407   218421.407000      26.477148      164360    49084273 1.547067e+06
    718835 ppac-pl1-s24-26                LocalLaplacianKernel(int, int, int, double, double, double const*, double*)   1000        142962279   142962.279000      17.329957      138680      927873 2.488830e+04
    718835 ppac-pl1-s24-26  HaloLaplacianKernel(int, int, int, double, double, double const*, double const*, double*)   1000          9012103     9012.103000       1.092451        7840       14160 6.788032e+02
    718835 ppac-pl1-s24-26                                                   NormKernel2(int, double const*, double*)   1001          2873884     2871.012987       0.348374         920       69000 2.128750e+03
    718835 ppac-pl1-s24-26                                                                    __amd_rocclr_copyBuffer   1001          1266281     1265.015984       0.153499         560        4121 5.188882e+02
    718835 ppac-pl1-s24-26                                                             __amd_rocclr_fillBufferAligned      1             3560     3560.000000       0.000432        3560        3560          NaN
```


## Next steps

Try to use rocpd for some more example from HPCTrainingExamples repository.

**Finally, try to profile your own application!**

