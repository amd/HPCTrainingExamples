
NOTE: this example is work in progress
# ROCm&trade; Systems Profiler aka `rocprof-sys`

NOTE: extensive documentation on how to use `rocprof-sys` for the [GhostExchange examples](https://github.com/amd/HPCTrainingExamples/tree/main/MPI-examples/GhostExchange) is also available as `README.md` in this exercises repo. Here, we show how to use `rocprof-sys` tools considering the example in [https://github.com/amd/HPCTrainingExamples/tree/main/Pragma_Examples/OpenMP/CXX/8_jacobi](https://github.com/amd/HPCTrainingExamples/tree/main/Pragma_Examples/OpenMP/CXX/8_jacobi).

In this series of examples, we will demonstrate profiling with `rocprof-sys` on a platform using an AMD Instinct&trade; MI250X GPU. ROCm 6.3.2 release includes the `rocprofiler-systems` packge that you can install.

Note that the focus of this exercise is on `rocprof-sys` profiler, not on how to achieve optimal performance.

First, start by cloning HPCTrainingExamples repository and loading ROCm:

```
git clone https://github.com/amd/HPCTrainingExamples.git
```

## Environment setup

For this training, one requires recent ROCm (>=6.3) which contains `rocprof-sys`, as well as an MPI installation.

Follow the environment setup for the OpenMP C++ jacobi training example (if not done already previously).
Check if rocprof-sys-run can be found in the loaded rocm version.
```
rocprof-sys-run --version
```
If it shows you a version, you are good to go, if it shows
```
rocprof-sys-run: command not found
```
Additionally load
```
module load rocprofiler-systems
```
then you should be able to see a reasonable output of a version for rocprof-sys-run.

## Build and run

No profiling yet, just check that the code compiles and runs correctly.

```
cd HPCTrainingExamples/Pragma_Examples/OpenMP/CXX/8_jacobi/2_jacobi_targetdata
 ./Jacobi -g 1 1
```

The above run should show output that looks like this:

```
Topology size: 1 x 1
Local domain size (current node): 4096 x 4096
Global domain size (all nodes): 4096 x 4096
Rank 0 selecting device 0 on host TheraC63
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
Total Jacobi run time: 1.2876 sec.
Measured lattice updates: 13.03 GLU/s (total), 13.03 GLU/s (per process)
Measured FLOPS: 221.51 GFLOPS (total), 221.51 GFLOPS (per process)
Measured device bandwidth: 1.25 TB/s (total), 1.25 TB/s (per process)
```

## `rocprof-sys` config

First, generate the `rocprof-sys` configuration file, and ensure that this file is known to `rocprof-sys`. 

```
rocprof-sys-avail -G ~/.rocprofsys.cfg
export ROCPROFSYS_CONFIG_FILE=~/.rocprofsys.cfg
```

Second, inspect configuration file, possibly changing some variables. For example, one can modify the following lines:

```
ROCPROFSYS_PROFILE                                  = true
ROCPROFSYS_USE_ROCTX                                = true
ROCPROFSYS_SAMPLING_CPUS                            = 0
```

You can see what flags can be included in the config file by doing:

```
rocprof-sys-avail --categories rocprofsys
```

To add brief descriptions, use the `-bd` option:

```
rocprof-sys-avail -bd --categories rocprofsys
```

Note that the list of flags displayed by the commands above may not include all actual flags that can be set in the config. For a full list of options, please read the [rocprof-sys documentation](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/index.html).

You can also create a configuration file with description per option. Beware, this is quite verbose:

```
rocprof-sys-avail -G ~/rocprofsys_all.cfg --all
```

## Instrument application binary

You can instrument the binary, and inspect which functions were instrumented (note that you need to change `<TIMESTAMP>` according to your generated folder path). 

```
rocprof-sys-instrument -o ./Jacobi_hip.inst -- ./Jacobi_hip
for f in $(ls rocprofsys-Jacobi_hip.inst-output/<TIMESTAMP>/instrumentation/*.txt); do echo $f; cat $f; echo "##########"; done
```

Currently `rocprof-sys` will instrument by default only the functions with >1024 instructions, so you may need to change it by using `-i #inst` or by adding `--function-include function_name` to select the functions you are interested in. Check more options using `rocprof-sys-instrument --help` or by reading the [rocprof-sys documentation](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/index.html).

Let's instrument the most important Jacobi kernels.

```
rocprof-sys-instrument --function-include 'Jacobi_t::Run' 'JacobiIteration' -o ./Jacobi_hip.inst -- ./Jacobi_hip
```

The output should show that only these functions have been instrumented:

```
...
[rocprof-sys][exe] Finding instrumentation functions...
[rocprof-sys][exe]    1 instrumented funcs in JacobiIteration.hip
[rocprof-sys][exe]    1 instrumented funcs in JacobiRun.hip
[rocprof-sys][exe]    1 instrumented funcs in Jacobi_hip
...
```

This can also be verified with:

```
$ cat rocprofsys-Jacobi_hip.inst-output/<TIMESTAMP>/instrumentation/instrumented.txt

  StartAddress   AddressRange  #Instructions  Ratio Linkage Visibility  Module                      Function                                     FunctionSignature
      0x226440            332             71   4.68  unknown    unknown JacobiIteration.hip         JacobiIteration                              JacobiIteration
      0x224ad0            677            146   4.64  unknown    unknown JacobiRun.hip               Jacobi_t::Run                                Jacobi_t::Run
      0x226370            205             38   5.39  unknown    unknown Jacobi_hip                  __device_stub__JacobiIterationKernel         __device_stub__JacobiIterationKernel
```

## Run instrumented binary

Now that we have a new application binary where the most important functions are instrumented, we can profile it using `rocprof-sys-run` under the `mpirun` environment.

```
mpirun -np 1 rocprof-sys-run -- ./Jacobi_hip.inst -g 1 1
```

Check the command line output generated by `rocprof-sys-run`, it contains some useful overviews and **paths to generated files**. Observe that the overhead to the application runtime is small. If you had previously set `ROCPROFSYS_PROFILE=true`, inspect `wall_clock-0.txt` which includes information on the function calls made in the code, such as how many times these calls have been called (`COUNT`) and the time in seconds they took in total (`SUM`).

**In many cases, simply checking the wall_clock files might be sufficient for your profiling!**

If it is not, continue by visualizing the trace.

## Visualizing traces using `Perfetto`

Copy generated `perfetto-trace-0.proto` file to your local machine, and using the Chrome browser open the web page [https://ui.perfetto.dev/](https://ui.perfetto.dev/):

Click `Open trace file` and select the `perfetto-trace-0.proto` file. Below, you can see an example of how the trace file would be visualized on `Perfetto`:

![jacobi_hip-perfetto_screenshot](https://hackmd.io/_uploads/BkgSH-E0A.png)


If there is an error opening trace file, try using an older `Perfetto` version, e.g., by opening the web page [https://ui.perfetto.dev/v46.0-35b3d9845/#!/](https://ui.perfetto.dev/v46.0-35b3d9845/#!/).

## Additional features
### Flat profiles

Append advanced option `ROCPROFSYS_FLAT_PROFILE=true` to `~/.rocprofsys.cfg` or prepend it to the `mpirun` command:

```
ROCPROFSYS_FLAT_PROFILE=true mpirun -np 1 rocprof-sys-run -- ./Jacobi_hip.inst -g 1 1
```

`wall_clock-0.txt` file now shows overall time in seconds for each function.

Note the significant total execution time for `hipMemcpy` and `Jacobi_t::Run` calls.

### Hardware counters

To see a list of all the counters for all the devices on the node, do:

```
rocprof-sys-avail --all
```

Select the counter you are interested in, and then declare them in your configuration file (or prepend to your `mpirun` command):

```
ROCPROFSYS_ROCM_EVENTS = VALUUtilization,FetchSize
```

Run the instrumented binary, and you will observe an output file for each hardware counter specified. You should also see a row for each hardware counter in the `Perfetto` trace generated by `rocprof-sys`.

Note that you do not have to instrument again after making changes to the config file. Just running the instrumented binary picks up the changes.

```
ROCPROFSYS_ROCM_EVENTS=VALUUtilization,FetchSize mpirun -np 1 rocprof-sys-run -- ./Jacobi_hip.inst -g 1 1
cat rocprof-sys-Jacobi_hip.inst-output/<TIMESTAMP>/rocprof-device-0-VALUUtilization-0.txt
```

### Sampling

To reduce the overhead of profiling, one can use call stack sampling. Set the following in your configuration file (or prepend to your `mpirun` command):

```
ROCPROFSYS_USE_SAMPLING = true
ROCPROFSYS_SAMPLING_FREQ = 100
```

Execute the instrumented binary, inspect `sampling*` files and visualize the `Perfetto` trace:

```
mpirun -np 1 rocprof-sys-run -- ./Jacobi_hip.inst -g 1 1
ls rocprofsys-Jacobi_hip.inst-output/<TIMESTAMP>/* | grep sampling
```

### Profiling multiple MPI processes

Run the instrumented binary with multiple MPI ranks. Note separate output files for each rank, including `perfetto-trace-*.proto` and `wall_clock-*.txt` files.

```
mpirun -np 2 rocprof-sys-run -- ./Jacobi_hip.inst -g 2 1
```

Inspect output text files. Then visualize `perfetto-trace-*.proto` files in `Perfetto`. Note that one can merge multiple trace files into a single one using simple concatenation:

```
cat perfetto-trace-*.proto > merged.proto
```

## Next steps

Try to use `rocprof-sys` to profile [GhostExchange examples](https://github.com/amd/HPCTrainingExamples/tree/main/MPI-examples/GhostExchange).

**Finally, try to profile your own application!**

