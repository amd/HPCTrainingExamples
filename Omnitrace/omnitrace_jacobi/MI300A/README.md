# Omnitrace
NOTE: extensive documentation on how to use Omnitrace for the [GhostExchange examples](https://github.com/amd/HPCTrainingExamples/tree/main/MPI-examples/GhostExchange) is also available as `README.md` in this exercises repo. Here, we show how to use Omnitrace tools considering the example in HPCTrainingExamples/HIP/jacobi.

In this series of examples, we will demonstrate profiling with Omnitrace on a platform using an AMD Instinct&trade; MI300 GPU, hosted in a training container in AAC6 environment. ROCm releases (6.2+) now include Omnitrace. Please install the additional package called `omnitrace` along with ROCm to find the Omnitrace binaries in the `${ROCM_PATH}/bin` directory.

Note that the focus of this exercise is on Omnitrace profiler, not on how to achieve optimal performance on MI300A.

First, start by cloning HPCTrainingExamples repository:

```
git clone https://github.com/amd/HPCTrainingExamples.git
```

## Environment setup

In this training, Omnitrace is available either as a part of the ROCm release or as a stand-alone AMDResearch tool. For all users having access to ROCm 6.2 or newer, we recommend using Omnitrace from ROCm.

```
module load rocm
```

## Build and run

No profiling yet, just check that the code compiles and runs correctly.

```
cd HPCTrainingExamples/HIP/jacobi
make
mpirun -np 1 ./Jacobi_hip -g 1 1
```

The `mpirun` should show output that looks like this:

```
Topology size: 1 x 1
Local domain size (current node): 4096 x 4096
Global domain size (all nodes): 4096 x 4096
Rank 0 selecting device 0 on host 1dfeb1ac07e2
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
Total Jacobi run time: 0.7315 sec.
Measured lattice updates: 22.94 GLU/s (total), 22.94 GLU/s (per process)
Measured FLOPS: 389.91 GFLOPS (total), 389.91 GFLOPS (per process)
Measured device bandwidth: 2.20 TB/s (total), 2.20 TB/s (per process)
```

## Omnitrace config

First, generate the Omnitrace configuration file, and ensure that this file is known to Omnitrace. 

```
omnitrace-avail -G ~/.omnitrace.cfg
export OMNITRACE_CONFIG_FILE=~/.omnitrace.cfg
```

Second, inspect configuration file, possibly changing some variables. For example, one can modify the following lines:

```
OMNITRACE_PROFILE                                  = true
OMNITRACE_USE_MPIP                                 = false
OMNITRACE_USE_ROCTX                                = true
OMNITRACE_SAMPLING_CPUS                            = 0
```

You can see what flags can be included in the config file by doing:

```
omnitrace-avail --categories omnitrace
```

To add brief descriptions, use the `-bd` option:

```
omnitrace-avail -bd --categories omnitrace
```

Note that the list of flags displayed by the commands above may not include all actual flags that can be set in the config. For a full list of options, please read the [Omnitrace documentation](https://rocm.docs.amd.com/projects/omnitrace/en/latest/index.html). 

You can also create a configuration file with description per option. Beware, this is quite verbose:

```
omnitrace-avail -G ~/omnitrace_all.cfg --all
```

## Instrument application binary

You can instrument the binary, and inspect which functions were instrumented (note that you need to change `<TIMESTAMP>` according to your generated folder path). 

```
omnitrace-instrument -o ./Jacobi_hip.inst -- ./Jacobi_hip
for f in $(ls omnitrace-Jacobi_hip.inst-output/<TIMESTAMP>/instrumentation/*.txt); do echo $f; cat $f; echo "##########"; done
```

Currently Omnitrace will instrument by default only the functions with >1024 instructions, so you may need to change it by using `-i #inst` or by adding `--function-include function_name` to select the functions you are interested in. Check more options using `omnitrace-instrument --help` or by reading the [Omnitrace documentation](https://rocm.docs.amd.com/projects/omnitrace/en/latest/index.html).

Let's instrument the most important Jacobi kernels.

```
omnitrace-instrument --function-include 'Jacobi_t::Run' 'JacobiIteration' -o ./Jacobi_hip.inst -- ./Jacobi_hip
```

The output should show that only those functions have been instrumented:

```
[omnitrace][exe] Finding instrumentation functions...
[omnitrace][exe]    1 instrumented funcs in JacobiIteration.hip
[omnitrace][exe]    1 instrumented funcs in JacobiRun.hip
[omnitrace][exe]    1 instrumented funcs in Jacobi_hip
```

This can also be verified with:

```
$ cat omnitrace-Jacobi_hip.inst-output/<TIMESTAMP>/instrumentation/instrumented.txt

  StartAddress   AddressRange  #Instructions  Ratio Linkage Visibility  Module                      Function                                     FunctionSignature
      0x226440            332             71   4.68  unknown    unknown JacobiIteration.hip         JacobiIteration                              JacobiIteration
      0x224ad0            677            146   4.64  unknown    unknown JacobiRun.hip               Jacobi_t::Run                                Jacobi_t::Run
      0x226370            205             38   5.39  unknown    unknown Jacobi_hip                  __device_stub__JacobiIterationKernel         __device_stub__JacobiIterationKernel
```

## Run instrumented binary

Now that we have a new application binary where the most important functions are instrumented, we can run it under the `mpirun` environment.

```
mpirun -np 1 omnitrace-run -- ./Jacobi_hip.inst -g 1 1
```

Check the command line output generated by Omnitrace, it contains some useful overviews and **paths to generated files**. Observe that the overhead to the application runtime is small. If you had previously set `OMNITRACE_PROFILE=true`, inspect `wall_clock-0.txt` which includes information on the function calls made in the code, such as how many times these calls have been called (`COUNT`) and the time in seconds they took in total (`SUM`).

**In many cases, simply checking the wall_clock files might be sufficient for your profiling!**

If it is not, continue by visualizing the trace.

## Visualizing traces using `Perfetto`

Copy generated `perfetto-trace-0.proto` file to your local machine, and using the Chrome browser open the web page [https://ui.perfetto.dev/](https://ui.perfetto.dev/):

```
scp -i <path/to/ssh/key> -P <port_number> <username>@aac6.amd.com:~/<path/to/proto/file> .
```

Click `Open trace file` and select the `perfetto-trace-0.proto` file. Below, you can see an example of how the trace file would be visualized on `Perfetto`:

![jacobi_hip-perfetto_screenshot](https://hackmd.io/_uploads/BkgSH-E0A.png)


If there is an error opening one trace file, try using an older `Perfetto` version, e.g., by opening the web page [https://ui.perfetto.dev/v46.0-35b3d9845/#!/](https://ui.perfetto.dev/v46.0-35b3d9845/#!/).

## Additional features
### Flat profiles

Append advanced option `OMNITRACE_FLAT_PROFILE=true` to `~/.omnitrace.cfg` or prepend it to the `mpirun` command:

```
OMNITRACE_FLAT_PROFILE=true mpirun -np 1 omnitrace-run -- ./Jacobi_hip.inst -g 1 1
```

`wall_clock-0.txt` file now shows overall time in seconds for each function.

Note the significant total execution time for `hipMemcpy` and `Jacobi_t::Run` calls.

### Hardware counters

To see a list of all the counters for all the devices on the node, do:

```
omnitrace-avail --all
```

Select the counter you are interested in, and then declare them in your configuration file (or prepend to your `mpirun` command):

```
OMNITRACE_ROCM_EVENTS = VALUUtilization,FetchSize
```

Run the instrumented binary, and you will observe an output file for each hardware counter specified. You should also see a row for each hardware counter in the `Perfetto` trace generated by Omnitrace.

Note that you do not have to instrument again after making changes to the config file. Just running the instrumented binary picks up the changes.

```
OMNITRACE_ROCM_EVENTS=VALUUtilization,FetchSize mpirun -np 1 omnitrace-run -- ./Jacobi_hip.inst -g 1 1
cat omnitrace-Jacobi_hip.inst-output/<TIMESTAMP>/rocprof-device-0-VALUUtilization-0.txt
```

### Sampling

To reduce the overhead of profiling, one can use sampling. Set the following in your configuration file (or prepend to your `mpirun` command):

```
OMNITRACE_USE_SAMPLING = true
OMNITRACE_SAMPLING_FREQ = 100
```

Execute the instrumented binary, inspect `sampling*` files and visualize the `Perfetto` trace:

```
mpirun -np 1 omnitrace-run -- ./Jacobi_hip.inst -g 1 1
ls omnitrace-Jacobi_hip.inst-output/<TIMESTAMP>/* | grep sampling
```

### Profiling multiple MPI processes

Run the instrumented binary with multiple MPI ranks. Note separate output files for each rank, including `perfetto-trace-*.proto` and `wall_clock-*.txt` files.

```
mpirun -np 2 omnitrace-run -- ./Jacobi_hip.inst -g 2 1
```

Inspect output text files. Then visualize `perfetto-trace-*.proto` files in `Perfetto`.

## Next steps

Try to use Omnitrace to profile [GhostExchange examples](https://github.com/amd/HPCTrainingExamples/tree/main/MPI-examples/GhostExchange).

**Finally, try to profile your own application!**
