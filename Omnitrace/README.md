\pagebreak{}

# Omnitrace

***NOTE***: extensive documentation on how to use `omnitrace` for the `GhostExchange_Array` example is now available as `README.md` files in the exercises repo. While the testing has been done on Frontier in that documentation, most of the `omnitrace` tools apply in the same way, hence it could provide additional training matieral.

Here, we show how to use `omnitrace` tools considering the example in `HPCTrainingExamples/HIP/jacobi`.

## Initial Setup

Setup environment:

```bash
module purge
module load omnitrace gcc/13
```

Next, create a configuration file for `omnitrace`: 

```bash
omnitrace-avail -G ~/omnitrace.cfg
```

If you do not provide a path to the config file, it will generate one in the current directory:  `./omnitrace-config.cfg`. This config file contains several flags that can be modified to turn on or off several options that impact the visualization of the traces in Perfetto. You can see what flags can be included in the config file by doing:

```bash
omnitrace-avail --categories omnitrace
```

To add brief descriptions, use the `-bd` option:

```bash
omnitrace-avail -bd --categories omnitrace
```

Note that the list of flags displayed by the commands above may not include all actual flags that can be set in the config.

You can also create a configuration file with description per option. Beware, this is quite verbose:

```bash
omnitrace-avail -G ~/omnitrace_all.cfg --all
```

Next you have to declare that you want to use this configuration file. Note, this is only necessary if you had provided a custom path and/or filename for the config file when you created it.

```bash
export OMNITRACE_CONFIG_FILE=~/omnitrace.cfg
```

## Setup Jacobi Example

Go to the Jacobi code in the examples repo:

```bash
cd ~/HPCTrainingExamples/HIP/jacobi
```

Compile the code:

```bash
make
```

Execute the binary to make sure it runs successfully:
<! --Note: To get rid of `Read -1, expected 4136, errno = 1` add `--mca pml ucx --mca pml_ucx_tls ib,sm,tcp,self,cuda,rocm` to the `mpirun` command line -->

```bash
mpirun -np 1 ./Jacobi_hip -g 1 1
```

## Runtime Instrumentation

Run the code with `omnitrace-instrument` to perform runtime instrumentation: this will produce a series of directories whose name is define by the time they were crated. In one of these directories, you can find the `wall_clock-<proc_ID>.txt` file, which includes information on the function calls made in the code, such as how many times these calls have been called (`COUNT`) and the time in seconds they took in total (`SUM`):

```bash
mpirun -np 1 omnitrace-instrument -- ./Jacobi_hip -g 1 1
```

The above command produces a folder called `instrumentation` that contains the  `available.txt` file, which shows all the functions that can be instrumented. To instrument a specific function, include the `--function-include <fnc>` option in the `omnitrace-instrument` command, for example:

```bash
mpirun -np 1 omnitrace-instrument -v 1 -I 'Jacobi_t::Run' 'JacobiIteration' -- ./Jacobi_hip -g 1 1
```

The output provided by the above command will show that only those functions have bene instrumented:

```bash
[...]
[omnitrace][exe]    1 instrumented funcs in JacobiIteration.hip
[omnitrace][exe]    1 instrumented funcs in JacobiRun.hip
[omnitrace][exe]    1 instrumented funcs in Jacobi_hip
[omnitrace][exe]    2 instrumented funcs in librocprofiler-register.so.0.3.0
[...]
```

Alternatively, you can use the `--print-available functions` option as shown below. The `--simulate` option will exit after outputting the diagnostics, the `- v` option is for verbose output: 

(NOTE: the output of the next command may be lengthy, you may want to pipe it to a file using >> out.txt at the end of the line to make searching it easier afterwards.)

```bash
mpirun -np 1 omnitrace-instrument -v 1 --simulate --print-available functions -- ./Jacobi_hip -g 1 1
```

## Binary Rewrite

You can create an instrumented binary using `omnitrace-instrument`: notice that this doesn't take very long to run:

```bash
omnitrace-instrument -o ./Jacobi_hip.inst -- ./Jacobi_hip
```

Execute the new instrumented binary using the `omnitrace-run` command inside `mpirun`. This is the recommended way to profile MPI applications as `omnitrace` will **separate the output files for each rank**:

```bash
mpirun -np 1 omnitrace-run -- ./Jacobi_hip.inst -g 1 1
```

To see the list of the instrumented GPU calls, make sure to turn on the `OMNITRACE_PROFILE` flag in your config file:

```bash
OMNITRACE_PROFILE                                  = true
```

Running the instrumented binary again, you can see that it generated a few extra files. One of those has a list of instrumented GPU calls and durations of those calls:

```bash
cat omnitrace-Jacobi_hip.inst-output/<TIMESTAMP>/roctracer-0.txt
```

## Debugging omnitrace-run

If you get errors when you run an instrumented binary or when you run with runtime instrumentation, add the following options `--monochrome -v 2 --debug` and try: this would give you additional debug information to assist you in figuring out where the problem may lie:

```
mpirun -np 1 omnitrace-run --monochrome -v 1 --debug -- ./Jacobi_hip.inst -g 1 1
``` 

## Visualization

Copy the `perfetto-trace-0.proto` to your local machine, and using the Chrome browser open the web page [https://ui.perfetto.dev/](https://ui.perfetto.dev/):

```bash
scp -i <path/to/ssh/key> -P <port_number> <username>@aac1.amd.com:~/<path/to/proto/file> .
```

Click `Open trace file` and select the `.proto` file. Below, you can see an example of how a `.proto` file would be visualized on Perfetto:

![image](https://user-images.githubusercontent.com/109979778/225769857-900aa6dd-1c7a-440f-82ab-872dcc09d73c.png)

## Hardware Counters

To see a list of all the counters for all the devices on the node, do:

```bash
omnitrace-avail --all
```

Declare in your configuration file:

```bash
OMNITRACE_ROCM_EVENTS = VALUUtilization,FetchSize
```

Check again:

```bash
grep OMNITRACE_ROCM_EVENTS $OMNITRACE_CONFIG_FILE
```

Run the instrumented binary, and you will observe an output file for each hardware counter specified. You should also see a row for each hardware counter in the perfetto trace generated by Omnitrace.

Note that you do not have to instrument again after making changes to the config file. Just running the instrumented binary picks up the changes you make in the config file. Ensure that the 
`OMNITRACE_CONFIG_FILE` environment variable is pointing to your config file.

```bash
mpirun -np 1 omnitrace-run -- ./Jacobi_hip.inst -g 1 1
```

The output should show something like this:

```bash
...]> Outputting 'omnitrace-Jacobi_hip.inst-output/<TIMESTAMP>/rocprof-device-0-VALUUtilization-0.json'
...]> Outputting 'omnitrace-Jacobi_hip.inst-output/<TIMESTAMP>/rocprof-device-0-VALUUtilization-0.txt'
...]> Outputting 'omnitrace-Jacobi_hip.inst-output/<TIMESTAMP>/rocprof-device-0-FetchSize-0.json'
...]> Outputting 'omnitrace-Jacobi_hip.inst-output/<TIMESTAMP>/rocprof-device-0-FetchSize-0.txt'
```

If you do not want to see the details for every CPU core, modify the config file to select only what you want to see, say CPU cores 0-2 only:

```bash
OMNITRACE_SAMPLING_CPUS                            = 0-2
```

Now running the instrumented binary again will show significantly fewer CPU lines in the profile:

```bash
mpirun -np 1 omnitrace-run -- ./Jacobi_hip.inst -g 1 1
```

## Profiling Multiple Ranks

Run the instrumented binary with multiple ranks. You'll find multiple `perfetto-trace-*.proto` files, one for each rank (note that depending on your system it may be necessary to do a `salloc` prior to the command below to ensure enough resources ara available):

```bash
mpirun -np 2 omnitrace-run -- ./Jacobi_hip.inst -g 2 1
```

You can visualize them separately in `Perfetto`, or combine them using `cat` and visualize them in the same `Perfetto` window (trace concatenation is not available in all `omnitrace` versions):

```bash
cat perfetto-trace-0.proto perfetto-trace-1.proto > allprocesses.proto
```

## Sampling

Set the following in your configuration file:

```bash
OMNITRACE_USE_SAMPLING = true
OMNITRACE_SAMPLING_FREQ = 100
```

Execute the instrumented binary and visualize the perfetto trace:

```bash
mpirun -np 1 omnitrace-run -- ./Jacobi_hip.inst -g 1 1
```

Scroll down to the very bottom to see the sampling output. Those traces will be annotated with a `(S)` as well.

## Kernel Timings

Open the `wall_clock-0.txt` file:

```bash
cat omnitrace-Jacobi_hip.inst-output/<TIMESTAMP>/wall_clock-0.txt
```

In order to see the kernel durations aggregated in your configuration file, make sure to set in your config file or in the environment:

```bash
OMNITRACE_PROFILE = true
OMNITRACE_FLAT_PROFILE = true
```

Execute the code and check the `wall_clock-0.txt` file again. Instead of updating the config file, you can also set the environment variables to achieve the same effect.

```bash
OMNITRACE_PROFILE=true OMNITRACE_FLAT_PROFILE=true mpirun -np 1 omnitrace-run -- ./Jacobi_hip.inst -g 1 1
```
