# Ghost Exchange: Original Implementation

This example shows a CPU-only implementation, and how to use Omnitrace to trace it.

## Environment: Frontier

We have developed and tested these examples for Frontier, using these modules:

```
module load cce/17.0.0
module load rocm/5.7.0
module load omnitrace/1.11.2
module load craype-accel-amd-gfx90a cmake/3.23.2
```

## Building and Running

To build and run this initial implementation do the following:

```
cd Orig
mkdir build; cd build;
cmake ..
make -j8
srun -N1 -n4 -c7 --gpu-bind=closest -A <account> -t 05:00 ./GhostExchange -x 2 -y 2 -i 20000 -j 20000 -h 2 -t -c -I 100
```

## Instrumenting with Binary-rewrite

Before instrumenting and running with Omnitrace, we need to make sure our default configuration file is generated with:

```
omnitrace-avail -G ~/.omnitrace.cfg
export OMNITRACE_CONFIG_FILE=~/.omnitrace.cfg
```

Note that `~/.omnitrace.cfg` is the default place Omnitrace will look for a configuration file, but 
you can point it to a different configuration file using the environment variable `OMNITRACE_CONFIG_FILE`.

It is recommended to use `omnitrace-instrument` to output an instrumented binary since our application uses MPI. This way, tracing output will appear in separate files by default. We can instrument and run with these commands:

```
omnitrace-instrument -o ./GhostExchange.inst -- ./GhostExchange
srun -N1 -n4 -c7 --gpu-bind=closest -A <account> -t 05:00 omnitrace-run -- ./GhostExchange.inst -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100
```

Note: it is necessary to run with `omnitrace-run` when running an instrumented binary.

## Initial Trace

Below is a screenshot of a trace obtained for this example:
<p><img src="orig_0.png"/></p>
<p><img src="orig_1.png"/></p>

In this screenshot, we see Omnitrace is showing CPU frequency data for every core.
To have Omnitrace only show CPU frequency for a single CPU core, add this to `~/.omnitrace.cfg`:

```
OMNITRACE_SAMPLING_CPUS                            = 0
```

and re-run the command from before, no need to re-instrument:

```
srun -N1 -n4 -c7 --gpu-bind=closest -A <account> -t 05:00 omnitrace-run -- ./GhostExchange.inst -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100
```

Now we see that only one instance of CPU frequency is reported:

<p><img src="orig_3_sample_1cpu.png"/></p>

Zooming in, we see instrumented MPI activity:

<p><img src="orig_2_zoom_in.png"/></p>

We can alter the Omnitrace configuration to see MPI overheads measured numerically. Add this to `~/.omnitrace.cfg`:

```
OMNITRACE_PROFILE                                  = true
```

Then re-running our same instrumented binary gives us a few new output files, look for `wall_clock-0.txt`:

<p><img src="profile.png"/></p>

Here, we see a hierarchical view of overheads, to flatten the profile to see total count and mean duration for each MPI call, add this to `~/.omnitrace.cfg`:

```
OMNITRACE_FLAT_PROFILE                             = true
```

Re-running `omnitrace-run` with our intrumented binary will now produce a `wall_clock-0.txt` file that looks like this:

<p><img src="flat_profile.png"/></p>

We can see the number of times each MPI function was called, and the time associated with each.
