ROCm Systems Profiler (Formerly Omnitrace)

ROCm Systems Profiler is an application tracing utility, as the name suggests, and can be used with application instrumentation or via application sampling.  In most AI use cases, instrumentation is impractical due to the many-component python ecosystem.  This guide will focus on the utility of application sampling with omnitrace in python applications.

ROCm Systems Profiler sampling has a wide range of hardware counters available for inspection.  The package includes an easy to use utility called `rocprof-sys-avail` which can list available counters as well as filter for specific categories.  

ROCm Systems Profiler can be configured to target specific hardware counters via configuration file, which can be passed on the command line via -c config.cfg syntax to the `rocprof-sys-sample` executable.  The full list of hardware counters available is extensive, however, a default configuration can be obtained with the `rocprof-sys-avail` tool.  A default configuration can be generated and inspected with `rocprof-sys-avail -G <name>.cfg -d`.  It is recommended to configure via config file and not via environment variables.  Especially if the primary source of consumption for the outputs will be via interactive viewers (such as Perfetto) the configuration file will automatically consolidate the traces into useful .proto files.

For machine learning applications, where sampling is the key tool for obtaining performance metrics, it is advantegous to disable the rocprofiler in the generated config.  The syntax to search for is `ROCPROFSYS_USE_ROCPROFILER=false`.

To use ROCm Systems Profiler to profile a workload, once the configuration file is generated the application can be launched.  Once the trace is collected, you can view it in the perfetto tool at [https://ui.perfetto.dev/](ui.perfetto.dev).  Note that due to a recent bug, it may be necessary to view traces at [https://ui.perfetto.dev/v46.0-35b3d9845/](https://ui.perfetto.dev/v46.0-35b3d9845/)

Launching a single-process profiling job can be done as seen in `rocm-systems-profiler.sh`:

```bash 
rocprof-sys-sample -c $RSP_CFG \
-I  kokkosp mpip ompt rocm-smi rocprofiler roctracer roctx rw-locks spin-locks -- \
python3 ${PROFILER_TOP_DIR}/train_cifar_100.py --batch-size 256 --max-steps 20 \
--data-path ${PROFILER_TOP_DIR}/data
```

Here, the sampling uses many external tools and will produce a fine grained view of compute parameters, as seen below:

![ROCm System trace](omnitrace-single-process.png)

The trace above lets you zoom in and select kernels of interest - in this small model, no kernel stands out as particularly troublesome except that they are all very very short running kernels.  In this case, it's due to the very small size of the data and just not enough work to keep the GPU busy:

![ROCm System trace Zoomed](omnitrace-singleProcess2.png)

ROCm Systems Profiler is compatible with Slurm and will generate one profile per process.  The usual way to combine slurm, omnitrace, and a python executable is the following:

```bash
srun --ntasks 4 \
rocprof-sys-sample -c $RSP_CFG \
-I  kokkosp mpip ompt rocm-smi rocprofiler roctracer roctx rw-locks spin-locks -- \
python3 ${PROFILER_TOP_DIR}/train_cifar_100.py --batch-size 256 --max-steps 20 \
--data-path ${PROFILER_TOP_DIR}/data
```

It's easiest to analyze just one trace from a distributed run, again in perfetto.  Here, we can clearly see RCCL kernels launching concurrent operations with compute kernels:

![ROCm System trace - RCCL](omnitrace-MultiProcess.png)

Zooming in on the RCCL kernels shows they are very long lived compared to the compute kernels:

![ROCm System trace - RCCL Zoom](omnitrace-MultiProcess-Zoom.png)

This is likely not causing bottlenecks: this profiling introduces significant overhead and artificially inflates the duration of these kernels.

ROCm Systems Profiler also  offers the ability to sample the call stack of the application.  This can provide additional information regarding CPU and GPU usage statistics, concurrent with your application kernels.  To run in this mode for python applications, launch it with the following arguments:

```bash
rocprof-sys-sample -c $RSP_CFG \
-I roctracer -- \
python3 ${PROFILER_TOP_DIR}/train_cifar_100.py --batch-size 256 --max-steps 20 \
--data-path ${PROFILER_TOP_DIR}/data
```

You can see, now, the C++ calls from both python and pytorch leading to kernels on the GPU:

![ROCm System trace - call stack](omnitrace-stack-singleProcess.png)

And zooming in on these, you can see exactly which low level rocBLAS kernels are launched in each higher level function:
![ROCm System trace - call stack](omnitrace-stack-singleProcessZoom.png)
