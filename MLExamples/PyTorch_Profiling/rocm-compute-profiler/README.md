# ROCm Compute Profiler (formerly Omniperf)

ROCm Compute Profiler (formerly Omniperf) is a tool for kernel level performance analysis, which can help analyze hardware performance of applications and kernels. ROCm Compute Profiler builds on top of the tools available in ROCProfiler, automating workload analysis.  During a profiling run, `rocprof-compute` (which is the name of the binary) will rerun the application multiple times to capture different hardware metrics – it’s typically optimal to focus the profiled workload to a smaller, representative run that will highlight bottlenecks while providing useful and actionable information for workload optimization.

Currently, rocprof-compute is available for single process workloads and scale out via slurm only.  Full integration into other job launchers is a work in progress.  Additionally,  rocprof-compute is being integrated with rocprofv3 to enable more features and better integration into the broader suite of ROCm tools.  Roofline capture is not yet enabled for MI300X and MI300A, but will be available in a future release.

For today’s workloads, the technique to capture profiles with rocprof-compute is to use the `rocprof-compute profiler -–name <name> -- script.sh`.  It is currently essential to use a wrapper script to encapsulate all arguments of the script into one argument rocprof-compute, though this will be relaxed in a future release.

## ROCm Compute Profiler Requirements

ROCm Compute Profiler has python requirements that need to be installed to perform a profiling run and analysis.  You can easily capture the right installation dependicies with `pip` using the `requirements.txt` file, found in the install location.  By default, it is here:

```
${ROCM_PATH}/libexec/rocprofiler-compute
```

## ROCm Compute Profiler Example

An example of a profiling run is shown below

```bash
rocprof-compute profile --name cifar_100_single_proc -- \
${PROFILER_TOP_DIR}/single_process/base_script.sh
```

In this case, the application name is `cifar_100_single_proc`, and the output is automatically stored in the folder `workloads/cifar_100_single_proc`.  You can analyze this output with the omniperf standalone CLI analyzer, or via a GUI interface as described in the documentation.

Via the command line analyzer, you can generate a high level view of the application’s performance with the “Speed of Light” summary configuration.  In general, the metrics captured can be overly comprehensive, making it challenging to select exactly which metrics are desired.   For an example application targeting a simple machine learning application, we can inspect the compute efficiency targeting blocks 2.1.2 to 2.1.5:

```
rocprof-compute analyze -p workloads/cifar_100_single_proc/MI300A_A1/ -b 2.1.2 2.1.3 2.1.4 2.1.5

2. System Speed-of-Light
2.1 Speed-of-Light
╒═════════════╤═══════════════════╤═════════╤════════╤════════════╤═══════════════╕
│ Metric_ID   │ Metric            │     Avg │ Unit   │       Peak │   Pct of Peak │
╞═════════════╪═══════════════════╪═════════╪════════╪════════════╪═══════════════╡
│ 2.1.2       │ MFMA FLOPs (BF16) │    0.00 │ Gflop  │ 1961164.80 │          0.00 │
├─────────────┼───────────────────┼─────────┼────────┼────────────┼───────────────┤
│ 2.1.3       │ MFMA FLOPs (F16)  │ 4303.42 │ Gflop  │ 1961164.80 │          0.22 │
├─────────────┼───────────────────┼─────────┼────────┼────────────┼───────────────┤
│ 2.1.4       │ MFMA FLOPs (F32)  │    0.00 │ Gflop  │  122572.80 │          0.00 │
├─────────────┼───────────────────┼─────────┼────────┼────────────┼───────────────┤
│ 2.1.5       │ MFMA FLOPs (F64)  │    0.00 │ Gflop  │  122572.80 │          0.00 │
╘═════════════╧═══════════════════╧═════════╧════════╧════════════╧═══════════════╛
```

As seen above, this workload in fp16 did in fact execute in fp16, but with incredibly poor compute efficiency.  We can dig deeper into this execution with block 10.1 (Compute Units Instruction Mix) and 11.1 (Compute Units – Compute Pipeline Speed of Light):

```
10. Compute Units - Instruction Mix
10.1 Overall Instruction Mix
╒═════════════╤══════════╤════════╤═══════╤══════════╤════════════════╕
│ Metric_ID   │ Metric   │    Avg │   Min │      Max │ Unit           │
╞═════════════╪══════════╪════════╪═══════╪══════════╪════════════════╡
│ 10.1.0      │ VALU     │ 180.31 │  2.03 │ 12516.00 │ Instr per wave │
├─────────────┼──────────┼────────┼───────┼──────────┼────────────────┤
│ 10.1.1      │ VMEM     │  17.68 │  0.01 │  1024.00 │ Instr per wave │
├─────────────┼──────────┼────────┼───────┼──────────┼────────────────┤
│ 10.1.2      │ LDS      │  28.19 │  0.00 │  1536.00 │ Instr per wave │
├─────────────┼──────────┼────────┼───────┼──────────┼────────────────┤
│ 10.1.3      │ MFMA     │   8.42 │  0.00 │  4096.00 │ Instr per wave │
├─────────────┼──────────┼────────┼───────┼──────────┼────────────────┤
│ 10.1.4      │ SALU     │  77.24 │  3.50 │  3158.50 │ Instr per wave │
├─────────────┼──────────┼────────┼───────┼──────────┼────────────────┤
│ 10.1.5      │ SMEM     │   3.58 │  1.00 │   145.72 │ Instr per wave │
├─────────────┼──────────┼────────┼───────┼──────────┼────────────────┤
│ 10.1.6      │ Branch   │  11.79 │  1.00 │   583.88 │ Instr per wave │
╘═════════════╧══════════╧════════╧═══════╧══════════╧════════════════╛


╒═════════════╤═══════════════════╤═════════╤════════╤════════════╤═══════════════╕
│ Metric_ID   │ Metric            │     Avg │ Unit   │       Peak │   Pct of Peak │
╞═════════════╪═══════════════════╪═════════╪════════╪════════════╪═══════════════╡
│ 11.1.0      │ VALU FLOPs        │  145.35 │ Gflop  │   61286.40 │          0.24 │
├─────────────┼───────────────────┼─────────┼────────┼────────────┼───────────────┤
│ 11.1.1      │ VALU IOPs         │  563.01 │ Giop   │   61286.40 │          0.92 │
├─────────────┼───────────────────┼─────────┼────────┼────────────┼───────────────┤
│ 11.1.2      │ MFMA FLOPs (BF16) │    0.00 │ Gflop  │ 1961164.80 │          0.00 │
├─────────────┼───────────────────┼─────────┼────────┼────────────┼───────────────┤
│ 11.1.3      │ MFMA FLOPs (F16)  │ 4303.42 │ Gflop  │ 1961164.80 │          0.22 │
├─────────────┼───────────────────┼─────────┼────────┼────────────┼───────────────┤
│ 11.1.4      │ MFMA FLOPs (F32)  │    0.00 │ Gflop  │  122572.80 │          0.00 │
├─────────────┼───────────────────┼─────────┼────────┼────────────┼───────────────┤
│ 11.1.5      │ MFMA FLOPs (F64)  │    0.00 │ Gflop  │  122572.80 │          0.00 │
├─────────────┼───────────────────┼─────────┼────────┼────────────┼───────────────┤
│ 11.1.6      │ MFMA IOPs (INT8)  │    0.00 │ Giop   │ 3922329.60 │          0.00 │
╘═════════════╧═══════════════════╧═════════╧════════╧════════════╧═══════════════╛
```

As is already known about this workload, it has simply not enough input data and compute requirements to keep the GPU pipelines fed.  
