Note: The following was tested with a therock 7.15 pre-release nightly build from 1st July 2026 on MI300A.

# Profiling the ShallowWater HIPStdPar Example with `rocprof-compute`

This exercise demonstrates how to use `rocprof-compute` (ROCm Compute Profiler) to profile a C++ standard parallelism (stdpar) application targeting AMD GPUs.This exercise uses the [ShallowWater_StdPar](https://github.com/amd/HPCTrainingExamples/tree/main/HIPStdPar/CXX/ShallowWater_StdPar) example, a 2D shallow water equation solver that uses `std::for_each` and `std::transform_reduce` with `std::execution::par_unseq` to offload computation to the GPU.

The exercise progressively introduces:
1. Roofline-only profiling
2. Analyzing the roofline and selecting the correct data type
3. Iteration multiplexing to reduce the number of application runs
4. Comparing profiles
5. Full hardware counter profiling


This exercise runs on a single MI300A GPU on a compute node. Allocate a node before proceeding:

# Setup
```
salloc -N 1 --gpus=1 -p <your_partition>
```

Load the ROCm module. The exact module name depends on your system:

```
module load therock/therock-dist-linux-gfx94X-dcgpu-7.15.0a20260701
```

Install the Python dependencies required by rocprof-compute:

```
python3 -m pip install --user -r $(dirname $(which rocprof-compute))/../libexec/rocprofiler-compute/requirements.txt
```
Note: If you have the dependencies installed once, you can skip this step.

Make the installed packages available to the Python interpreter (neccessary every time you have a fresh environment):

```
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH
```

Set the compiler and enable unified memory, which is required for HIPStdPar managed memory on MI300A:

```
export CXX=amdclang++
export HSA_XNACK=1
```

Verify that rocprof-compute is available:

```
rocprof-compute --version
```

## Build and run the application

Navigate to the ShallowWater stdpar example and build it:

```
cd HPCTrainingExamples/HIPStdPar/CXX/ShallowWater_StdPar
make clean && make
```

Run the application to verify it works correctly:

```
./ShallowWater
```
If you have a node with more than one numa domain make sure to run with affinity set through e.g. `ROCR_VISIBLE_DEVICES=0 numactl -C 0 -m 0` or using the --cpu-bind and --gpu-bind options when running with mpirun or srun (may heavily depend on your system what is supported).
In the following affinity for aac6 is assumed but it can vary for different systems! It is important for reproducible performance to get the affinity right.

The output should show iterations progressing with conservation of mass:

```
Iteration:00000, Time:0.000000, Timestep:0.048059 Total mass:399200.000000
Iteration:00100, Time:0.032798, Timestep:0.032798 Total mass:399200.000000
Iteration:00200, Time:0.060447, Timestep:0.027649 Total mass:399200.000000
...
Iteration:01900, Time:0.419123, Timestep:0.019992 Total mass:399200.000000
Iteration:02000, Time:0.441816, Timestep:0.022692 Total mass:399200.000000
 Flow finished in 0.169651 seconds

```

## Step 1: Roofline-only profiling

The `--roof-only` flag collects the achievable memory bandwidth and floating-point operation rates of the kernels on the GPU, then plots where the application's kernels fall relative to the GPUs hardware limits plotted as FLOPs vs. arithmetic intensity. It skips the full multi-pass hardware counter collection, which makes it significantly faster (3 passes of the whole app execution instead of 13).

Pin the process to GPU 0 and NUMA node 0 to get consistent measurements:

```
ROCR_VISIBLE_DEVICES=0 numactl -C 0 -m 0 rocprof-compute profile --output-directory Profile-ShallowWater --roof-only -- ./ShallowWater
```

The `--output-directory` argument supports several substitution keywords that can be useful for automated workflows and MPI jobs:

| Keyword | Substitution |
|---|---|
| `%hostname%` | Host name |
| `%gpumodel%` | GPU model name |
| `%rank%` | MPI process rank |
| `%env{NAME}%` | Value of environment variable `NAME` |

The default output path is `<cwd>/workloads/<name>/%gpumodel%`. The name has to be specified with `-n <name>` if there is no `--output-directory` specified.

## Step 2: Analyze the roofline

After profiling, inspect the results:

```
rocprof-compute analyze -p Profile-ShallowWater/0
```

The roofline is displayed both as ASCII art in the terminal and as an HTML file in the output directory. Open the HTML for a more readable interactive chart.

**Important:** The default roofline shows FP32 performance ceilings. ShallowWater uses FP64 arithmetic throughout, so the default view gives a misleading picture of the kernels' efficiency. Select the correct data type explicitly:

```
rocprof-compute analyze --roofline-data-type FP64 -p Profile-ShallowWater/0
```

The FP64 roofline places the kernels relative to the correct hardware limits.


## Step 3: Iteration multiplexing

By default, rocprof-compute runs the application multiple times (one pass per hardware counter group) to collect all metrics. For an application like ShallowWater this is fast, but for longer-running codes this can be expensive.

The `--iteration-multiplexing` flag interleaves counter collection across a single application run by switching the active counter group on each kernel invocation, instead of re-running the application for each group. This saves running the application the full number of passes:

```
ROCR_VISIBLE_DEVICES=0 numactl -C 0 -m 0 rocprof-compute profile --iteration-multiplexing --output-directory Profile-ShallowWater-multiplexing --roof-only -- ./ShallowWater
```

Analyze the multiplexed profile, again selecting FP64 for the correct view:

```
rocprof-compute analyze --roofline-data-type FP64 -p Profile-ShallowWater-multiplexing/0
```

## Step 4: Comparing profiles

rocprof-compute can compare two profiles side by side. The first `-p` argument is the baseline; the second is the comparison. This is useful for checking whether multiplexing introduces any differences in the reported kernel statistics (block 0):

```
rocprof-compute analyze -p Profile-ShallowWater-multiplexing/0 -p Profile-ShallowWater/0
```

The output table shows the baseline value, the comparison value, and the percentage difference for each metric. For example we can see any discrepancies between the multiplexed and non-multiplexed runs beyond expected statistical effects. This will confirm if the interleaving counter collection across iterations is representative. Note also that kernels which are called not often enough are automatically dropped if iteration multiplexing is active. This can be useful to avoid looking at initialization kernels which may be expensive in a short profiling run but otherwise not.

## Step 5: Full hardware counter profiling

To collect the complete set of hardware performance counters (not just roofline data), remove `--roof-only`. Add `--no-roof` if you specifically do not want roofline data:

```
ROCR_VISIBLE_DEVICES=0 numactl -C 0 -m 0 rocprof-compute profile --iteration-multiplexing -n Profile-ShallowWater-all -- ./ShallowWater
```

Analyze the full profile:

```
rocprof-compute analyze -p workloads/Profile-ShallowWater-all/0/
```

The full profile includes memory traffic, occupancy, instruction mix, cache utilization, and many other metrics organized into numbered blocks in the analyze output.

Usually you are interested in a specific kernel. Select the kernel with `-k` or dispatch `-d` number from the stats in block 0 in the analysis. Both during collection as well as analysis one can filter for a kernel name where also a partial name is sufficient. This can be useful if an interesting kernel name was identified with rocprofv3 (compare linked example below). 


## Notes and known issues

- `-n` vs `--output-directory` in therock 7.15 early nightlies:** In the 1st July 2026 therock 7.15 pre-release build, using `--output-directory` for full profiling (non-roof-only) causes a path mismatch where the `results.db` file cannot be found during the analyze step. Use `-n <name>` instead, which places results under `workloads/<name>/%gpumodel%/` and is correctly resolved by the analyzer.

- rocprof-compute does not support kernel-rename and/or pause and resume roctx markers. Names of kernels have to be identified using other tools (see other exercises using the same example).

- `ROCPROF=rocprofv3` fallback: If the native rocprofiler-sdk backend fails on your system for whatever reason, you can currently (therock 7.14 and 7.15 pre-release build) try to fall back to the rocprofv3 backend by:`export ROCPROF=rocprofv3`. This allows rocprof-compute to proceed using rocprofv3 instead of the native tool. However, the fallback backend does **not** support `--iteration-multiplexing`, so the application will be run once per counter group.
- Affinity is important when you run rocprof-compute with MPI. Make sure you leave some threads on the CPU for rocprof helper threads. Recommendation: pin to at least 4 more cores than you would need otherwise when running the app without the profiler.
- When running with MPI wrap mpirun or srun command including stting the affinity as you would run the whole app around rocprof-compute.
- When running with MPI make sure to include %rank% in the naming of folders, otherwise all processes may write to the same folder leading to hangs or crashes or corrupt data.

## References

- [rocprof-compute documentation](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/)
- [Rocprofv3 HIPStdPar example](https://github.com/amd/HPCTrainingExamples/tree/main/Rocprofv3/HIPStdPar)
- [rocprofiler-systems ShallowWater HIPStdPar example](https://github.com/amd/HPCTrainingExamples/tree/main/rocprofiler-systems/ShallowWater/HIPStdPar)
- [ShallowWater_StdPar source](https://github.com/amd/HPCTrainingExamples/tree/main/HIPStdPar/CXX/ShallowWater_StdPar)
