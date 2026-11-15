# Novice shallow-water profiling on AAC7

We work through the [novice stages](novice/README.md) on an MI300A, from `0_baseline` to
`5_vectorized_loads`. Each stage README says which measurement to take. This page is
the AAC7 setup and the commands we run there.

## Setup

On the login node:

```bash
cd Profiling-by-example/shallow-water
cp env_aac7.sh env.sh
./setup_rocprof_compute_venv.sh
```

`env.sh` uses the SPX partition `192C4G1H_MI300A_RHEL9_A1`, loads `rocm-new/10.1.0` and
`cray-python/3.12.12`, and activates `~/rocprof-compute-venv`. Each batch script sources
`env.sh` when the job starts.

## Running a stage

Each stage directory has `fom.sh` and `profile.sh`. They hold the Slurm request and
the commands for that stage. We submit them from the stage directory. `submit.sh`
reads the partition from `env.sh`:

```bash
cd novice/0_baseline
../../submit.sh fom.sh
../../submit.sh profile.sh
```

`fom.sh` asks for one GPU and 30 minutes. It builds and runs:

```bash
make
./shallow
```

`./shallow` prints the throughput. The log is `fom_<jobid>.out`.

`profile.sh` asks for one GPU and two hours. It runs the `rocprofv3` commands from
that stage's README. It then collects and reports the roofline, using the stage
directory as the workload name:

```bash
rocprof-compute profile -n 0_baseline --roof-only --device 0 -k compute_rhs \
    --iteration-multiplexing -- ./shallow
rocprof-compute analyze -p workloads/0_baseline/0
```

The log is `profile_<jobid>.out`. The roofline plot is
`workloads/0_baseline/0/empirRoof_gpu-0_FP32.html`. We repeat both submissions in each
later stage, with that stage's directory name in place of `0_baseline`.

ROCm 10.1.0 rescales the roofline axes in each plot, so we compare stages by the
plotted values rather than by position on the page. ROCm 10.2.0 keeps the axes fixed
across plots, and we will move to it once it is public.
