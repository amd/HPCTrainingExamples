# Novice shallow-water profiling on AAC6

We work through the [novice stages](novice/README.md) on an MI300A, from `0_baseline` to
`5_vectorized_loads`. Each stage README says which measurement to take. This page is
the AAC6 setup and the commands we run there.

## Setup

On the login node:

```bash
cd Profiling-by-example/shallow-water
cp env_aac6.sh env.sh
```

We set `SLURM_PARTITION` in `env.sh` to our partition. Then:

```bash
./setup_rocprof_compute_venv.sh
```

`env.sh` loads `rocm/10.2.0a20260921` and activates `~/rocprof-compute-venv`.
`rocprof-compute analyze` uses it. Each batch script sources `env.sh` when the job
starts.

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

The log is `profile_<jobid>.out`. We repeat both submissions in each later stage.
