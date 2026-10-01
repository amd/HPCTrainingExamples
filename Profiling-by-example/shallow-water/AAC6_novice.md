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
source env.sh
salloc -N 1 -p "$SLURM_PARTITION" --gpus=1 -t 02:00:00
```

`env.sh` loads `rocm/10.2.0a20260921` and activates `~/rocprof-compute-venv`. The
allocation keeps that environment. `rocprof-compute analyze` uses it.

## Commands

```bash
cd novice/0_baseline
make
./shallow
```

`./shallow` prints the throughput. The stage README gives the `rocprofv3` command for
that step. Every stage also collects a roofline. The workload name is the stage
directory:

```bash
rocprof-compute profile -n 0_baseline --roof-only --device 0 -k compute_rhs \
    --iteration-multiplexing -- ./shallow
rocprof-compute analyze -p workloads/0_baseline/0
```

Each later stage starts the same way: `make`, then `./shallow`. The `rocprofv3`
command for that stage is in its README.
