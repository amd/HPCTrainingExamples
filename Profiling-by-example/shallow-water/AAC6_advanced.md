# Advanced shallow-water profiling on AAC6

We work through the [advanced stages](advanced/README.md) on one MI300A node, from
`0_baseline` to `6_2d_decomposition`. Each stage README says which measurement to take.
This page is the AAC6 setup and the commands we run there.

## Setup

On the login node:

```bash
cd Profiling-by-example/shallow-water
cp env_aac6.sh env.sh
```

We set `SLURM_PARTITION` in `env.sh` to our partition. The template binds one rank per
NUMA domain, which is the SPX layout. On a single-NUMA node we switch `GPU_BIND` to
`../gpu_bind_cpx.sh` and `MPI_BIND` to `--map-by slot`.

```bash
./setup_rocprof_compute_venv.sh
source env.sh
salloc -N 1 -p "$SLURM_PARTITION" --exclusive --gres=gpu:4 -t 02:00:00
```

`env.sh` loads `rocm/10.2.0a20260921` and Open MPI, and activates
`~/rocprof-compute-venv`. The allocation keeps that environment. The job is exclusive
so each rank gets CPU cores next to its GPU.

## Commands

```bash
cd advanced/0_baseline
make
for n in 1 2 4; do
    mpirun -n $n --map-by ppr:1:numa --bind-to numa ../gpu_bind.sh ./shallow_mpi
done
```

On an MI300A node configured in CPX mode, we use the CPX binding script:

```bash
mpirun -n $n --map-by slot ../gpu_bind_cpx.sh ./shallow_mpi
```

The stage README gives the `rocprofv3`, `rocprof-compute`, and `rocprof-sys` commands
for that step. We run them with the same `mpirun` line. Each later stage starts with
`make`, then that launch.
