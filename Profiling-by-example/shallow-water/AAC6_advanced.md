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
```

`env.sh` loads `rocm/10.2.0a20260921` and Open MPI, and activates
`~/rocprof-compute-venv`. Each batch script sources `env.sh` when the job starts.

For NIC counter profiling in stages 5 and 6, we set `ROCPROFSYS_NETWORK_INTERFACE`
in `env.sh` to the node's HPC interface name. We find that name on a compute node:

```bash
rocprof-sys-avail -H -r net
```

The stage 6 README has the collection recipe. The NIC runs in `profile.sh` stay
commented out until a two-node job is available.

## Running a stage

Each stage directory has `fom.sh` and `profile.sh`. They hold the Slurm request and
the commands for that stage. We submit them from the stage directory. `submit.sh`
reads the partition from `env.sh`:

```bash
cd advanced/0_baseline
../../submit.sh fom.sh
../../submit.sh profile.sh
```

`fom.sh` asks for one exclusive node with four GPUs and allows two hours. The
exclusive node gives each rank CPU cores next to its GPU. The script builds, then
runs at 1, 2, and 4 ranks:

```bash
make
for n in 1 2 4; do
    mpirun -n $n --map-by ppr:1:numa --bind-to numa ../gpu_bind.sh ./shallow_mpi
done
```

On an MI300A node configured in CPX mode, that launch uses the CPX binding script:

```bash
mpirun -n $n --map-by slot ../gpu_bind_cpx.sh ./shallow_mpi
```

The log is `fom_<jobid>.out`.

`profile.sh` allows two hours on the same exclusive node. It runs the `rocprofv3`,
`rocprof-compute`, and `rocprof-sys` commands from that stage's README, with the same
`mpirun` binding. The log is `profile_<jobid>.out`. We repeat both submissions in
each later stage.
