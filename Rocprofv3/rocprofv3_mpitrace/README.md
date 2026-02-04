# Quick Start Guide: Using mpitrace with rocprofv3

This guide demonstrates how to use [mpitrace](https://github.com/sfantao/mpitrace-install) with `rocprofv3` to profile MPI communication alongside GPU kernel execution.

## Installation

### Prerequisites

Load required modules:
```bash
module load rocm gcc openmpi
```

### Build mpitrace

```bash
git clone git@github.com:sfantao/mpitrace-install.git
cd mpitrace-install
./create-for-env-gcc-and-mpi
```

This downloads and installs mpitrace and its dependencies. Important libraries:
- `mpitrace/src/libmpitrace.so`
- `mpitrace/roctx/libmpitrace-legacy.so` (for legacy `rocprof` v1)
- `mpitrace/roctx/libmpitrace.so` (for `rocprofv3` - **default**)

## Helper scripts

Create the following helper scripts in your working directory:

### `helper_mpitrace.sh`

```bash
#!/bin/bash
export LD_PRELOAD=$HOME/software/mpitrace-install/mpitrace/roctx/libmpitrace.so
export TRACE_ALL_EVENTS=yes
export TRACE_ALL_TASKS=yes
export TRACE_BUFFER_SIZE=48000000
$*
```

### `helper_rocprofv3.sh`

```bash
#!/bin/bash
if [[ -n ${OMPI_COMM_WORLD_RANK+z} ]]; then
  # mpich
  export MPI_RANK=${OMPI_COMM_WORLD_RANK}
elif [[ -n ${MV2_COMM_WORLD_RANK+z} ]]; then
  # ompi
  export MPI_RANK=${MV2_COMM_WORLD_RANK}
elif [[ -n ${SLURM_PROCID+z} ]]; then
    # mpich via srun
    export MPI_RANK=${SLURM_PROCID}
fi
pid="$$"
outdir="rank.${MPI_RANK}"
outfile=${outdir}
eval "rocprofv3 -d ${outdir} -o ${outfile} $*"
```

Make them executable:
```bash
chmod +x helper_mpitrace.sh helper_rocprofv3.sh
```

## Example 1: Jacobi MPI application

### Build the example

```bash
module load rocm openmpi
git clone git@github.com:amd/HPCTrainingExamples.git
cd HPCTrainingExamples/HIP/jacobi

make clean
make
```

### Profile with mpitrace + rocprofv3

```bash
NUM_GPUS=2 RANK_STRIDE=16 mpirun -np 2 --bind-to none \
  ./helper_mpitrace.sh ./helper_rocprofv3.sh \
  --runtime-trace --output-format pftrace -- \
  ./Jacobi_hip -g 2 1

cat rank*/*pftrace > ranks_all.pftrace
```

### View results

Open `ranks_all.pftrace` in [Perfetto UI](https://ui.perfetto.dev/). The timeline will show:
- **MPI calls** from mpitrace: `MPI_Isend`, `MPI_Irecv`, `MPI_Waitall`
- **User-defined rocTX markers**: `MPI Exchange::Halo Exchange`, `Halo H2D::Halo Exchange`
- **GPU kernels**: `LocalLaplacianKernel()`
- **Overlap** between computation and communication

![Jacobi trace showing MPI calls and GPU kernels](attachments/image_35377.png)

### Additional output files

- `mpi_profile.<pid>.<rank>` - Summary files (rank 0 has main summary)
- `<pid>.trc` - Trace files viewable with `tracerview` app (requires additional build step)

## Key points

1. **mpitrace library**: Use `roctx/libmpitrace.so` for `rocprofv3` (default)
2. **Legacy support**: Use `roctx/libmpitrace-legacy.so` for legacy `rocprof` v1
3. **Trace merging**: Combine per-rank traces with `cat rank*/*pftrace > ranks_all.pftrace`
4. **Visualization**: View merged traces in [Perfetto UI](https://ui.perfetto.dev/)

## References

- [mpitrace repository](https://github.com/sfantao/mpitrace-install)
