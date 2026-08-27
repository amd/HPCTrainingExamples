#!/bin/bash
#SBATCH --job-name=cg_rocsys
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --time=00:20:00
#SBATCH --output=rocsys_%j.out
#SBATCH --error=rocsys_%j.out
set -u
date; echo "Node(s): ${SLURM_JOB_NODELIST:-$(hostname)}"
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
module purge 2>/dev/null
module load "${ROCM_MODULE:-rocm/6.4.1}"
module load "${ROCSYS_MODULE:-rocprofiler-systems/6.4.1}" 2>/dev/null || module load rocprofiler-systems
module load "${OPENMPI_MODULE:-openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4}" 2>/dev/null || module load openmpi
chmod +x gpu_bind.sh 2>/dev/null

# Build with the rocprofiler-sdk roctx library (NOT classic -lroctx64): rocprof-sys
# is a rocprofiler-sdk tool and captures its ranges via the `marker_api` domain, so
# the roctx phases (cg_iteration, dot_allreduce, halo_wait_recv, spmv_on_proc, ...)
# become searchable in Perfetto exactly like the rocprofv3 --marker-trace figures.
echo "=== build ROCTX=1 (rocprofiler-sdk-roctx) ==="; make clean >/dev/null 2>&1
make ROCTX=1 ROCTX_LIB=-lrocprofiler-sdk-roctx 2>&1 | tail -2 || { echo BUILD FAILED; exit 1; }

OUT=rocsys_timeline; rm -rf "$OUT"; mkdir -p "$OUT"
SEED=12345
MAT=src/Dubcova2.pm

# CPU involvement knobs (see docs/profilers/rocprofiler-systems.md):
#   USE_SAMPLING=ON       -> host call-stack sampling (per-thread CPU rows)
#   USE_MPIP=true         -> MPI wrappers on the host thread (default ON): the
#                            MPI_Allreduce / MPI_Waitall / MPI_Isend blocks that the
#                            CPU spends the halo exchange + dot-allreduce in.
#   USE_PROCESS_SAMPLING=OFF drops the ~5 per-core counter rows so the GPU-kernel
#                            queues sit directly under the host thread (readable fig).
export ROCPROFSYS_USE_SAMPLING=ON
export ROCPROFSYS_SAMPLING_FREQ=1000
export ROCPROFSYS_SAMPLING_DELAY=0.0
export ROCPROFSYS_USE_PROCESS_SAMPLING=OFF
export ROCPROFSYS_TIME_OUTPUT=OFF
export ROCPROFSYS_PERFETTO_FILL_POLICY=ring_buffer

# One rank-0 trace per transport so the host-side comparison (§4 of the guide) is
# reproducible: isend (GPU-Aware MPI), staged (host-path), rccl (GPU-native).
for METHOD in ${METHODS:-isend staged rccl}; do
  echo "=== 4-rank $METHOD, rocprof-sys on rank 0 -> perfetto proto ==="
  CG_SEED=$SEED mpirun -n 4 --oversubscribe ./gpu_bind.sh bash -c '
    R=${OMPI_COMM_WORLD_RANK:-0}
    export ROCPROFSYS_OUTPUT_PATH='"$OUT/${METHOD}_rank"'$R
    if [ "$R" = 0 ]; then
      exec rocprof-sys-run -- ./cg_gpu '"$MAT"' '"$METHOD"'
    else
      exec ./cg_gpu '"$MAT"' '"$METHOD"'
    fi' > "$OUT/$METHOD.log" 2>&1
  echo "rc=$?"; grep -E "solve time|comm total|halo exch|allreduce" "$OUT/$METHOD.log" | head
  f=$(find "$OUT/${METHOD}_rank0" -name '*.proto' | head -1)
  [ -n "$f" ] && cp "$f" "$OUT/cg_${METHOD}_n4.proto" && \
    echo "  cg_${METHOD}_n4.proto <- $f ($(stat -c %s "$f") bytes)"
done

echo "=== roctx / host labels captured (isend) ==="
strings "$OUT/cg_isend_n4.proto" 2>/dev/null | \
  grep -oE 'cg_iteration|dot_allreduce|halo_wait_recv|halo_wait_send|halo_post_gather|spmv_on_proc|spmv_off_proc|MPI_Allreduce|MPI_Waitall|MPI_Isend' | \
  sort | uniq -c
date
echo "DONE -> $OUT/  (open cg_<method>_n4.proto in Perfetto; search a roctx/halo label)"
