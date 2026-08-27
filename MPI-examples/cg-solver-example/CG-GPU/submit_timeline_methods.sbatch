#!/bin/bash
#SBATCH --job-name=cg_tlmethods
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --time=00:20:00
#SBATCH --output=tlmethods_%j.out
#SBATCH --error=tlmethods_%j.out
set -u
date; echo "Node(s): ${SLURM_JOB_NODELIST:-$(hostname)}"
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
module purge 2>/dev/null
module load "${ROCM_MODULE:-rocm/6.4.1}"
module load "${OPENMPI_MODULE:-openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4}" 2>/dev/null || module load openmpi
chmod +x gpu_bind.sh 2>/dev/null
echo "=== build ROCTX=1 ==="; make clean >/dev/null 2>&1
make ROCTX=1 2>&1 | tail -2 || { echo BUILD FAILED; exit 1; }

OUT=timeline_methods; rm -rf "$OUT"; mkdir -p "$OUT"
SEED=12345
ITERS=${CG_MAX_ITERS:-5}
MAT=src/Dubcova2.pm

# Light trace: kernels + roctx markers + memory copies (NO --sys-trace HSA/HIP
# firehose) so each pftrace is small and the timeline is just the iterations.
# --memory-copy-trace is important for `staged`, which shows D<->H copies.
run_method () {
  local method="$1"
  echo "=== 4-rank $method, $ITERS iterations -> pftrace (rank 0) ==="
  CG_SEED=$SEED CG_MAX_ITERS=$ITERS mpirun -n 4 ./gpu_bind.sh bash -c '
    R=${OMPI_COMM_WORLD_RANK:-0}
    if [ "$R" = 0 ]; then
      exec rocprofv3 --kernel-trace --marker-trace --memory-copy-trace \
        --output-format pftrace -d '"$OUT/$method"' -- ./cg_gpu '"$MAT"' '"$method"'
    else
      exec ./cg_gpu '"$MAT"' '"$method"'
    fi' > "$OUT/$method.log" 2>&1
  echo "rc=$?"; grep -E "capping|iterations|Max iter|Converged|residual" "$OUT/$method.log" | head -4
  local f
  f=$(find "$OUT/$method" -name '*.pftrace' | head -1)
  [ -n "$f" ] && cp "$f" "$OUT/cg_${method}_n4.pftrace" && \
    echo "cg_${method}_n4.pftrace <- $f ($(stat -c %s "$f") bytes)"
}

run_method isend
run_method staged
run_method rccl

echo "=== pftrace files ==="
ls -la "$OUT"/*.pftrace 2>/dev/null
date; echo "DONE -> $OUT/"
