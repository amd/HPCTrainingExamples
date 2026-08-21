#!/bin/bash
#SBATCH --job-name=cg_pftrace
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --time=00:15:00
#SBATCH --output=pftrace_%j.out
#SBATCH --error=pftrace_%j.out
set -u
date; echo "Node(s): ${SLURM_JOB_NODELIST:-$(hostname)}"
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
module purge 2>/dev/null
module load "${ROCM_MODULE:-rocm/6.4.1}"
module load "${OPENMPI_MODULE:-openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4}" 2>/dev/null || module load openmpi
chmod +x gpu_bind.sh 2>/dev/null
echo "=== build ROCTX=1 ==="; make clean >/dev/null 2>&1
make ROCTX=1 2>&1 | tail -2 || { echo BUILD FAILED; exit 1; }

OUT=timeline_pftrace; rm -rf "$OUT"; mkdir -p "$OUT"
SEED=12345
ITERS=${CG_MAX_ITERS:-5}

# Light trace: kernels + roctx markers + memory copies (NO --sys-trace HSA/HIP
# firehose) so the pftrace is small and the timeline is just the iterations.
TRACE="--kernel-trace --marker-trace --memory-copy-trace"

echo "=== 1-rank, $ITERS iterations, $TRACE -> pftrace ==="
CG_SEED=$SEED CG_MAX_ITERS=$ITERS rocprofv3 $TRACE \
  --output-format pftrace -d "$OUT/n1" -- ./cg_gpu src/Dubcova2.pm isend > "$OUT/n1.log" 2>&1
echo "rc=$?"; grep -E "capping|iterations|Max iter" "$OUT/n1.log"

echo "=== 4-rank, $ITERS iterations, $TRACE -> pftrace (rank 0) ==="
CG_SEED=$SEED CG_MAX_ITERS=$ITERS mpirun -n 4 ./gpu_bind.sh bash -c '
  R=${OMPI_COMM_WORLD_RANK:-0}
  if [ "$R" = 0 ]; then
    exec rocprofv3 --kernel-trace --marker-trace --memory-copy-trace --output-format pftrace -d '"$OUT"'/n4 -- ./cg_gpu src/Dubcova2.pm isend
  else
    exec ./cg_gpu src/Dubcova2.pm isend
  fi' > "$OUT/n4.log" 2>&1
echo "rc=$?"; grep -E "capping|iterations|Max iter" "$OUT/n4.log"

echo "=== pftrace files ==="
find "$OUT/n1" "$OUT/n4" -name '*.pftrace' -printf '%p  %s bytes\n' 2>/dev/null | sort
# copy to stable names
for r in n1 n4; do
  f=$(find "$OUT/$r" -name '*.pftrace' | head -1)
  [ -n "$f" ] && cp "$f" "$OUT/cg_${r}.pftrace" && echo "cg_${r}.pftrace <- $f"
done
date; echo "DONE -> $OUT/"
