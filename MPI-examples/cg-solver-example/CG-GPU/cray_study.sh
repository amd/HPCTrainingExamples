#!/bin/bash
# =============================================================================
# Communication + affinity study driver for the CG-GPU solver on an HPE Cray EX
# MI300A node (PrgEnv-amd + cray-mpich, GPU-aware via craype-accel-amd-gfx942).
#
# Mirrors the AAC6 / OpenMPI study (CG-GPU/STUDY_REPORT.md) but uses srun +
# cray-mpich.  Runs every communication variant REPEATS times with a fixed RHS
# seed, verifies convergence, and prints per-run and averaged timing.
#
# Env overrides:
#   MATRIX    (default src/Dubcova2.pm)
#   METHODS   (default: staged isend rccl alltoallv_staged alltoallv)
#   REPEATS   (default 5)
#   NUM_RANKS (default 4)
#   CG_SEED   (default 12345)
#   JOBID     srun --jobid to reuse an existing allocation (optional)
#   BIND      1 = use gpu_bind.sh + --cpu-bind=none (default); 0 = no binding
# =============================================================================
set -u

REPO_DIR="$(dirname "$(readlink -fm "$0")")"
cd "$REPO_DIR" || exit 1

MATRIX=${MATRIX:-src/Dubcova2.pm}
METHODS=${METHODS:-"staged isend rccl alltoallv_staged alltoallv"}
REPEATS=${REPEATS:-5}
NUM_RANKS=${NUM_RANKS:-4}
export CG_SEED=${CG_SEED:-12345}
export MPICH_GPU_SUPPORT_ENABLED=1
BIND=${BIND:-1}

SRUN="srun"
[ -n "${JOBID:-}" ] && SRUN="srun --jobid=$JOBID"

if [ "$BIND" -eq 1 ]; then
   LAUNCH="$SRUN -N1 -n $NUM_RANKS --gpus-per-node=4 --cpu-bind=none ./gpu_bind.sh"
else
   LAUNCH="$SRUN -N1 -n $NUM_RANKS --gpus-per-node=4"
fi

echo "=== CG-GPU Cray study ==="
echo "matrix=$MATRIX ranks=$NUM_RANKS repeats=$REPEATS seed=$CG_SEED bind=$BIND"
echo "launch: $LAUNCH ./cg_gpu"
echo

printf '%-18s %8s %6s %9s %9s %9s %9s %9s\n' \
   method run iters solve comm halo allred compute

for m in $METHODS; do
   ssum=0; csum=0; hsum=0; asum=0; psum=0; n=0
   for r in $(seq 1 "$REPEATS"); do
      OUT=$($LAUNCH ./cg_gpu "$MATRIX" "$m" 2>&1 | tr -d '\0')
      IT=$( echo "$OUT" | grep "iterations to converge" | tail -1 | awk '{print $1}')
      SV=$( echo "$OUT" | grep "CG solve time:"  | tail -1 | awk '{print $4}')
      CM=$( echo "$OUT" | grep "comm total:"     | tail -1 | awk '{print $3}')
      HL=$( echo "$OUT" | grep "halo exchange:"  | tail -1 | awk '{print $3}')
      AR=$( echo "$OUT" | grep "dot allreduce:"  | tail -1 | awk '{print $3}')
      CP=$( echo "$OUT" | grep "compute (rest):" | tail -1 | awk '{print $3}')
      printf '%-18s %8s %6s %9s %9s %9s %9s %9s\n' \
         "$m" "$r" "${IT:-NA}" "${SV:-NA}" "${CM:-NA}" "${HL:-NA}" "${AR:-NA}" "${CP:-NA}"
      if [ -n "$SV" ]; then
         ssum=$(awk -v a="$ssum" -v b="$SV" 'BEGIN{print a+b}')
         csum=$(awk -v a="$csum" -v b="$CM" 'BEGIN{print a+b}')
         hsum=$(awk -v a="$hsum" -v b="$HL" 'BEGIN{print a+b}')
         asum=$(awk -v a="$asum" -v b="$AR" 'BEGIN{print a+b}')
         psum=$(awk -v a="$psum" -v b="$CP" 'BEGIN{print a+b}')
         n=$((n+1))
      fi
   done
   if [ "$n" -gt 0 ]; then
      printf '%-18s %8s %6s %9.4f %9.4f %9.4f %9.4f %9.4f\n' \
         "$m" "AVG(n=$n)" "-" \
         "$(awk -v s=$ssum -v n=$n 'BEGIN{print s/n}')" \
         "$(awk -v s=$csum -v n=$n 'BEGIN{print s/n}')" \
         "$(awk -v s=$hsum -v n=$n 'BEGIN{print s/n}')" \
         "$(awk -v s=$asum -v n=$n 'BEGIN{print s/n}')" \
         "$(awk -v s=$psum -v n=$n 'BEGIN{print s/n}')"
   fi
   echo
done
echo "=== done ==="
