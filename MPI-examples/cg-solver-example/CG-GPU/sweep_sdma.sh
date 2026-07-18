#!/bin/bash
# =============================================================================
# SDMA vs blit-kernel copy-engine sweep (single ROCm version).
#
# The HSA runtime can service host<->device (and some device<->device) copies
# either with the dedicated SDMA/DMA engines or with shader "blit" kernels:
#
#   HSA_ENABLE_SDMA=0        force blit kernels (shader copies)
#   HSA_ENABLE_SDMA=1        use SDMA engines (runtime default)
#   HSA_ENABLE_SDMA_GANG=1   gang multiple SDMA engines for one copy
#
# This only touches the *copy* path, so it should move communication time
# (halo exchange, especially the host-staged variants' D<->H copies) while
# leaving the rocSPARSE/rocBLAS compute time essentially unchanged.
#
# The binary is identical across configs, so we build ONCE and only vary the
# runtime environment.  Fixed seed + min-of-REPEATS for low-noise comparison.
#
# Env overrides:
#   ROCM_VER  ROCm module version   (default: 6.4.3)
#   METHODS   comm variants         (default: all 5)
#   REPEATS   runs per config       (default: 5)
#   CG_SEED   fixed RHS seed        (default: 12345)
#   MATRIX    matrix file           (default: src/Dubcova2.pm)
# =============================================================================
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}" || { echo "cannot cd to submit dir"; exit 1; }

ROCM_VER=${ROCM_VER:-6.4.3}
METHODS=${METHODS:-"staged isend rccl alltoallv_staged alltoallv"}
REPEATS=${REPEATS:-5}
RANKS=${SLURM_NTASKS:-4}
MATRIX=${MATRIX:-src/Dubcova2.pm}
export CG_SEED=${CG_SEED:-12345}

# SDMA configs:  label  "VAR=val VAR=val ..."
CONFIGS=(
  "blit|HSA_ENABLE_SDMA=0 HSA_ENABLE_SDMA_GANG=0"
  "sdma|HSA_ENABLE_SDMA=1 HSA_ENABLE_SDMA_GANG=0"
  "sdma_gang|HSA_ENABLE_SDMA=1 HSA_ENABLE_SDMA_GANG=1"
)

chmod +x gpu_bind.sh 2>/dev/null

module purge >/dev/null 2>&1
module load "rocm/$ROCM_VER" >/dev/null 2>&1 || { echo "no rocm/$ROCM_VER module"; exit 1; }
module load openmpi           >/dev/null 2>&1 || { echo "no openmpi module"; exit 1; }
export OMPI_CXX=hipcc

echo "SDMA sweep: rocm=$ROCM_VER ranks=$RANKS repeats=$REPEATS seed=$CG_SEED matrix=$MATRIX"
echo "Node: ${SLURM_JOB_NODELIST:-$(hostname)}"
echo

make clean >/dev/null 2>&1
if ! make >/tmp/sdma_build.log 2>&1; then
   echo "build failed; see /tmp/sdma_build.log"; tail -20 /tmp/sdma_build.log; exit 1
fi

printf '%-16s %-10s %6s %10s %10s %10s %10s %10s\n' \
       method config iters "solve(s)" "comm(s)" "halo(s)" "allred(s)" "compute(s)"
printf '%-16s %-10s %6s %10s %10s %10s %10s %10s\n' \
       ---------------- ---------- ------ ---------- ---------- ---------- ---------- ----------

for method in $METHODS; do
   for entry in "${CONFIGS[@]}"; do
      label=${entry%%|*}
      vars=${entry#*|}
      # Build the -x list and an env prefix for the runtime copy-engine flags.
      xargs=""
      for kv in $vars; do
         export "${kv?}"
         xargs="$xargs -x ${kv%%=*}"
      done

      runs=""
      iters=""
      for r in $(seq 1 "$REPEATS"); do
         OUT=$(mpirun --bind-to none --oversubscribe -x CG_SEED $xargs -n "$RANKS" \
               ./gpu_bind.sh ./cg_gpu "$MATRIX" "$method" 2>&1 | tr -d '\0')
         s=$(echo "$OUT"  | grep "CG solve time:" | awk '{print $4}')
         c=$(echo "$OUT"  | grep "comm total:"    | awk '{print $3}')
         h=$(echo "$OUT"  | grep "halo exchange:" | awk '{print $3}')
         a=$(echo "$OUT"  | grep "dot allreduce:" | awk '{print $3}')
         it=$(echo "$OUT" | grep "iterations to converge" | awk '{print $1}')
         [ -n "$it" ] && iters="$it"
         [ -n "$s" ] && [ -n "$c" ] && runs="$runs$s $c $h $a"$'\n'
      done

      if [ -z "$runs" ]; then
         printf '%-16s %-10s %6s %10s %10s %10s %10s %10s\n' \
                "$method" "$label" "${iters:--}" "-" "-" "-" "-" "run-fail"
         continue
      fi

      # Keep the min-solve run; compute = solve - comm.
      read -r s c h a <<<"$(printf '%s' "$runs" | sort -g | head -1)"
      comp=$(awk -v s="$s" -v c="$c" 'BEGIN{printf "%.4f", s-c}')
      printf '%-16s %-10s %6s %10s %10s %10s %10s %10s\n' \
             "$method" "$label" "${iters:--}" "$s" "$c" "$h" "$a" "$comp"
   done
   echo
done

echo "Done. (Per method/config: minimum solve-time run over $REPEATS; compute = solve - comm.)"
