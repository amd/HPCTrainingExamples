#!/bin/bash
# =============================================================================
# Sweep ROCm versions with ONE communication method to isolate the compute-path
# performance of the CG solver across toolchains.
#
# Runs inside a single SLURM allocation (one node) so results are directly
# comparable: same GPUs, sequential builds (no shared-binary race), fixed seed.
# For each ROCm version it rebuilds cg_gpu, runs the chosen method REPEATS times
# with per-rank GPU+NUMA binding, and reports the *minimum* solve time (warm,
# low-noise) with its communication / compute split (compute = solve - comm).
#
# Env overrides:
#   METHOD    communication variant           (default: isend)
#   VERSIONS  space-separated ROCm versions   (default: representative set)
#   REPEATS   runs per version                (default: 3)
#   CG_SEED   fixed RHS seed                  (default: 12345)
#   MATRIX    matrix file                     (default: src/Dubcova2.pm)
#   SPMV      SpMV path(s) to build/compare   (default: v1; use "v1 v2" or "v2")
#             v1 = classic rocsparse_spmv, v2 = rocsparse_v2_spmv (ROCm 7.x only)
# =============================================================================
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}" || { echo "cannot cd to submit dir"; exit 1; }

METHOD=${METHOD:-isend}
REPEATS=${REPEATS:-3}
RANKS=${SLURM_NTASKS:-4}
MATRIX=${MATRIX:-src/Dubcova2.pm}
SPMV=${SPMV:-v1}
export CG_SEED=${CG_SEED:-12345}
VERSIONS=${VERSIONS:-"6.3.4 6.4.1 6.4.3 7.0.2 7.1.1 7.2.0 7.2.2 7.2.4 7.13.0"}

chmod +x gpu_bind.sh 2>/dev/null
echo "Sweep: method=$METHOD ranks=$RANKS repeats=$REPEATS seed=$CG_SEED matrix=$MATRIX spmv='$SPMV'"
echo "Node: ${SLURM_JOB_NODELIST:-$(hostname)}"
echo

RESULTS=""
printf '%-10s %-4s %6s %10s %10s %10s\n' rocm spmv iters "solve(s)" "comm(s)" "compute(s)"
printf '%-10s %-4s %6s %10s %10s %10s\n' ---------- ---- ------ ---------- ---------- ----------

for v in $VERSIONS; do
   module purge >/dev/null 2>&1
   if ! module load "rocm/$v" >/dev/null 2>&1; then
      printf '%-10s %-4s %6s %10s %10s %10s\n' "$v" "-" "-" "-" "-" "no-module"
      continue
   fi
   if ! module load openmpi >/dev/null 2>&1; then
      printf '%-10s %-4s %6s %10s %10s %10s\n' "$v" "-" "-" "-" "-" "no-openmpi"
      continue
   fi
   export OMPI_CXX=hipcc

   for mode in $SPMV; do
      case "$mode" in
         v2) mkflag="SPMV_V2=1" ;;
         *)  mkflag="" ;;
      esac
      make clean >/dev/null 2>&1
      if ! make $mkflag >"/tmp/sweep_build_${v}_${mode}.log" 2>&1; then
         printf '%-10s %-4s %6s %10s %10s %10s\n' "$v" "$mode" "-" "-" "-" "build-fail"
         continue
      fi

      # Run REPEATS times; keep the row with the minimum solve time.
      runs=""
      iters=""
      for r in $(seq 1 "$REPEATS"); do
         OUT=$(mpirun --bind-to none --oversubscribe -x CG_SEED -n "$RANKS" \
               ./gpu_bind.sh ./cg_gpu "$MATRIX" "$METHOD" 2>&1 | tr -d '\0')
         s=$(echo "$OUT" | grep "CG solve time:" | awk '{print $4}')
         c=$(echo "$OUT" | grep "comm total:"    | awk '{print $3}')
         it=$(echo "$OUT" | grep "iterations to converge" | awk '{print $1}')
         [ -n "$it" ] && iters="$it"
         [ -n "$s" ] && [ -n "$c" ] && runs="$runs$s $c"$'\n'
      done

      if [ -z "$runs" ]; then
         printf '%-10s %-4s %6s %10s %10s %10s\n' "$v" "$mode" "${iters:--}" "-" "-" "run-fail"
         continue
      fi

      # Pick min-solve run; compute = solve - comm.
      read -r minsolve mincomm <<<"$(printf '%s' "$runs" | sort -g | head -1)"
      comp=$(awk -v s="$minsolve" -v c="$mincomm" 'BEGIN{printf "%.4f", s-c}')
      printf '%-10s %-4s %6s %10s %10s %10s\n' "$v" "$mode" "${iters:--}" "$minsolve" "$mincomm" "$comp"
      RESULTS="${RESULTS}${v} ${mode} ${iters} ${minsolve} ${mincomm} ${comp}"$'\n'
   done
done

echo
echo "Done. (Per version/spmv: minimum solve time over $REPEATS runs; compute = solve - comm.)"
