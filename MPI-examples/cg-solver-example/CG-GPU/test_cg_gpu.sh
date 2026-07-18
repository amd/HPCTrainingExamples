#!/bin/bash
# =============================================================================
# Test for the distributed GPU Conjugate Gradient solver (CG-GPU)
#
# Based on CG-GPU/README.md: builds ./cg_gpu and runs it across every
# communication variant, checking that each one converges to a small residual.
#
#   staged | isend | rccl | alltoallv_staged | alltoallv | staged_unified |
#   alltoallv_unified
#
# The CG algorithm is identical for all seven variants; only the SpMV ghost
# exchange differs, so every variant must converge with a residual below the
# solver's tolerance (tol = 1e-6 * initial_residual).
#
# Environment overrides:
#   MATRIX     matrix file to solve      (default: src/Dubcova2.pm)
#   NUM_RANKS  number of MPI ranks       (default: min(4, GPU count))
#   RES_TOL    max acceptable residual   (default: 1e-3)
#   METHODS    space-separated variants  (default: all seven)
# =============================================================================

set -u

# ---------------------------------------------------------------------------
# Toolchain / modules  (mirrors the HPCTrainingExamples test convention)
# ---------------------------------------------------------------------------
if [[ -n "${CRAYPE_VERSION:-}" || -f /etc/cray-release ]]; then
   IS_CRAY=1
else
   IS_CRAY=0
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
      echo "rocm module is not loaded; loading default rocm module"
      module load rocm
   fi
fi

# ---------------------------------------------------------------------------
# Require at least one GPU, otherwise skip (matches other GPU tests)
# ---------------------------------------------------------------------------
GPU_COUNT=$(rocminfo 2>/dev/null | grep -c "Device Type:             GPU")
if [ "${GPU_COUNT:-0}" -lt 1 ]; then
   echo "No GPU detected -- Skip"
   exit 0
fi

# ---------------------------------------------------------------------------
# MPI launcher
# ---------------------------------------------------------------------------
if [ "$IS_CRAY" -eq 1 ]; then
   MPIRUN=srun
   # cray-mpich needs this at runtime to accept GPU pointers in MPI calls
   # (the GPU-aware "isend"/"alltoallv" variants); harmless for the others.
   export MPICH_GPU_SUPPORT_ENABLED=1
   # --cpu-bind=none lets the per-rank numactl wrapper (gpu_bind.sh) pin each
   # rank to its GPU-local NUMA node.  Without it, srun's default per-task
   # cpuset makes numactl --cpunodebind fail (sched_setaffinity: Invalid
   # argument) on the ranks whose target NUMA lies outside their cgroup.
   MPIRUN_OPTS="--cpu-bind=none"
else
   module -t list 2>&1 | grep -q "^openmpi"
   if [ $? -eq 1 ]; then
      module load openmpi
   fi
   MPIRUN=mpirun
   # Let the solver's own hipSetDevice(rank % num_gpus) handle GPU placement and
   # avoid OpenMPI's default core binding, which fails inside tight SLURM
   # allocations ("more cpus than are available in your allocation").
   MPIRUN_OPTS="--bind-to none --oversubscribe"
fi
MPIRUN_OPTS="${MPIRUN_OPTS:-}"

# One rank per GPU, capped at 4 (README example uses 4 ranks).
NUM_RANKS=${NUM_RANKS:-$(( GPU_COUNT < 4 ? GPU_COUNT : 4 ))}

REPO_DIR="$(dirname "$(readlink -fm "$0")")"
cd "$REPO_DIR" || { echo "FAIL: cannot cd to $REPO_DIR"; exit 1; }

MATRIX=${MATRIX:-src/Dubcova2.pm}
RES_TOL=${RES_TOL:-1e-3}
METHODS=${METHODS:-"staged isend rccl alltoallv_staged alltoallv staged_unified alltoallv_unified"}

# The '*_unified' methods exploit the MI300A single address space: the GPU accesses
# malloc'd host send/recv buffers coherently via XNACK (no staging copies, no
# GPU-Aware MPI). This must be enabled before HSA init, so export it for all ranks.
export HSA_XNACK=${HSA_XNACK:-1}

# Optional fixed RHS seed (CG_SEED). When set, every method solves the identical
# system, so iteration counts become comparable across methods. OpenMPI needs it
# explicitly forwarded to the ranks.
SEED_OPTS=""
if [ -n "${CG_SEED:-}" ]; then
   export CG_SEED
   [ "$MPIRUN" = "mpirun" ] && SEED_OPTS="-x CG_SEED"
   echo "Using fixed RHS seed: CG_SEED=$CG_SEED"
fi

# Optional per-rank GPU+NUMA binding wrapper (e.g. GPU_BIND=./gpu_bind.sh).
# When set, each rank is pinned to one GPU and its NUMA-local CPUs/memory.
# Requires an allocation that owns whole NUMA nodes (e.g. sbatch --exclusive).
BIND_WRAP=${GPU_BIND:-}
if [ -n "$BIND_WRAP" ]; then
   [ -x "$BIND_WRAP" ] || chmod +x "$BIND_WRAP" 2>/dev/null
   echo "Using per-rank binding wrapper: $BIND_WRAP"
fi

if [ ! -f "$MATRIX" ]; then
   echo "FAIL: matrix file '$MATRIX' not found"
   exit 1
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo "=== Building cg_gpu ==="
make clean >/dev/null 2>&1
if ! make; then
   echo "FAIL: build failed"
   exit 1
fi
[ -x ./cg_gpu ] || { echo "FAIL: cg_gpu was not produced"; exit 1; }

# ---------------------------------------------------------------------------
# Helper to launch the solver for one communication variant
# ---------------------------------------------------------------------------
run_method() {
   local method="$1"
   # tr -d '\0' drops stray NUL bytes some MPI/ROCm layers emit, which would
   # otherwise trigger "ignored null byte" warnings in command substitution.
   $MPIRUN $MPIRUN_OPTS $SEED_OPTS -n "$NUM_RANKS" $BIND_WRAP ./cg_gpu "$MATRIX" "$method" 2>&1 | tr -d '\0'
}

# ---------------------------------------------------------------------------
# Run every communication variant and check convergence
# ---------------------------------------------------------------------------
echo "=== Running CG-GPU: matrix=$MATRIX ranks=$NUM_RANKS res_tol=$RES_TOL ==="
FAILURES=0
# Timing summary rows: "method iters wall solve comm halo allreduce result"
SUMMARY_ROWS=""

for method in $METHODS; do
   echo "--- method=$method ---"
   T0=$(date +%s.%N)
   OUT=$(run_method "$method")
   T1=$(date +%s.%N)
   WALL=$(awk -v a="$T0" -v b="$T1" 'BEGIN{ printf "%.4f", b - a }')
   echo "$OUT"
   echo "  total run time (build excluded, incl. matrix I/O + setup): ${WALL} s"

   ITERS=$(echo "$OUT" | grep "iterations to converge" | tail -1 | awk '{print $1}')
   SOLVE=$(echo "$OUT" | grep "CG solve time:"  | tail -1 | awk '{print $4}')
   COMM=$( echo "$OUT" | grep "comm total:"     | tail -1 | awk '{print $3}')
   HALO=$( echo "$OUT" | grep "halo exchange:"  | tail -1 | awk '{print $3}')
   ALLR=$( echo "$OUT" | grep "dot allreduce:"  | tail -1 | awk '{print $3}')
   RES=$(  echo "$OUT" | grep "2-norm of residual:" | tail -1 | awk '{print $NF}')

   # Determine pass/fail: must converge (not hit cap) with a small residual.
   RESULT="PASS"
   if echo "$OUT" | grep -q "Max iterations reached"; then
      RESULT="FAIL(maxiter)"
   elif ! echo "$OUT" | grep -q "iterations to converge"; then
      RESULT="FAIL(no-run)"
   elif [ -z "$RES" ]; then
      RESULT="FAIL(no-res)"
   else
      OK=$(awk -v r="$RES" -v t="$RES_TOL" \
           'BEGIN{ if (r==r+0 && r>=0 && r<t) print 1; else print 0 }')
      [ "$OK" -eq 1 ] || RESULT="FAIL(res=$RES)"
   fi
   [ "$RESULT" = "PASS" ] || FAILURES=$((FAILURES + 1))
   echo "RESULT[$method]: $RESULT (residual=${RES:-NA}, iters=${ITERS:-NA})"

   SUMMARY_ROWS+=$(printf '%-18s %6s %9s %9s %9s %9s %9s  %s\n' \
       "$method" "${ITERS:-NA}" "${WALL:-NA}" "${SOLVE:-NA}" \
       "${COMM:-NA}" "${HALO:-NA}" "${ALLR:-NA}" "$RESULT")
   SUMMARY_ROWS+=$'\n'
done

echo "========================================"
echo "Timing summary (seconds; solve/comm/halo/allreduce = max across ranks):"
printf '%-18s %6s %9s %9s %9s %9s %9s  %s\n' \
   method iters wall solve comm halo allred result
printf '%s' "$SUMMARY_ROWS"
echo "========================================"

if [ "$FAILURES" -eq 0 ]; then
   echo "ALL CG-GPU METHODS PASSED"
   make clean >/dev/null 2>&1
   exit 0
else
   echo "CG-GPU TEST FAILED ($FAILURES method(s) failed)"
   exit 1
fi
