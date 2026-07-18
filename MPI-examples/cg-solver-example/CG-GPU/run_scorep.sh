#!/bin/bash
# =============================================================================
# Score-P measurement of the distributed GPU Conjugate Gradient solver.
#
# Builds an instrumented ./cg_gpu (make SCOREP=1), runs one communication
# variant under the Score-P measurement system, and prints a text summary
# (scorep-score + cube_dump).  Produces a CUBE4 profile (profile.cubex) and an
# OTF2 trace (traces.otf2) that you can open in CubeGUI / a trace viewer -- see
# ../docs/profilers/scorep.md.
#
# GPU + HIP kernel capture is validated on ROCm 6.4.x with scorep/9.4 (the HIP
# adapter).  On ROCm 7.2.x the ROCm adapter aborts, so GPU capture is disabled
# automatically there and only the MPI dimension is recorded (set SCOREP_GPU=0
# to force MPI-only on any version).
#
# Run inside a GPU allocation, e.g.:
#   srun -p <partition> --exclusive --gres=gpu:4 -t 00:20:00 ./run_scorep.sh
#
# Environment overrides:
#   ROCM_VERSION    rocm module version     (default: 6.4.3)
#   SCOREP_VERSION  scorep module version   (default: 9.4)
#   METHOD          communication variant   (default: isend)
#   NUM_RANKS       number of MPI ranks     (default: min(4, GPU count))
#   MATRIX          matrix file             (default: src/Dubcova2.pm)
#   EXPDIR          experiment directory    (default: scorep_cg_<method>)
#   SCOREP_GPU      1=enable ROCm/HIP GPU capture, 0=MPI-only (default: auto)
# =============================================================================

set -u

REPO_DIR="$(dirname "$(readlink -fm "$0")")"
cd "$REPO_DIR" || { echo "FAIL: cannot cd to $REPO_DIR"; exit 1; }

ROCM_VERSION=${ROCM_VERSION:-6.4.3}
SCOREP_VERSION=${SCOREP_VERSION:-9.4}
METHOD=${METHOD:-isend}
MATRIX=${MATRIX:-src/Dubcova2.pm}

# ---------------------------------------------------------------------------
# Modules: rocm + (patched hipBLASLt if present) + openmpi + scorep
# ---------------------------------------------------------------------------
module purge >/dev/null 2>&1
module load rocm/"$ROCM_VERSION" 2>/dev/null || module load rocm 2>/dev/null
# Tuned hipBLASLt for performance runs, where this ROCm build provides one.
if module avail hipblaslt/patched 2>&1 | grep -q 'hipblaslt/patched'; then
   module load hipblaslt/patched 2>/dev/null
fi
module list 2>&1 | grep -q 'hipblaslt/patched' \
   && echo '[ok] hipblaslt/patched active' \
   || echo '[info] no patched hipBLASLt for this ROCm build'
module load openmpi 2>/dev/null
module load scorep/"$SCOREP_VERSION" 2>/dev/null || module load scorep 2>/dev/null

command -v scorep >/dev/null || { echo "FAIL: scorep not on PATH"; exit 1; }
echo "[info] $(scorep --version 2>&1 | head -1) | rocm/$ROCM_VERSION"

# ---------------------------------------------------------------------------
# GPU count / ranks
# ---------------------------------------------------------------------------
GPU_COUNT=$(rocminfo 2>/dev/null | grep -c "Device Type:             GPU")
if [ "${GPU_COUNT:-0}" -lt 1 ]; then echo "No GPU detected -- Skip"; exit 0; fi
NUM_RANKS=${NUM_RANKS:-$(( GPU_COUNT < 4 ? GPU_COUNT : 4 ))}
EXPDIR=${EXPDIR:-scorep_cg_${METHOD}}

# ---------------------------------------------------------------------------
# Decide GPU capture.  The Score-P ROCm adapter aborts on ROCm 7.2.x, so
# default to MPI-only there and enable GPU capture on 6.x.
# ---------------------------------------------------------------------------
if [ -z "${SCOREP_GPU:-}" ]; then
   case "$ROCM_VERSION" in
      6.*) SCOREP_GPU=1 ;;
      *)   SCOREP_GPU=0 ;;
   esac
fi

# ---------------------------------------------------------------------------
# Build instrumented binary
# ---------------------------------------------------------------------------
echo "=== Building instrumented cg_gpu (make SCOREP=1) ==="
make clean >/dev/null 2>&1
if ! make SCOREP=1 >/tmp/cg_scorep_build.$$.log 2>&1; then
   tail -20 /tmp/cg_scorep_build.$$.log; echo "FAIL: build failed"; exit 1
fi
[ -x ./cg_gpu ] || { echo "FAIL: cg_gpu not produced"; exit 1; }

# ---------------------------------------------------------------------------
# Score-P measurement configuration
# ---------------------------------------------------------------------------
export HSA_XNACK=${HSA_XNACK:-1}
export SCOREP_ENABLE_PROFILING=true      # -> profile.cubex (CUBE4 summary)
export SCOREP_ENABLE_TRACING=true        # -> traces.otf2   (OTF2 event trace)
export SCOREP_TOTAL_MEMORY=${SCOREP_TOTAL_MEMORY:-64M}
export SCOREP_EXPERIMENT_DIRECTORY="$EXPDIR"
if [ "$SCOREP_GPU" = "1" ]; then
   export SCOREP_ROCM_ENABLE=yes
   echo "[info] ROCm/HIP GPU capture: ENABLED (SCOREP_ROCM_ENABLE=yes)"
else
   unset SCOREP_ROCM_ENABLE
   echo "[info] ROCm/HIP GPU capture: DISABLED (MPI-only; ROCm $ROCM_VERSION)"
fi
rm -rf "$EXPDIR"

echo "=== Running: method=$METHOD ranks=$NUM_RANKS matrix=$MATRIX ==="
mpirun --bind-to none --oversubscribe -n "$NUM_RANKS" \
       ./cg_gpu "$MATRIX" "$METHOD" 2>&1 | tr -d '\0' \
   | grep -aiE "method=|Initial residual|iterations to converge|2-norm of residual|CG solve time|comm total" \
   || true

# ---------------------------------------------------------------------------
# Text summaries
# ---------------------------------------------------------------------------
if [ ! -f "$EXPDIR/profile.cubex" ]; then
   echo "WARN: $EXPDIR/profile.cubex not written (measurement may have aborted)."
   ls -la "$EXPDIR" 2>/dev/null
   exit 1
fi

echo; echo "=== scorep-score: region-type breakdown ==="
scorep-score "$EXPDIR/profile.cubex"

# cube_stat gives a clean flat per-routine time profile.  (Prefer it over
# cube_dump -c region, which can hang on some CUBE builds; guard with timeout.)
echo; echo "=== cube_stat: flat region time profile ==="
timeout 30 cube_stat -p -m time "$EXPDIR/profile.cubex" 2>/dev/null \
   || echo "(cube_stat unavailable; use scorep-score above)"

echo; echo "=== Artifacts ==="
echo "  profile : $REPO_DIR/$EXPDIR/profile.cubex   (open in CubeGUI)"
echo "  trace   : $REPO_DIR/$EXPDIR/traces.otf2     (open in a trace viewer)"
echo "  See ../docs/profilers/scorep.md for viewing over VNC/noVNC/X11."
