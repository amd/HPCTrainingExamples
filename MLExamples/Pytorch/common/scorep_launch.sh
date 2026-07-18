#!/bin/bash
# =============================================================================
# Launch a torchrun PyTorch training script under Score-P (per-rank profiles).
#
# Instruments the Python training loop with Score-P and writes one CUBE4 profile
# (and optional OTF2 trace) per rank.  Because torchrun runs the script in-process,
# we use `torchrun --no-python` to launch `python -m scorep <script>` per worker
# (via scorep_worker.sh), each with its own SCOREP_EXPERIMENT_DIRECTORY.
#
# NOTE ON GPU CAPTURE.  The PyTorch module is built on ROCm 7.2.x, where the
# Score-P ROCm adapter aborts (see MPI-examples/.../profilers/scorep.md).  So this
# wrapper captures the **Python training-loop regions** (dataloader / forward /
# backward / optimizer step), *not* GPU kernels or RCCL.  Use torch.profiler,
# rocprofv3, or rocprofiler-systems for GPU/RCCL detail on ROCm 7.2.x.
#
# Prerequisite (once, on a login node — it has network):
#   module load rocm openmpi pytorch scorep
#   python -m venv --system-site-packages ~/scorep-venvs/ml
#   source ~/scorep-venvs/ml/bin/activate && pip install scorep
#
# Usage (inside a GPU allocation):
#   NPROC=2 ./scorep_launch.sh ddp_resnet_bench.py -a resnet50 -b 128 --warmup 2 --iters 5
#
# Environment overrides:
#   NPROC            GPUs / workers                    (default: min(2, GPU count))
#   VENV             Score-P venv to activate          (default: ~/scorep-venvs/ml)
#   SCOREP_EXP_BASE  base dir for per-rank profiles    (default: scorep_<script>)
#   SCOREP_TRACE     1 = also write an OTF2 trace      (default: 0, profile only)
#   SCOREP_PY_ARGS   args to `python -m scorep`        (default: --nocompiler --mpp=none)
# =============================================================================
set -u

if [ "$#" -lt 1 ]; then
   echo "usage: [NPROC=N] $0 <script.py> [script args...]"; exit 2
fi
SCRIPT="$1"; shift
COMMON="$(dirname "$(readlink -fm "$0")")"

# --- Score-P venv (scorep Python bindings layered on the pytorch module) --------
VENV="${VENV:-$HOME/scorep-venvs/ml}"
if [ -f "$VENV/bin/activate" ]; then
   # shellcheck disable=SC1091
   source "$VENV/bin/activate"
else
   echo "[warn] Score-P venv '$VENV' not found; assuming 'scorep' is importable."
   echo "       Create it once on a login node:"
   echo "         python -m venv --system-site-packages $VENV"
   echo "         source $VENV/bin/activate && pip install scorep"
fi
python -c "import scorep" 2>/dev/null || { echo "FAIL: 'import scorep' failed in $(which python)"; exit 1; }

# --- GPU count / workers --------------------------------------------------------
GPU_COUNT=$(rocminfo 2>/dev/null | grep -c "Device Type:             GPU")
if [ "${GPU_COUNT:-0}" -lt 1 ]; then echo "No GPU detected -- Skip"; exit 0; fi
NPROC="${NPROC:-$(( GPU_COUNT < 2 ? GPU_COUNT : 2 ))}"

# --- Score-P measurement config -------------------------------------------------
export HSA_XNACK="${HSA_XNACK:-1}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"   # keep conv autotune out of the window
export SCOREP_ENABLE_PROFILING=true
export SCOREP_ENABLE_TRACING="$( [ "${SCOREP_TRACE:-0}" = 1 ] && echo true || echo false )"
export SCOREP_TOTAL_MEMORY="${SCOREP_TOTAL_MEMORY:-256M}"
unset SCOREP_ROCM_ENABLE                              # GPU adapter aborts on ROCm 7.2.x
export SCOREP_ML=1                                    # activate scorep_ml.region() in the scripts
# --nopython: record only the hand-placed user regions (automatic Python
# instrumentation of PyTorch is far too heavy to be usable).
export SCOREP_PY_ARGS="${SCOREP_PY_ARGS:---nopython --nocompiler --mpp=none}"

base_default="scorep_$(basename "$SCRIPT" .py)"
export SCOREP_EXP_BASE="${SCOREP_EXP_BASE:-$base_default}"
rm -rf "$SCOREP_EXP_BASE"
mkdir -p "$SCOREP_EXP_BASE"   # Score-P creates rank_N/ but not the parent

echo "[info] python=$(which python)"
echo "[info] scorep py=$(python -c 'import scorep;print(scorep.__version__)')"
echo "[info] workers=$NPROC  script=$SCRIPT  exp_base=$SCOREP_EXP_BASE  trace=$SCOREP_ENABLE_TRACING"

# --- Launch: one `python -m scorep <script>` per rank via torchrun --no-python --
python -m torch.distributed.run --standalone --no-python --nproc_per_node="$NPROC" \
   "$COMMON/scorep_worker.sh" "$SCRIPT" "$@"
rc=$?

echo; echo "=== per-rank experiment directories ==="
ls -d "$SCOREP_EXP_BASE"/rank_* 2>/dev/null

# --- Text summary for rank 0 ----------------------------------------------------
R0="$SCOREP_EXP_BASE/rank_0/profile.cubex"
if [ -f "$R0" ]; then
   echo; echo "=== scorep-score (rank 0) ==="; scorep-score "$R0" 2>/dev/null | head -20
   echo; echo "=== cube_stat: flat region time profile (rank 0) ==="
   timeout 30 cube_stat -p -m time "$R0" 2>/dev/null | head -25 \
      || echo "(cube_stat unavailable; use scorep-score above)"
else
   echo "WARN: $R0 not written (measurement may have aborted)."
fi
exit $rc
