#!/bin/bash

# Regression check for the Origami analytical kernel-selection layer in
# hipBLASLt 7.x. Companion test to
# hipblaslt_regression_check.sh, which probes returnAlgoCount=0 events
# on the DEFAULT (TENSILE_SOLUTION_SELECTION_METHOD=0) dispatch path.
# This script probes the analytical (TENSILE_SOLUTION_SELECTION_METHOD=2) path specifically.
#
# Background. Starting in ROCm 7.0 hipBLASLt ships an analytical kernel
# selector ("Origami"). The latency table (origami::hardware_t::
# INSTRUCTION_MAP) lives in libhipblaslt.so
#
# NOTE the hipblaslt/patched overlay done with
# https://github.com/amd/HPCTrainingDock/blob/main/rocm/scripts/hipblaslt_patch_setup.sh
# DOES affect the warning count -- not
# by editing INSTRUCTION_MAP (it can't), but by inserting equality
# rows in the bias-fused .dat that short-circuit ExactLogicLibrary
# BEFORE the ExperimentalStreamK / Prediction row is reached. Verified
# 2026-05-20: overlay loaded -> 0 warnings; overlay unloaded -> 81460
# warnings, same shape, same env. So this probe MUST unload the
# overlay or it will report a false PASS.
#
# Origami is opt-in via TENSILE_SOLUTION_SELECTION_METHOD=2 and TENSILE_STREAMK_DYNAMIC_GRID=5
# For details: https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/reference/env-variables.html
# Prediction library using a compiled-in cost model that consults
# INSTRUCTION_MAP per (MI_M, MI_N, MI_K, dtype). When that table is
# missing the entry for a tuple actually used by a candidate, Origami
# emits this exact diagnostic to STDERR (not the hipBLASLt structured
# log):
#
#     Warning: Latency not found for MI_M=<M>, MI_N=<N>, MI_K=<K>,
#              mi_input_type=<dtype>. Returning latency value of 32
#              (really slow).
#
# and substitutes a 32-cycle fallback latency. The fallback biases the
# analytical model against every kernel that uses that MFMA shape, so
# missing entries are NOT a benign warning -- they actively degrade
# kernel selection.
#
#
# Datatypes (HIPBLASLT_ORIGAMI_DTYPES env var, comma-separated):
#   fp16,fp32,bf16   (default)
#
# This test assumes PyTorch has been
# installed with:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/pytorch_setup.sh

if ! type module >/dev/null 2>&1; then
   [ -r /etc/profile.d/lmod.sh ]         && . /etc/profile.d/lmod.sh
   [ -r /usr/share/lmod/lmod/init/bash ] && . /usr/share/lmod/lmod/init/bash
fi

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
   echo "rocm module is not loaded"
   echo "loading default rocm module"
   module load rocm
fi
module load pytorch 2>/dev/null

# CTest-friendly SKIP: pytorch must be importable.
if ! python3 -c "import torch" 2>/dev/null; then
   echo "ORIGAMI CHECK: SKIPPED (pytorch module not available or torch import failed)"
   exit 77
fi

export TENSILE_SOLUTION_SELECTION_METHOD=2
export ANALYTICAL_GEMM_HEURISTICS=1
export TENSILE_STREAMK_DYNAMIC_GRID=5

# Enable light hipBLASLt logging so we can post-hoc see which .so /
# .dat was loaded (line "[initialize] Using ..." or
# "[initialize] HIPBLASLT_TENSILE_LIBPATH not set: Using ...").
ORIGAMI_HIPBLASLT_LOG=$(mktemp -t hipblaslt_origami.XXXXXX.log)
export HIPBLASLT_LOG_LEVEL=3
export HIPBLASLT_LOG_FILE="${ORIGAMI_HIPBLASLT_LOG}"
: > "${HIPBLASLT_LOG_FILE}"
# Do NOT set ANALYTICAL_GEMM_DEBUG=1 by default -- it dumps Origami's
# entire INSTRUCTION_MAP to stdout, ~15k lines per process, which
# drowns the PASS/FAIL signal. Set HIPBLASLT_ORIGAMI_DEBUG=1 to opt in.
if [ "${HIPBLASLT_ORIGAMI_DEBUG:-0}" = "1" ]; then
   export ANALYTICAL_GEMM_DEBUG=1
fi

echo
echo "--- hipBLASLt Origami regression-check environment ---"
echo "  TENSILE_SOLUTION_SELECTION_METHOD = ${TENSILE_SOLUTION_SELECTION_METHOD}"
echo "  ANALYTICAL_GEMM_HEURISTICS        = ${ANALYTICAL_GEMM_HEURISTICS}"
echo "  TENSILE_STREAMK_DYNAMIC_GRID      = ${TENSILE_STREAMK_DYNAMIC_GRID}"
echo "  ANALYTICAL_GEMM_DEBUG             = ${ANALYTICAL_GEMM_DEBUG:-<unset>}"
echo "  HIPBLASLT_LOG_FILE                = ${HIPBLASLT_LOG_FILE}"
echo "  HIPBLASLT_TENSILE_LIBPATH         = ${HIPBLASLT_TENSILE_LIBPATH:-<unset, will use default <rocm>/lib/hipblaslt/library>}"
echo "  HIPBLASLT_OVERLAY                 = ${HIPBLASLT_OVERLAY:-<unset>}"
echo "  ROCM_PATH                         = ${ROCM_PATH:-<unset>}"
if type module >/dev/null 2>&1; then
   echo "  loaded modules                    =" $(module -t list 2>&1 | tr '\n' ' ')
fi
echo "------------------------------------------------------"
echo

if [ -n "${HIPBLASLT_ORIGAMI_ARCH:-}" ]; then
   GFX_ARCH="${HIPBLASLT_ORIGAMI_ARCH}"
elif command -v rocminfo >/dev/null 2>&1; then
   GFX_ARCH=$(rocminfo 2>/dev/null \
      | grep -E '^\s*Name:\s*gfx' \
      | sed -e 's/^\s*Name:\s*//' -e 's/\s*$//' -e 's/:.*//' \
      | sort -u | head -1)
fi
GFX_ARCH=${GFX_ARCH:-unknown}
echo "detected GPU arch: ${GFX_ARCH}"

DTYPES_RAW=${HIPBLASLT_ORIGAMI_DTYPES:-fp16,fp32,bf16}

OVERALL_FAIL=0
PER_DTYPE_RESULT=""
LOGS_TO_CLEAN="${ORIGAMI_HIPBLASLT_LOG}"
cleanup_logs() {
   if [ -n "${LOGS_TO_CLEAN}" ]; then
      # shellcheck disable=SC2086
      rm -f ${LOGS_TO_CLEAN}
   fi
}
trap cleanup_logs EXIT

# Single-dtype probe. Forks a python subprocess with stderr redirected
# to a per-dtype tempfile, runs ONE nn.Linear forward+backward at the
# canonical ResNet-FC shape (256 x 2048 -> 100). That single call
# triggers Origami's full candidate-scoring loop (tens of thousands
# of get_mi_latency invocations), so if the MI-shape gap exists for
# this dtype, the tempfile will contain thousands of identical
# warnings; if it doesn't, the tempfile is empty.
run_dtype_probe() {
   local DTYPE="$1"
   local ERR_LOG
   ERR_LOG=$(mktemp -t "hipblaslt_origami_${DTYPE}.XXXXXX.log")
   LOGS_TO_CLEAN="${LOGS_TO_CLEAN} ${ERR_LOG}"

   echo
   echo "=========================================================="
   echo "  ORIGAMI PROBE  dtype=${DTYPE}  arch=${GFX_ARCH}"
   echo "=========================================================="

   # Capture stdout AND stderr. Origami's warning is emitted via
   # std::cerr on the rocm/7.2.0 build we tested, but other builds /
   # versions may have routed it through std::clog (line-buffered) or
   # rerouted to stdout. Merging streams avoids a silent miss.
   python3 - "${DTYPE}" >"${ERR_LOG}" 2>&1 <<'PY'
import sys
import torch
import torch.nn as nn

dtype_label = sys.argv[1] if len(sys.argv) > 1 else "fp16"
_DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp64": torch.float64,
}
if dtype_label not in _DTYPE_MAP:
    print(
        f"HIPBLASLT_ORIGAMI_CHECK: unknown dtype '{dtype_label}' "
        f"(expected one of: {', '.join(sorted(_DTYPE_MAP))})",
        file=sys.stderr,
    )
    sys.exit(2)
if not torch.cuda.is_available():
    print("HIPBLASLT_ORIGAMI_CHECK: no GPU available, skipping",
          file=sys.stderr)
    sys.exit(77)

torch_dtype = _DTYPE_MAP[dtype_label]
dev = torch.device("cuda")
torch.manual_seed(0)

# Canonical workload point. Same shape used by the equality-lookup
# regression test's warmup, so the two tests probe the same family.
# The shape's role here is just to make Origami's scoring loop run --
# nothing about the specific (M, N, K) values is special for the
# Origami test, what matters is that the candidate kernels are
# scored under the requested dtype.
net = nn.Linear(2048, 100).to(dtype=torch_dtype, device=dev)
x   = torch.randn(256, 2048, dtype=torch_dtype, device=dev)
out = net(x); out.sum().backward()
torch.cuda.synchronize()
print(f"  [{dtype_label}] forward+backward OK")
PY
   local RC_PY=$?

   if [ ${RC_PY} -eq 77 ]; then
      echo "ORIGAMI CHECK [dtype=${DTYPE}]: SKIPPED (no GPU)"
      return 77
   fi
   if [ ${RC_PY} -ne 0 ]; then
      echo "ORIGAMI CHECK [dtype=${DTYPE}]: FAILED (python harness rc=${RC_PY})"
      echo "  -- last 20 stderr lines:"
      tail -20 "${ERR_LOG}" | sed 's/^/     /'
      return 1
   fi

   # Count Origami warnings, and dedup the (MI_M, MI_N, MI_K, dtype)
   # tuples so the operator sees exactly which entries are missing in
   # INSTRUCTION_MAP rather than a wall of 60k identical lines.
   local NUM_WARN UNIQUE_TUPLES
   NUM_WARN=$(grep -c 'Latency not found for MI_M=' "${ERR_LOG}")
   echo
   echo "--- Origami stderr summary [dtype=${DTYPE}] ---"
   echo "  total 'Latency not found' warnings : ${NUM_WARN}"

   echo
   echo "--- hipBLASLt library actually used [dtype=${DTYPE}] ---"
   echo "  HIPBLASLT_TENSILE_LIBPATH (at probe time) = ${HIPBLASLT_TENSILE_LIBPATH:-<unset>}"
   if [ -s "${ORIGAMI_HIPBLASLT_LOG}" ]; then
      INIT_LINE=$(grep -E '\[initialize\]' "${ORIGAMI_HIPBLASLT_LOG}" | head -2)
      if [ -n "${INIT_LINE}" ]; then
         echo "${INIT_LINE}" | sed 's/^/  /'
      else
         echo "  (no [initialize] line in log -- showing first 5 lines instead:)"
         head -5 "${ORIGAMI_HIPBLASLT_LOG}" | sed 's/^/    /'
      fi
   else
      echo "  WARN: hipBLASLt log is empty -- LOG_LEVEL=3 was set but"
      echo "        nothing was emitted. hipBLASLt may not have been"
      echo "        invoked at all by this probe, in which case the"
      echo "        PASS verdict below is meaningless."
   fi
   echo "----------------------------------------------------"

   if [ "${NUM_WARN}" -eq 0 ]; then
      echo "  no MI-shape gaps observed for this dtype on ${GFX_ARCH}."
      echo "ORIGAMI CHECK [dtype=${DTYPE}]: PASSED"
      return 0
   fi

   echo "  distinct missing (MI_M, MI_N, MI_K, mi_input_type) tuples:"
   UNIQUE_TUPLES=$(grep -hoE 'MI_M=[0-9]+, MI_N=[0-9]+, MI_K=[0-9]+, mi_input_type=[A-Za-z0-9_]+' \
                       "${ERR_LOG}" | sort -u)
   echo "${UNIQUE_TUPLES}" | sed 's/^/     /'
   echo "ORIGAMI CHECK [dtype=${DTYPE}]: FAILED"
   echo "  -> file an upstream ticket asking AMD to add these entries to"
   echo "     origami::hardware_t::INSTRUCTION_MAP for ${GFX_ARCH}."
   return 1
}

# Main dtype loop.
IFS=',' read -r -a DTYPES_ARR <<< "${DTYPES_RAW}"
ALL_SKIP=1
for DT in "${DTYPES_ARR[@]}"; do
   DT_TRIM=$(echo "${DT}" | tr -d '[:space:]')
   case "${DT_TRIM}" in
      fp16|fp32|bf16|fp64) ;;
      *) echo "ORIGAMI CHECK: unknown dtype '${DT_TRIM}' (expected one of: fp16, fp32, bf16, fp64)"
         OVERALL_FAIL=1
         PER_DTYPE_RESULT="${PER_DTYPE_RESULT}  ${DT_TRIM}: BAD_DTYPE\n"
         continue;;
   esac
   run_dtype_probe "${DT_TRIM}"
   RC=$?
   case ${RC} in
      0)  PER_DTYPE_RESULT="${PER_DTYPE_RESULT}  ${DT_TRIM}: PASS\n"
          ALL_SKIP=0;;
      77) PER_DTYPE_RESULT="${PER_DTYPE_RESULT}  ${DT_TRIM}: SKIP (no GPU)\n";;
      *)  PER_DTYPE_RESULT="${PER_DTYPE_RESULT}  ${DT_TRIM}: FAIL\n"
          OVERALL_FAIL=1
          ALL_SKIP=0;;
   esac
done

echo
echo "=========================================================="
echo "  ORIGAMI CHECK summary (dtypes = ${DTYPES_RAW}, arch = ${GFX_ARCH})"
echo "=========================================================="
echo -en "${PER_DTYPE_RESULT}"
echo

if [ "${ALL_SKIP}" -eq 1 ]; then
   echo "ORIGAMI CHECK: SKIPPED  (every dtype reported no-GPU)"
   exit 77
fi
if [ ${OVERALL_FAIL} -eq 0 ]; then
   echo "ORIGAMI CHECK: PASSED  (zero MI-shape gaps across every dtype probed)"
   exit 0
else
   echo "ORIGAMI CHECK: FAILED  (one or more dtypes hit a missing INSTRUCTION_MAP entry; see per-dtype lists above)"
   exit 1
fi
