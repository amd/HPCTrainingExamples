#!/bin/bash

# Regression check for the Origami analytical kernel-selection layer in
# hipBLASLt 7.x on AMD CDNA accelerators. Companion test to
# hipblaslt_regression_check.sh, which probes returnAlgoCount=0 events
# on the DEFAULT (TENSILE_SOLUTION_SELECTION_METHOD=0) dispatch path.
# This script probes the analytical (TSSM=2) path specifically.
#
# Background. Starting in ROCm 7.0 hipBLASLt ships an analytical kernel
# selector ("Origami"). The latency table (origami::hardware_t::
# INSTRUCTION_MAP) lives in libhipblaslt.so itself as a compiled-in
# static initialiser, NOT in any .dat file -- verified by
# `nm -C libhipblaslt.so | grep origami::hardware_t::INSTRUCTION_MAP`.
#
# Two on-host call paths reach origami::select_best_macro_tile_size
# (which is what emits the "Latency not found" warning via
# origami::compute_total_latency -> origami::hardware_t::get_mi_latency):
#
#   (a) Tensile path: ExactLogicLibrary walks the rows of a loaded .dat
#       file; the ExperimentalStreamK row, when enabled, hands the
#       problem to a PredictionLibrary, which calls origami directly
#       (tensilelite/include/Tensile/PredictionLibrary.hpp ~L165). This
#       row is gated by Debug::useExperimentalSelection()==2, i.e.
#       enabled when TENSILE_SOLUTION_SELECTION_METHOD=2.
#   (b) rocroller path: handle->useRocRoller branch in tensile_host.cpp.
#       Dormant in our environment (useRocRoller defaults to -1; the
#       useRocRoller(handle, prob) check returns false unless the user
#       opts in via HIPBLASLT_USE_ROCROLLER=1 AND the problem matches a
#       narrow shape filter). We have NEVER observed origami warnings
#       come through this path on this cluster.
#
# Path (a) is the one this test exercises. Empirically (2026-05-20,
# rocm/7.2.0, unpatched .dat) the warnings on Linear(2048, 100) fp16
# come 100% from the BACKWARD call, which lands on the
# _ExperimentalStreamK_..._Ailk_Bljk_..._gfx942.dat file (the only .dat
# whose sole inner row is `FreeSizeMatching -> Prediction`); the
# winning kernel is Cijk_Ailk_Bljk_..._SK3_..., confirming the
# Prediction-library / Origami selection actually shipped a kernel.
#
# The hipblaslt/patched overlay DOES affect the warning count -- not
# by editing INSTRUCTION_MAP (it can't), but by inserting equality
# rows in the bias-fused .dat that short-circuit ExactLogicLibrary
# BEFORE the ExperimentalStreamK / Prediction row is reached. Verified
# 2026-05-20: overlay loaded -> 0 warnings; overlay unloaded -> 81460
# warnings, same shape, same env. So this probe MUST unload the
# overlay or it will report a false PASS.
#
# Origami is opt-in via TENSILE_SOLUTION_SELECTION_METHOD=2. Once
# engaged, Origami scores every candidate kernel handed to it by the
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
# This script engages Origami, sweeps a small workload across the
# requested datatypes, captures stderr per-dtype, and reports any
# datatype whose stderr contains the warning. Distinct (MI_M, MI_N,
# MI_K, dtype) tuples that fired the warning are deduplicated and
# printed under the dtype banner -- that list is exactly what an AMD-
# side fix needs to populate INSTRUCTION_MAP with.
#
# Datatypes (HIPBLASLT_ORIGAMI_DTYPES env var, comma-separated):
#   fp16,fp32,bf16   (default)
#
#     fp16  -- known to FAIL on ROCm 7.2.x on gfx942: INSTRUCTION_MAP
#              is missing (MI_M=16, MI_N=16, MI_K=32, mi_input_type=Half)
#              (the v_mfma_f32_16x16x32_f16 MFMA -- canonical fp16 MFMA
#              on CDNA3). PASS once AMD adds this entry.
#     fp32  -- known to PASS on ROCm 7.2.x on gfx942: all Float entries
#              for the MI shapes actually used by gfx942 kernels are
#              present.
#     bf16  -- known to PASS on ROCm 7.2.x on gfx942: the BFloat16 MI
#              entries the loaded kernels actually use (notably
#              (16,16,16) and (32,32,8)) are populated. No warnings on
#              a dedicated bf16 probe at the same shape (2026-05-20).
#
# fp64 is INTENTIONALLY EXCLUDED from the default sweep. Reason
# (verified on rocm/7.2.0 + PyTorch 2.9.1, 2026-05):
#   - No `_HA_Bias_*_DD_*_Contraction_*_gfx942.dat` ships in any rocm
#     7.x release; Tensile has no fp64 bias-fused kernels on gfx942.
#   - SLURM probe of nn.Linear(fp64) shows 0 hipBLASLt algorithm-
#     heuristic calls -- PyTorch dispatches fp64 GEMMs through rocBLAS
#     dgemm + ATen bias-add, never through hipBLASLt.
#   - Origami is a layer INSIDE hipBLASLt's algorithm selector; if
#     hipBLASLt is not called, Origami is not engaged, and any
#     Origami INSTRUCTION_MAP latency-entry deficit for the Double
#     dtype is irrelevant to nn.Linear(fp64) performance on this
#     stack.
# Users who explicitly opt in (HIPBLASLT_ORIGAMI_DTYPES=fp64) can
# still run the sweep; the harness accepts it but the result is not
# meaningful for nn.Linear's actual fp64 dispatch path.
#
# Exit code: 0 PASS (zero datatypes triggered the warning), 1 FAIL
# (one or more datatypes triggered the warning), 77 SKIP (no GPU).
# Greppable output:
#   "ORIGAMI CHECK [dtype=fp16]: PASSED" / "FAILED",
#   "ORIGAMI CHECK: PASSED / FAILED" summary.
#
# NOTE: like its companion test, this script assumes PyTorch has been
# installed per:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/pytorch_setup.sh
# and that the relevant ROCm modulefile auto-loads any deployed
# hipblaslt/patched overlay. The overlay is NOT independent of this
# test: it suppresses the warning storm by short-circuiting
# ExactLogicLibrary at the EqualityMatching row before the
# ExperimentalStreamK Prediction row (which is what calls origami)
# gets a chance to run. The overlay can't and doesn't edit
# INSTRUCTION_MAP -- it just makes the test's repro shape not visit
# the broken Prediction path. So this test must unload the overlay
# to observe the upstream gap.

# Robust against `bash <path>` invocation (Lmod's `module` shell
# function may not be present in a fresh non-interactive subshell).
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

# Engage Origami unconditionally for the lifetime of this script.
# These env vars are read once at first call into the relevant code
# path (static-local cache via __cxa_guard_acquire), so they MUST be
# set before the python subprocess starts.
#
# Rationale per env var, grounded in:
#   docs : https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/reference/env-variables.html#origami-with-stream-k-configuration
#   bin  : `strings libhipblaslt.so.1 | grep -cx <ENV>` (gfx942, rocm/7.2.0)
#
#   TENSILE_SOLUTION_SELECTION_METHOD=2
#     docs: "Origami with Stream-K (enables Origami solution selection
#            for consistent performance)".
#     bin : 1 occurrence in libhipblaslt.so.1.
#     Without this Origami is not engaged at all; the test would
#     probe the default tuned-library path.
#
#   ANALYTICAL_GEMM_HEURISTICS=1
#     bin : 1 occurrence in libhipblaslt.so.1 (read by
#           origami::hardware_t::read_heuristics_env_var, per nm -C).
#     Gates Origami's analytical heuristics on. Default behaviour
#     when unset is observed to be enabled; set explicitly so the
#     test does not depend on default state.
#
#   TENSILE_STREAMK_DYNAMIC_GRID=5
#     docs: "The Stream-K algorithm uses the Origami
#            select_best_grid_size function".
#     bin : 1 occurrence in libhipblaslt.so.1.
#     src : tensilelite/include/Tensile/AMDGPU.hpp:256-260 reads this
#           env var; default when unset is 6. ContractionSolution::
#           getSKGrid switches on the value at kernel launch.
#     LOAD-BEARING EMPIRICALLY: only with TSDG=5 does the warning
#     storm fire on the unpatched library; with =6 or unset, no
#     warning -- the test would report PASS even on a stack with the
#     known (MI_M=16, MI_N=16, MI_K=32, Half) gap. Verified 2026-05
#     on rocm/7.2.0 gfx942. The branch that decides whether the
#     launch-time origami call reaches get_mi_latency lives inside
#     the closed `origami::streamk::select_grid` -- we observe the
#     effect, not the mechanism.
#
# NOTE on scope: this test detects a LATENT bug. The default
# selector (TSSM=0) does not engage Origami, so real PyTorch
# workloads on rocm/7.2.x are not hitting the missing-entry path
# unless the user explicitly opts into TSSM=2. The test is still
# worth keeping because AMD documents TSSM=2 as the "consistent
# performance" mode and recommends users adopt it; the bug will
# bite once users follow that recommendation. A FAILED verdict
# here is "an upstream gap exists", not "your workload is slow
# because of this".
export TENSILE_SOLUTION_SELECTION_METHOD=2
export ANALYTICAL_GEMM_HEURISTICS=1
export TENSILE_STREAMK_DYNAMIC_GRID=5
# Required overlay-unload. The patched overlay inserts equality rows
# that match the probe shape (Linear(2048, 100), batch 256) directly
# in the forward-direction bias-fused .dat AND carries 2 backward
# equality rows for this exact shape, so with the overlay loaded
# ExactLogicLibrary short-circuits at EqualityMatching and never
# reaches the ExperimentalStreamK / Prediction row that calls
# origami -- 0 warnings = false PASS. Verified 2026-05-20: overlay
# loaded -> 0 warnings; overlay unloaded -> 81460 warnings on the
# same shape, same env.
#
# We do NOT silence the unload's error. If the unload fails (e.g. a
# downstream module like pytorch depends on hipblaslt/patched), the
# script must SAY SO -- a silently-failed unload caused us to misread
# an earlier test run.
#
# After the unload we also unset the env vars the modulefile set
# (HIPBLASLT_TENSILE_LIBPATH, HIPBLASLT_OVERLAY) and post-run we
# report what's still in module list / what the env vars are now, so
# the operator can see whether the unload actually took.

# Enable light hipBLASLt logging so we can post-hoc see WHICH .so /
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

# Detect GPU arch for diagnostic context. The Origami INSTRUCTION_MAP is
# per-arch, so a future regression that lands a gfx950 entry but skips
# gfx942 (or vice versa) wants a clear "this is the arch we tested"
# label in the output.
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

   # Resolved-library diagnostic. The PASS verdict is ambiguous
   # between "no Origami gap exists" and "Origami wasn't engaged at
   # all" unless we can also show WHICH library was loaded and that
   # hipBLASLt actually did some work. We print three things:
   #   1. The [initialize] line, if found (names the library loaded).
   #   2. The current value of HIPBLASLT_TENSILE_LIBPATH (which the
   #      patched modulefile sets and our `unset` may have failed to
   #      clear).
   #   3. The first 5 lines of the log unconditionally, so even when
   #      hipBLASLt logs at a different level / with different tags
   #      the operator can see what was captured.
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
