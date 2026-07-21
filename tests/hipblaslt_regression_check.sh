#!/bin/bash

# Note, this test is checking the default track
# which for MI300A is the case when TENSILE_SOLUTION_SELECTION_METHOD=0
# NOTE that on MI350, the TENSILE_SOLUTION_SELECTION_METHOD flag has no effect
# and StreamK is always the track that is picked
# For details: https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/reference/env-variables.html#origami-with-stream-k-configuration

# Fast regression check for hipBLASLt heuristic-fall-through bugs.
# Exercises the three GEMM families (forward + 2
# backward variants) of `nn.Linear(hidden, num_classes)` over a small
# (batch_size, num_classes, hidden) sweep, captures the hipBLASLt API
# log, and flags any `returnAlgoCount=0` event.
#
#
# Modes (HIPBLASLT_REGRESS_SWEEP env var):
#   quick   (default)  12 configs / 36 GEMM tuples, ~10 s.
#   full               120 configs / 360 GEMM tuples, ~80 s.
#
# Datatypes (HIPBLASLT_REGRESS_DTYPES env var, comma-separated):
#   fp16,fp32,bf16   (default)
#
# Each dtype consults a DIFFERENT Tensile .dat library on disk and a
# DIFFERENT solution catalogue:
#   fp16  -> TensileLibrary_HH_HH_..._gfx942.dat
#   fp32  -> TensileLibrary_SS_SS_..._gfx942.dat
#   bf16  -> TensileLibrary_BB_BB_..._gfx942.dat
#
# fp64 is INTENTIONALLY EXCLUDED: it is not normally dispatched
# through hipBLASLt (parser sees zero rocblaslt_matmul_algo_get_heuristic
# events), and every shipped TensileLibrary_DD_DD_..._gfx942.dat is a
# single TruePred->Matching catch-all row that cannot return zero
# solutions anyway. Either reason alone makes returnAlgoCount=0
# unreachable, so fp64 would PASS vacuously.
# fp32 stays: the API IS exercised, and some SS_SS placeholders ship
# with EqualityMatching-only rows (no catch-all) where a real miss
# is structurally possible.
#
# Detection is architecture-agnostic: any (M, N, K, tA, tB) tuple for
# which hipBLASLt's heuristic returns zero solutions counts as a
# regression on whatever arch the run is on. The shape sweep (i.e. what shapes are checked)
# is arch-aware -- the script reads `rocminfo` and picks the (bs, nc,
# hidden) ranges that match what could be a production workload pattern of the
# target arch:
#     gfx942 (CDNA3, MI300A/X)  : ResNet-style FC shapes
#                                 -- centered on (256, 100, 2048),
#                                 the single_process.sh point that
#                                 surfaced the original 2026-05 bug.
#     gfx950 (CDNA4, MI355X)    : Llama-/Mistral-style FFN + output
#                                 projection shapes (bs = tokens,
#                                 nc in {ffn-dim, vocab-tile},
#                                 hidden in {4096, 8192}).
#     other / unknown           : small blended fallback (warns).
# Override the auto-detected arch with HIPBLASLT_REGRESS_ARCH=gfx95X
# (useful for dry-running the gfx950 sweep on a gfx942 node).
#
# Why these shapes? On gfx942 (MI300A) the original bug shows up as
# 3 specific (M, N, K) tuples per training step:
#     (100,  256, 2048) tA=T tB=N  -- forward
#     (2048, 256,  100) tA=N tB=N  -- backward grad
#     (2048, 100,  256) tA=N tB=T  -- backward weight
# These aren't synthetic / corner-case shapes -- they're the canonical
# signature of a CNN classifier head: nn.Linear(features=2048,
# num_classes=N) at batch=B. The 2048 comes from the ResNet-50/-101/
# -152 backbone output channels after AvgPool (He et al. 2016, "Deep
# Residual Learning for Image Recognition", Table 1; torchvision impl:
# `torchvision/models/resnet.py` -> `self.fc = nn.Linear(512 *
# block.expansion, num_classes)` with `block.expansion=4` for
# Bottleneck = 2048). N=100 is CIFAR-100; the same head shape applies
# to ImageNet (N=1000), all classifier-head ViTs, and most fine-tuning
# pipelines. So the patched + probed shapes are the bread-and-butter
# of CV transfer learning, not an esoteric corner.
#
# Runtime: ~10 s on a healthy / patched system, regardless of partition
# (SPX or CPX). On a regressed system each broken shape adds ~6 s of
# Tensile dynamic-compile stall, which makes the test slower and is
# itself a signal to inspect the failing-shapes list it prints.
#
# NOTE: this test assumes PyTorch has been installed according
# to the instructions available in the model installation repo:
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

if ! python3 -c "import torch" 2>/dev/null; then
   echo "REGRESSION CHECK: SKIPPED (pytorch module not available or torch import failed)"
   exit 77
fi

# Self-diagnostic header
echo
echo "--- hipBLASLt regression-check environment ---"
echo "  HIPBLASLT_TENSILE_LIBPATH = ${HIPBLASLT_TENSILE_LIBPATH:-<unset, will use default <rocm>/lib/hipblaslt/library>}"
echo "  HIPBLASLT_OVERLAY         = ${HIPBLASLT_OVERLAY:-<unset>}"
echo "  ROCM_PATH                 = ${ROCM_PATH:-<unset>}"
if type module >/dev/null 2>&1; then
   echo "  loaded modules            =" $(module -t list 2>&1 | tr '\n' ' ')
fi
echo "----------------------------------------------"
echo

# HIPBLASLT_REGRESS_ARCH override: changes which SHAPE FAMILY is
# probed, NOT which .dat library hipBLASLt opens. The .dat library is
# selected by the actual hardware at runtime -- setting
# HIPBLASLT_REGRESS_ARCH=gfx950 on a gfx942 node will probe Llama-style
# shapes against the gfx942 heuristic library (a useful but distinct
# question), not against a gfx950 library that doesn't exist on that
# node. To actually test a gfx950 .dat library, you need gfx950
# hardware.
if [ -n "${HIPBLASLT_REGRESS_ARCH:-}" ]; then
   GFX_ARCH="${HIPBLASLT_REGRESS_ARCH}"
elif command -v rocminfo >/dev/null 2>&1; then
   GFX_ARCH=$(rocminfo 2>/dev/null \
      | grep -E '^\s*Name:\s*gfx' \
      | sed -e 's/^\s*Name:\s*//' -e 's/\s*$//' -e 's/:.*//' \
      | sort -u | head -1)
fi
GFX_ARCH=${GFX_ARCH:-unknown}
echo "detected GPU arch: ${GFX_ARCH}"

# hipBLASLt LOG_LEVEL semantics (NOT a simple verbosity counter --
# higher levels are bitmask-progressive over distinct log channels):
#     1 -> ERROR
#     2 -> ERROR + TRACE
#     3 -> ERROR + TRACE + HINTS
#     4 -> + INFO            <- [Info][initialize] line lives here
#     5 -> + API             <- [Api][rocblaslt_matmul_algo_get_heuristic]
#                                returnAlgoCount=N lines live here
# We need BOTH the API channel (to see heuristic misses) and the INFO
# channel (to confirm which .dat library was opened), so LEVEL=5 is
# the minimum that gives the parser anything to work with. 
export HIPBLASLT_LOG_LEVEL=5

# Sweep selector:
#
#   quick   (default): 12 configs around the canonical workload point.
#           Detects ANY heuristic miss in that neighbourhood --
#           whether it's a patched shape that broke or a new shape we
#           never patched. The test is intentionally STRICTER than
#           the current overlay's coverage, so a FAIL on a patched
#           system is itself useful signal: it means the upstream
#           gap is wider than what we've patched and the overlay
#           should be extended. Don't dilute this by narrowing it
#           just to satisfy CTest -- the failures are real.
#
#   full    120 configs. Same intent as `quick`, maximum coverage.
SWEEP=${HIPBLASLT_REGRESS_SWEEP:-full}

# Datatypes to sweep. Each dtype consults an INDEPENDENT Tensile .dat
# library on disk, so each needs INDEPENDENT regression coverage:
# a passing fp16 run says nothing about whether the fp32 or bf16
# equality tables have the same shape family populated. Default covers
# the three dtypes that actually dispatch through hipBLASLt for
# nn.Linear(bias=True) on gfx942 / rocm-7.x.
#
DTYPES_RAW=${HIPBLASLT_REGRESS_DTYPES:-fp16,fp32,bf16}

OVERALL_FAIL=0
PER_DTYPE_RESULT=""
LOGS_TO_CLEAN=""
cleanup_logs() {
   if [ -n "${LOGS_TO_CLEAN}" ]; then
      # shellcheck disable=SC2086
      rm -f ${LOGS_TO_CLEAN}
   fi
}
trap cleanup_logs EXIT

# Single-dtype run: takes the dtype label (fp16|fp32) and re-runs the
# Python harness + parser pipeline against its own log file. Most of
# the body is identical to the pre-2026-05-18 single-dtype script;
# the only change is the dtype is passed through as argv[3] of the
# Python harness so .half() vs .float() can be selected, and the log
# is per-dtype so the parser's miss list isn't cross-contaminated.
run_dtype_sweep() {
   local DTYPE="$1"

   local LOG
   LOG=$(mktemp -t "hipblaslt_regress_${DTYPE}.XXXXXX.log")
   LOGS_TO_CLEAN="${LOGS_TO_CLEAN} ${LOG}"
   export HIPBLASLT_LOG_FILE=${LOG}

   echo
   echo "=========================================================="
   echo "  REGRESSION SWEEP  dtype=${DTYPE}  sweep=${SWEEP}  arch=${GFX_ARCH}"
   echo "=========================================================="

   python3 - "${SWEEP}" "${GFX_ARCH}" "${DTYPE}" <<'PY'
import os, sys, time, itertools
import torch
import torch.nn as nn

sweep = sys.argv[1] if len(sys.argv) > 1 else "quick"
arch  = sys.argv[2] if len(sys.argv) > 2 else "unknown"
dtype_label = sys.argv[3] if len(sys.argv) > 3 else "fp16"
_DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp64": torch.float64,
}
if dtype_label not in _DTYPE_MAP:
    print(
        f"HIPBLASLT_REGRESSION_CHECK: unknown dtype '{dtype_label}' "
        f"(expected one of: {', '.join(sorted(_DTYPE_MAP))})",
        file=sys.stderr,
    )
    sys.exit(2)
torch_dtype = _DTYPE_MAP[dtype_label]
if not torch.cuda.is_available():
    print("HIPBLASLT_REGRESSION_CHECK: no GPU available, skipping", file=sys.stderr)
    sys.exit(77)  # CTest "skip" convention

# Per-arch sweep ranges. The (M, N, K) tuples that fall out of
# nn.Linear(hidden, num_classes) forward+backward on batch=bs are:
#     fwd:        (num_classes, bs, hidden)
#     bwd grad:   (hidden, bs, num_classes)
#     bwd weight: (hidden, num_classes, bs)
# Pick (bs, num_classes, hidden) ranges that cover the production
# workload pattern for the target arch.
if arch == "gfx942":
    # CDNA3 (MI300A/X): ResNet-final-FC pattern (original regression).
    # The sweep is intentionally wider than the existing overlay's
    # coverage so the test surfaces upstream-heuristic family gaps,
    # not just the 3 cells the overlay patches.
    if sweep == "full":
        batches = [32, 64, 128, 256, 512, 1024]
        classes = [10, 100, 256, 1000, 2048]
        hiddens = [512, 1024, 2048, 4096]
    else:                                       # quick
        batches = [64, 128, 256, 512]
        classes = [10, 100, 1000]
        hiddens = [2048]
elif arch == "gfx950":
    # CDNA4 (MI355X): LLM-style FFN / attention output projection
    # shapes. hidden = model dim, nc = ffn-dim or vocab-tile, bs =
    # tokens/microbatch. Sized to span Llama-3-8B (hidden=4096),
    # Llama-3-70B (hidden=8192), Mistral-7B (hidden=4096).
    if sweep == "full":
        batches = [1, 8, 32, 128, 1024, 4096]
        classes = [4096, 8192, 11008, 14336, 32768]
        hiddens = [2048, 4096, 8192]
    else:                                       # quick
        batches = [1, 32, 512, 4096]
        classes = [4096, 11008, 32768]
        hiddens = [4096]
else:
    # Unknown / future arch: small blended fallback.
    print(f"  WARN: unrecognized arch '{arch}', using generic fallback sweep",
          file=sys.stderr)
    batches = [64, 256, 1024]
    classes = [100, 1000, 4096]
    hiddens = [2048]

dev = torch.device("cuda")
configs = list(itertools.product(batches, classes, hiddens))
print(f"arch={arch} sweep={sweep} dtype={dtype_label}: {len(configs)} configurations, "
      f"{len(configs)*3} GEMM (M,N,K) tuples")
print(f"  batches={batches}")
print(f"  classes={classes}")
print(f"  hiddens={hiddens}")
print(f"  device     = {torch.cuda.get_device_name(0)}")
print(f"  torch.hip  = {torch.version.hip}")

torch.manual_seed(0)
# Warmup at the canonical single_process.sh point (bs=256, nc=100,
# hd=2048). We pick this shape specifically because:
#   (a) for fp16 the patched overlay covers it with an exact-match
#       row, so the warmup itself acts as a one-line confirmation
#       that the patch is actually being consulted -- if even this
#       shape misses, the env said the patch was loaded but it
#       wasn't taking effect;
#   (b) it triggers hipBLASLt initialization (emits the
#       `[Info][initialize] Using ...` line into the log) which our
#       library-path diagnostic keys off;
#   (c) using a shape that we KNOW is a heuristic hit (for fp16)
#       means the warmup doesn't pollute the miss list with a
#       synthetic (M=10, N=8, K=64) entry that has nothing to do
#       with the workload under test. For fp32 we keep the same
#       shape for diagnostic parity, even though it may itself be
#       a heuristic miss against the unpatched SS_SS .dat -- the
#       warmup MISS contributing to the count is part of the
#       signal, not noise.
_warm = nn.Linear(2048, 100).to(dtype=torch_dtype, device=dev)
_warm(torch.randn(256, 2048, dtype=torch_dtype, device=dev)).sum().backward()
torch.cuda.synchronize()

t0 = time.time()
for bs, nc, hd in configs:
    net = nn.Linear(hd, nc).to(dtype=torch_dtype, device=dev)
    x   = torch.randn(bs, hd, dtype=torch_dtype, device=dev)
    t   = time.time()
    out = net(x)
    out.sum().backward()
    torch.cuda.synchronize()
    dt  = time.time() - t
    print(f"  [{dtype_label}] bs={bs:>4} nc={nc:>4} hd={hd:>4}  fwd+bwd {dt*1000:>7.1f} ms")
print(f"total compute time: {time.time()-t0:.2f} s")
PY
   local RC_PY=$?

   if [ ${RC_PY} -eq 77 ]; then
      echo "REGRESSION CHECK [dtype=${DTYPE}]: SKIPPED (no GPU)"
      return 77
   fi
   if [ ${RC_PY} -ne 0 ]; then
      echo "REGRESSION CHECK [dtype=${DTYPE}]: FAILED (python harness rc=${RC_PY})"
      return 1
   fi

   # Sanity-check the log file is non-empty BEFORE running the parser.
   echo
   echo "--- hipBLASLt log sanity check [dtype=${DTYPE}] ---"
   echo "  log file       = ${LOG}"
   if [ -s "${LOG}" ]; then
      local LOG_BYTES LOG_LINES API_CALLS
      LOG_BYTES=$(stat -c %s "${LOG}" 2>/dev/null || echo "?")
      LOG_LINES=$(wc -l < "${LOG}")
      echo "  size           = ${LOG_BYTES} bytes, ${LOG_LINES} lines"
      echo "  unique tags    =" $(grep -oE '\[(Error|Trace|Hints|Info|Api|Profile)\]' "${LOG}" | sort -u | tr '\n' ' ')
      API_CALLS=$(grep -c 'rocblaslt_matmul_algo_get_heuristic' "${LOG}" 2>/dev/null || echo 0)
      echo "  algo_get_heuristic calls captured = ${API_CALLS}"
      if [ "${API_CALLS}" -eq 0 ]; then
         echo "  WARN: no algo_get_heuristic events in log -- parser will report"
         echo "        zero misses regardless of true state. Check HIPBLASLT_LOG_LEVEL."
      fi
   else
      echo "  log file is empty or missing -- hipBLASLt logging is not configured."
      echo "  Expected env: HIPBLASLT_LOG_LEVEL=${HIPBLASLT_LOG_LEVEL:-<unset>},"
      echo "                HIPBLASLT_LOG_FILE=${HIPBLASLT_LOG_FILE:-<unset>}."
   fi

   # Extract the authoritative library path from the hipBLASLt init line.
   echo
   echo "--- hipBLASLt library path actually used (from log) [dtype=${DTYPE}] ---"
   local INIT_LINE
   INIT_LINE=$(grep -m1 'initialize' "${LOG}" 2>/dev/null)
   if [ -n "${INIT_LINE}" ]; then
      echo "  ${INIT_LINE}"
      if echo "${INIT_LINE}" | grep -q 'rocm-patches'; then
         echo "  -> overlay ACTIVE (patched .dat files in use)"
      else
         echo "  -> overlay NOT in use (default upstream .dat files)"
      fi
   else
      echo "  (no [initialize] line in log)"
      # If we're at LOG_LEVEL>=4 (so Info channel is on) and the file is
      # non-empty but still has no init line, that's worth flagging: it
      # may mean hipBLASLt never actually ran (e.g. all GEMMs went via
      # rocBLAS) which is a third class of false PASS.
      if [ "${HIPBLASLT_LOG_LEVEL:-0}" -ge 4 ] && [ -s "${LOG}" ]; then
         echo "  WARN: log is non-empty + LOG_LEVEL=${HIPBLASLT_LOG_LEVEL} >= 4 (INFO on),"
         echo "        yet no [initialize] line. hipBLASLt may not have been"
         echo "        invoked by the probe -- check first 5 log lines:"
         head -5 "${LOG}" | sed 's/^/        /'
      fi
   fi
   echo "-------------------------------------------------------"
   echo

   # Parse the log: every `returnAlgoCount=0` is a heuristic miss.  For
   # each miss, reconstruct (M, N, K) from the 3 layout_create lines that
   # immediately precede it. 
   local MISSES RC_PARSE
   MISSES=$(python3 - "${LOG}" <<'PY'
import re, sys, os
log = sys.argv[1]
re_lay  = re.compile(r'rocblaslt_matrix_layout_create.*rows=(\d+) cols=(\d+) ld=\d+')
re_tA   = re.compile(r'MATMUL_DESC_TRANSA.*bufData=0x([0-9a-fA-F]+)')
re_tB   = re.compile(r'MATMUL_DESC_TRANSB.*bufData=0x([0-9a-fA-F]+)')
re_miss = re.compile(r'rocblaslt_matmul_algo_get_heuristic.*returnAlgoCount=(\d+)')
tA = tB = None
layouts = []
seen = {}
if not os.path.exists(log):
    sys.exit("log file not found")
with open(log) as f:
    for ln in f:
        m = re_tA.search(ln);  ta = int(m.group(1),16) if m else None
        m = re_tB.search(ln);  tb = int(m.group(1),16) if m else None
        if ta is not None: tA = ta
        if tb is not None: tB = tb
        m = re_lay.search(ln)
        if m:
            layouts.append((int(m.group(1)), int(m.group(2))))
            if len(layouts) > 3: layouts = layouts[-3:]
        m = re_miss.search(ln)
        if m and int(m.group(1)) == 0 and len(layouts) >= 3:
            a, b, c = layouts[-3:]
            M, N = c
            K = a[0] if a[1] == M else a[1] if a[0] == M else max(a)
            op = lambda x: 'N' if x == 0x6f else 'T' if x == 0x70 else f'0x{x:02x}' if x else '?'
            key = (M, N, K, op(tA), op(tB))
            seen[key] = seen.get(key, 0) + 1
            layouts = []
for (M,N,K,a,b), n in sorted(seen.items()):
    print(f"  MISS x{n}  M={M:<5} N={N:<5} K={K:<5} tA={a} tB={b}")
sys.exit(0 if not seen else 1)
PY
)
   RC_PARSE=$?

   echo "${MISSES}"

   if [ ${RC_PARSE} -eq 0 ]; then
      echo "REGRESSION CHECK [dtype=${DTYPE}]: PASSED  (zero heuristic misses across the sweep)"
      return 0
   else
      echo "REGRESSION CHECK [dtype=${DTYPE}]: FAILED  (heuristic misses listed above)"
      echo "  -> rebuild / extend the hipblaslt/patched overlay for these shapes."
      echo "  -> patcher: hpctd repo, rocm/scripts/hipblaslt_patch_setup.sh"
      return 1
   fi
}

# Main dtype loop: comma-separated DTYPES_RAW -> "fp16 fp32" etc.
IFS=',' read -r -a DTYPES_ARR <<< "${DTYPES_RAW}"
ALL_SKIP=1
for DT in "${DTYPES_ARR[@]}"; do
   DT_TRIM=$(echo "${DT}" | tr -d '[:space:]')
   case "${DT_TRIM}" in
      fp16|fp32|bf16|fp64) ;;
      *) echo "REGRESSION CHECK: unknown dtype '${DT_TRIM}' (expected one of: fp16, fp32, bf16, fp64)"
         OVERALL_FAIL=1
         PER_DTYPE_RESULT="${PER_DTYPE_RESULT}  ${DT_TRIM}: BAD_DTYPE\n"
         continue;;
   esac
   run_dtype_sweep "${DT_TRIM}"
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
echo "  REGRESSION CHECK summary (dtypes = ${DTYPES_RAW})"
echo "=========================================================="
echo -en "${PER_DTYPE_RESULT}"
echo

if [ "${ALL_SKIP}" -eq 1 ]; then
   echo "REGRESSION CHECK: SKIPPED  (every dtype reported no-GPU)"
   exit 77
fi
if [ ${OVERALL_FAIL} -eq 0 ]; then
   echo "REGRESSION CHECK: PASSED  (zero heuristic misses across every dtype)"
   exit 0
else
   echo "REGRESSION CHECK: FAILED  (one or more dtypes had heuristic misses; see per-dtype list above)"
   exit 1
fi
