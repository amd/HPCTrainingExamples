#!/bin/bash
# Regression test for cupy/cupy#9742 (HIP __shfl_*_sync 32-bit-mask bug).
#
# What this test does
# -------------------
# 1. Forces a cold kernel cache (CUPY_CACHE_DIR -> fresh tempdir) so we
#    actually exercise the HIPRTC compile path; a warm cache would mask
#    the bug.
# 2. Exercises three independent entry points into the affected scan
#    kernel on arrays large enough to require multi-block scans:
#       * ``arr[bool_mask]`` (the upstream issue's reproducer)
#       * ``cp.cumsum(arr)``
#       * ``cp.nonzero(arr)``
# 3. Synchronizes the stream after each call so any HIPRTC compile
#    error surfaces inline (CuPy lazily JITs on first call).
# 4. Checks correctness against the obvious NumPy ground truth.
# 5. Prints exactly ``Success`` on pass (matches PASS_REGULAR_EXPRESSION
#    in tests/CMakeLists.txt) or ``Failure: <reason>`` on any failure.
#
# Refs:  https://github.com/cupy/cupy/issues/9742  (bug report)
#        https://github.com/cupy/cupy/pull/9748    (fix, merged 2026-04-28)
#        https://github.com/cupy/cupy/pull/9897    (v14 backport)
#
# This test must run on a GPU compute node.  Same constraint and same
# module-loading pattern as the existing cupy_check_array_sum.sh /
# cupy_check_gpu_access.sh tests in this directory.
#
# NOTE: this test assumes CuPy has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/cupy_setup.sh

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   if [ -z "$CXX" ]; then
      export CXX=`which CC`
   fi
   if [ -z "$CC" ]; then
      export CC=`which cc`
   fi
   if [ -z "$FC" ]; then
      export FC=`which ftn`
   fi
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
fi

module load cupy

python3 - <<'__CUPY_REPRO_PYEOF__'
import os, sys, tempfile, traceback

# Force a cold kernel cache so JIT compile actually runs.  CuPy reads
# CUPY_CACHE_DIR at import time -- this MUST be set before `import cupy`.
_cache_dir = tempfile.mkdtemp(prefix="cupy_hip_shfl_mask_cache_")
os.environ["CUPY_CACHE_DIR"] = _cache_dir

try:
    try:
        import numpy as np
        import cupy as cp
    except Exception as exc:
        print(f"Failure: cupy/numpy import failed: {exc!r}")
        sys.exit(1)

    try:
        n_dev = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        print(f"Failure: no CUDA/HIP device available: {exc!r}")
        sys.exit(1)
    if n_dev < 1:
        print("Failure: cupy reports zero GPUs")
        sys.exit(1)

    is_hip = bool(getattr(cp.cuda.runtime, "is_hip", False)) or \
             bool(getattr(cp.cuda.runtime, "_is_hip_environment", False))
    print(f"[cupy_hip_shfl_mask] cupy={getattr(cp, '__version__', '?')} "
          f"hip={'yes' if is_hip else 'no'} cache_dir={_cache_dir}")

    # ── Step 1: arr[bool_mask] -- the upstream issue's exact reproducer ──
    def _exercise_boolean_indexing():
        n = 2048
        x = cp.arange(n, dtype=cp.int32)
        mask = (x & 1) == 0
        y = x[mask]
        cp.cuda.Stream.null.synchronize()
        expected = np.arange(0, n, 2, dtype=np.int32)
        got = cp.asnumpy(y)
        if got.shape != expected.shape or not (got == expected).all():
            raise AssertionError(
                f"boolean indexing produced wrong result: "
                f"shape={got.shape} (expected {expected.shape}), "
                f"first 5: {got[:5].tolist()} (expected {expected[:5].tolist()})"
            )

    # ── Step 2: cumsum -- goes through _cupy_bsum_shfl scan kernel ──
    def _exercise_cumsum():
        n = 4096
        x = cp.ones(n, dtype=cp.int64)
        y = cp.cumsum(x)
        cp.cuda.Stream.null.synchronize()
        expected_last = n
        got_last = int(cp.asnumpy(y[-1]))
        if got_last != expected_last:
            raise AssertionError(
                f"cumsum produced wrong tail value: {got_last} != {expected_last}"
            )

    # ── Step 3: nonzero -- invokes _nonzero_kernel_incomplete_scan ──
    def _exercise_nonzero():
        n = 4096
        x = cp.zeros(n, dtype=cp.int32)
        x[::2] = 1
        (idx,) = cp.nonzero(x)
        cp.cuda.Stream.null.synchronize()
        expected = np.arange(0, n, 2, dtype=idx.dtype.type)
        got = cp.asnumpy(idx)
        if got.shape != expected.shape or not (got == expected).all():
            raise AssertionError(
                f"nonzero produced wrong indices: "
                f"shape={got.shape} (expected {expected.shape}), "
                f"first 5: {got[:5].tolist()} (expected {expected[:5].tolist()})"
            )

    def _classify_compile_error(exc):
        """If exc looks like the cupy#9742 HIPRTC mask error, return a tag."""
        text = str(exc) + "\n" + "".join(
            traceback.format_exception_only(type(exc), exc))
        needles = (
            "The mask must be a 64",       # hiprtc_runtime.h static_assert
            "sizeof(MaskT) == 8",          # static_assert source line
            "__shfl_xor_sync",
            "__shfl_up_sync",
            "__shfl_down_sync",
        )
        if any(n in text for n in needles):
            return "HIPRTC rejected 32-bit mask in __shfl_*_sync (cupy#9742)"
        return None

    steps = (
        ("boolean indexing arr[mask]", _exercise_boolean_indexing),
        ("cumsum",                     _exercise_cumsum),
        ("nonzero",                    _exercise_nonzero),
    )
    for label, fn in steps:
        try:
            fn()
        except Exception as exc:
            tag = _classify_compile_error(exc)
            if tag is not None:
                print(f"Failure: {label}: {tag}")
            else:
                print(f"Failure: {label}: {exc!r}")
            traceback.print_exc()
            sys.exit(1)
        print(f"[cupy_hip_shfl_mask]   {label}: ok")

    print("Success")
    sys.exit(0)
finally:
    try:
        import shutil
        shutil.rmtree(_cache_dir, ignore_errors=True)
    except Exception:
        pass
__CUPY_REPRO_PYEOF__
_rc=$?

module unload cupy

exit ${_rc}
