#!/bin/bash

# This test runs the multi gpu test from FTorch using the system
# gfortran toolchain. For the AMD-toolchain (amdclang + amdflang)
# variant, see ftorch_multigpu_test_amdflang.sh.

# NOTE: this test assumes FTorch has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/ftorch_setup.sh
#
# NOTE: if the default rocm module bus-errors on `import torch` /
# `rocm-smi` (observed with rocm/7.2.0 and rocm/7.2.1 on MI300A as
# of 2026-05), pre-load a known-working rocm version such as
# `module load rocm/7.0.2` before invoking this test.

# Surface failures: prior versions of this test had no `set -e`, so
# every per-step failure (SIGBUS in pt2ts.py, dispatcher errors in
# multigpu_infer_python.py, exit-1 from multigpu_infer_fortran on a
# single-GPU node) was silently masked because the trailing
# `deactivate` returned 0. The test would report success even when no
# GPU work ever ran. With set -e + pipefail, real failures abort the
# script; the few KNOWN-flaky steps are wrapped in explicit `|| ...`
# blocks so they don't shadow real regressions.
set -eo pipefail

# Make the script robust to being invoked via `bash <path>` (a fresh
# non-interactive subshell where Lmod's `module` shell function may
# not be imported even when exported with `export -f module`). Sourcing
# Lmod's bash init re-defines `module` in this shell.
if ! type module >/dev/null 2>&1; then
   [ -r /etc/profile.d/lmod.sh ]            && . /etc/profile.d/lmod.sh
   [ -r /usr/share/lmod/lmod/init/bash ]    && . /usr/share/lmod/lmod/init/bash
fi

# `set -e + pipefail` is incompatible with the legacy
#     module -t list | grep -q "^rocm"; if [ $? -eq 1 ]; then ...
# idiom because a non-match makes the pipeline fail and `set -e` aborts
# before the if-test runs. Use `if ! <pipeline>` instead -- inside an
# if-condition, set -e is suspended for the tested command.
if ! module -t list 2>&1 | grep -q "^rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load ftorch

# pytorch_setup.sh now exports PYTORCH_ROCM_ARCH from the modulefile.
# As a safety net for older pytorch installs (built before that fix),
# default it to gfx942 (the MI300A on sh5 nodes). Without this,
# find_package(Torch) -> Caffe2Config -> LoadHIP.cmake hard-errors:
#   "No GPU arch specified for ROCm build. Please use PYTORCH_ROCM_ARCH"
# See pytorch_setup.sh modulefile heredoc for the durable fix.
if [ -z "${PYTORCH_ROCM_ARCH:-}" ]; then
   _GFX=""
   if command -v rocminfo >/dev/null 2>&1; then
      _GFX=$(rocminfo 2>/dev/null \
             | grep -E '^\s*Name:\s*gfx' \
             | sed -e 's/^\s*Name:\s*//' -e 's/\s*$//' \
             | sort -u \
             | paste -sd';' - || true)
   fi
   export PYTORCH_ROCM_ARCH="${_GFX:-gfx942}"
   unset _GFX
   echo "ftorch_multigpu_test: exporting PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}"
fi

BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT

cd ${BUILD_DIR}

git clone https://github.com/Cambridge-ICCS/FTorch.git ftorch_test
cd ftorch_test/examples/07_MultiGPU

# Workarounds for upstream FTorch examples/07_MultiGPU bugs (commit a3aac61):
#   1. simplenet.py initialises Linear weights inside `torch.inference_mode()`,
#      which makes PyTorch >=2.9 raise "Inference tensors do not track version
#      counter." when the model is later evaluated by pt2ts.py.
#   2. The example's CMakeLists.txt declares `LANGUAGES Fortran` only, but the
#      transitive Torch->Caffe2->FindHIP dependency reads CMAKE_CXX_COMPILER_ID
#      (unset without CXX), producing: `if given arguments: "STREQUAL" "Clang"`.
#   3. pt2ts.py writes `saved_multigpu_model_<dev>.pt` while
#      multigpu_infer_python.py reads `torchscript_multigpu_model_<dev>.pt`.
sed -i 's/with torch\.inference_mode():/with torch.no_grad():/' simplenet.py
sed -i 's/LANGUAGES Fortran)/LANGUAGES Fortran CXX)/' CMakeLists.txt

python3 -m venv ftorch_test
source ftorch_test/bin/activate
# pt2ts.py: creates saved_multigpu_model_hip.pt -- required by the
# Fortran inference step below. Hard-fail if this step fails (no `||`).
python3 pt2ts.py --device_type hip
ln -sf saved_multigpu_model_hip.pt torchscript_multigpu_model_hip.pt
# multigpu_infer_python.py: pure-Python smoke test independent of FTorch.
# Currently broken on rocm-7.0.2 + pytorch-v2.9.1 with
#   NotImplementedError: Could not run 'aten::empty_strided' with
#                        arguments from the 'HIP' backend
# This is a pytorch dispatcher issue: aten::empty_strided is registered
# under DispatchKey::CUDA (the legacy hip-as-cuda alias path) but not
# under DispatchKey::HIP. The Python TorchScript .to('hip') path uses
# the explicit HIP dispatch key, hitting the gap; the FTorch (libtorch
# C++) path uses CUDA-as-HIP and works fine. See readelf comparison in
# libtorch_hip.so: HIP key has 816 occurrences vs CUDA key 18,276.
# Tracker: this is upstream pytorch+ROCm, not specific to our build.
# Keep going so we still exercise the Fortran build + run, which is
# what this test is really about.
python3 multigpu_infer_python.py --device_type hip || \
   echo "[ftorch_multigpu_test] WARN: multigpu_infer_python.py failed (known pytorch HIP-dispatch-key gap; continuing to Fortran build)"
mkdir build && cd build
cmake ..
make -j
# multigpu_infer_fortran iterates over devices 0..N-1; on a single-GPU
# node (typical sh5 layout) the device-1 attempt errors with
#   [ERROR]: invalid device index 1 for device count
# and the executable exits non-zero. Device 0 still PASSES (the actual
# multigpu coupling check). Capture the rc so we can a) still tear down
# the venv with `deactivate`, and b) consider the device-1-only failure
# a soft failure when at least one device PASSED.
set +e
./multigpu_infer_fortran hip ../saved_multigpu_model_hip.pt | tee fortran_test.out
fortran_rc=${PIPESTATUS[0]}
set -e
deactivate
if [ ${fortran_rc} -ne 0 ]; then
   if grep -q "PASSED" fortran_test.out; then
      echo "[ftorch_multigpu_test] multigpu_infer_fortran exit ${fortran_rc} but PASSED on device 0; treating as success on this single-GPU node"
   else
      echo "[ftorch_multigpu_test] FAIL: multigpu_infer_fortran exit ${fortran_rc}, no PASSED line found" >&2
      exit ${fortran_rc}
   fi
fi
