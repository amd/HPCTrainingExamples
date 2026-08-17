#!/bin/bash

# This test runs the numba-hip example to verify
# numba-hip is installed and can execute kernels on AMD GPUs

# NOTE: this test assumes numba-hip has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/hip-python_setup.sh

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

module load hip-python

# numba-hip JIT-compiles kernels by linking a COMGR-generated device
# library (built with the ROCm SDK's LLVM) using the LLVM bundled in the
# rocm-llvm-python wheel. When the SDK's LLVM is newer than the wheel's,
# that link fails and there is no installable matching wheel. The
# hip-python modulefile sets NUMBA_HIP_DEVICE_JIT_UNSUPPORTED=1 in that
# case; skip rather than hard-fail (CTest matches the message below via
# SKIP_REGULAR_EXPRESSION).
if [ "${NUMBA_HIP_DEVICE_JIT_UNSUPPORTED:-0}" = "1" ]; then
   echo "SKIP: numba-hip device JIT unsupported (SDK LLVM newer than rocm-llvm-python LLVM)"
   exit 0
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
EXAMPLE_DIR="$REPO_DIR/Python/hip-python"
WORK_DIR=$(mktemp -d)

cleanup() {
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

# Copy files to work directory
cp "$EXAMPLE_DIR/numba-hip.py" "$WORK_DIR/"

cd "$WORK_DIR"

python3 ./numba-hip.py 2>/dev/null | grep -q 'PASSED' && echo 'Success' || echo 'Failure'
