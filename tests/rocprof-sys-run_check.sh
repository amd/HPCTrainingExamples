#!/bin/bash

# This test checks that the rocprof-sys-run
# (formerly omnitrace-run)
# binary exists and it is able to run and complete

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

if command -v rocprof-sys-run &> /dev/null; then
    echo "rocprof-sys-run found at: $(which rocprof-sys-run)"
else
    echo "loading rocprofiler-systems module"
    module load rocprofiler-systems
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
pushd ${REPO_DIR}/HIP/Stream_Overlap/0-Orig/
BUILD_DIR=$(mktemp -d build_XXXXXX)
cd ${BUILD_DIR}
cmake ../
make -j

# rocprof-sys >= v1.10 (ROCm 10.2) defaults to RocPD output (USE_ROCPD=true,
# USE_PERFETTO=false); this check looks for the perfetto tracing-session output,
# so enable the perfetto backend explicitly.
export ROCPROFSYS_USE_PERFETTO=1

rocprof-sys-run -- ./compute_comm_overlap 2

cd ..
rm -rf ${BUILD_DIR}

popd

