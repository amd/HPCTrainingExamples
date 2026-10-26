#!/bin/bash

# This test runs the JAX lax_vmap_op_test.py pytest file on the GPU,
# taking the test sources from the same repo and commit as the
# installed jax package

# NOTE: this test assumes JAX has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/jax_setup.sh


module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module -t list 2>&1 | grep -q "^jax"
if [ $? -eq 1 ]; then
  module load jax
fi

BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cd ${BUILD_DIR}

# ROCm/jax source builds are versioned X.Y.Z.devN+<sha>, PyPI releases X.Y.Z
read -r JAX_V JAX_SHA < <(python3 -c 'import jax,re; m=re.fullmatch(r"(\d+\.\d+\.\d+)(?:\.dev\d+\+([0-9a-f]{7,40}))?", jax.__version__); print(*(m.groups("") if m else ("?", "")))')
if [ "${JAX_V}" = "?" ]; then
  echo "Failure: cannot tell which repo jax $(python3 -c 'import jax; print(jax.__version__)') was built from"
  exit 1
fi
if [ -n "${JAX_SHA}" ]; then
  echo "jax ${JAX_V}+${JAX_SHA}: tests from ROCm/jax rocm-jaxlib-v${JAX_V} @ ${JAX_SHA}"
  git clone -q --filter=tree:0 --no-checkout --branch rocm-jaxlib-v${JAX_V} https://github.com/ROCm/jax.git jax || exit 1
  cd jax
  git sparse-checkout set tests build && git -c advice.detachedHead=false checkout -q ${JAX_SHA} || exit 1
else
  echo "jax ${JAX_V}: tests from jax-ml/jax jax-v${JAX_V}"
  git -c advice.detachedHead=false clone -q --depth 1 --branch jax-v${JAX_V} --filter=blob:none --sparse https://github.com/jax-ml/jax.git jax || exit 1
  cd jax
  git sparse-checkout set tests build || exit 1
fi

# appended last so the jax module's numpy/scipy are not shadowed
grep -vE '^\s*(matplotlib|auditwheel|scipy-stubs|google-benchmark)' build/test-requirements.txt > ${BUILD_DIR}/requirements.txt
pip3 install -q --target=${BUILD_DIR}/deps -r ${BUILD_DIR}/requirements.txt || exit 1
export PYTHONPATH=${PYTHONPATH}:${BUILD_DIR}/deps

export JAX_PLATFORMS=rocm
export JAX_SKIP_SLOW_TESTS=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python3 -m pytest -q -n 4 --tb=line tests/lax_vmap_op_test.py > ${BUILD_DIR}/pytest.log 2>&1
rc=$?
npass=$(grep -oE '[0-9]+ passed' ${BUILD_DIR}/pytest.log | tail -1 | cut -d' ' -f1)
tail -n 20 ${BUILD_DIR}/pytest.log
# pytest also returns 0 when every test is skipped
if [ ${rc} -eq 0 ] && [ "${npass:-0}" -gt 0 ]; then
  echo "Success"
else
  echo "Failure (pytest rc=${rc}, passed=${npass:-0})"
fi
