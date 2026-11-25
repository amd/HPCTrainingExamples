#!/bin/bash

# Smoke test for the instructions in MLExamples/README_apptainer.md.
# Usage: apptainer_ml_test.sh <case>
#
# Runs a short example inside an Apptainer container the way the README describes
# (--cleanenv, venv with --system-site-packages plus a .pth file pointing to the
# image's PyTorch) and prints "Success" if the example finishes.
#
# The image is not pulled by the test (conversion is slow and memory hungry). Pull it once
# as described in the README and point APPTAINER_SIF to it (default: $HOME/rocm-pytorch.sif).
# The test is skipped (exit 77) if apptainer, a GPU or the image is missing.

CASE=$1
SIF=${APPTAINER_SIF:-$HOME/rocm-pytorch.sif}
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

skip() {
  echo "SKIPPED: $1"
  exit 77
}

case "${CASE}" in
  tinytransformer_v1)
    TOP=TinyTransformer; SUB=version1_pytorch_baseline
    CMD="python3 tiny_llama_v1.py --batch-size 4 --seq-len 128 --num-layers 2 --num-steps 5 --profile-dir ./out"
    SENTINEL="completed successfully" ;;
  tinytransformer_v2)
    TOP=TinyTransformer; SUB=version2_pytorch_fused
    CMD="python3 tiny_llama_v2.py --batch-size 4 --seq-len 128 --num-layers 2 --num-steps 5 --profile-dir ./out"
    SENTINEL="completed successfully" ;;
  tinytransformer_v3)
    TOP=TinyTransformer; SUB=version3_triton
    CMD="python3 tiny_llama_v3.py --batch-size 4 --seq-len 128 --num-layers 2 --num-steps 5"
    SENTINEL="completed successfully" ;;
  tinytransformer_v4)
    TOP=TinyTransformer; SUB=version4_pytorch_sdpa
    CMD="python3 tiny_llama_v4.py --batch-size 4 --seq-len 128 --num-layers 2 --num-steps 5"
    SENTINEL="completed successfully" ;;
  tinyopenfold_v1)
    TOP=TinyOpenFold; SUB=version1_pytorch_baseline
    CMD="python3 tiny_openfold_v1.py --num-blocks 2 --batch-size 2 --num-steps 5 --no-data-parallel --profile-dir ./out"
    SENTINEL="completed successfully" ;;
  tinyopenfold_v2)
    TOP=TinyOpenFold; SUB=version2_pytorch_fused
    CMD="python3 tiny_openfold_v2.py --num-blocks 2 --batch-size 2 --num-steps 5 --profile-dir ./out"
    SENTINEL="completed successfully" ;;
  tinyopenfold_v3)
    TOP=TinyOpenFold; SUB=version3_triton
    CMD="python3 tiny_openfold_v3.py --num-blocks 2 --batch-size 2 --num-steps 5"
    SENTINEL="completed successfully" ;;
  tinyopenfold_v3_correctness)
    TOP=TinyOpenFold; SUB=version3_triton
    CMD="python3 test_correctness.py"
    SENTINEL="All tests PASSED" ;;
  inference_benchmark)
    TOP=inference_benchmark; SUB=.
    CMD="python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 16 --iterations 5"
    SENTINEL="OK: finished running benchmark" ;;
  *)
    echo "Failure: unknown case '${CASE}'"
    exit 1 ;;
esac

command -v apptainer >/dev/null 2>&1 || skip "apptainer not found"
[ -e /dev/kfd ] || skip "no AMD GPU (/dev/kfd) on this node"
[ -f "${SIF}" ] || skip "container image ${SIF} not found (see MLExamples/README_apptainer.md)"

SCRATCH=$(mktemp -d)
trap "rm -rf ${SCRATCH}" EXIT

# Work on a copy so that outputs and the venv do not end up in the repository
cp -r "${REPO_DIR}/MLExamples/${TOP}" "${SCRATCH}/"
mkdir -p "${SCRATCH}/cache"

# Same steps as in README_apptainer.md, section 3
cat > "${SCRATCH}/inner.sh" <<'EOS'
set -eo pipefail
export IMAGE_SP=$(python3 -c 'import torch, os; print(os.path.dirname(os.path.dirname(torch.__file__)))')
python3 -m venv --system-site-packages --without-pip ./venv-pt
VENV_SP=$(./venv-pt/bin/python3 -c 'import site; print(site.getsitepackages()[0])')
echo "${IMAGE_SP}" > "${VENV_SP}/_container_torch.pth"
source ./venv-pt/bin/activate

if [ -f requirements.txt ]; then
  python3 -m pip install --no-build-isolation -r requirements.txt || { echo "SKIPPED: pip install failed (no network access?)"; exit 77; }
fi

python3 -c '
import os, torch
assert os.path.realpath(os.path.dirname(os.path.dirname(torch.__file__))) == os.path.realpath(os.environ["IMAGE_SP"]), "venv does not use the container PyTorch: " + torch.__file__
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print("Torch", torch.__version__, "HIP", torch.version.hip, "device", torch.cuda.get_device_name(0))
'

eval "${TEST_CMD}" 2>&1 | tee run.log
grep -q "${TEST_SENTINEL}" run.log
echo "Success"
EOS

apptainer exec --cleanenv \
  --bind "${SCRATCH}:/work" --pwd "/work/${TOP}/${SUB}" \
  --env TEST_CMD="${CMD}" --env TEST_SENTINEL="${SENTINEL}" \
  --env TORCH_HOME=/work/cache --env XDG_CACHE_HOME=/work/cache \
  --env TRITON_CACHE_DIR=/work/cache/triton \
  --env MIOPEN_USER_DB_PATH=/work/cache/miopen --env MIOPEN_CUSTOM_CACHE_DIR=/work/cache/miopen \
  --env PYTHONIOENCODING=utf-8 \
  "${SIF}" bash /work/inner.sh
rc=$?

if [ ${rc} -eq 77 ]; then
  exit 77
elif [ ${rc} -ne 0 ]; then
  echo "Failure (rc=${rc})"
  exit ${rc}
fi
