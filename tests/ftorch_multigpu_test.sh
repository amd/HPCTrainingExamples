#!/bin/bash

# This test runs the multi gpu test from FTorch 

# NOTE: this test assumes FTorch has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/ftorch_setup.sh

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  # NOTE: if the default rocm module bus-errors on `import torch` /
  # `rocm-smi` (observed with rocm/7.2.0 on MI300A as of 2026-05),
  # pre-load a known-working rocm version (e.g. `module load rocm/7.0.2`)
  # before invoking this test.
  module load rocm
fi
module load ftorch

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
python3 pt2ts.py --device_type hip
ln -sf saved_multigpu_model_hip.pt torchscript_multigpu_model_hip.pt
python3 multigpu_infer_python.py --device_type hip
mkdir build && cd build
cmake ..
make -j
./multigpu_infer_fortran hip ../saved_multigpu_model_hip.pt
deactivate
