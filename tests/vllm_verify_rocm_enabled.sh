#!/bin/bash

# This test checks that vLLM is ROCm-enabled: it resolves to the ROCm
# platform (RocmPlatform / is_rocm) and its compiled kernels load.

# NOTE: this test assumes vLLM has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/vllm_setup.sh


module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load vllm

python3 <<EOF
import sys

try:
    from vllm.platforms import current_platform
except Exception as e:
    print("ERROR: could not import vllm.platforms:", e)
    sys.exit(1)

name = type(current_platform).__name__

is_rocm_attr = getattr(current_platform, "is_rocm", None)
try:
    is_rocm = is_rocm_attr() if callable(is_rocm_attr) else bool(is_rocm_attr)
except Exception:
    is_rocm = False

print("vLLM platform class : %s" % name)
print("vLLM is_rocm        : %s" % is_rocm)

try:
    print("vLLM device name    : %s" % current_platform.get_device_name(0))
except Exception as e:
    print("vLLM device name    : (unavailable: %s)" % e)

# Compiled kernels are ABI-locked to torch; import them so a mismatch fails
# the test instead of being a non-fatal warning inside vLLM.
compiled_ok = True
for mod in ("vllm._C", "vllm._rocm_C"):
    try:
        __import__(mod)
        print("vLLM compiled ops   : %s loaded OK" % mod)
    except Exception as e:
        compiled_ok = False
        print("vLLM compiled ops   : %s FAILED to load: %s" % (mod, e))

rocm_platform = is_rocm or name == "RocmPlatform"

if rocm_platform and compiled_ok:
    print("RESULT: vLLM is ROCm-enabled (%s)" % name)
else:
    if not rocm_platform:
        print("RESULT: vLLM is NOT ROCm-enabled -- fell back to %s (no AMD GPU backend)" % name)
    else:
        print("RESULT: vLLM is NOT ROCm-enabled -- %s selected but its compiled ROCm kernels "
              "failed to load (torch/vLLM ABI mismatch)" % name)
    sys.exit(1)
EOF
