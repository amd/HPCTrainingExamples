#!/bin/bash

# This test checks that vLLM has been built with ROCm support and
# resolves to the ROCm platform at runtime (RocmPlatform / is_rocm),
# rather than silently falling back to UnspecifiedPlatform (which would
# import fine but never select the AMD GPU backend).

# NOTE: this test assumes vLLM has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/vllm_setup.sh


module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
# The vllm modulefile prereqs rocm and loads its bound pytorch itself.
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

# Best-effort device name so a manual runner sees the actual GPU vLLM will use.
try:
    print("vLLM device name    : %s" % current_platform.get_device_name(0))
except Exception as e:
    print("vLLM device name    : (unavailable: %s)" % e)

# vLLM resolves the ROCm backend via the amdsmi plugin; if that is missing it
# silently falls back to UnspecifiedPlatform -- imports fine but never uses the
# AMD GPU. is_rocm / RocmPlatform is the canonical "ROCm-enabled" signal.
if is_rocm or name == "RocmPlatform":
    print("RESULT: vLLM is ROCm-enabled (%s)" % name)
else:
    print("RESULT: vLLM is NOT ROCm-enabled -- fell back to %s (no AMD GPU backend)" % name)
    sys.exit(1)
EOF
