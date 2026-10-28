#!/usr/bin/env bash
# Run INSIDE the ROCm 10 PyTorch container. No PYTHONPATH or sourcing required.
# Usage: bash setup_env_rocm10_pytorch_venv.sh [new-environment-directory]
# Default: $HOME/venvs/physicsnemo-rocm10
# The environment remains dependent on this container's Python and packages.
set -euo pipefail

CONTAINER_PYTHON=/opt/venv/bin/python3
ENV_DIR="${1:-$HOME/venvs/physicsnemo-rocm10}"
if [ "$#" -gt 1 ]; then
    echo 'Usage: bash setup_env_rocm10_pytorch_venv.sh [new-environment-directory]' >&2
    exit 2
fi
if [ ! -x "$CONTAINER_PYTHON" ]; then
    echo 'Run this script inside the ROCm 10 PyTorch container with /opt/venv/bin/python3.' >&2
    exit 1
fi
if [ -e "$ENV_DIR" ]; then
    echo "Destination already exists: $ENV_DIR. Choose a new directory." >&2
    exit 1
fi

# Old PYTHONPATH-based installs must not influence setup or its verification.
# This only changes the installer process, not the calling shell.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

"$CONTAINER_PYTHON" -c '
import torch
assert torch.version.hip and "rocm10.0.0" in torch.__version__, "Expected the ROCm 10 PyTorch image"
print("Container torch:", torch.__version__, "from", torch.__file__)
'

"$CONTAINER_PYTHON" -m venv "$ENV_DIR"
ENV_DIR="$(cd "$ENV_DIR" && pwd)"
PY="$ENV_DIR/bin/python3"

# Creating a venv from /opt/venv does not automatically inherit /opt/venv's
# packages. Register those directories explicitly in the NEW venv, including
# their .pth files (needed by some container packages). The image stays read-only.
"$CONTAINER_PYTHON" - "$PY" "$ENV_DIR" <<'PY'
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
from importlib.metadata import version

python, env_dir = sys.argv[1:]
destination = Path(subprocess.check_output(
    [python, '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'],
    text=True).strip())
paths = sorted({sysconfig.get_path('purelib'), sysconfig.get_path('platlib')})
lines = ['import site; site.addsitedir(' + repr(path) + ')' for path in paths]
(destination / 'container_packages.pth').write_text('\n'.join(lines) + '\n', encoding='utf-8')
pins = {name: version(name) for name in ('torch', 'torchvision')}
(Path(env_dir) / 'container-constraints.txt').write_text(
    ''.join(f'{name}=={value}\n' for name, value in pins.items()), encoding='utf-8')
(Path(env_dir) / 'container-packages.json').write_text(
    json.dumps({'site_packages': paths, 'versions': pins}, indent=2) + '\n', encoding='utf-8')
PY

"$PY" -m pip install --no-deps nvidia-physicsnemo==2.1.1 warp-lang tensordict timm
# Pins prevent a resolver from replacing the image's ROCm torch/torchvision.
"$PY" -m pip install --constraint "$ENV_DIR/container-constraints.txt" \
    scipy numpy pyvista tabulate tensorboard torchinfo tqdm rich einops \
    omegaconf hydra-core requests h5py onnx treelib termcolor gitpython \
    s3fs cftime pandas jaxtyping nvtx matplotlib jupyterlab ipykernel \
    huggingface_hub safetensors pyvers cloudpickle orjson importlib-metadata \
    typing_extensions filelock networkx sympy jinja2 fsspec urllib3 packaging

"$PY" - "$ENV_DIR" <<'PY' | tee "$ENV_DIR/INSTALL_LOG.txt"
import json
from pathlib import Path
import sys
import torch
import torchvision
import physicsnemo
import scipy
import pyvista
import hydra
import omegaconf
import warp
import tensordict
import timm
from importlib.metadata import version

pins = json.loads((Path(sys.argv[1]) / 'container-packages.json').read_text())['versions']
for name, expected in pins.items():
    assert version(name) == expected, f'{name} changed from the container version'
assert str(Path(torch.__file__).resolve()).startswith('/opt/venv/'), 'Torch must come from the container'
print('Python:', sys.executable)
print('Torch:', torch.__version__, 'from', torch.__file__)
print('PhysicsNeMo:', physicsnemo.__version__)
print('SciPy:', scipy.__version__)
print('GPU available:', torch.cuda.is_available())
print('Dependency imports passed; no PYTHONPATH required.')
PY
"$PY" -m pip freeze > "$ENV_DIR/installed-packages.txt"
printf '\nShared environment ready for training/inference, design optimization, and inverse fitting.\n'
printf 'Inside this same container, from the physics_nemo directory:\n'
printf '  export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500\n'
printf 'Exercise 1 (training/inference; Adam for the MI300A toy loop):\n'
printf '  FORCE_ADAM_MI300A=1 %q -u 01_training_and_inference/workshop_crash_surrogate.py --epochs 30\n' "$PY"
printf 'Exercise 2 (design optimization):\n'
printf '  %q -u 02_design_optimization/optimize.py --limit-mm 350 --output-dir 02_design_optimization/results/first_run\n' "$PY"
printf 'Exercise 3 (inverse fitting), from physics_nemo:\n'
printf '  %q -u 03_inverse_problems/infer.py --output-dir 03_inverse_problems/results/first_run\n' "$PY"
printf 'Optional activation:\n  source %q\n' "$ENV_DIR/bin/activate"
