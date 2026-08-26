#!/bin/bash
# One-time login-node setup for novice profile.sh on systems without a
# roofline-extractor module. Clone the repo first, then point ROOFLINE_EXTRACTOR
# at it in env.sh.
#
#   git clone https://github.com/AMD-HPC/rooflineExtractor.git ~/rooflineExtractor
#   export ROOFLINE_EXTRACTOR=$HOME/rooflineExtractor
#   ./setup_roofline_extractor_venv.sh

set -e
SW_ROOT="$(cd "$(dirname "$0")" && pwd)"
: "${ROOFLINE_EXTRACTOR:?Set ROOFLINE_EXTRACTOR to your rooflineExtractor checkout}"

export ROOFLINE_VENV="${ROOFLINE_VENV:-${HOME}/roofline-venv}"

python3 -m venv "${ROOFLINE_VENV}"
source "${ROOFLINE_VENV}/bin/activate"
python3 -m pip install -r "${ROOFLINE_EXTRACTOR}/requirements.txt"

echo "Created ${ROOFLINE_VENV}"
