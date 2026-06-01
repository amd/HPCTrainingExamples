#!/bin/bash

set -e

ROCMVERSION=7.2.2
module load rocm/$ROCMVERSION

python3 -m venv aivenv --system-site-packages
source aivenv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

