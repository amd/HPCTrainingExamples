#!/bin/bash

module purge
module load rocm
module load hpctoolkit

pip install pipx
pipx install 'meson>=1.3.2'
export PATH=$HOME/.local/bin:$PATH

cd ${HPCTOOLKIT_PATH}
meson test

pipx uninstall meson
pip uninstall -y pipx





