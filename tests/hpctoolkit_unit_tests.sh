#!/bin/bash

module purge
module load rocm
module load hpctoolkit

pip install pipx
pipx install 'meson>=1.3.2'
export PATH=$HOME/.local/bin:$PATH
git clone https://gitlab.com/hpctoolkit/hpctoolkit.git
cd hpctoolkit
meson test

cd ..
rm -rf hpctoolkit
pipx uninstall meson
pip uninstall -y pipx





