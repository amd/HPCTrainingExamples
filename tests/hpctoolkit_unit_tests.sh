#!/bin/bash

module purge
module load rocm
module load hpctoolkit

cd ${HPCTOOLKIT_PATH}
meson test




