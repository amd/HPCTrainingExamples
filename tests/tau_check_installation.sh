#!/bin/bash

module purge
module load rocm
module load tau

git clone https://github.com/UO-OACISS/tau2.git
cd tau2

./tau_validate ${TAU_LIB}

