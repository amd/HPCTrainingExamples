#!/bin/bash

mkdir tsp
git clone https://github.com/pkestene/tsp
cd tsp
git checkout 51587
wget -q https://raw.githubusercontent.com/ROCm/roc-stdpar/main/data/patches/tsp/TSP.patch

patch -p1 < TSP.patch

cd stdpar

export HSA_XNACK=1
module load amdclang
export STDPAR_CXX=$CXX
export ROCM_GPU=`rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'`
export STDPAR_TARGET=${ROCM_GPU}

export AMD_LOG_LEVEL=3

if [[ ${ROCM_GPU} =~ "gfx9" ]]; then
   sed -i -e '/--hipstdpar/s/--hipstdpar /--hipstdpar -lstdc++ /' Makefile
else
   sed -i -e '/--hipstdpar/s/--hipstdpar /--hipstdpar --hipstdpar-interpose-alloc -lstdc++ /' Makefile
fi

make tsp_clang_stdpar_gpu
./tsp_clang_stdpar_gpu

make clean
cd ../..
rm -rf tsp
