#!/bin/bash

wget https://github.com/OpenMP/Examples/archive/refs/tags/v5.2.1.tar.gz
tar -xzvf v5.2.1.tar.gz
cd Examples-5.2.1/sources
sed -i -e 's/^comp_c=""/comp_c="$CC"/' \
       -e 's/^comp_cpp=""/comp_cpp="$CXX"/' \
       -e 's/^comp_f=""/comp_f="$FC"/' \
       -e 's/^omp_flag=""/omp_flag="-fopenmp --offload-arch=native"/' \
     eval_codes

module load amdclang
./eval_codes > amdclang_openmp.out

module load clacc
./eval_codes > clacc_openmp.out

module load aomp
./eval_codes > aomp_openmp.out

diff3 aomp_openmp.out clacc_openmp.out amdclang_openmp.out

sed -e '1,$s!/opt/rocmplus-5.7.2/aomp_18.0-0/bin/!!' aomp_openmp.out >aomp_openmp_edited.out
sed -e '1,$s!/opt/rocm-5.7.2/llvm/bin/!!' -e '1,$s/amdflang/flang/' amdclang_openmp.out > amdclang_openmp_edited.out

diff aomp_openmp_edited.out amdclang_openmp_edited.out
