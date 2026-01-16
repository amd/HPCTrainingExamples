#!/bin/bash

pushd $(dirname $0)

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

#Slurm interactive test

#salloc -N 1 -p LocalQ --gpus=1 -t 10:00

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
cd ${REPO_DIR}/HIP/vectorAdd
make vectoradd
./vectoradd
make clean
rm -rf build
mkdir build && cd build
cmake ..
make
./vectoradd
cd ..
rm -rf build
cd

# need to add slurm example

cd ${REPO_DIR}/HIP/hip-stream
make stream
./stream
make clean
rm -rf build
mkdir build && cd build
cmake ..
make
./stream
cd ..
rm -rf build
cd

cd ${REPO_DIR}/HIP/saxpy
make saxpy
./saxpy
make clean
rm -rf build
mkdir build && cd build
cmake ..
make
./saxpy
cd ..
rm -rf build
cd

cd ${REPO_DIR}/HIP/jacobi

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load openmpi

rm -rf build
mkdir build && cd build
cmake ..
make

#salloc -p LocalQ --gpus=2 -n 2 -t 00:10:00
mpirun -n 2 ./Jacobi_hip

cd ..
rm -rf build
cd

cd ${REPO_DIR}/HIPIFY/mini-nbody/cuda
hipify-perl -examine nbody-orig.cu

hipify-perl nbody-orig.cu > nbody-orig.cpp
hipcc -DSHMOO -I../ nbody-orig.cpp -o nbody-orig

./nbody-orig

make clean
cd

wget https://asc.llnl.gov/sites/asc/files/2020-09/pennant-singlenode-cude.tgz
tar -xzvf pennant-singlenode-cude.tgz

cd PENNANT

hipexamine-perl.sh
hipconvertinplace-perl.sh
mv src/HydroGPU.cu src/HydroGPU.hip

sed -i -e 's/CUDA/HIP/' Makefile
sed -i -e '/CXX/s/icpc/hipcc/' Makefile
sed -i -e 's/HIPC/HIPCC/' Makefile
sed -i -e 's/nvcc/hipcc/' Makefile
sed -i -e '/-arch=sm_21/s/HIPCCFLAGS/HIPCCFLAGS_CUDA/' Makefile
sed -i -e '/HIPCCFLAGS +=/s/^/#/' Makefile
sed -i -e 's/-fast//' Makefile
sed -i -e 's/-fno-alias//' Makefile
sed -i -e 's/cu/hip/' Makefile
sed -i -e '/LD/s/$(CXX)/hipcc/' Makefile
sed -i -e 's/^LDFLAGS/#LDFLAGS/' Makefile

sed -i -e 's/cu/hip/' -e '29,32s/^/#/' -e '51,51s/^/#/' -e 's/CUDAC/CXX/g' -e 's/-arch=sm_21 --ptxas-options=-v/-g -std=c++14 -munsafe-fp-atomics/' -e 's/-G -lineinfo//' -e 's/nvcc/hipcc/' -e 's/-L\$(CUDA_INSTALL_PATH)\/lib64 -lcudart//' Makefile

sed -i -e 's/__CUDACC__/__HIPCC__/' -e '75,100s/#else/#elif defined(__CUDACC__)/' -e '22,22a#include <hip/hip_runtime.h>' src/Vec2.hh

sed -i -e '/hip_runtime/d' -e '724,724a#ifdef __CUDACC__' -e '738,738a#endif' src/HydroGPU.hip

cd PENNANT
make
build/pennant test/leblanc/leblanc.pnt

cp ${REPO_DIR}/HIP/saxpy/Makefile .
cp ${REPO_DIR}/HIP/saxpy/CMakeLists.txt .
cp Makefile Makefile.portable
cp CMakeLists.txt CMakeLists.txt.portable
cp ~/Makefile .
cp ~/CMakeLists.txt .

cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/1_saxpy/6_saxpy_targetdata
module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load amdclang
make
./saxpy
make clean

cd

cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/2_reduction/1_reduction_solution
module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load amdclang
make
./freduce
make clean

popd
