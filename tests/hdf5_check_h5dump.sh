#!/bin/bash

if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
   if [ -z "$CXX" ]; then
      export CXX=`which CC`
   fi
   if [ -z "$CC" ]; then
      export CC=`which cc`
   fi
   if [ -z "$FC" ]; then
      export FC=`which ftn`
   fi
else
   module list 2>&1 | grep -q -w "rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
   module load amdflang-new >& /dev/null
   if [ "$?" == "1" ]; then
      module load amdclang
   fi
fi

if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
   module load cray-hdf5-parallel
else
   module load hdf5
fi
module load openmpi

if [[ `which mpicc | wc -l` -eq 0 ]]; then
   # this means MPI is not found, but this is a test for parallel HDF5, so we skip
   echo "Skip"
fi


git clone https://github.com/essentialsofparallelcomputing/Chapter16.git

pushd Chapter16/HDF5Examples/hdf5block2d

sed -i '37i target_link_libraries(hdf5block2d m)' CMakeLists.txt
sed -i '37i target_link_libraries(hdf5block2d z)' CMakeLists.txt

mkdir build && cd build && cmake -DHDF5_IS_PARALLEL=ON .. && make

mpirun -n 4 ./hdf5block2d

h5dump -y example.hdf5

popd

rm -rf Chapter16
