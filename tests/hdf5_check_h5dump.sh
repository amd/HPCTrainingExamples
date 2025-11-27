#!/bin/bash


if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load hdf5
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




