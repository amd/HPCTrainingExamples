#!/bin/bash
#echo "Removing old version if it it exists"
rm -rf rocHPCG
#echo "Getting master branch version from https://github.com/ROCm/rocHPCG"
git clone -b master https://github.com/ROCm/rocHPCG
#echo "Building HPCG code"
module load amdclang openmpi rocm
cd rocHPCG/
./install.sh \
  	--with-mpi=/opt/rocmplus-6.1.1/openmpi \
 	--with-rocm=$ROCM_PATH \
  	--gpu-aware-mpi=ON \
  	--with-openmp=ON \
  	--with-memmgmt=ON \
  	--with-memdefrag=ON
echo "running HPCG"
mpirun -n 4 build/release/bin/rochpcg 560 280 280 1800 |& tee rochpcg.out
echo ""
grep "THIS IS NOT A VALID RUN" rochpcg.out
FOM=`grep Final rochpcg.out |awk '{print $3}'`
echo "FOM: $FOM"


