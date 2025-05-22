#!/bin/bash

usage()
{
    echo ""
    echo "--help : prints this message"
    echo "--remove-after-run : remove all dependencies and builds after run"
    echo ""
    exit
}

send-error()
{
    usage
    echo -e "\nError: ${@}"
    exit 1
}

reset-last()
{
   last() { send-error "Unsupported argument :: ${1}"; }
}

n=0
while [[ $# -gt 0 ]]
do
   case "${1}" in
      "--remove-after-run")
          shift
          REMOVE_AFTER=1
          reset-last
          ;;
     "--help")
          usage
          ;;
      "--*")
          send-error "Unsupported argument at position $((${n} + 1)) :: ${1}"
          ;;
      *)
         last ${1}
         ;;
   esac
   n=$((${n} + 1))
   shift
done


REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
pushd $REPO_DIR

cd Libraries/RocSolverRf

mkdir dependencies && cd dependencies


module load rocm

mkdir lapack_install
export LAPACK_PATH=$PWD/lapack_install
mkdir suitesparse_install
export SUITESPARSE_PATH=$PWD/suitesparse_install

git clone https://github.com/Reference-LAPACK/lapack
cd lapack
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$LAPACK_PATH -DBUILD_SHARED_LIBS=ON ..
make -j
make install

cd ../../

LD_LIBRARY_PATH_BACKUP=$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LAPACK_PATH/lib

git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git --branch v7.10.1
cd SuiteSparse
sed -i 's/umfpack\;paru\;rbio/umfpack\;rbio/g' CMakeLists.txt
mkdir build_test && cd build_test
CMAKE_OPTIONS="-DBLA_VENDOR=LAPACK"
cmake -DCMAKE_INSTALL_PREFIX=$SUITESPARSE_PATH ..
make -j
make install
cd ../../../

export SS_LIB_DIR=$SUITESPARSE_PATH/lib
export SS_INCLUDE_DIR=$SUITESPARSE_PATH/include/suitesparse

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SS_LIB_DIR

make

wget https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_5_k101.tar.gz
tar -xvf  af_5_k101.tar.gz
./klu_example --matrix1 af_5_k101/af_5_k101.mtx --matrix2 af_5_k101/af_5_k101.mtx --matrix3 af_5_k101/af_5_k101.mtx

if [[ "${REMOVE_AFTER}" == "1" ]]; then
   make clean
   rm -rf dependencies
   rm -rf af_5_*
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_BACKUP
   popd
fi



