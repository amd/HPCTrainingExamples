#!/bin/bash

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   if [ -z "$CXX" ]; then
      export CXX=`which CC`
   fi
   if [ -z "$CC" ]; then
      export CC=`which cc`
   fi
   if [ -z "$FC" ]; then
      export FC=`which ftn`
   fi
   if [ -z "$HIPCC" ]; then
      export HIPCC=`which hipcc`
   fi
else
   module -t list 2>&1 | grep -q "^rocm"
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

# Module + NetCDF compile/link flags.
#   * Cray + Cray (CCE) compiler: cray-netcdf-hdf5parallel + the ftn wrapper
#     inject the paths automatically -> no explicit flags, keep ftn.
#   * Cray + AMD compiler, or off-Cray: build with the compiler the custom
#     netcdf-fortran module was built with -- nf-config --fc, the from-source
#     mpich-wrappers mpifort (wrapping flang-new). Do NOT keep the Cray ftn
#     wrapper for the AMD compiler: ftn always links cray-mpich's
#     /opt/cray/pe/lib64/libmpifort_amd.so.12 (even for serial code), which was
#     compiled with CLASSIC flang and hard-NEEDs libpgmath.so / libflang.so --
#     absent in the flang-new toolchain, so the binary dies at startup with
#     "libpgmath.so: cannot open shared object file". mpifort links the
#     mpich-wrappers libmpifort_amd (flang-new, self-contained) instead.
#     nf-config also reports the right -I (netcdf.mod; ftn/amdflang ignore
#     CPATH/FPATH for .mod lookup) and -l flags. nf-config --flibs emits
#     -lnetcdff -lnetcdf but omits netcdf-c's -L (separate prefix); the netcdf
#     modules put both lib dirs on LIBRARY_PATH so the link resolves.
if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   FTN_VERSION=$("$FC" --version 2>&1)
   if [[ "$FTN_VERSION" == *Cray* ]]; then
      module load cray-netcdf-hdf5parallel
      NETCDF_LIBS=""
   else
      module load netcdf-fortran
      FC=`nf-config --fc`
      NETCDF_LIBS="$(nf-config --fflags) $(nf-config --flibs)"
   fi
else
   module load netcdf-fortran
   FC=`nf-config --fc`
   NETCDF_LIBS="$(nf-config --fflags) $(nf-config --flibs)"
fi

SRC_DIR=$(pwd)
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cd ${BUILD_DIR}

git clone https://github.com/Unidata/netcdf-fortran.git
$FC ./netcdf-fortran/examples/F90/pres_temp_4D_wr.F90 ${NETCDF_LIBS} -o pres_temp_4D_wr
$FC ./netcdf-fortran/examples/F90/pres_temp_4D_rd.F90 ${NETCDF_LIBS} -o pres_temp_4D_rd
./pres_temp_4D_wr
./pres_temp_4D_rd
