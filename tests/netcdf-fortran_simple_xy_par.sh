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
   if [ "$PE_ENV" = "AMD" ]; then
      echo "Using the AMD compiler"
      # Custom amdflang-capable NetCDF build (not the Cray-authored module).
      module load netcdf-c
      module load netcdf-fortran
   elif [ "$PE_ENV" = "CRAY" ]; then
      echo "Using the Cray compiler"
      # Cray-authored module; the ftn wrapper injects NetCDF paths for the
      # active PrgEnv, so no explicit flags are needed.
      module load cray-netcdf-hdf5parallel
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
   module load netcdf-c
   module load netcdf-fortran
   module load openmpi
fi

if [[ ${HDF5_ENABLE_PARALLEL} == "OFF" ]]; then
   # NETCDF has not been built with parallel I/O support
   echo "Skip"
fi

if [ -n "${CRAY_MPICH_VERSION:-}" ]; then
  echo "Detected Cray MPICH: using srun launcher"
  MPIRUN="srun"
else
  MPIRUN="mpirun"
fi

# NetCDF compile/link flags.
#   * Cray PrgEnv-cray (cray-netcdf-hdf5parallel): the ftn wrapper injects the
#     NetCDF include/lib paths automatically -> no explicit flags, keep ftn.
#   * Everywhere else -- Cray PrgEnv-amd-new with the custom netcdf-c /
#     netcdf-fortran modules, or a non-Cray AMD system -- nf-config reports the
#     right -I (netcdf.mod lives in netcdf-fortran/include; ftn/amdflang do not
#     honor CPATH/FPATH for .mod lookup) and -l flags. nf-config --flibs emits
#     -lnetcdff -lnetcdf but omits netcdf-c's -L (netcdf-c is a separate install
#     prefix); the netcdf modules now put both lib dirs on LIBRARY_PATH so the
#     link resolves without an explicit -L. flang-new (AMD_COMPILER_TYPE=DEFAULT,
#     set by the amd-new / PrgEnv-amd-new modules) links its Fortran runtime
#     statically, so the binary needs no libpgmath.so/libflang.so at run time.
#     On non-Cray, use the compiler netcdf-fortran was built with (nf-config
#     --fc); on Cray keep the ftn wrapper set above.
if [ "$PE_ENV" = "CRAY" ]; then
   NETCDF_LIBS=""
else
   if [[ -z "$CRAYPE_VERSION" && ! -f /etc/cray-release ]]; then
      FC=`nf-config --fc`
   fi
   NETCDF_LIBS="$(nf-config --fflags) $(nf-config --flibs)"
fi

SRC_DIR=$(pwd)
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cd ${BUILD_DIR}

git clone https://github.com/Unidata/netcdf-fortran.git
$FC  ./netcdf-fortran/examples/F90/simple_xy_par_wr.F90 -o simple_xy_par_wf ${NETCDF_LIBS}
${MPIRUN} -n 4 --oversubscribe ./simple_xy_par_wf
ncdump simple_xy_par.nc
