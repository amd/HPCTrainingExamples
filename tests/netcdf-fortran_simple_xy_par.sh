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

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   if [ "$PE_ENV" = "AMD" ]; then
      # The Cray ftn wrapper does NOT auto-inject the custom amdflang
      # netcdf-fortran module's paths (only the Cray-authored cray-netcdf
      # module is wired into ftn). amdflang does not honor FPATH/CPATH for
      # Fortran .mod lookup, so netcdf.mod is not found and every nf90_* symbol
      # is undeclared. Pass the include dir (where netcdf.mod lives) and link
      # libs explicitly. netcdf-c lives in a SEPARATE prefix (NETCDF_C_ROOT)
      # from netcdf-fortran, so both -L dirs are required; nf-config --flibs
      # alone emits -lnetcdf without netcdf-c's -L (ld.lld: cannot find
      # -lnetcdf). Mirrors netcdf-fortran_pres_temp_4D.sh.
      NETCDF_LIBS="-I${NETCDF_F_ROOT}/include -L${NETCDF_F_ROOT}/lib -lnetcdff -L${NETCDF_C_ROOT}/lib -lnetcdf"
   else
      # Cray PrgEnv + cray-netcdf-hdf5parallel: the ftn wrapper injects paths.
      NETCDF_LIBS=""
   fi
else
   NETCDF_LIBS="-I${NETCDF_F_ROOT}/include -L${NETCDF_F_ROOT}/lib -lnetcdff -L${PNETCDF_ROOT}/lib -lpnetcdf"
   # use the compiler used to build netcdf-fortran
   FC=`nf-config --fc`
fi

SRC_DIR=$(pwd)
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cd ${BUILD_DIR}

git clone https://github.com/Unidata/netcdf-fortran.git
$FC  ./netcdf-fortran/examples/F90/simple_xy_par_wr.F90 -o simple_xy_par_wf ${NETCDF_LIBS}
${MPIRUN} -n 4 --oversubscribe ./simple_xy_par_wf
ncdump simple_xy_par.nc
