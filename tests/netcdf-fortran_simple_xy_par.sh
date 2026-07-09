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

# Compiler, NetCDF flags, and MPI launcher are all keyed on the compiler/MPI
# actually in use -- they MUST stay matched:
#
#   * Cray PrgEnv-cray (PE_ENV=CRAY, cray-netcdf-hdf5parallel): the ftn wrapper
#     injects the NetCDF include/lib paths and links cray-mpich, which is
#     PMI-integrated with srun. Keep ftn (FC set above) + srun, no explicit
#     NetCDF flags.
#
#   * AMD compiler (PE_ENV=AMD on Cray, or any non-Cray AMD system): build with
#     the compiler the custom netcdf-c/netcdf-fortran modules were built with --
#     nf-config --fc, which is the from-source mpich-wrappers mpifort (wrapping
#     flang-new). Do NOT use the Cray ftn wrapper here: ftn links cray-mpich's
#     /opt/cray/pe/lib64/libmpifort_amd.so.12, which was compiled with CLASSIC
#     flang and hard-NEEDs libpgmath.so / libflang.so. Those .so are absent in
#     the flang-new toolchain, so every rank dies at startup with
#     "libpgmath.so: cannot open shared object file". mpifort instead links the
#     mpich-wrappers libmpifort_amd (flang-new, self-contained) and supplies a
#     flang-new-format mpi.mod. nf-config also reports the right -I (netcdf.mod
#     lives in netcdf-fortran/include) and -l flags.
#     Launch with that MPI's OWN mpiexec/mpirun: srun cannot wire up the
#     from-source MPICH (each rank would get a singleton MPI_COMM_WORLD), exactly
#     like the HIP Jacobi / TAU tests.
if [ "$PE_ENV" = "CRAY" ]; then
   NETCDF_LIBS=""
   MPIRUN="srun"
else
   FC=`nf-config --fc`
   NETCDF_LIBS="$(nf-config --fflags) $(nf-config --flibs)"
   MPI_BINDIR=$(dirname "$(command -v "$FC" 2>/dev/null)" 2>/dev/null)
   if [ -n "${MPI_BINDIR}" ] && [ -x "${MPI_BINDIR}/mpirun" ]; then
      MPIRUN="${MPI_BINDIR}/mpirun"
   elif [ -n "${MPI_BINDIR}" ] && [ -x "${MPI_BINDIR}/mpiexec" ]; then
      MPIRUN="${MPI_BINDIR}/mpiexec"
   elif [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
      MPIRUN="srun"
   else
      MPIRUN="mpirun"
   fi
fi

# --oversubscribe is an OpenMPI flag (lets >1 rank share a slot); MPICH's hydra
# mpiexec rejects it. Only add it for OpenMPI.
MPI_OPTS=""
if command -v ompi_info >/dev/null 2>&1; then
   MPI_OPTS="--oversubscribe"
fi

SRC_DIR=$(pwd)
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cd ${BUILD_DIR}

git clone https://github.com/Unidata/netcdf-fortran.git
$FC  ./netcdf-fortran/examples/F90/simple_xy_par_wr.F90 -o simple_xy_par_wf ${NETCDF_LIBS}
${MPIRUN} -n 4 ${MPI_OPTS} ./simple_xy_par_wf
ncdump simple_xy_par.nc
